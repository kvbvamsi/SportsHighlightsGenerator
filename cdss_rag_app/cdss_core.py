"""
cdss_core.py
------------
Core, UI-independent logic for the AI-Powered Clinical Decision Support System (CDSS).

Responsibilities:
  - Load a medicine knowledge base from JSON (structured) and/or PDF (unstructured) sources
  - Build / persist / reload a FAISS vector store using Ollama embeddings
  - Retrieve relevant medicines for a patient case via a RAG chain (Llama 3.2 via Ollama)
  - Fall back to a direct (non-retrieved) LLM call when the knowledge base has no good match,
    and clearly label such answers as LLM-generated / not verified
  - Interpret lab report values against reference ranges
  - Extract lab values from raw PDF/report text (regex based, offline, no LLM required)
  - Screen for emergency ("red-flag") symptoms
  - Run a naive contraindication check against patient history/allergies

This module has no Streamlit dependency so it can be unit-tested or reused (e.g. in the notebook,
a CLI, or a different front end) independently of app.py.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
EMBEDDING_ENDPOINT = f"{OLLAMA_BASE_URL}/api/embeddings"

# FAISS default distance is L2 (lower = more similar). Tune per embedding model / dataset.
SIMILARITY_THRESHOLD = 1.0

REQUIRED_MEDICINE_FIELDS = [
    "medicine_name", "generic_name", "drug_class", "disease_condition",
    "common_symptoms", "relevant_lab_markers", "standard_dosage",
    "contraindications", "side_effects",
]


# ---------------------------------------------------------------------------
# Ollama connectivity check
# ---------------------------------------------------------------------------

def ollama_is_reachable(base_url: str = OLLAMA_BASE_URL) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class OllamaEmbeddings(Embeddings):
    """Thin wrapper around Ollama's /api/embeddings endpoint."""

    def __init__(self, model: str = OLLAMA_MODEL, url: str = EMBEDDING_ENDPOINT):
        self.model = model
        self.url = url

    def _embed(self, text: str):
        payload = {"model": self.model, "prompt": text}
        response = requests.post(
            self.url, headers={"Content-Type": "application/json"},
            data=json.dumps(payload), timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama embeddings error {response.status_code}: {response.text}")
        data = response.json()
        if "embedding" not in data:
            raise RuntimeError(f"'embedding' key missing in Ollama response: {data}")
        return data["embedding"]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def get_llm(model: str = OLLAMA_MODEL, temperature: float = 0.0) -> Ollama:
    return Ollama(model=model, temperature=temperature)


# ---------------------------------------------------------------------------
# Structured (JSON) knowledge base loading
# ---------------------------------------------------------------------------

def _normalize_record(rec: dict) -> dict:
    """Fill any missing expected fields with 'not specified' so downstream text rendering never KeyErrors."""
    out = dict(rec)
    for field_name in REQUIRED_MEDICINE_FIELDS:
        out.setdefault(field_name, "not specified")
    return out


def load_medicine_json(source) -> list[dict]:
    """
    Load structured medicine records from a JSON file.
    `source` may be a filesystem path (str/Path), a file-like object, or raw bytes/str.
    Expected format: a JSON array of objects, each with (at minimum) `medicine_name` and
    `disease_condition`. Missing optional fields are filled with 'not specified'.
    """
    if isinstance(source, (str, Path)) and Path(source).exists():
        raw = Path(source).read_text(encoding="utf-8")
    elif isinstance(source, (bytes, bytearray)):
        raw = source.decode("utf-8")
    elif hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
    else:
        raw = str(source)

    data = json.loads(raw)
    if isinstance(data, dict):
        # allow either a bare list or {"medicines": [...]}
        data = data.get("medicines", [data])
    if not isinstance(data, list):
        raise ValueError("Medicine JSON must be a list of records (or {'medicines': [...]}).")

    records = [_normalize_record(r) for r in data]
    for r in records:
        if r["medicine_name"] == "not specified":
            raise ValueError(f"Record missing required field 'medicine_name': {r}")
    return records


def record_to_text(row: dict) -> str:
    """Render a structured medicine record as a natural-language passage for embedding."""
    return (
        f"Medicine: {row['medicine_name']} ({row['generic_name']}), class: {row['drug_class']}.\n"
        f"Indicated for: {row['disease_condition']}.\n"
        f"Common symptoms treated: {row['common_symptoms']}.\n"
        f"Relevant lab markers: {row['relevant_lab_markers']}.\n"
        f"Standard dosage: {row['standard_dosage']}.\n"
        f"Contraindications: {row['contraindications']}.\n"
        f"Side effects: {row['side_effects']}."
    )


def json_records_to_documents(records: list[dict]) -> list[Document]:
    return [
        Document(page_content=record_to_text(r), metadata={**r, "source_type": "structured_json"})
        for r in records
    ]


# ---------------------------------------------------------------------------
# Unstructured (PDF) knowledge base loading
# ---------------------------------------------------------------------------

def extract_text_from_pdf(source) -> str:
    """
    Extract raw text from a PDF. `source` may be a filesystem path, a file-like object
    (e.g. Streamlit's UploadedFile), or raw bytes.
    """
    if isinstance(source, (bytes, bytearray)):
        reader = PdfReader(io.BytesIO(source))
    else:
        reader = PdfReader(source)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def pdf_to_documents(source, source_name: str, chunk_size: int = 500, chunk_overlap: int = 60) -> list[Document]:
    """
    Extract + chunk a PDF (e.g. a drug leaflet, clinical guideline, or formulary excerpt) into
    Documents for the vector store. Unlike the structured JSON path, these chunks carry no
    guaranteed schema — they supplement the curated knowledge base with free-text medical content.
    """
    text = extract_text_from_pdf(source)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return [
        Document(
            page_content=chunk,
            metadata={"source_type": "pdf_unstructured", "source_name": source_name, "medicine_name": None},
        )
        for chunk in chunks
    ]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def build_knowledge_base(documents: list[Document], embeddings: Embeddings) -> FAISS:
    if not documents:
        raise ValueError("No documents supplied to build the knowledge base.")
    return FAISS.from_documents(documents, embeddings)


def add_documents(vectorstore: FAISS, documents: list[Document]) -> FAISS:
    if documents:
        vectorstore.add_documents(documents)
    return vectorstore


def save_knowledge_base(vectorstore: FAISS, persist_dir: str) -> None:
    vectorstore.save_local(persist_dir)


def load_knowledge_base(persist_dir: str, embeddings: Embeddings) -> FAISS:
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)


# ---------------------------------------------------------------------------
# RAG chain + LLM fallback
# ---------------------------------------------------------------------------

CLINICAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a clinical decision support assistant helping a healthcare professional.\n"
        "Use ONLY the medicine information in the context below to answer. Do not use outside knowledge.\n"
        "If the context does not contain a medicine that adequately matches the patient's condition, "
        "respond with exactly the single token: NOT_FOUND_IN_DATABASE\n\n"
        "Context:\n{context}\n\n"
        "Patient case:\n{question}\n\n"
        "If the context is sufficient, respond with:\n"
        "1. Recommended medicine(s) and why they match the case\n"
        "2. Standard dosage\n"
        "3. Key contraindications / warnings relevant to this patient\n\n"
        "Answer:"
    ),
)

FALLBACK_PROMPT_TEMPLATE = (
    "You are a clinical decision support assistant. A patient case was NOT found in the curated, "
    "verified medicine database. Using your general medical knowledge, suggest an appropriate class "
    "of medicine / treatment approach for the following case. Be concise (medicine class, general "
    "dosing principle, and key precautions). Do not fabricate a specific curated source.\n\n"
    "Patient case:\n{question}\n\nAnswer:"
)


def build_qa_chain(llm: Ollama, vectorstore: FAISS, k: int = 3) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={"prompt": CLINICAL_PROMPT},
        return_source_documents=True,
    )


def get_recommendation(vectorstore: FAISS, qa_chain: RetrievalQA, llm: Ollama, query: str,
                        similarity_threshold: float = SIMILARITY_THRESHOLD) -> dict:
    """Retrieval-first, LLM-fallback-second. Always labels the source of the answer."""
    scored_docs = vectorstore.similarity_search_with_score(query, k=3)
    best_score = scored_docs[0][1] if scored_docs else float("inf")

    result = qa_chain.invoke({"query": query})
    answer = result["result"].strip()

    used_fallback = ("NOT_FOUND_IN_DATABASE" in answer) or (best_score > similarity_threshold)

    matched_medicines = [
        d.metadata.get("medicine_name") for d, _ in scored_docs if d.metadata.get("medicine_name")
    ]
    matched_sources = [
        d.metadata.get("source_name", "curated_database") for d, _ in scored_docs
    ]

    if not used_fallback:
        return {
            "source": "RETRIEVED_FROM_DATABASE",
            "answer": answer,
            "matched_medicines": matched_medicines,
            "matched_sources": matched_sources,
            "best_similarity_score": best_score,
        }

    fallback_answer = llm.invoke(FALLBACK_PROMPT_TEMPLATE.format(question=query))
    return {
        "source": "LLM_GENERATED_NOT_VERIFIED",
        "answer": (
            "\u26a0\ufe0f GENERATED BY LLM \u2014 NOT RETRIEVED FROM THE VERIFIED MEDICINE DATABASE. "
            "This suggestion has not been cross-checked against curated clinical data and MUST be "
            "reviewed by a licensed clinician before any action.\n\n" + fallback_answer.strip()
        ),
        "matched_medicines": [],
        "matched_sources": [],
        "best_similarity_score": best_score,
    }


# ---------------------------------------------------------------------------
# Lab report interpretation
# ---------------------------------------------------------------------------

LAB_REFERENCE_RANGES = {
    "Fasting Glucose":   {"unit": "mg/dL", "low": 70,   "high": 99,   "high_meaning": "suggestive of Diabetes Mellitus / Hyperglycemia"},
    "HbA1c":             {"unit": "%",     "low": 4.0,  "high": 5.6,  "high_meaning": "suggestive of Diabetes Mellitus (poor glycemic control)"},
    "Total Cholesterol": {"unit": "mg/dL", "low": 100,  "high": 199,  "high_meaning": "suggestive of Hyperlipidemia"},
    "LDL":               {"unit": "mg/dL", "low": 0,    "high": 99,   "high_meaning": "suggestive of Dyslipidemia / cardiovascular risk"},
    "Triglycerides":     {"unit": "mg/dL", "low": 0,    "high": 149,  "high_meaning": "suggestive of Hypertriglyceridemia"},
    "Hemoglobin":        {"unit": "g/dL",  "low": 12.0, "high": 17.5, "low_meaning": "suggestive of Anemia"},
    "WBC Count":         {"unit": "/uL",   "low": 4000, "high": 11000,"high_meaning": "suggestive of active infection / inflammation"},
    "Creatinine":        {"unit": "mg/dL", "low": 0.6,  "high": 1.3,  "high_meaning": "suggestive of renal impairment"},
    "TSH":               {"unit": "mIU/L", "low": 0.4,  "high": 4.0,  "high_meaning": "suggestive of Hypothyroidism", "low_meaning": "suggestive of Hyperthyroidism"},
    "CRP":                {"unit": "mg/L", "low": 0,    "high": 10,   "high_meaning": "suggestive of infection / inflammation"},
    "ESR":                {"unit": "mm/hr","low": 0,    "high": 22,   "high_meaning": "suggestive of infection / inflammation"},
}

# synonyms used when scanning free-text (PDF-extracted) lab reports
LAB_SYNONYMS = {
    "Fasting Glucose":   ["fasting glucose", "fasting blood sugar", "fbs", "glucose fasting"],
    "HbA1c":             ["hba1c", "hb a1c", "glycated hemoglobin", "glycosylated hemoglobin"],
    "Total Cholesterol": ["total cholesterol", "cholesterol total", "cholesterol, total"],
    "LDL":               ["ldl cholesterol", "ldl-c", "ldl"],
    "Triglycerides":     ["triglycerides", "tg"],
    "Hemoglobin":        ["hemoglobin", "haemoglobin", "hb\\b"],
    "WBC Count":         ["wbc count", "total leukocyte count", "tlc", "white blood cell count", "wbc"],
    "Creatinine":        ["serum creatinine", "creatinine"],
    "TSH":               ["tsh", "thyroid stimulating hormone"],
    "CRP":               ["crp", "c-reactive protein", "c reactive protein"],
    "ESR":               ["esr", "erythrocyte sedimentation rate"],
}


def interpret_lab_report(lab_values: dict) -> list[str]:
    """Compare raw lab values (dict of {test_name: numeric_value}) against reference ranges."""
    findings = []
    for test, value in lab_values.items():
        ref = LAB_REFERENCE_RANGES.get(test)
        if ref is None:
            findings.append(f"{test}: {value} (no reference range configured, reported as-is)")
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            findings.append(f"{test}: {value} (could not parse as a number, skipped)")
            continue
        if value > ref["high"]:
            meaning = ref.get("high_meaning", "abnormal, clinical correlation advised")
            findings.append(f"{test}: {value} {ref['unit']} \u2014 HIGH (normal {ref['low']}-{ref['high']}) \u2014 {meaning}")
        elif value < ref["low"]:
            meaning = ref.get("low_meaning", "abnormal, clinical correlation advised")
            findings.append(f"{test}: {value} {ref['unit']} \u2014 LOW (normal {ref['low']}-{ref['high']}) \u2014 {meaning}")
        else:
            findings.append(f"{test}: {value} {ref['unit']} \u2014 within normal range")
    return findings


def extract_lab_values_from_text(text: str) -> dict:
    """
    Best-effort, offline (no LLM) extraction of known lab test values from raw report text
    (e.g. text extracted from an uploaded PDF lab report). Looks for `<test name> ... <number>`
    patterns and returns the first match per test. This is intentionally simple/regex-based so it
    works even when Ollama is unavailable; the caller can let the user review/edit before use.
    """
    found = {}
    normalized = text.replace("\n", " ")
    for canonical, synonyms in LAB_SYNONYMS.items():
        for syn in synonyms:
            pattern = re.compile(syn + r"[^0-9\-]{0,20}(-?\d+\.?\d*)", re.IGNORECASE)
            m = pattern.search(normalized)
            if m:
                try:
                    found[canonical] = float(m.group(1))
                except ValueError:
                    pass
                break
    return found


# ---------------------------------------------------------------------------
# Safety layer: emergency red-flag screening
# ---------------------------------------------------------------------------

RED_FLAG_KEYWORDS = [
    "chest pain", "crushing chest", "difficulty breathing", "shortness of breath at rest",
    "slurred speech", "facial droop", "one sided weakness", "loss of consciousness",
    "unconscious", "severe bleeding", "uncontrolled bleeding", "seizure",
    "suicidal", "self harm", "coughing blood", "blue lips", "severe allergic reaction",
    "anaphylaxis", "unable to breathe",
]


def check_red_flags(symptoms: str) -> Optional[str]:
    text = (symptoms or "").lower()
    matched = [kw for kw in RED_FLAG_KEYWORDS if kw in text]
    if matched:
        return (
            "EMERGENCY WARNING: symptom description matches high-risk red-flag pattern(s): "
            f"{', '.join(matched)}. This system will NOT recommend outpatient medication. "
            "Advise immediate emergency care / call local emergency services."
        )
    return None


# ---------------------------------------------------------------------------
# Naive contraindication check
# ---------------------------------------------------------------------------

def check_contraindications(vectorstore: FAISS, matched_medicine_names: list[str], medical_history: str) -> list[str]:
    """Keyword-level cross-reference of patient history/allergies against each matched medicine's
    contraindications field (only meaningful for structured-JSON-sourced medicines)."""
    warnings = []
    history_lower = (medical_history or "").lower()
    if not matched_medicine_names or not history_lower:
        return warnings

    # pull contraindications back out of the vectorstore's docstore metadata
    for name in matched_medicine_names:
        for doc_id in list(vectorstore.docstore._dict.keys()):  # type: ignore[attr-defined]
            doc = vectorstore.docstore._dict[doc_id]  # type: ignore[attr-defined]
            if doc.metadata.get("medicine_name") == name and doc.metadata.get("contraindications"):
                terms = [t.strip() for t in doc.metadata["contraindications"].split(",")]
                for term in terms:
                    key_words = [w for w in term.split() if len(w) > 4]
                    if key_words and any(w in history_lower for w in key_words):
                        warnings.append(f"POSSIBLE CONTRAINDICATION: {name} vs patient history term '{term}'")
                break
    return warnings


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

@dataclass
class PatientCase:
    symptoms: str
    medical_history: str = ""
    diagnosed_disease: Optional[str] = None
    lab_values: dict = field(default_factory=dict)


def run_cdss_pipeline(vectorstore: FAISS, qa_chain: RetrievalQA, llm: Ollama, case: PatientCase) -> dict:
    red_flag = check_red_flags(case.symptoms)
    if red_flag:
        return {"status": "EMERGENCY", "message": red_flag}

    lab_findings = interpret_lab_report(case.lab_values) if case.lab_values else []

    query_parts = [f"Symptoms: {case.symptoms}."]
    if case.diagnosed_disease:
        query_parts.append(f"Diagnosed disease: {case.diagnosed_disease}.")
    if case.medical_history:
        query_parts.append(f"Medical history: {case.medical_history}.")
    if lab_findings:
        query_parts.append("Lab findings: " + "; ".join(lab_findings) + ".")
    query = " ".join(query_parts)

    recommendation = get_recommendation(vectorstore, qa_chain, llm, query)
    contraindication_warnings = check_contraindications(
        vectorstore, recommendation["matched_medicines"], case.medical_history
    )

    return {
        "status": "OK",
        "query_used": query,
        "lab_findings": lab_findings,
        "source": recommendation["source"],
        "recommendation": recommendation["answer"],
        "matched_medicines": recommendation["matched_medicines"],
        "matched_sources": recommendation["matched_sources"],
        "contraindication_warnings": contraindication_warnings,
    }
