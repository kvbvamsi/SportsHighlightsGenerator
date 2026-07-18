# AI-Powered Clinical Decision Support System (CDSS) Using RAG

**Project Documentation**
Version 1.0 | GenAI Applications Project

---

## 1. Overview

### 1.1 Objective

This project implements an intelligent **Clinical Decision Support System (CDSS)** that recommends
appropriate medicines to a healthcare professional based on a patient's:

- Symptoms
- Medical history / known allergies
- Laboratory report values
- Diagnosed disease (if known)

The system is built on a **Retrieval-Augmented Generation (RAG)** architecture: it first tries to
answer from a curated, verifiable medicine knowledge base, and only falls back to the open-weight
**Llama 3.2** model (served locally via **Ollama**) when the knowledge base has no adequate match —
in which case the answer is explicitly labeled as LLM-generated rather than retrieved.

### 1.2 Problem Being Solved

Clinicians spend significant time cross-referencing lab values, symptoms, and drug references before
arriving at a medication decision. Key challenges this project addresses architecturally:

| Challenge | How the system addresses it |
|---|---|
| Large volume of lab parameters | A reference-range engine automatically flags abnormal values and states their clinical implication |
| Overlapping medicine indications | Semantic retrieval (embeddings) matches a case description to the closest indication, not just keyword overlap |
| Knowledge base is never complete | LLM fallback covers cases outside the curated data, but is clearly labeled as unverified |
| Drug interactions / contraindications | A keyword-based safety check cross-references patient history against each matched medicine |
| Data entry friction | JSON and PDF ingestion (both knowledge base and patient input) instead of hardcoded data |

### 1.3 Key Design Decision: Retrieval-First, LLM-Fallback-Second

This is the core architectural requirement and shapes every other design choice: **the system must
never silently substitute a generic LLM answer for a verified one.** Concretely:

1. Every query is run against the FAISS-retrieved knowledge base first, using a prompt that forces
   the model to answer *only* from the retrieved context.
2. If the model can't find a match, it is instructed to emit a literal sentinel token
   (`NOT_FOUND_IN_DATABASE`) instead of guessing.
3. Independently, the raw vector-similarity score of the best match is also checked against a
   threshold — this catches cases where the model might otherwise hallucinate a plausible-sounding
   answer from weak context.
4. Only if *either* signal indicates no good match does the system call the LLM directly, and the
   response is prefixed with an explicit warning and tagged `LLM_GENERATED_NOT_VERIFIED` in the
   returned data structure — this label is programmatically checkable, not just a string in prose.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT UI (app.py)                        │
│  ┌────────────────────────┐        ┌──────────────────────────────────┐  │
│  │ Knowledge Base Builder  │        │        Patient Case Input        │  │
│  │  - Upload JSON          │        │  - Manual form                   │  │
│  │  - Upload PDF           │        │  - Upload patient case JSON      │  │
│  │  - Bundled sample data  │        │  - Upload lab report PDF         │  │
│  └───────────┬──────────────┘        └───────────────┬──────────────────┘  │
└──────────────┼───────────────────────────────────────┼─────────────────────┘
               │                                        │
               ▼                                        ▼
┌──────────────────────────────┐        ┌──────────────────────────────────┐
│   DATA INGESTION LAYER        │        │   PRE-PROCESSING LAYER            │
│  load_medicine_json()         │        │  extract_text_from_pdf()          │
│  pdf_to_documents()           │        │  extract_lab_values_from_text()   │
│  record_to_text()             │        │  interpret_lab_report()           │
└───────────────┬────────────────┘        │  check_red_flags()                │
                │                          └───────────────┬──────────────────┘
                ▼                                          │
┌──────────────────────────────┐                           │
│   EMBEDDING + VECTOR STORE    │                           │
│  OllamaEmbeddings (Llama 3.2) │                           │
│  FAISS.from_documents()       │                           │
└───────────────┬────────────────┘                           │
                │                                            │
                ▼                                            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        RAG RETRIEVAL + LLM FALLBACK                        │
│  qa_chain = RetrievalQA(llm, retriever, CLINICAL_PROMPT)                   │
│  best_score = vectorstore.similarity_search_with_score(query)              │
│  IF answer == NOT_FOUND_IN_DATABASE OR best_score > THRESHOLD:             │
│      -> call llm.invoke(FALLBACK_PROMPT)  =>  LLM_GENERATED_NOT_VERIFIED   │
│  ELSE:                                                                      │
│      -> use retrieved answer               =>  RETRIEVED_FROM_DATABASE     │
└───────────────────────────────┬───────────────────────────────────────────┘
                                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                             SAFETY LAYER                                   │
│  check_contraindications() — patient history vs matched medicine data      │
└───────────────────────────────┬───────────────────────────────────────────┘
                                 ▼
                     Labeled recommendation returned to UI
```

### 2.2 Component Summary

| Layer | Module / Function | Purpose |
|---|---|---|
| Data ingestion | `load_medicine_json`, `pdf_to_documents` | Load knowledge base from JSON and/or PDF, no hardcoding |
| Text rendering | `record_to_text` | Converts a structured record into an embeddable natural-language passage |
| Embeddings | `OllamaEmbeddings` | Wraps Ollama's `/api/embeddings` endpoint for LangChain compatibility |
| Vector store | `build_knowledge_base`, `save_knowledge_base`, `load_knowledge_base` | FAISS index build + optional disk persistence |
| RAG chain | `build_qa_chain`, `CLINICAL_PROMPT` | Retrieval-constrained answer generation |
| Fallback logic | `get_recommendation` | Confidence-gated switch between retrieval and raw LLM |
| Lab interpretation | `LAB_REFERENCE_RANGES`, `interpret_lab_report` | Flags abnormal values with clinical meaning |
| PDF lab extraction | `LAB_SYNONYMS`, `extract_lab_values_from_text` | Regex-based, offline extraction of lab values from report text |
| Safety | `RED_FLAG_KEYWORDS`, `check_red_flags` | Emergency symptom screen, runs before any recommendation |
| Safety | `check_contraindications` | Keyword cross-reference of patient history vs. medicine contraindications |
| Orchestration | `PatientCase`, `run_cdss_pipeline` | Ties every layer together into one callable pipeline |
| UI | `app.py` | Streamlit front end: knowledge-base builder + patient case input + results |

---

## 3. Methodology

### 3.1 Data Ingestion — JSON (Structured Knowledge Base)

Medicine records are no longer hardcoded in the source code. They are loaded at runtime from a
user-supplied or bundled JSON file, validated, and defaulted where fields are missing.

```python
REQUIRED_MEDICINE_FIELDS = [
    "medicine_name", "generic_name", "drug_class", "disease_condition",
    "common_symptoms", "relevant_lab_markers", "standard_dosage",
    "contraindications", "side_effects",
]

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
        data = data.get("medicines", [data])   # allow {"medicines": [...]} too
    if not isinstance(data, list):
        raise ValueError("Medicine JSON must be a list of records (or {'medicines': [...]}).")

    records = [_normalize_record(r) for r in data]
    for r in records:
        if r["medicine_name"] == "not specified":
            raise ValueError(f"Record missing required field 'medicine_name': {r}")
    return records
```

**Design note:** accepting a path, a file-like object (Streamlit's `UploadedFile`), *or* raw bytes
means the same function serves both the CLI/notebook context and the Streamlit upload widget without
branching logic elsewhere.

### 3.2 Data Ingestion — PDF (Unstructured Knowledge Base)

Unstructured references — drug leaflets, formulary excerpts, clinical guideline PDFs — are extracted
with `pypdf` and chunked with LangChain's recursive splitter, so they can be embedded into the *same*
vector store as the structured JSON records.

```python
def extract_text_from_pdf(source) -> str:
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
```

### 3.3 Rendering Structured Records for Embedding

Rather than embedding raw tabular fields, each medicine record is rendered as a natural-language
paragraph — embedding models capture semantic meaning far better from prose than from a flat
key/value dump.

```python
def record_to_text(row: dict) -> str:
    return (
        f"Medicine: {row['medicine_name']} ({row['generic_name']}), class: {row['drug_class']}.\n"
        f"Indicated for: {row['disease_condition']}.\n"
        f"Common symptoms treated: {row['common_symptoms']}.\n"
        f"Relevant lab markers: {row['relevant_lab_markers']}.\n"
        f"Standard dosage: {row['standard_dosage']}.\n"
        f"Contraindications: {row['contraindications']}.\n"
        f"Side effects: {row['side_effects']}."
    )
```

### 3.4 Embeddings — Ollama Wrapper

LangChain's `Embeddings` base class is implemented against Ollama's local `/api/embeddings` endpoint,
so no external embedding API or key is required.

```python
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
```

### 3.5 Vector Store — FAISS

Both JSON-derived and PDF-derived `Document` objects are indexed together, and the index can be
persisted to disk so the (relatively expensive) embedding step doesn't need to repeat on every app
restart.

```python
def build_knowledge_base(documents: list[Document], embeddings: Embeddings) -> FAISS:
    if not documents:
        raise ValueError("No documents supplied to build the knowledge base.")
    return FAISS.from_documents(documents, embeddings)

def save_knowledge_base(vectorstore: FAISS, persist_dir: str) -> None:
    vectorstore.save_local(persist_dir)

def load_knowledge_base(persist_dir: str, embeddings: Embeddings) -> FAISS:
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
```

### 3.6 The Clinical RAG Prompt

The retrieval chain uses a custom `PromptTemplate` that does two things simultaneously: (1) restricts
the model to the retrieved context only, and (2) defines a machine-checkable "no match" signal.

```python
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

def build_qa_chain(llm: Ollama, vectorstore: FAISS, k: int = 3) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={"prompt": CLINICAL_PROMPT},
        return_source_documents=True,
    )
```

### 3.7 Confidence-Gated LLM Fallback

This function is the heart of the "retrieve first, generate second, always disclose" requirement.
Two independent signals — the sentinel token and the FAISS similarity score — must both indicate a
good match before a database-sourced answer is returned.

```python
FALLBACK_PROMPT_TEMPLATE = (
    "You are a clinical decision support assistant. A patient case was NOT found in the curated, "
    "verified medicine database. Using your general medical knowledge, suggest an appropriate class "
    "of medicine / treatment approach for the following case. Be concise (medicine class, general "
    "dosing principle, and key precautions). Do not fabricate a specific curated source.\n\n"
    "Patient case:\n{question}\n\nAnswer:"
)

def get_recommendation(vectorstore, qa_chain, llm, query: str,
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

    if not used_fallback:
        return {
            "source": "RETRIEVED_FROM_DATABASE",
            "answer": answer,
            "matched_medicines": matched_medicines,
            "best_similarity_score": best_score,
        }

    fallback_answer = llm.invoke(FALLBACK_PROMPT_TEMPLATE.format(question=query))
    return {
        "source": "LLM_GENERATED_NOT_VERIFIED",
        "answer": (
            "⚠️ GENERATED BY LLM — NOT RETRIEVED FROM THE VERIFIED MEDICINE DATABASE. "
            "This suggestion has not been cross-checked against curated clinical data and MUST be "
            "reviewed by a licensed clinician before any action.\n\n" + fallback_answer.strip()
        ),
        "matched_medicines": [],
        "best_similarity_score": best_score,
    }
```

**Why two signals instead of one?** The sentinel token relies on the LLM following instructions
correctly; the similarity-score check is a deterministic backstop that fires even if the model
ignores the instruction and answers anyway from weak or irrelevant context.

### 3.8 Lab Report Interpretation

Raw numeric lab values are converted into clinically meaningful findings by comparison against a
reference-range table, rather than being passed to the model as bare numbers.

```python
LAB_REFERENCE_RANGES = {
    "Fasting Glucose":   {"unit": "mg/dL", "low": 70,   "high": 99,   "high_meaning": "suggestive of Diabetes Mellitus / Hyperglycemia"},
    "HbA1c":             {"unit": "%",     "low": 4.0,  "high": 5.6,  "high_meaning": "suggestive of Diabetes Mellitus (poor glycemic control)"},
    "Hemoglobin":        {"unit": "g/dL",  "low": 12.0, "high": 17.5, "low_meaning": "suggestive of Anemia"},
    "TSH":               {"unit": "mIU/L", "low": 0.4,  "high": 4.0,  "high_meaning": "suggestive of Hypothyroidism", "low_meaning": "suggestive of Hyperthyroidism"},
    # ... Total Cholesterol, LDL, Triglycerides, WBC Count, Creatinine, CRP, ESR
}

def interpret_lab_report(lab_values: dict) -> list[str]:
    """Compare raw lab values (dict of {test_name: numeric_value}) against reference ranges."""
    findings = []
    for test, value in lab_values.items():
        ref = LAB_REFERENCE_RANGES.get(test)
        if ref is None:
            findings.append(f"{test}: {value} (no reference range configured, reported as-is)")
            continue
        value = float(value)
        if value > ref["high"]:
            meaning = ref.get("high_meaning", "abnormal, clinical correlation advised")
            findings.append(f"{test}: {value} {ref['unit']} — HIGH (normal {ref['low']}-{ref['high']}) — {meaning}")
        elif value < ref["low"]:
            meaning = ref.get("low_meaning", "abnormal, clinical correlation advised")
            findings.append(f"{test}: {value} {ref['unit']} — LOW (normal {ref['low']}-{ref['high']}) — {meaning}")
        else:
            findings.append(f"{test}: {value} {ref['unit']} — within normal range")
    return findings
```

### 3.9 Extracting Lab Values From an Uploaded PDF

When a patient's lab report is uploaded as a PDF, the raw text is scanned with per-test synonym
patterns to pull out numeric values — deliberately regex-based (not LLM-based) so it works even if
Ollama is briefly unavailable, and so the result is deterministic and reviewable.

```python
LAB_SYNONYMS = {
    "Fasting Glucose": ["fasting glucose", "fasting blood sugar", "fbs", "glucose fasting"],
    "HbA1c":           ["hba1c", "hb a1c", "glycated hemoglobin", "glycosylated hemoglobin"],
    "Hemoglobin":      ["hemoglobin", "haemoglobin", "hb\\b"],
    # ... one synonym list per test in LAB_REFERENCE_RANGES
}

def extract_lab_values_from_text(text: str) -> dict:
    """
    Best-effort, offline (no LLM) extraction of known lab test values from raw report text.
    Looks for `<test name> ... <number>` patterns and returns the first match per test.
    """
    found = {}
    normalized = text.replace("\n", " ")
    for canonical, synonyms in LAB_SYNONYMS.items():
        for syn in synonyms:
            pattern = re.compile(syn + r"[^0-9\-]{0,20}(-?\d+\.?\d*)", re.IGNORECASE)
            m = pattern.search(normalized)
            if m:
                found[canonical] = float(m.group(1))
                break
    return found
```

The Streamlit UI never trusts this extraction blindly — extracted values are shown in an editable
`st.data_editor` table so the user can correct or remove anything before it's used (see §4.3).

### 3.10 Safety Layer — Emergency Red-Flag Screening

Before any retrieval or generation happens, symptom text is screened for patterns that call for
emergency care rather than an outpatient medicine suggestion. This runs unconditionally, first, in
the pipeline.

```python
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
```

### 3.11 Safety Layer — Contraindication Check

For any medicine actually retrieved from the database, its `contraindications` field is
cross-referenced against the patient's stated history/allergies using keyword overlap.

```python
def check_contraindications(vectorstore: FAISS, matched_medicine_names: list[str], medical_history: str) -> list[str]:
    warnings = []
    history_lower = (medical_history or "").lower()
    if not matched_medicine_names or not history_lower:
        return warnings

    for name in matched_medicine_names:
        for doc_id in list(vectorstore.docstore._dict.keys()):
            doc = vectorstore.docstore._dict[doc_id]
            if doc.metadata.get("medicine_name") == name and doc.metadata.get("contraindications"):
                terms = [t.strip() for t in doc.metadata["contraindications"].split(",")]
                for term in terms:
                    key_words = [w for w in term.split() if len(w) > 4]
                    if key_words and any(w in history_lower for w in key_words):
                        warnings.append(f"POSSIBLE CONTRAINDICATION: {name} vs patient history term '{term}'")
                break
    return warnings
```

> This is intentionally a lightweight demonstration, not a real drug-interaction engine — see
> §6 Limitations.

### 3.12 Orchestration — The End-to-End Pipeline

Every layer above is composed into a single callable that the UI (or any other caller) invokes with
one `PatientCase` object.

```python
@dataclass
class PatientCase:
    symptoms: str
    medical_history: str = ""
    diagnosed_disease: Optional[str] = None
    lab_values: dict = field(default_factory=dict)


def run_cdss_pipeline(vectorstore, qa_chain, llm, case: PatientCase) -> dict:
    # 1. Safety screen first, always
    red_flag = check_red_flags(case.symptoms)
    if red_flag:
        return {"status": "EMERGENCY", "message": red_flag}

    # 2. Lab interpretation
    lab_findings = interpret_lab_report(case.lab_values) if case.lab_values else []

    # 3. Build a rich clinical query from all available context
    query_parts = [f"Symptoms: {case.symptoms}."]
    if case.diagnosed_disease:
        query_parts.append(f"Diagnosed disease: {case.diagnosed_disease}.")
    if case.medical_history:
        query_parts.append(f"Medical history: {case.medical_history}.")
    if lab_findings:
        query_parts.append("Lab findings: " + "; ".join(lab_findings) + ".")
    query = " ".join(query_parts)

    # 4. Retrieval-first / LLM-fallback recommendation
    recommendation = get_recommendation(vectorstore, qa_chain, llm, query)

    # 5. Contraindication check
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
        "contraindication_warnings": contraindication_warnings,
    }
```

---

## 4. Streamlit Application (`app.py`)

### 4.1 Cached Resources

`OllamaEmbeddings` and the `Ollama` LLM client are cached with `st.cache_resource` so they aren't
re-instantiated on every Streamlit rerun (Streamlit reruns the whole script on each interaction).

```python
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return core.OllamaEmbeddings()

@st.cache_resource(show_spinner=False)
def get_llm():
    return core.get_llm()
```

### 4.2 Knowledge Base Builder (Sidebar)

Users choose any combination of the bundled sample JSON, their own uploaded JSON file(s), and
uploaded PDF references. Everything is merged into one document list and embedded in a single pass.

```python
json_files = st.file_uploader("Upload medicine JSON file(s)", type=["json"], accept_multiple_files=True)
pdf_files = st.file_uploader("Upload PDF drug leaflets / guidelines", type=["pdf"], accept_multiple_files=True)

if st.button("🔨 Build / Rebuild Knowledge Base", type="primary"):
    all_json_records = []
    if use_sample:
        all_json_records.extend(core.load_medicine_json(SAMPLE_MEDICINES_PATH))
    for f in json_files or []:
        all_json_records.extend(core.load_medicine_json(f))

    documents = core.json_records_to_documents(all_json_records)
    for f in pdf_files or []:
        documents.extend(core.pdf_to_documents(f, source_name=f.name))

    embeddings = get_embeddings()
    vectorstore = core.build_knowledge_base(documents, embeddings)
    st.session_state.vectorstore = vectorstore
    st.session_state.qa_chain = core.build_qa_chain(get_llm(), vectorstore)
```

### 4.3 Patient Lab Report Upload + Editable Review

Extracted values are never used silently — they're loaded into a `st.data_editor` so the clinician
can correct or remove any misread value before it feeds into the pipeline.

```python
lab_pdf = st.file_uploader("Upload lab report PDF", type=["pdf"])
if lab_pdf is not None:
    text = core.extract_text_from_pdf(lab_pdf)
    extracted = core.extract_lab_values_from_text(text)
    st.session_state.extracted_lab_values = extracted

df = pd.DataFrame([{"Test": k, "Value": v} for k, v in st.session_state.extracted_lab_values.items()])
edited = st.data_editor(
    df, num_rows="dynamic",
    column_config={
        "Test": st.column_config.SelectboxColumn(options=LAB_TEST_NAMES, required=True),
        "Value": st.column_config.NumberColumn(required=True),
    },
)
lab_values = {row["Test"]: row["Value"] for _, row in edited.iterrows() if row["Test"]}
```

### 4.4 Running the Pipeline and Displaying Results

```python
case = core.PatientCase(
    symptoms=symptoms,
    medical_history=medical_history,
    diagnosed_disease=diagnosed_disease or None,
    lab_values=lab_values,
)
result = core.run_cdss_pipeline(st.session_state.vectorstore, st.session_state.qa_chain, get_llm(), case)

if result["status"] == "EMERGENCY":
    st.error(result["message"], icon="🚨")
else:
    if result["source"] == "RETRIEVED_FROM_DATABASE":
        st.success(f"Source: **{result['source']}**", icon="✅")
    else:
        st.warning(f"Source: **{result['source']}**", icon="⚠️")
    st.markdown(result["recommendation"])
    for w in result["contraindication_warnings"]:
        st.error(w, icon="⚠️")
```

---

## 5. Data Schemas

### 5.1 Medicine Knowledge Base (JSON)

```json
{
  "medicine_name": "Metformin",
  "generic_name": "Metformin HCl",
  "drug_class": "Biguanide",
  "disease_condition": "Type 2 Diabetes Mellitus",
  "common_symptoms": "increased thirst, frequent urination, fatigue, blurred vision",
  "relevant_lab_markers": "Fasting Glucose >= 126 mg/dL, HbA1c >= 6.5%",
  "standard_dosage": "500 mg twice daily with meals, titrate up to 2000 mg/day max",
  "contraindications": "severe renal impairment, metabolic acidosis, pregnancy (relative)",
  "side_effects": "GI upset, diarrhea, vitamin B12 deficiency (long term)"
}
```

A file may contain a bare JSON array of such objects, or `{"medicines": [ ... ]}`. Any missing
optional field is auto-filled with `"not specified"`; only `medicine_name` is strictly required.

### 5.2 Patient Case (JSON)

```json
{
  "symptoms": "increased thirst, frequent urination, fatigue, blurred vision",
  "medical_history": "No known drug allergies. No prior renal disease. Family history of diabetes.",
  "diagnosed_disease": "Type 2 Diabetes Mellitus",
  "lab_report": {
    "Fasting Glucose": 158,
    "HbA1c": 7.8,
    "Hemoglobin": 13.2
  }
}
```

### 5.3 PDF Inputs

| PDF type | Handling |
|---|---|
| Knowledge-base reference (drug leaflet, guideline) | Extracted → chunked (500 chars, 60 overlap) → embedded as unstructured `Document`s |
| Patient lab report | Extracted → regex-scanned for known test names → editable table → merged into `PatientCase.lab_values` |

---

## 6. Testing & Validation Performed

Since this environment has no outbound access to Ollama/PyPI package mirrors, validation was split
into what could and couldn't be executed directly:

| Component | Test performed | Result |
|---|---|---|
| All `.py` files | `ast.parse()` syntax validation | Pass |
| `interpret_lab_report` | Ran against sample abnormal/normal values | Correctly flagged HIGH/LOW with clinical meaning |
| `check_red_flags` | Tested against an emergency case and a benign case | Correctly triggered / correctly did not trigger |
| `record_to_text` | Rendered a sample medicine record | Produced expected natural-language passage |
| `check_contraindications` | Penicillin-allergy history vs. Amoxicillin-Clavulanate | Correctly flagged; no false positive on unrelated history |
| `extract_text_from_pdf` + `extract_lab_values_from_text` | Ran against the generated `sample_lab_report.pdf` | All 8 test values correctly extracted |
| FAISS / Ollama-dependent paths (`build_knowledge_base`, `get_recommendation`, `run_cdss_pipeline` end-to-end, Streamlit UI) | Not executable in this sandbox (no network to Ollama or the LangChain/FAISS/Streamlit package indexes) | **Requires validation on your local machine with Ollama running** |

**Recommended local test sequence before relying on this for a demo/submission:**
1. `streamlit run app.py`, build the KB with only the sample JSON.
2. Run the bundled diabetes case → expect `RETRIEVED_FROM_DATABASE`.
3. Run a gout case → expect `LLM_GENERATED_NOT_VERIFIED`.
4. Add `sample_drug_leaflet_allopurinol.pdf` to the KB, rebuild, rerun the gout case → expect it to
   flip to `RETRIEVED_FROM_DATABASE`.
5. Upload `sample_lab_report.pdf` in the patient form and confirm the extracted table matches the PDF.
6. Enter an emergency-symptom case (e.g. chest pain) → expect the pipeline to short-circuit before
   any retrieval happens.

---

## 7. Limitations

- **Not a certified medical device.** Educational prototype only; no regulatory validation.
- **Contraindication checker is keyword-based**, not a real drug-interaction database — it will miss
  interactions that aren't phrased with overlapping words.
- **Lab-value PDF extraction is regex-based** and tuned to reports formatted like the bundled sample
  (test name near its value); irregular report layouts may need manual correction, which is why the
  UI always surfaces an editable table rather than using extracted values directly.
- **Similarity threshold is dataset/embedding-model specific** (`SIMILARITY_THRESHOLD` in
  `cdss_core.py`) and should be recalibrated if the embedding model or knowledge base changes
  significantly.
- **No authentication, audit logging, or PHI handling controls** — do not connect to real patient data
  without adding these.

## 8. Future Enhancements

- Replace the hand-curated sample JSON with a real formulary import (the ingestion pipeline is already
  schema-driven, so this needs no code changes).
- Replace the naive contraindication checker with a structured drug-interaction dataset/API.
- Log every `LLM_GENERATED_NOT_VERIFIED` response to a review queue; once a clinician validates it,
  promote it into the curated JSON knowledge base.
- Add conversation memory so a clinician can refine a case interactively rather than resubmitting.
- Add authentication and an audit trail before any real deployment.

---

## 9. Repository Layout

```
cdss_rag_app/
├── app.py                     # Streamlit UI
├── cdss_core.py                # RAG pipeline, data loaders, lab/safety logic (no UI dependency)
├── requirements.txt
├── README.md                   # setup & usage instructions
├── PROJECT_DOCUMENTATION.md    # this document
└── sample_data/
    ├── medicines_sample.json
    ├── sample_patient_case.json
    ├── sample_lab_report.pdf
    └── sample_drug_leaflet_allopurinol.pdf
```
