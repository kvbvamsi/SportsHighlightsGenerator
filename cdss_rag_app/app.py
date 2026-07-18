"""
Streamlit front end for the AI-Powered Clinical Decision Support System (CDSS).

Run with:
    streamlit run app.py

Requires a local Ollama server running llama3.2 (see README.md).
"""

import json
import traceback

import pandas as pd
import streamlit as st

import cdss_core as core

st.set_page_config(page_title="Clinical Decision Support System", page_icon="🩺", layout="wide")

SAMPLE_MEDICINES_PATH = "sample_data/medicines_sample.json"
SAMPLE_PATIENT_CASE_PATH = "sample_data/sample_patient_case.json"
PERSIST_DIR = "faiss_index"

LAB_TEST_NAMES = list(core.LAB_REFERENCE_RANGES.keys())


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return core.OllamaEmbeddings()


@st.cache_resource(show_spinner=False)
def get_llm():
    return core.get_llm()


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

for key, default in [
    ("vectorstore", None),
    ("qa_chain", None),
    ("kb_doc_count", 0),
    ("kb_medicine_count", 0),
    ("extracted_lab_values", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar: Knowledge base management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📚 Knowledge Base")

    ollama_ok = core.ollama_is_reachable()
    if ollama_ok:
        st.success("Ollama server reachable")
    else:
        st.error("Ollama server not reachable at localhost:11434.\nStart it with `ollama serve` "
                  "and make sure `ollama pull llama3.2` has been run.")

    st.markdown("**1. Structured data (JSON)**")
    use_sample = st.checkbox("Include bundled sample dataset (18 medicines)", value=True)
    json_files = st.file_uploader(
        "Upload medicine JSON file(s)", type=["json"], accept_multiple_files=True, key="json_uploader"
    )

    st.markdown("**2. Unstructured references (PDF)**")
    pdf_files = st.file_uploader(
        "Upload PDF drug leaflets / guidelines", type=["pdf"], accept_multiple_files=True, key="pdf_uploader"
    )

    build_clicked = st.button("🔨 Build / Rebuild Knowledge Base", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        f"Current knowledge base: **{st.session_state.kb_doc_count}** document chunks "
        f"({st.session_state.kb_medicine_count} structured medicine records)."
    )

    if build_clicked:
        if not ollama_ok:
            st.error("Cannot build the knowledge base: Ollama is not reachable.")
        else:
            try:
                with st.spinner("Loading records and building embeddings... this can take a minute."):
                    all_json_records = []
                    if use_sample:
                        all_json_records.extend(core.load_medicine_json(SAMPLE_MEDICINES_PATH))
                    for f in json_files or []:
                        all_json_records.extend(core.load_medicine_json(f))

                    documents = core.json_records_to_documents(all_json_records)

                    for f in pdf_files or []:
                        documents.extend(core.pdf_to_documents(f, source_name=f.name))

                    if not documents:
                        st.warning("No data selected. Check the sample dataset box or upload a file.")
                    else:
                        embeddings = get_embeddings()
                        vectorstore = core.build_knowledge_base(documents, embeddings)
                        llm = get_llm()
                        qa_chain = core.build_qa_chain(llm, vectorstore)

                        st.session_state.vectorstore = vectorstore
                        st.session_state.qa_chain = qa_chain
                        st.session_state.kb_doc_count = len(documents)
                        st.session_state.kb_medicine_count = len(all_json_records)

                st.success(f"Knowledge base built: {len(documents)} chunks "
                           f"({len(all_json_records)} structured records + "
                           f"{len(documents) - len(all_json_records)} PDF chunks).")
            except Exception as e:
                st.error(f"Failed to build knowledge base: {e}")
                st.code(traceback.format_exc())

    if st.session_state.vectorstore is not None:
        if st.button("💾 Save knowledge base to disk", use_container_width=True):
            try:
                core.save_knowledge_base(st.session_state.vectorstore, PERSIST_DIR)
                st.success(f"Saved to ./{PERSIST_DIR}")
            except Exception as e:
                st.error(f"Save failed: {e}")

    if st.button("📂 Load previously saved knowledge base", use_container_width=True):
        try:
            embeddings = get_embeddings()
            vectorstore = core.load_knowledge_base(PERSIST_DIR, embeddings)
            llm = get_llm()
            st.session_state.vectorstore = vectorstore
            st.session_state.qa_chain = core.build_qa_chain(llm, vectorstore)
            st.session_state.kb_doc_count = vectorstore.index.ntotal
            st.success("Knowledge base loaded from disk.")
        except Exception as e:
            st.error(f"No saved knowledge base found, or load failed: {e}")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🩺 AI-Powered Clinical Decision Support System")
st.caption(
    "Retrieval-Augmented Generation over a medicine knowledge base, with Llama 3.2 (via Ollama) as a "
    "clearly-labeled fallback when the database has no match."
)
st.warning(
    "⚠️ Educational prototype only — not a certified medical device. All output must be reviewed by a "
    "licensed clinician before any real-world use.",
    icon="⚠️",
)

tab_case, tab_about = st.tabs(["🩺 New Patient Case", "ℹ️ About & Sample Files"])

# ---------------------------------------------------------------------------
# Tab: New patient case
# ---------------------------------------------------------------------------

with tab_case:
    if st.session_state.vectorstore is None:
        st.info("Build the knowledge base from the sidebar first (☚), or load a previously saved one.")

    input_method = st.radio(
        "Patient case input method", ["Manual entry", "Upload patient case JSON"], horizontal=True
    )

    symptoms = ""
    medical_history = ""
    diagnosed_disease = ""
    lab_values = {}

    if input_method == "Upload patient case JSON":
        case_file = st.file_uploader(
            "Upload patient case JSON (symptoms, medical_history, diagnosed_disease, lab_report)",
            type=["json"], key="case_json_uploader",
        )
        if case_file is not None:
            try:
                case_data = json.loads(case_file.read().decode("utf-8"))
                symptoms = case_data.get("symptoms", "")
                medical_history = case_data.get("medical_history", "")
                diagnosed_disease = case_data.get("diagnosed_disease", "") or ""
                lab_values = case_data.get("lab_report", {}) or {}
                st.success("Patient case JSON loaded.")
                st.json(case_data)
            except Exception as e:
                st.error(f"Could not parse JSON: {e}")

    else:
        col1, col2 = st.columns(2)
        with col1:
            symptoms = st.text_area(
                "Symptoms", placeholder="e.g. increased thirst, frequent urination, fatigue", height=100
            )
            diagnosed_disease = st.text_input("Diagnosed disease (optional)", placeholder="e.g. Type 2 Diabetes Mellitus")
        with col2:
            medical_history = st.text_area(
                "Medical history / known allergies",
                placeholder="e.g. No known drug allergies. No prior renal disease.", height=100,
            )

        st.markdown("**Lab report**")
        lab_input_method = st.radio(
            "Lab input method", ["Upload lab report PDF", "Enter manually", "Skip"], horizontal=True
        )

        if lab_input_method == "Upload lab report PDF":
            lab_pdf = st.file_uploader("Upload lab report PDF", type=["pdf"], key="lab_pdf_uploader")
            if lab_pdf is not None:
                try:
                    text = core.extract_text_from_pdf(lab_pdf)
                    extracted = core.extract_lab_values_from_text(text)
                    st.session_state.extracted_lab_values = extracted
                    with st.expander("View extracted raw text"):
                        st.text(text)
                except Exception as e:
                    st.error(f"Could not read PDF: {e}")

            if st.session_state.extracted_lab_values:
                st.caption("Review / correct the auto-extracted values before submitting:")
                df = pd.DataFrame(
                    [{"Test": k, "Value": v} for k, v in st.session_state.extracted_lab_values.items()]
                )
                edited = st.data_editor(
                    df, num_rows="dynamic", use_container_width=True, key="lab_pdf_editor",
                    column_config={
                        "Test": st.column_config.SelectboxColumn(options=LAB_TEST_NAMES, required=True),
                        "Value": st.column_config.NumberColumn(required=True),
                    },
                )
                lab_values = {row["Test"]: row["Value"] for _, row in edited.iterrows() if row["Test"]}

        elif lab_input_method == "Enter manually":
            st.caption("Add one row per lab test:")
            default_df = pd.DataFrame({"Test": pd.Series(dtype="str"), "Value": pd.Series(dtype="float")})
            edited = st.data_editor(
                default_df, num_rows="dynamic", use_container_width=True, key="lab_manual_editor",
                column_config={
                    "Test": st.column_config.SelectboxColumn(options=LAB_TEST_NAMES, required=True),
                    "Value": st.column_config.NumberColumn(required=True),
                },
            )
            lab_values = {row["Test"]: row["Value"] for _, row in edited.iterrows() if row["Test"]}

    st.divider()
    run_clicked = st.button("🔎 Get Recommendation", type="primary", disabled=st.session_state.vectorstore is None)

    if run_clicked:
        if not symptoms.strip():
            st.error("Please enter the patient's symptoms.")
        else:
            case = core.PatientCase(
                symptoms=symptoms,
                medical_history=medical_history,
                diagnosed_disease=diagnosed_disease or None,
                lab_values={k: v for k, v in (lab_values or {}).items()},
            )
            try:
                with st.spinner("Running red-flag screen, retrieval, and (if needed) LLM fallback..."):
                    result = core.run_cdss_pipeline(
                        st.session_state.vectorstore, st.session_state.qa_chain, get_llm(), case
                    )

                if result["status"] == "EMERGENCY":
                    st.error(result["message"], icon="🚨")
                else:
                    if result["source"] == "RETRIEVED_FROM_DATABASE":
                        st.success(f"Source: **{result['source']}** "
                                   f"(matched: {', '.join(result['matched_medicines']) or 'n/a'})", icon="✅")
                    else:
                        st.warning(f"Source: **{result['source']}**", icon="⚠️")

                    if result["lab_findings"]:
                        with st.expander("Lab findings", expanded=True):
                            for f in result["lab_findings"]:
                                if "HIGH" in f or "LOW" in f:
                                    st.markdown(f"🔴 {f}")
                                else:
                                    st.markdown(f"🟢 {f}")

                    st.markdown("### Recommendation")
                    st.markdown(result["recommendation"])

                    if result["contraindication_warnings"]:
                        st.markdown("### Safety Warnings")
                        for w in result["contraindication_warnings"]:
                            st.error(w, icon="⚠️")

                    with st.expander("Debug: query sent to the RAG pipeline"):
                        st.code(result["query_used"])
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Tab: About / sample files
# ---------------------------------------------------------------------------

with tab_about:
    st.markdown(
        """
### How this works
1. **Knowledge base** — structured medicine records (JSON) are rendered into natural-language passages
   and embedded with Ollama; PDF references (drug leaflets, guidelines) are chunked and embedded the
   same way, so both feed the same FAISS vector store.
2. **Retrieval** — for each patient case, the top matching passages are retrieved and given to Llama 3.2
   as context. If the model can't find a good match, it returns a sentinel the app detects.
3. **LLM fallback** — when nothing relevant is retrieved (sentinel *or* a poor similarity score), the app
   calls Llama 3.2 directly for a general recommendation, and the UI labels it **LLM_GENERATED_NOT_VERIFIED**
   so it's never confused with a database-backed answer.
4. **Safety layer** — an emergency red-flag screen runs before anything else, and a keyword-based
   contraindication check cross-references the patient's history against matched medicines.

### Sample files (in `sample_data/`)
- `medicines_sample.json` — 18 structured medicine records to seed the knowledge base.
- `sample_drug_leaflet_allopurinol.pdf` — an unstructured PDF leaflet for Allopurinol/Gout, deliberately
  **not** in the JSON dataset. Upload it alongside the sample JSON to see gout cases move from
  LLM-fallback to database-retrieved.
- `sample_lab_report.pdf` — a synthetic lab report you can upload via "Upload lab report PDF" to test
  the PDF lab-value extractor.
- `sample_patient_case.json` — a ready-made patient case for the "Upload patient case JSON" input method.

Bring your own data by uploading additional JSON files (same schema as `medicines_sample.json`) or PDFs
in the sidebar — no code changes needed.
"""
    )

    try:
        with open(SAMPLE_MEDICINES_PATH, "rb") as f:
            st.download_button("⬇️ Download sample medicines.json", f, file_name="medicines_sample.json")
    except FileNotFoundError:
        pass

    try:
        with open(SAMPLE_PATIENT_CASE_PATH, "rb") as f:
            st.download_button("⬇️ Download sample_patient_case.json", f, file_name="sample_patient_case.json")
    except FileNotFoundError:
        pass
