# AI-Powered Clinical Decision Support System (CDSS) — RAG + Llama 3.2

An educational GenAI project: a Streamlit app that recommends medicines for a patient case using
Retrieval-Augmented Generation over a medicine knowledge base, falling back to a raw Llama 3.2
(via Ollama) call — clearly labeled as such — when the knowledge base has no good match.

> ⚠️ **This is a course/portfolio prototype, not a certified medical device.** Do not use it to make
> real treatment decisions. All output must be reviewed by a licensed clinician.

## Project layout

```
cdss_rag_app/
├── app.py                 # Streamlit UI
├── cdss_core.py            # All RAG / lab-interpretation / safety logic (no Streamlit dependency)
├── requirements.txt
├── README.md
└── sample_data/
    ├── medicines_sample.json               # 18 structured medicine records (knowledge base seed)
    ├── sample_patient_case.json            # Example patient case for the "upload JSON" input path
    ├── sample_lab_report.pdf               # Synthetic lab report to test PDF lab-value extraction
    └── sample_drug_leaflet_allopurinol.pdf # Unstructured PDF leaflet, NOT in the JSON dataset
```

## 1. Prerequisites

Install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.2
ollama serve          # keep this running; default endpoint http://localhost:11434
```

## 2. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## 4. Using the app

### Build the knowledge base (sidebar)
- Leave **"Include bundled sample dataset"** checked to seed the vector store with the 18 sample
  medicines, and/or upload your own `medicines_*.json` file(s) in the same schema.
- Optionally upload PDF drug leaflets / clinical guideline excerpts — these are chunked and embedded
  alongside the structured records, so the RAG can retrieve from both.
- Click **"Build / Rebuild Knowledge Base"**.
- Optionally **"Save knowledge base to disk"** so you don't have to rebuild it (re-embed everything)
  every time you restart the app — next time, click **"Load previously saved knowledge base"** instead.

### Try the bundled demo
1. Build the KB with only the sample dataset checked (no PDFs).
2. Enter symptoms for **gout** (see `sample_data/sample_patient_case.json` for the diabetes example, or
   try: *"sudden severe pain and swelling in the big toe joint, redness, warmth to touch"*). Since gout
   isn't in the sample JSON, you should see `LLM_GENERATED_NOT_VERIFIED`.
3. Now also upload `sample_data/sample_drug_leaflet_allopurinol.pdf` and rebuild the KB.
4. Re-run the same gout case — it should now come back `RETRIEVED_FROM_DATABASE`, matched against the
   Allopurinol leaflet content. This demonstrates PDF ingestion actually expanding what the RAG can answer.

### Enter a patient case
Two ways:
- **Manual entry**: type symptoms / medical history / diagnosed disease, then either upload a lab report
  PDF (values are auto-extracted with a regex parser and shown in an editable table for you to confirm/
  correct), or add lab test rows manually, or skip labs entirely.
- **Upload patient case JSON**: use `sample_data/sample_patient_case.json` as a template
  (`symptoms`, `medical_history`, `diagnosed_disease`, `lab_report`).

Click **"Get Recommendation"**. The app will:
1. Screen for emergency ("red-flag") symptoms first — if matched, it stops and tells you to seek
   emergency care instead of suggesting a medicine.
2. Interpret any lab values against reference ranges.
3. Retrieve from the knowledge base; fall back to a labeled LLM-generated answer if nothing matches well.
4. Run a keyword-based contraindication check against the patient's stated history/allergies.

## 5. Bringing your own data

- **JSON schema** for medicine records (see `sample_data/medicines_sample.json`):
  ```json
  {
    "medicine_name": "Metformin",
    "generic_name": "Metformin HCl",
    "drug_class": "Biguanide",
    "disease_condition": "Type 2 Diabetes Mellitus",
    "common_symptoms": "increased thirst, frequent urination, fatigue",
    "relevant_lab_markers": "Fasting Glucose >= 126 mg/dL, HbA1c >= 6.5%",
    "standard_dosage": "500 mg twice daily with meals",
    "contraindications": "severe renal impairment, metabolic acidosis",
    "side_effects": "GI upset, diarrhea"
  }
  ```
  You can upload a JSON file containing either a bare array of such objects, or `{"medicines": [...]}`.
  Missing optional fields are auto-filled with `"not specified"`.

- **PDFs**: any drug leaflet, formulary excerpt, or clinical guideline PDF — text is extracted and
  chunked automatically. No schema required.

## 6. Notes / limitations

- The similarity threshold that triggers the LLM fallback (`SIMILARITY_THRESHOLD` in `cdss_core.py`) is
  tuned loosely for the `llama3.2` embedding model; recalibrate if you switch embedding models.
- The contraindication checker is a naive keyword matcher for demonstration — do not treat it as a real
  drug-interaction engine.
- The PDF lab-value extractor is regex-based and works best on reports formatted like
  `sample_lab_report.pdf` (test name near its numeric value); always let the user review the extracted
  table before submitting.
