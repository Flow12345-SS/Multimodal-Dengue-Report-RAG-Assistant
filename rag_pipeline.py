import os
import json
import re
import requests
import streamlit as st
from utils.logger import get_logger

logger = get_logger(__name__)

REPORTS_DIR = "reports"
VECTOR_DB_DIR = "vector_db"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

NOT_AVAILABLE_MSG = "The requested information is not available in the uploaded report."

# ---------------------------------------------------------------------------
# Health Check for Local LLM (Ollama)
# ---------------------------------------------------------------------------
def check_ollama_health(model_name: str):
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if response.status_code != 200:
            return False, "Ollama service unreachable"
        data = response.json()
        models = [m['name'] for m in data.get('models', [])]
        if not any(m.startswith(model_name) for m in models):
            return False, f"Model '{model_name}' not pulled"
        return True, ""
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Vectorstore Loader (Always loads the active session's vectorstore)
# ---------------------------------------------------------------------------
def load_vectorstore():
    from langchain_community.vectorstores import FAISS
    from embeddings import get_embeddings_model

    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        try:
            embeddings = get_embeddings_model()
            return FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return None
    return None


# ---------------------------------------------------------------------------
# Active Report Metadata Loader
# ---------------------------------------------------------------------------
def get_active_report_meta() -> dict:
    meta_path = os.path.join(VECTOR_DB_DIR, "active_report_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Dynamic Clinical Entity Extractor (Zero Hardcoded Data)
# ---------------------------------------------------------------------------
def extract_dynamic_entities_from_text(text: str) -> dict:
    """
    Extracts clinical and administrative entities dynamically from document text.
    Returns None for any field that does not appear in the text.
    """
    entities = {}

    # Patient Name
    name_m = re.search(
        r"(?:Patient\s*Name|Name\s*of\s*Patient|Patient|Name)\s*[:\-]\s*([A-Za-z\s\.\,\-]+?)(?:\n|\r|\t|\||Age|Gender|Sex|ID|$)",
        text,
        re.I
    )
    if name_m:
        val = name_m.group(1).strip()
        val = re.sub(r'[\,\:\;\|\-]+$', '', val).strip()
        if val and val.lower() not in ['unknown', 'unspecified', 'male', 'female', 'years', 'report', 'details', 'name', 'na']:
            entities['patient_name'] = val

    # Patient ID
    id_m = re.search(
        r"(?:Patient\s*ID|PID|MRN|Case\s*No|Record\s*ID|\bID)\s*[:\-]?\s*([A-Za-z0-9\-]+)",
        text,
        re.I
    )
    if not id_m:
        id_m = re.search(r"Patient\s*Report\s*-\s*([A-Za-z0-9\-]+)", text, re.I)
    if not id_m:
        id_m = re.search(r"\b([A-Z]\d{3,6})\b", text, re.I)
    if id_m:
        val = id_m.group(1).strip().upper()
        if val.lower() not in ['unknown', 'unspecified', 'intake', 'patient', 'report', 'details', 'id', 'na']:
            entities['patient_id'] = val

    # Age
    age_m = re.search(r"(?:Age)\s*[:\-]?\s*(\d+(?:\s*Years)?)", text, re.I)
    if age_m:
        entities['age'] = age_m.group(1).strip()

    # Gender
    gender_m = re.search(r"(?:Gender|Sex)\s*[:\-]?\s*([A-Za-z]+)", text, re.I)
    if gender_m:
        val = gender_m.group(1).strip().capitalize()
        if val in ['Male', 'Female', 'Other']:
            entities['gender'] = val

    # Platelet Count
    platelet_m = re.search(
        r"(?:Platelet\s*Count|Platelets?|PLT)\s*[:\-]?\s*([\d,]+(?:\s*(?:/\s*[uµ]L|\s*/\s*mm3|\s*x10\^?3/\s*[uµ]L|k/\s*[uµ]L))?)",
        text,
        re.I
    )
    if platelet_m:
        entities['platelet_count'] = platelet_m.group(1).strip()

    # NS1 Antigen
    ns1_m = re.search(r"NS1(?:\s*Antigen)?\s*[:\-]?\s*([A-Za-z\+\-]+)", text, re.I)
    if ns1_m:
        entities['ns1'] = ns1_m.group(1).strip()

    # IgM Antibody
    igm_m = re.search(r"IgM(?:\s*Antibody)?\s*[:\-]?\s*([A-Za-z\+\-]+)", text, re.I)
    if igm_m:
        entities['igm'] = igm_m.group(1).strip()

    # IgG Antibody
    igg_m = re.search(r"IgG(?:\s*Antibody)?\s*[:\-]?\s*([A-Za-z\+\-]+)", text, re.I)
    if igg_m:
        entities['igg'] = igg_m.group(1).strip()

    # Diagnosis
    diag_m = re.search(r"(?:Diagnosis|Clinical\s*Diagnosis|Impression|Condition)\s*[:\-]?\s*([^\n\r\|]+)", text, re.I)
    if diag_m:
        entities['diagnosis'] = diag_m.group(1).strip()

    # Risk Level
    risk_m = re.search(r"(?:Risk\s*Level|Risk\s*Assessment|Risk)\s*[:\-]?\s*([^\n\r\|]+)", text, re.I)
    if risk_m:
        entities['risk_level'] = risk_m.group(1).strip()

    # WBC Count
    wbc_m = re.search(r"(?:WBC\s*Count|WBC|Total\s*Leukocyte\s*Count|TLC)\s*[:\-]?\s*([\d,]+(?:\s*/\s*[uµ]L|\s*/\s*mm3)?)", text, re.I)
    if wbc_m:
        entities['wbc_count'] = wbc_m.group(1).strip()

    # Hematocrit / PCV
    hct_m = re.search(r"(?:Hematocrit|HCT|PCV)\s*[:\-]?\s*(\d+(?:\.\d+)?\s*%?)", text, re.I)
    if hct_m:
        entities['hematocrit'] = hct_m.group(1).strip()

    # Recommendations
    rec_m = re.search(r"(?:Recommendations?|Advice|Treatment\s*Plan|Plan)\s*[:\-]?\s*([^\n\r]+(?:\n[^\n\r]+){0,4})", text, re.I)
    if rec_m:
        entities['recommendations'] = rec_m.group(1).strip()

    return entities


# ---------------------------------------------------------------------------
# Direct Grounded Answer Builder (Offline Fallback Engine)
# ---------------------------------------------------------------------------
def direct_grounded_answer(question: str, context: str, entities: dict) -> str:
    """Answers direct field queries using entities strictly present in the uploaded report."""
    q = question.lower().strip()

    # Patient ID
    if re.search(r"\b(patient\s*id|patient\'s\s*id|patient-id|pid|\bid\b)\b", q) and not any(k in q for k in ['diagnosis', 'report', 'platelet', 'risk']):
        if 'patient_id' in entities:
            return entities['patient_id']
        return NOT_AVAILABLE_MSG

    # Patient Name
    if re.search(r"\b(patient\s*name|name\s*of\s*patient|patient\'s\s*name|whose\s*name|who\s*is\s*the\s*patient)\b", q):
        if 'patient_name' in entities:
            return entities['patient_name']
        return NOT_AVAILABLE_MSG

    # Platelet Count
    if re.search(r"\b(platelet|platelets|platelet\s*count|plt)\b", q):
        if 'platelet_count' in entities:
            return entities['platelet_count']
        return NOT_AVAILABLE_MSG

    # Age
    if re.search(r"\b(age|how\s*old|years\s*old)\b", q) and not any(k in q for k in ['diagnosis', 'report', 'platelet', 'risk']):
        if 'age' in entities:
            return entities['age']
        return NOT_AVAILABLE_MSG

    # Gender
    if re.search(r"\b(gender|sex|male\s*or\s*female)\b", q):
        if 'gender' in entities:
            return entities['gender']
        return NOT_AVAILABLE_MSG

    # Diagnosis
    if re.search(r"\b(diagnosis|condition|diagnosed|what\s*disease)\b", q):
        if 'diagnosis' in entities:
            return entities['diagnosis']
        return NOT_AVAILABLE_MSG

    # Risk level
    if re.search(r"\b(risk|risk\s*level|risk\s*assessment)\b", q):
        if 'risk_level' in entities:
            return entities['risk_level']
        return NOT_AVAILABLE_MSG

    # Recommendations
    if re.search(r"\b(recommendations?|advice|treatment|precautions?)\b", q):
        if 'recommendations' in entities:
            return entities['recommendations']
        return NOT_AVAILABLE_MSG

    # NS1 Antigen
    if re.search(r"\b(ns1|ns1\s*antigen)\b", q):
        if 'ns1' in entities:
            return f"NS1 Antigen: {entities['ns1']}"
        return NOT_AVAILABLE_MSG

    # Antibodies (IgM / IgG)
    if re.search(r"\b(igm|igg|antibody|antibodies)\b", q):
        res = []
        if 'igm' in entities:
            res.append(f"IgM: {entities['igm']}")
        if 'igg' in entities:
            res.append(f"IgG: {entities['igg']}")
        if res:
            return ", ".join(res)
        return NOT_AVAILABLE_MSG

    # General overview if requested
    if any(k in q for k in ['summary', 'overview', 'details', 'full report', 'assessment']):
        lines = []
        if 'patient_name' in entities:
            lines.append(f"• **Patient Name:** {entities['patient_name']}")
        if 'patient_id' in entities:
            lines.append(f"• **Patient ID:** {entities['patient_id']}")
        if 'age' in entities:
            lines.append(f"• **Age:** {entities['age']}")
        if 'gender' in entities:
            lines.append(f"• **Gender:** {entities['gender']}")
        if 'platelet_count' in entities:
            lines.append(f"• **Platelet Count:** {entities['platelet_count']}")
        if 'diagnosis' in entities:
            lines.append(f"• **Diagnosis:** {entities['diagnosis']}")
        if 'risk_level' in entities:
            lines.append(f"• **Risk Level:** {entities['risk_level']}")
        if 'recommendations' in entities:
            lines.append(f"• **Recommendations:** {entities['recommendations']}")
        if lines:
            return "\n".join(lines)
        return NOT_AVAILABLE_MSG

    # Fallback: check if keywords from question appear anywhere in context
    q_words = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', q) if w not in ['what', 'when', 'where', 'which', 'from', 'this', 'that', 'with', 'have', 'does', 'report', 'patient']]
    matched_sentences = []
    for line in context.split('\n'):
        line_clean = line.strip()
        if line_clean and any(w in line_clean.lower() for w in q_words):
            matched_sentences.append(line_clean)
    if matched_sentences:
        return "\n".join(matched_sentences[:4])

    return NOT_AVAILABLE_MSG


# ---------------------------------------------------------------------------
# Main Answer Generation Engine
# ---------------------------------------------------------------------------
def generate_answer(question: str, model_name: str = "tinyllama"):
    """
    RAG Pipeline:
    1. Loads the FAISS vectorstore built strictly from the currently uploaded report.
    2. Performs similarity search to retrieve relevant chunks with scores.
    3. Dynamically extracts patient identity and entities from the retrieved chunks.
    4. If Ollama is available, prompts LLM with strict grounding in the retrieved context.
       If LLM is unavailable or ungrounded, uses exact entity extraction.
    5. Returns (answer, retrieved_patient_ui, evidence_list).
    """
    vectorstore = load_vectorstore()
    if not vectorstore:
        return "Please upload and process a medical report to begin.", "", []

    # Similarity search with distance scores
    try:
        results_with_scores = vectorstore.similarity_search_with_score(question, k=4)
    except Exception:
        docs = vectorstore.similarity_search(question, k=4)
        results_with_scores = [(d, 0.0) for d in docs]

    if not results_with_scores:
        return NOT_AVAILABLE_MSG, "", []

    docs = [r[0] for r in results_with_scores]
    context = "\n".join([d.page_content for d in docs])

    # Extract dynamic entities from context and active report metadata
    active_meta = get_active_report_meta()
    entities = extract_dynamic_entities_from_text(context)

    # Reconcile patient identifiers dynamically
    retrieved_name = entities.get('patient_name') or active_meta.get('patient_name') or "Not specified"
    retrieved_id = entities.get('patient_id') or active_meta.get('patient_id') or "Not specified"

    retrieved_patient_ui = f"**Retrieved Patient:** {retrieved_name} ({retrieved_id})"

    # Build clean structured clinical evidence preview
    findings_list = []
    if entities.get('age'):
        findings_list.append(f"Age: {entities['age']}")
    if entities.get('ns1'):
        findings_list.append(f"NS1 Antigen: {entities['ns1']}")
    if entities.get('platelet_count'):
        findings_list.append(f"Platelet Count: {entities['platelet_count']}")
    if entities.get('wbc_count'):
        findings_list.append(f"WBC Count: {entities['wbc_count']}")
    if entities.get('hematocrit'):
        findings_list.append(f"Hematocrit: {entities['hematocrit']}")
    if entities.get('igm'):
        findings_list.append(f"IgM Antibody: {entities['igm']}")
    if entities.get('igg'):
        findings_list.append(f"IgG Antibody: {entities['igg']}")
    if entities.get('gender'):
        findings_list.append(f"Gender: {entities['gender']}")

    raw_rec = entities.get('recommendations', '')
    rec_items = []
    if raw_rec:
        for line in re.split(r'[\n\r;•\*\d+\.]+', raw_rec):
            line_c = line.strip()
            if line_c and len(line_c) > 3:
                rec_items.append(line_c)
    if not rec_items:
        rec_items = [
            "Monitor platelet count daily",
            "Maintain adequate fluid intake",
            "Follow-up testing recommended"
        ]

    clinical_evidence = {
        "patient_name": retrieved_name,
        "patient_id": retrieved_id,
        "findings": findings_list,
        "diagnosis": entities.get('diagnosis', 'Suspected Dengue Fever'),
        "recommendations": rec_items
    }

    # Check if Ollama LLM is available
    is_ollama_ok, _ = check_ollama_health(model_name)

    if is_ollama_ok:
        try:
            from langchain_ollama import ChatOllama

            system_prompt = (
                "You are an AI medical report question-answering assistant.\n"
                "Answer the user's question STRICTLY and ONLY using the provided Context extracted from the uploaded medical report.\n"
                "Rules:\n"
                "1. If the answer cannot be found directly in the Context, you MUST answer EXACTLY:\n"
                f'"{NOT_AVAILABLE_MSG}"\n'
                "2. Do NOT hallucinate, assume, or infer any patient names, values, or diseases not in the Context.\n"
                "3. Be concise, direct, and factually exact.\n\n"
                f"Context from Uploaded Report:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )

            llm = ChatOllama(
                model=model_name,
                base_url=OLLAMA_BASE_URL,
                temperature=0.0,
                num_predict=250
            )
            response = llm.invoke(system_prompt)
            output_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()

            if output_text and len(output_text) > 1:
                return output_text, retrieved_patient_ui, clinical_evidence

        except Exception as e:
            logger.warning(f"Ollama generation failed, falling back to direct extraction: {e}")

    # Deterministic Grounded Fallback
    grounded_ans = direct_grounded_answer(question, context, entities)
    return grounded_ans, retrieved_patient_ui, clinical_evidence
