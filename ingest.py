import os
import shutil
import re
from utils.logger import get_logger

logger = get_logger(__name__)

REPORTS_DIR = "reports"
VECTOR_DB_DIR = "vector_db"

def init_directories():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

def clean_directories():
    """Safely wipes old files inside the directories without deleting the root folders."""
    import gc
    gc.collect()
    print("Cleaning directories...")
    
    if os.path.exists(REPORTS_DIR):
        print("Cleaning reports directory...")
        for f in os.listdir(REPORTS_DIR):
            file_path = os.path.join(REPORTS_DIR, f)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print("Deleted:", file_path)
                except PermissionError:
                    print(f"Warning: Could not delete {file_path} due to PermissionError.")
                    pass
    else:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
    if os.path.exists(VECTOR_DB_DIR):
        print("Cleaning vector_db directory...")
        for f in os.listdir(VECTOR_DB_DIR):
            file_path = os.path.join(VECTOR_DB_DIR, f)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print("Deleted:", file_path)
                except PermissionError:
                    print(f"Warning: Could not delete {file_path} due to PermissionError.")
                    pass
    else:
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)

def extract_metadata(text):
    patient_id = "Unknown"
    patient_name = "Unknown"
    
    id_match = re.search(r"(?:Patient\s*ID|PID|\bID|Patient\s*Report)\s*[:\-]?\s*([A-Z0-9\-]+)", text, re.IGNORECASE)
    if not id_match:
        id_match = re.search(r'\b(D\d{3,4}|PID-\d+)\b', text, re.IGNORECASE)
    if id_match:
        val = id_match.group(1).strip()
        if val.lower() not in ['unknown', 'unspecified', 'intake', 'patient']:
            patient_id = val.upper()
        
    name_match = re.search(r"(?:Patient\s*Name|Name)\s*[:\-]\s*([A-Za-z\s]+)", text, re.IGNORECASE)
    if name_match:
        val = name_match.group(1).split('\n')[0].strip()
        if val.lower() not in ['unknown', 'unspecified', 'male', 'female', 'years']:
            patient_name = val
        
    return {"patient_id": patient_id, "patient_name": patient_name}

def ingest_documents():
    """
    Reads PDFs, extracts metadata, creates chunk, and builds a clean FAISS index.
    Returns (success_boolean, list_of_indexed_patients)
    """
    import time
    start = time.time()
    
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from embeddings import get_embeddings_model
    from privacy import mask_sensitive_data
    
    pdf_files = [f for f in os.listdir(REPORTS_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No PDF files found in '{REPORTS_DIR}' directory.")
        return False, []
        
    all_chunks = []
    indexed_patients = []
    seen_patients = set()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    for file in pdf_files:
        file_path = os.path.join(REPORTS_DIR, file)
        print(f"\n[DEBUG] Uploaded File Processing: {file}")
        
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            for doc_idx, doc in enumerate(docs):
                page_text = doc.page_content.strip()
                if not page_text:
                    continue
                
                # Extract page-level metadata first
                page_meta = extract_metadata(page_text)
                default_pid = page_meta.get('patient_id', 'Unknown')
                default_pname = page_meta.get('patient_name', 'Unknown')

                # Only split if there are multiple separate reports in a single page
                report_headers = list(re.finditer(r'(?:DENGUE\s*PATIENT\s*REPORT|Patient\s*Report\s*-\s*D\d+)', page_text, flags=re.IGNORECASE))
                if len(report_headers) > 1:
                    patient_sections = []
                    for idx, h in enumerate(report_headers):
                        start_pos = h.start()
                        end_pos = report_headers[idx + 1].start() if idx + 1 < len(report_headers) else len(page_text)
                        patient_sections.append(page_text[start_pos:end_pos])
                else:
                    patient_sections = [page_text]
                    
                for section in patient_sections:
                    if not section.strip():
                        continue
                        
                    meta = extract_metadata(section)
                    pid = meta.get('patient_id')
                    if not pid or pid == "Unknown":
                        pid = default_pid
                    pname = meta.get('patient_name')
                    if not pname or pname == "Unknown":
                        pname = default_pname
                    
                    if pname != "Unknown" and pname != "Unspecified":
                        patient_key = f"{pname.lower()}_{pid.upper()}"
                        if patient_key not in seen_patients:
                            seen_patients.add(patient_key)
                            indexed_patients.append({"name": pname, "id": pid})
                            print(f"[DEBUG] Indexed Patient: {pname} ({pid})")
                            
                    masked_text = mask_sensitive_data(section)
                    sub_chunks = text_splitter.split_text(masked_text)
                    
                    for sub_text in sub_chunks:
                        chunk = Document(page_content=sub_text, metadata={
                            'source_file': file,
                            'patient_id': pid,
                            'patient_name': pname
                        })
                        all_chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Error processing file {file}: {str(e)}")
            
    if all_chunks:
        embeddings = get_embeddings_model()
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        vectorstore.save_local(VECTOR_DB_DIR)

        # Save all indexed patients to indexed_patients.json and active_patient.json
        import json
        for d in [VECTOR_DB_DIR, REPORTS_DIR]:
            try:
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "indexed_patients.json"), "w", encoding="utf-8") as f:
                    json.dump(indexed_patients, f, indent=2)
                if indexed_patients:
                    with open(os.path.join(d, "active_patient.json"), "w", encoding="utf-8") as f:
                        json.dump({
                            "patient_name": indexed_patients[0]["name"],
                            "patient_id": indexed_patients[0]["id"],
                            "all_patients": indexed_patients,
                            "source_files": pdf_files
                        }, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save patient metadata in {d}: {e}")

        return True, indexed_patients
    else:
        return False, []
