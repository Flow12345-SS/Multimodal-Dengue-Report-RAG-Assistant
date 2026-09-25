import os
import shutil
import re
import json
from utils.logger import get_logger

logger = get_logger(__name__)

REPORTS_DIR = "reports"
VECTOR_DB_DIR = "vector_db"

def init_directories():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

def clean_directories():
    """Completely wipes old session reports and vector databases to ensure report isolation."""
    import gc
    gc.collect()
    logger.info("Cleaning session directories...")
    
    for dir_path in [REPORTS_DIR, VECTOR_DB_DIR]:
        if os.path.exists(dir_path):
            for item in os.listdir(dir_path):
                p = os.path.join(dir_path, item)
                try:
                    if os.path.isfile(p) or os.path.islink(p):
                        os.unlink(p)
                    elif os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Could not remove {p}: {e}")
        else:
            os.makedirs(dir_path, exist_ok=True)

def extract_text_from_file(file_path: str) -> str:
    """Extracts text from PDF, DOCX, and TXT files dynamically."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            pages_text = []
            for i, page in enumerate(reader.pages):
                page_content = page.extract_text() or ""
                pages_text.append(page_content)
            text = "\n".join(pages_text)
            
            # Scanned PDF fallback
            if not text.strip():
                try:
                    import fitz  # PyMuPDF if available
                    doc = fitz.open(file_path)
                    text = "\n".join([p.get_text() for p in doc])
                except ImportError:
                    pass
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")

    elif ext in [".docx", ".doc"]:
        try:
            import docx
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs.append(row_text)
            text = "\n".join(paragraphs)
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")

    elif ext in [".txt", ".csv", ".tsv"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            
    return text.strip()

def extract_dynamic_metadata(text: str) -> dict:
    """Dynamically extracts patient identifier fields from raw document text."""
    meta = {
        "patient_name": "Not specified",
        "patient_id": "Not specified"
    }
    
    # Dynamic Patient Name matching
    name_patterns = [
        r"(?:Patient\s*Name|Name\s*of\s*Patient|Patient|Name)\s*[:\-]\s*([A-Za-z\s\.\,\-]+?)(?:\n|\r|\t|\||Age|Gender|Sex|$)",
        r"(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+([A-Za-z\s]+)"
    ]
    for pat in name_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            # Clean trailing punctuation
            val = re.sub(r'[\,\:\;\|\-]+$', '', val).strip()
            # Filter noise words
            if val and val.lower() not in ['unknown', 'unspecified', 'male', 'female', 'years', 'report', 'details', 'name', 'na']:
                meta["patient_name"] = val
                break

    # Dynamic Patient ID matching
    id_patterns = [
        r"(?:Patient\s*ID|PID|MRN|Case\s*No|Record\s*ID|\bID)\s*[:\-]?\s*([A-Za-z0-9\-]+)",
        r"Patient\s*Report\s*-\s*([A-Za-z0-9\-]+)",
        r"\b([A-Z]\d{3,6})\b"
    ]
    for pat in id_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if val.lower() not in ['unknown', 'unspecified', 'intake', 'patient', 'report', 'details', 'id', 'na']:
                meta["patient_id"] = val.upper()
                break

    return meta

def ingest_documents():
    """
    Ingests all files currently present in reports/, builds a fresh FAISS vectorstore,
    and saves active document metadata.
    """
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from embeddings import get_embeddings_model
    from privacy import mask_sensitive_data

    uploaded_files = [
        f for f in os.listdir(REPORTS_DIR)
        if os.path.isfile(os.path.join(REPORTS_DIR, f)) and not f.endswith(".json")
    ]

    if not uploaded_files:
        logger.warning(f"No document files found in '{REPORTS_DIR}' to index.")
        return False, {}

    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    combined_metadata = {
        "files": uploaded_files,
        "patient_name": "Not specified",
        "patient_id": "Not specified",
        "chunk_count": 0
    }

    for file in uploaded_files:
        file_path = os.path.join(REPORTS_DIR, file)
        raw_text = extract_text_from_file(file_path)
        
        if not raw_text:
            continue

        # Extract dynamic metadata for this file
        file_meta = extract_dynamic_metadata(raw_text)
        if combined_metadata["patient_name"] == "Not specified" and file_meta["patient_name"] != "Not specified":
            combined_metadata["patient_name"] = file_meta["patient_name"]
        if combined_metadata["patient_id"] == "Not specified" and file_meta["patient_id"] != "Not specified":
            combined_metadata["patient_id"] = file_meta["patient_id"]

        # Privacy redaction on raw document
        masked_text = mask_sensitive_data(raw_text)

        chunks = text_splitter.split_text(masked_text)
        for idx, chunk_text in enumerate(chunks):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source_file": file,
                    "chunk_id": idx,
                    "patient_name": file_meta["patient_name"],
                    "patient_id": file_meta["patient_id"]
                }
            )
            all_chunks.append(doc)

    if not all_chunks:
        logger.warning("No readable text could be extracted from uploaded files.")
        return False, {}

    embeddings = get_embeddings_model()
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_DIR)

    combined_metadata["chunk_count"] = len(all_chunks)

    # Save active report metadata
    meta_path = os.path.join(VECTOR_DB_DIR, "active_report_meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(combined_metadata, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write session metadata: {e}")

    return True, combined_metadata
