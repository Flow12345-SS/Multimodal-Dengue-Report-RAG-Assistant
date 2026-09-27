import os
import streamlit as st

@st.cache_resource
def get_embeddings_model():
    """
    Initializes and returns the SentenceTransformers embedding model.
    Cached via Streamlit to prevent reloading on every script run.
    """
    import time
    from langchain_huggingface import HuggingFaceEmbeddings
    from utils.logger import get_logger
    logger = get_logger(__name__)
    
    logger.info("Loading embeddings model...")
    start = time.time()

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    encode_kwargs = {'normalize_embeddings': True}
    
    # 1. Try loading from local cache first (instant if already present)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu', 'local_files_only': True},
            encode_kwargs=encode_kwargs
        )
    except Exception as e:
        logger.info(f"Local embeddings cache not found ({e}), downloading for cloud deployment...")
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs=encode_kwargs
        )
    
    logger.info(f"Embeddings model loaded in {time.time() - start:.2f} seconds.")
    return embeddings
