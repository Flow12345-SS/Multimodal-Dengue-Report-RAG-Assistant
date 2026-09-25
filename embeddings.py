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
    model_kwargs = {'device': 'cpu'} # Change to 'cuda' if GPU is available
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    logger.info(f"Embeddings model loaded in {time.time() - start:.2f} seconds.")
    return embeddings
