import os
import re
import numpy as np
import streamlit as st
from langchain_core.embeddings import Embeddings
from utils.logger import get_logger

logger = get_logger(__name__)


class FastClinicalEmbeddings(Embeddings):
    """
    Self-contained, zero-network deterministic clinical embeddings.
    Guarantees 100% uptime on cloud hosting platforms (like Streamlit Cloud)
    without relying on external network downloads or facing LocalEntryNotFoundError.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.boost_words = {
            'diagnosis', 'platelet', 'count', 'ns1', 'antigen', 'igm', 'igg',
            'dengue', 'fever', 'risk', 'recommendations', 'patient', 'name',
            'age', 'gender', 'hematocrit', 'wbc', 'normal', 'thrombocytopenia'
        }

    def _embed(self, text: str) -> list[float]:
        import hashlib
        v = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r'\b[a-zA-Z0-9_\-\.]{2,}\b', str(text).lower())
        if not words:
            return v.tolist()
        for i, w in enumerate(words):
            weight = 3.0 if w in self.boost_words else 1.0
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 1) & 1 else -1.0
            v[idx] += sign * weight
            if i > 0:
                bg = f'{words[i-1]}_{w}'
                h_bg = int(hashlib.md5(bg.encode('utf-8')).hexdigest(), 16)
                idx_bg = h_bg % self.dim
                sign_bg = 1.0 if (h_bg >> 1) & 1 else -1.0
                v[idx_bg] += sign_bg * (weight * 1.5)
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        return v.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


def is_cloud_deployment() -> bool:
    """Detect if running in cloud hosting (e.g. Streamlit Cloud /mount/src)."""
    return (
        os.path.exists("/mount/src") or
        "STREAMLIT_SHARING_HOST" in os.environ or
        os.environ.get("STREAMLIT_DEPLOYMENT") == "1" or
        "SPACES_ZERO_GPU" in os.environ
    )


@st.cache_resource
def get_embeddings_model():
    """
    Initializes and returns the embeddings model.
    In cloud deployment (Streamlit Cloud), uses FastClinicalEmbeddings directly
    to prevent network timeouts, LocalEntryNotFoundError, and OSError.
    In local environment, uses HuggingFace SentenceTransformers with fallback.
    """
    # Cloud deployment: use zero-network FastClinicalEmbeddings immediately
    if is_cloud_deployment():
        logger.info("Cloud deployment detected (/mount/src). Using FastClinicalEmbeddings engine.")
        return FastClinicalEmbeddings()

    # Local development: try HuggingFace SentenceTransformers
    logger.info("Initializing embeddings model for local environment...")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        encode_kwargs = {'normalize_embeddings': True}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs=encode_kwargs
        )
        logger.info("Loaded HuggingFace SentenceTransformer embeddings successfully.")
        return embeddings
    except BaseException as e:
        logger.warning(f"HuggingFace Hub unavailable ({e}). Using FastClinicalEmbeddings fallback.")
        return FastClinicalEmbeddings()
