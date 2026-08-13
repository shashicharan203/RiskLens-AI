import numpy as np
from typing import List, Union

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class FinancialEmbeddingGenerator:
    """Embedding model wrapper using Sentence Transformers for custom RAG indexing."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dimension = 384
        self.model = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"Warning: Could not load SentenceTransformer '{model_name}': {e}. Using deterministic fallback embedder.")
                self.model = None

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate normalized float32 embeddings for text or list of texts."""
        if isinstance(texts, str):
            texts = [texts]
            
        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype(np.float32)
        else:
            return self._fallback_encode(texts)

    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        """Deterministic hashing fallback embedder ensuring zero-dependency compatibility."""
        embeddings = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = text.lower().split()
            for word in words:
                idx = abs(hash(word)) % self.dimension
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)
