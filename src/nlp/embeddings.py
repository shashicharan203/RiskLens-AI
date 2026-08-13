import numpy as np
from typing import List, Union

class FinancialEmbeddingGenerator:
    """Embedding model wrapper using Sentence Transformers for custom RAG indexing (Lazy-loaded)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dimension = 384
        self.model = None
        self._initialized = False

    def _get_model(self):
        """Lazy load SentenceTransformer on demand."""
        if not self._initialized:
            self._initialized = True
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                try:
                    self.dimension = self.model.get_embedding_dimension()
                except AttributeError:
                    self.dimension = self.model.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"Warning: Could not load SentenceTransformer '{self.model_name}': {e}. Using deterministic fallback embedder.")
                self.model = None
        return self.model

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate normalized float32 embeddings for text or list of texts."""
        if isinstance(texts, str):
            texts = [texts]
            
        model = self._get_model()
        if model is not None:
            try:
                embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings.astype(np.float32)
            except Exception as e:
                print(f"Warning: SentenceTransformer encode failed ({e}). Falling back to hashing embedder.")
                return self._fallback_encode(texts)
        else:
            return self._fallback_encode(texts)

    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        """Deterministic hashing fallback embedder ensuring zero-dependency & low-memory compatibility."""
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
