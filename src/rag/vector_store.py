import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

class FAISSVectorStore:
    """Manual FAISS vector store implementation for RAG retrieval."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.metadata_store: List[Dict[str, Any]] = []
        
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized cosine similarity
        else:
            print("FAISS not installed. Using numpy cosine similarity vector store fallback.")
            self.index = None
            self.vectors_np: Optional[np.ndarray] = None

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add text chunks and their embeddings to vector store index."""
        if len(chunks) != len(embeddings):
            raise ValueError("Length of chunks and embeddings must match.")
            
        embeddings_norm = embeddings.copy()
        # Ensure L2 normalization for Inner Product / Cosine Similarity
        norms = np.linalg.norm(embeddings_norm, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings_norm = embeddings_norm / norms
        
        if HAS_FAISS and self.index is not None:
            self.index.add(embeddings_norm.astype(np.float32))
        else:
            if self.vectors_np is None:
                self.vectors_np = embeddings_norm.astype(np.float32)
            else:
                self.vectors_np = np.vstack([self.vectors_np, embeddings_norm.astype(np.float32)])
                
        self.metadata_store.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        """Search top-k most similar document chunks."""
        if len(self.metadata_store) == 0:
            return []
            
        q_norm = query_vector.copy()
        if len(q_norm.shape) == 1:
            q_norm = q_norm.reshape(1, -1)
            
        norm = np.linalg.norm(q_norm, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        q_norm = (q_norm / norm).astype(np.float32)
        
        results = []
        
        if HAS_FAISS and self.index is not None:
            scores, indices = self.index.search(q_norm, min(top_k, len(self.metadata_store)))
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata_store):
                    item = dict(self.metadata_store[idx])
                    item["similarity_score"] = float(score)
                    results.append(item)
        else:
            sims = np.dot(self.vectors_np, q_norm.T).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            for idx in top_indices:
                item = dict(self.metadata_store[idx])
                item["similarity_score"] = float(sims[idx])
                results.append(item)
                
        return results

    def save(self, dir_path: str):
        """Serialize FAISS index and chunk metadata."""
        os.makedirs(dir_path, exist_ok=True)
        meta_path = os.path.join(dir_path, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, indent=2)
            
        if HAS_FAISS and self.index is not None:
            index_path = os.path.join(dir_path, "faiss_index.bin")
            faiss.write_index(self.index, index_path)
        else:
            np_path = os.path.join(dir_path, "vectors.npy")
            if self.vectors_np is not None:
                np.save(np_path, self.vectors_np)
                
        print(f"Vector store saved to {dir_path}")

    @classmethod
    def load(cls, dir_path: str, dimension: int = 384) -> "FAISSVectorStore":
        """Load serialized vector store."""
        instance = cls(dimension=dimension)
        meta_path = os.path.join(dir_path, "metadata.json")
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata store not found at {meta_path}")
            
        with open(meta_path, "r", encoding="utf-8") as f:
            instance.metadata_store = json.load(f)
            
        index_path = os.path.join(dir_path, "faiss_index.bin")
        np_path = os.path.join(dir_path, "vectors.npy")
        
        if HAS_FAISS and os.path.exists(index_path):
            instance.index = faiss.read_index(index_path)
        elif os.path.exists(np_path):
            instance.vectors_np = np.load(np_path)
            
        return instance
