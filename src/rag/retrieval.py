import os
import glob
import pandas as pd
from typing import List, Dict, Any, Optional

from src.nlp.document_parser import DocumentParser
from src.rag.chunking import CustomTextChunker
from src.nlp.embeddings import FinancialEmbeddingGenerator
from src.rag.vector_store import FAISSVectorStore

class RAGRetriever:
    """End-to-end RAG retrieval pipeline without LangChain."""

    def __init__(self, vector_store_dir: str = "models/vector_store"):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        self.vector_store_dir = os.path.join(root_dir, vector_store_dir)
        
        self.embedder = FinancialEmbeddingGenerator()
        self.chunker = CustomTextChunker(chunk_size=200, chunk_overlap=30)
        self.vector_store: Optional[FAISSVectorStore] = None
        
        self._init_vector_store()

    def _init_vector_store(self):
        """Load existing vector index or build index from dataset documents."""
        meta_path = os.path.join(self.vector_store_dir, "metadata.json")
        if os.path.exists(meta_path):
            try:
                self.vector_store = FAISSVectorStore.load(self.vector_store_dir, dimension=self.embedder.dimension)
                print(f"Loaded FAISS vector store with {len(self.vector_store.metadata_store)} documents.")
                return
            except Exception as e:
                print(f"Vector store load error: {e}. Rebuilding vector index...")
                
        self.build_index()

    def build_index(
        self, 
        docs_dir: str = "data/documents", 
        news_file: str = "data/financial_news.csv"
    ):
        """Parse documents and news articles, embed, and populate FAISS index."""
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        full_docs_dir = os.path.join(root_dir, docs_dir)
        full_news_file = os.path.join(root_dir, news_file)
        
        all_chunks = []
        
        # 1. Process documents directory
        if os.path.exists(full_docs_dir):
            for file_path in glob.glob(os.path.join(full_docs_dir, "*.*")):
                if os.path.isfile(file_path):
                    try:
                        pages = DocumentParser.parse_document(file_path)
                        chunks = self.chunker.chunk_document_pages(pages)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        print(f"Error parsing document {file_path}: {e}")
                        
        # 2. Process financial news dataset
        if os.path.exists(full_news_file):
            try:
                df_news = pd.read_csv(full_news_file)
                for idx, row in df_news.iterrows():
                    news_text = f"Title: {row.get('title', '')}. Content: {row.get('content', '')}"
                    chunks = self.chunker.chunk_text(
                        news_text,
                        metadata={
                            "document_name": f"News Article ({row.get('company', 'Market')})",
                            "page_number": 1
                        }
                    )
                    all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error parsing news CSV {full_news_file}: {e}")
                
        if not all_chunks:
            # Add default fallback knowledge chunk
            all_chunks = self.chunker.chunk_text(
                "LedgerMind Financial Risk Intelligence: High risk accounts exhibit debt-to-income > 0.50, credit utilization > 80%, and velocity > 10 transactions/hr.",
                metadata={"document_name": "LedgerMind Financial Report 2025", "page_number": 1}
            )
            
        # Encode & store
        contents = [c["content"] for c in all_chunks]
        embeddings = self.embedder.encode(contents)
        
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.vector_store.add_documents(all_chunks, embeddings)
        self.vector_store.save(self.vector_store_dir)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Embed query and search vector database for evidence chunks."""
        if self.vector_store is None:
            self.build_index()
            
        q_emb = self.embedder.encode([query])
        results = self.vector_store.search(q_emb, top_k=top_k)
        return results
