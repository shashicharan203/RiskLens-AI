import pytest
from src.rag.chunking import CustomTextChunker
from src.nlp.embeddings import FinancialEmbeddingGenerator
from src.rag.vector_store import FAISSVectorStore
from src.rag.retrieval import RAGRetriever
from src.llm.generator import CustomRAGGenerator

def test_custom_text_chunker():
    chunker = CustomTextChunker(chunk_size=10, chunk_overlap=2)
    text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10 Word11 Word12 Word13"
    chunks = chunker.chunk_text(text, metadata={"document_name": "test.txt", "page_number": 1})
    assert len(chunks) > 1
    assert "content" in chunks[0]

def test_faiss_vector_store():
    embedder = FinancialEmbeddingGenerator()
    store = FAISSVectorStore(dimension=embedder.dimension)
    
    texts = ["Corporate debt increased dramatically.", "Tech company revenue grew by 25%."]
    chunks = [
        {"chunk_id": "c1", "content": texts[0], "document_name": "doc1.txt", "page_number": 1},
        {"chunk_id": "c2", "content": texts[1], "document_name": "doc2.txt", "page_number": 1}
    ]
    
    embeddings = embedder.encode(texts)
    store.add_documents(chunks, embeddings)
    
    q_vec = embedder.encode(["debt loss liability"])
    results = store.search(q_vec, top_k=1)
    assert len(results) == 1
    assert "similarity_score" in results[0]

def test_rag_generator():
    generator = CustomRAGGenerator()
    evidence = [
        {"content": "Annual report shows debt obligations of $50M.", "document_name": "Annual Report", "page_number": 12}
    ]
    resp = generator.generate_response("Why is this risky?", evidence)
    assert "answer" in resp
    assert "Customer Risk" in resp["answer"] or "Policy Evidence" in resp["answer"]
    assert "Annual Report" in resp["answer"]
