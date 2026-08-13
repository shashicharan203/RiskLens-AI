import re
from typing import List, Dict, Any

class CustomTextChunker:
    """Manual text chunking implementation without LangChain."""

    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 40):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split raw text into overlapping token/word chunks with metadata."""
        if metadata is None:
            metadata = {}
            
        # Clean text
        text_clean = re.sub(r'\s+', ' ', text).strip()
        words = text_clean.split(' ')
        
        if len(words) <= self.chunk_size:
            return [{
                "chunk_id": f"{metadata.get('document_name', 'doc')}_p{metadata.get('page_number', 1)}_c0",
                "content": text_clean,
                "document_name": metadata.get("document_name", "unknown"),
                "page_number": metadata.get("page_number", 1),
                "chunk_index": 0
            }]
            
        chunks = []
        start_idx = 0
        chunk_count = 0
        step = self.chunk_size - self.chunk_overlap
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_str = ' '.join(chunk_words)
            
            chunks.append({
                "chunk_id": f"{metadata.get('document_name', 'doc')}_p{metadata.get('page_number', 1)}_c{chunk_count}",
                "content": chunk_str,
                "document_name": metadata.get("document_name", "unknown"),
                "page_number": metadata.get("page_number", 1),
                "chunk_index": chunk_count
            })
            
            chunk_count += 1
            start_idx += step
            if end_idx >= len(words):
                break
                
        return chunks

    def chunk_document_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk a list of document page dicts extracted from document parser."""
        all_chunks = []
        for page in pages:
            doc_name = page.get("document_name", "document")
            page_num = page.get("page_number", 1)
            content = page.get("content", "")
            
            page_chunks = self.chunk_text(
                content, 
                metadata={"document_name": doc_name, "page_number": page_num}
            )
            all_chunks.extend(page_chunks)
        return all_chunks
