import os
from typing import List, Dict, Any

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

class DocumentParser:
    """Document extraction utility for PDF and text financial reports using PyMuPDF."""

    @staticmethod
    def parse_pdf(pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text page by page from PDF file."""
        if not HAS_PYMUPDF:
            print("PyMuPDF (fitz) is not installed. Falling back to plain text reading if applicable.")
            return []
            
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
        pages_content = []
        doc = fitz.open(pdf_path)
        file_name = os.path.basename(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages_content.append({
                    "document_name": file_name,
                    "page_number": page_num + 1,
                    "content": text
                })
        doc.close()
        return pages_content

    @staticmethod
    def parse_text_file(text_path: str) -> List[Dict[str, Any]]:
        """Extract text from plain text file."""
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Text file not found at {text_path}")
            
        file_name = os.path.basename(text_path)
        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        return [{
            "document_name": file_name,
            "page_number": 1,
            "content": content
        }]

    @classmethod
    def parse_document(cls, file_path: str) -> List[Dict[str, Any]]:
        """Unified document parser dispatching based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return cls.parse_pdf(file_path)
        elif ext in [".txt", ".md", ".csv"]:
            return cls.parse_text_file(file_path)
        else:
            raise ValueError(f"Unsupported document format: {ext}")
