import os
import io
import tempfile
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Union

from src.nlp.document_parser import DocumentParser
from src.rag.chunking import CustomTextChunker

REQUIRED_TRANSACTION_COLS = [
    'transaction_amount', 'transaction_frequency', 'merchant_category', 'location'
]

OPTIONAL_TRANSACTION_COLS = [
    'time_pattern', 'is_night_transaction', 'is_weekend', 'account_age_days',
    'avg_monthly_income', 'debt_to_income', 'interest_rate', 'credit_utilization',
    'failed_login_attempts', 'device_risk_score'
]

VALID_MERCHANTS = [
    'Grocery', 'Utilities', 'Electronics', 'Luxury_Goods', 
    'Crypto_Exchange', 'Wire_Transfer', 'Gambling', 'Travel'
]

VALID_LOCATIONS = [
    'Domestic', 'International_LowRisk', 'International_HighRisk', 'Online_Unverified'
]

class InputHandler:
    """Input validation & file processing layer for transactions CSV and PDF/TXT documents."""

    @staticmethod
    def validate_and_clean_csv(file_buffer: Union[io.BytesIO, io.StringIO, str, pd.DataFrame]) -> Tuple[bool, str, pd.DataFrame]:
        """Validate transaction CSV columns, handle missing values, and sanitize data types."""
        try:
            if isinstance(file_buffer, pd.DataFrame):
                df = file_buffer.copy()
            else:
                df = pd.read_csv(file_buffer)
        except Exception as e:
            return False, f"Failed to parse CSV file: {str(e)}", pd.DataFrame()

        if df.empty:
            return False, "Uploaded CSV file is empty.", pd.DataFrame()

        # Normalize column names
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        # Check required columns
        missing_required = [col for col in REQUIRED_TRANSACTION_COLS if col not in df.columns]
        if missing_required:
            return False, f"Missing required columns in CSV: {', '.join(missing_required)}. Required columns are: {', '.join(REQUIRED_TRANSACTION_COLS)}", pd.DataFrame()

        # Fill missing optional columns with sensible defaults
        defaults = {
            'time_pattern': 14,
            'is_night_transaction': 0,
            'is_weekend': 0,
            'account_age_days': 180,
            'avg_monthly_income': 5000.0,
            'debt_to_income': 0.35,
            'interest_rate': 7.5,
            'credit_utilization': 0.50,
            'failed_login_attempts': 0,
            'device_risk_score': 0.20
        }

        for col, default_val in defaults.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                df[col] = df[col].fillna(default_val)

        # Sanitize data types and missing values in required fields
        df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce').fillna(100.0)
        df['transaction_frequency'] = pd.to_numeric(df['transaction_frequency'], errors='coerce').fillna(1).astype(int)
        
        # Clean categorical columns
        df['merchant_category'] = df['merchant_category'].astype(str).str.title().str.replace(" ", "_")
        df['merchant_category'] = df['merchant_category'].apply(
            lambda x: x if x in VALID_MERCHANTS else 'Grocery'
        )

        df['location'] = df['location'].astype(str).str.title().str.replace(" ", "_")
        df['location'] = df['location'].apply(
            lambda x: x if x in VALID_LOCATIONS else 'Domestic'
        )

        # Preserve or generate customer_id and customer_name
        if 'customer_id' not in df.columns:
            df['customer_id'] = [f"Customer {i+1:03d}" for i in range(len(df))]
        else:
            df['customer_id'] = df['customer_id'].astype(str).fillna("Customer 001")

        if 'customer_name' not in df.columns:
            df['customer_name'] = df['customer_id']
        else:
            df['customer_name'] = df['customer_name'].astype(str).fillna(df['customer_id'])

        return True, f"Successfully validated and cleaned {len(df)} transactions.", df

    @staticmethod
    def process_uploaded_document_bytes(
        file_name: str, 
        file_bytes: bytes, 
        retriever_instance
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """Parse uploaded PDF or TXT file from bytes, chunk text, embed, and index into FAISS vector store."""
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in ['.pdf', '.txt', '.md']:
            return False, f"Unsupported file format '{ext}'. Only PDF and TXT files are supported.", []

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            pages = DocumentParser.parse_document(tmp_path)
            os.remove(tmp_path)

            if not pages:
                return False, f"No text could be extracted from {file_name}.", []

            # Override document name with original uploaded file name
            for p in pages:
                p["document_name"] = file_name

            chunker = CustomTextChunker(chunk_size=200, chunk_overlap=30)
            chunks = chunker.chunk_document_pages(pages)

            if not chunks:
                return False, f"Could not create text chunks from {file_name}.", []

            contents = [c["content"] for c in chunks]
            embeddings = retriever_instance.embedder.encode(contents)

            retriever_instance.vector_store.add_documents(chunks, embeddings)
            retriever_instance.vector_store.save(retriever_instance.vector_store_dir)

            return True, f"Successfully parsed, embedded, and indexed {len(chunks)} text chunks from '{file_name}'.", chunks

        except Exception as e:
            return False, f"Error processing document '{file_name}': {str(e)}", []
