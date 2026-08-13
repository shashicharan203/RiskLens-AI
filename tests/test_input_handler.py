import io
import pytest
import pandas as pd
from src.data_processing.input_handler import InputHandler
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_validate_and_clean_csv_valid():
    csv_data = """transaction_amount,transaction_frequency,merchant_category,location,debt_to_income
1500.0,5,Crypto_Exchange,International_HighRisk,0.65
300.0,2,Grocery,Domestic,0.20
"""
    valid, msg, df_clean = InputHandler.validate_and_clean_csv(io.StringIO(csv_data))
    assert valid is True
    assert len(df_clean) == 2
    assert "transaction_amount" in df_clean.columns
    assert "is_night_transaction" in df_clean.columns  # Default column added

def test_validate_and_clean_csv_missing_cols():
    csv_data = """amount,frequency
1500.0,5
"""
    valid, msg, df_clean = InputHandler.validate_and_clean_csv(io.StringIO(csv_data))
    assert valid is False
    assert "Missing required columns" in msg

def test_upload_transactions_csv_api_endpoint():
    csv_bytes = b"transaction_amount,transaction_frequency,merchant_category,location\n2500.0,10,Wire_Transfer,International_HighRisk\n50.0,1,Grocery,Domestic\n"
    files = {"file": ("test_tx.csv", csv_bytes, "text/csv")}
    response = client.post("/upload-transactions-csv", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_transactions"] == 2
    assert "predictions" in data

def test_upload_document_api_endpoint():
    txt_bytes = b"LedgerMind Financial Risk Report: High risk accounts exhibit debt-to-income over 0.50."
    files = {"file": ("uploaded_policy.txt", txt_bytes, "text/plain")}
    response = client.post("/upload-document", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["document_name"] == "uploaded_policy.txt"
