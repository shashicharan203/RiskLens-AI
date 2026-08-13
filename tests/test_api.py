import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_risk_endpoint():
    payload = {
        "transaction_amount": 4500.0,
        "transaction_frequency": 12,
        "merchant_category": "Crypto_Exchange",
        "location": "International_HighRisk",
        "time_pattern": 2,
        "is_night_transaction": 1,
        "is_weekend": 0,
        "account_age_days": 180,
        "avg_monthly_income": 6000.0,
        "debt_to_income": 0.65,
        "interest_rate": 8.5,
        "credit_utilization": 0.88,
        "failed_login_attempts": 4,
        "device_risk_score": 0.85
    }
    response = client.post("/predict-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_level" in data

def test_explain_risk_endpoint():
    payload = {
        "transaction_amount": 4500.0,
        "transaction_frequency": 12,
        "merchant_category": "Crypto_Exchange",
        "location": "International_HighRisk",
        "time_pattern": 2,
        "is_night_transaction": 1,
        "is_weekend": 0,
        "account_age_days": 180,
        "avg_monthly_income": 6000.0,
        "debt_to_income": 0.65,
        "interest_rate": 8.5,
        "credit_utilization": 0.88,
        "failed_login_attempts": 4,
        "device_risk_score": 0.85
    }
    response = client.post("/explain-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "important_factors" in data

def test_retrieve_evidence_endpoint():
    payload = {
        "query": "Why is this company considered high risk?",
        "top_k": 2
    }
    response = client.post("/retrieve-evidence", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
