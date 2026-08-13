import pytest
import os
from src.models.human_review import HumanReviewStore
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_human_review_store(tmp_path):
    store_file = os.path.join(tmp_path, "test_reviews.json")
    store = HumanReviewStore(file_path=str(store_file))
    
    record = store.record_review(
        transaction_id="TXN_TEST_001",
        transaction_details={"amount": 5000.0},
        risk_score=0.88,
        anomaly_score=0.92,
        decision="REJECT",
        comments="High risk wire transfer",
        analyst_id="Analyst_Test"
    )
    
    assert record["decision"] == "REJECT"
    assert record["transaction_id"] == "TXN_TEST_001"
    assert os.path.exists(store_file)
    
    all_revs = store.get_all_reviews()
    assert len(all_revs) == 1
    assert all_revs[0]["analyst_id"] == "Analyst_Test"

def test_review_transaction_api_endpoint():
    payload = {
        "transaction_id": "TXN_API_999",
        "transaction_details": {"transaction_amount": 10000.0},
        "risk_score": 0.85,
        "anomaly_score": 0.78,
        "decision": "INVESTIGATE",
        "comments": "Flagged for manual compliance check",
        "analyst_id": "Analyst_Compliance"
    }
    response = client.post("/review-transaction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["review_record"]["decision"] == "INVESTIGATE"
    
    get_resp = client.get("/reviews")
    assert get_resp.status_code == 200
    reviews_data = get_resp.json()
    assert "reviews" in reviews_data
    assert len(reviews_data["reviews"]) > 0
