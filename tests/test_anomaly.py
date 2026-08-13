import pytest
import numpy as np
from src.models.autoencoder import normalize_anomaly_score
from src.models.predict import RiskPredictor

def test_normalize_anomaly_score():
    score_normal, status_normal = normalize_anomaly_score(raw_mse=0.1, baseline_mean=0.5, baseline_std=0.8)
    assert 0.0 <= score_normal <= 1.0
    assert status_normal in ["NORMAL", "SUSPICIOUS", "HIGHLY ANOMALOUS"]

    score_high, status_high = normalize_anomaly_score(raw_mse=3.5, baseline_mean=0.5, baseline_std=0.8)
    assert score_high >= 0.70
    assert status_high == "HIGHLY ANOMALOUS"

def test_anomaly_predict_integration():
    predictor = RiskPredictor()
    sample_high_risk = {
        "transaction_amount": 18000.0,
        "transaction_frequency": 35,
        "merchant_category": "Wire_Transfer",
        "location": "International_HighRisk",
        "time_pattern": 3,
        "is_night_transaction": 1,
        "is_weekend": 0,
        "account_age_days": 60,
        "avg_monthly_income": 4000.0,
        "debt_to_income": 0.85,
        "interest_rate": 14.0,
        "credit_utilization": 0.95,
        "failed_login_attempts": 6,
        "device_risk_score": 0.98
    }
    res = predictor.predict_single(sample_high_risk)
    assert "anomaly_score" in res
    assert "anomaly_status" in res
    assert "combined_risk_score" in res
    assert "combined_risk_level" in res
    assert "requires_human_review" in res
    assert res["requires_human_review"] is True
