import pytest
import os
import pandas as pd
from src.data_processing.generate_sample_data import generate_transactions
from src.models.train import train_and_save_pipeline
from src.models.predict import RiskPredictor
from src.explainability.shap_explainer import ShapRiskExplainer

def test_model_training_and_prediction(tmp_path):
    model_file = os.path.join(tmp_path, "test_risk_model.pkl")
    train_and_save_pipeline(output_model_path=str(model_file))
    assert os.path.exists(model_file)
    
    predictor = RiskPredictor(model_path=str(model_file))
    sample_payload = {
        "transaction_amount": 8500.0,
        "transaction_frequency": 18,
        "merchant_category": "Crypto_Exchange",
        "location": "International_HighRisk",
        "time_pattern": 3,
        "is_night_transaction": 1,
        "is_weekend": 0,
        "account_age_days": 120,
        "avg_monthly_income": 4500.0,
        "debt_to_income": 0.70,
        "interest_rate": 10.5,
        "credit_utilization": 0.90,
        "failed_login_attempts": 4,
        "device_risk_score": 0.88
    }
    
    res = predictor.predict_single(sample_payload)
    assert "risk_score" in res
    assert "risk_level" in res
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

def test_shap_explainer():
    explainer = ShapRiskExplainer()
    sample_payload = {
        "transaction_amount": 12000.0,
        "transaction_frequency": 25,
        "merchant_category": "Wire_Transfer",
        "location": "International_HighRisk",
        "time_pattern": 2,
        "is_night_transaction": 1,
        "is_weekend": 0,
        "account_age_days": 90,
        "avg_monthly_income": 5000.0,
        "debt_to_income": 0.75,
        "interest_rate": 12.0,
        "credit_utilization": 0.95,
        "failed_login_attempts": 5,
        "device_risk_score": 0.95
    }
    explanation = explainer.explain_transaction(sample_payload)
    assert "risk_score" in explanation
    assert "important_factors" in explanation
    assert len(explanation["important_factors"]) > 0
