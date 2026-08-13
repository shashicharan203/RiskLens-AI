import pytest
from src.nlp.news_sentiment import FinancialNewsAnalyzer
from src.recommendation.advisor import RiskAdvisorEngine
from src.simulation.what_if import WhatIfSimulator

def test_news_sentiment():
    analyzer = FinancialNewsAnalyzer()
    res = analyzer.analyze_news("Apex Financial Group faces quarterly loss as non-performing assets rise 18%.")
    assert "sentiment" in res
    assert "entity" in res
    assert "impact" in res
    assert res["impact"] in ["High Risk", "Medium Risk", "Low Risk"]

def test_recommendation_advisor():
    advisor = RiskAdvisorEngine()
    rec = advisor.generate_recommendations(risk_score=0.82, shap_factors=["+ High transaction amount"])
    assert "recommendations" in rec
    assert len(rec["recommendations"]) > 0

def test_what_if_simulation():
    simulator = WhatIfSimulator()
    base_tx = {
        "transaction_amount": 1000.0, "transaction_frequency": 5, "merchant_category": "Grocery",
        "location": "Domestic", "time_pattern": 14, "is_night_transaction": 0, "is_weekend": 0,
        "account_age_days": 365, "avg_monthly_income": 6000.0, "debt_to_income": 0.30,
        "interest_rate": 6.0, "credit_utilization": 0.40, "failed_login_attempts": 0, "device_risk_score": 0.10
    }
    sim = simulator.simulate_scenario(base_tx, {"interest_rate": 14.0})
    assert "baseline" in sim
    assert "scenario" in sim
    assert "delta" in sim
