import pytest
import pandas as pd
import numpy as np
from src.data_processing.generate_sample_data import generate_transactions
from src.data_processing.feature_engineering import FeatureEngineer
from src.data_processing.preprocess import DataPreprocessor
from src.models.risk_model import FinancialRiskClassifier
from src.models.autoencoder import train_autoencoder, evaluate_autoencoder

def test_model_evaluation_metrics():
    df = generate_transactions(num_samples=100)
    df_eng = FeatureEngineer.add_engineered_features(df)
    X = df_eng.drop(columns=['transaction_id', 'account_id', 'is_risk'], errors='ignore')
    y = df_eng['is_risk']
    
    preprocessor = DataPreprocessor()
    X_proc = preprocessor.fit_transform(X)
    
    for algo in ['logistic_regression', 'random_forest', 'xgboost']:
        clf = FinancialRiskClassifier(model_type=algo)
        clf.fit(X_proc, y)
        metrics = clf.evaluate(X_proc, y)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "confusion_matrix" in metrics
        assert "roc_curve" in metrics
        assert len(metrics["confusion_matrix"]) == 2

def test_autoencoder_evaluation_metrics():
    df = generate_transactions(num_samples=80)
    df_eng = FeatureEngineer.add_engineered_features(df)
    X = df_eng.drop(columns=['transaction_id', 'account_id', 'is_risk'], errors='ignore')
    y = df_eng['is_risk']
    
    preprocessor = DataPreprocessor()
    X_proc = preprocessor.fit_transform(X)
    
    model, _ = train_autoencoder(X_proc.values, epochs=5)
    ae_metrics = evaluate_autoencoder(model, X_proc.values, y.values)
    
    assert "accuracy" in ae_metrics
    assert "precision" in ae_metrics
    assert "recall" in ae_metrics
    assert "f1_score" in ae_metrics
    assert "roc_auc" in ae_metrics
    assert "confusion_matrix" in ae_metrics
