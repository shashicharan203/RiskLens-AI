import os
import joblib
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, Union

from src.data_processing.feature_engineering import FeatureEngineer
from src.models.autoencoder import compute_anomaly_scores, normalize_anomaly_score

class RiskPredictor:
    """Inference predictor for financial transaction risk scoring & anomaly detection."""

    def __init__(self, model_path: str = "models/risk_model.pkl"):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        full_path = os.path.join(root_dir, model_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Model artifact not found at {full_path}. Run `python src/models/train.py` first."
            )
            
        payload = joblib.load(full_path)
        self.model = payload['model']
        self.model_type = payload['model_type']
        self.preprocessor = payload['preprocessor']
        self.autoencoder = payload['autoencoder']
        self.feature_names = payload['feature_names']
        self.all_model_metrics = payload.get('all_model_metrics', {})
        self.ae_stats = payload.get('ae_baseline_stats', {'mean': 0.5, 'std': 0.8})

    def _determine_risk_level(self, score: float) -> str:
        if score >= 0.70:
            return "HIGH"
        elif score >= 0.35:
            return "MEDIUM"
        else:
            return "LOW"

    def predict_single(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Predict risk score, anomaly score, combined risk assessment, and human review flag."""
        df_single = pd.DataFrame([input_dict])
        
        # Feature Engineering
        df_eng = FeatureEngineer.add_engineered_features(df_single)
        
        # Preprocessing
        X_proc = self.preprocessor.transform(df_eng)
        
        # PyTorch Autoencoder Anomaly Score
        raw_ae_errors = compute_anomaly_scores(self.autoencoder, X_proc.values)
        raw_mse = float(raw_ae_errors[0])
        anomaly_score, anomaly_status = normalize_anomaly_score(
            raw_mse, 
            baseline_mean=self.ae_stats['mean'], 
            baseline_std=self.ae_stats['std']
        )
        
        X_proc['ae_anomaly_score'] = raw_ae_errors
        X_final = X_proc[self.feature_names]
        
        # Model Risk Prediction
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_final)
            risk_score = float(probs[0][1])
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X_final)
            risk_score = float(1.0 / (1.0 + np.exp(-scores[0])))
        else:
            risk_score = float(self.model.predict(X_final)[0])
            
        risk_score = round(float(np.clip(risk_score, 0.0, 1.0)), 2)
        risk_level = self._determine_risk_level(risk_score)
        
        # Combined Risk Assessment:
        # Combined score blends supervised ML probability (65% weight) with unsupervised deep AE anomaly magnitude (35% weight)
        combined_score = round(float(np.clip(0.65 * risk_score + 0.35 * anomaly_score, 0.0, 1.0)), 2)
        combined_level = self._determine_risk_level(combined_score)
        
        # Human-in-the-Loop Trigger
        requires_human_review = (risk_level == "HIGH" or anomaly_status == "HIGHLY ANOMALOUS" or combined_level == "HIGH")
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "anomaly_score": anomaly_score,
            "anomaly_status": anomaly_status,
            "combined_risk_score": combined_score,
            "combined_risk_level": combined_level,
            "requires_human_review": requires_human_review,
            "review_status": "Requires Human Review" if requires_human_review else "Standard Automated Clearance"
        }

    def predict_batch(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        """Predict risk & anomaly scores for a batch of transactions."""
        df_eng = FeatureEngineer.add_engineered_features(df_batch)
        X_proc = self.preprocessor.transform(df_eng)
        ae_scores = compute_anomaly_scores(self.autoencoder, X_proc.values)
        X_proc['ae_anomaly_score'] = ae_scores
        X_final = X_proc[self.feature_names]
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_final)[:, 1]
        else:
            probs = self.model.predict(X_final)
            
        df_res = df_batch.copy()
        df_res['risk_score'] = np.round(probs, 2)
        df_res['risk_level'] = df_res['risk_score'].apply(self._determine_risk_level)
        return df_res
