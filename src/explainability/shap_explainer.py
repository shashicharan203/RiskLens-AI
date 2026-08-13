import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.data_processing.feature_engineering import FeatureEngineer
from src.models.autoencoder import compute_anomaly_scores
from src.models.predict import RiskPredictor

FEATURE_DISPLAY_MAP = {
    "transaction_amount": "High transaction amount",
    "transaction_frequency": "Unusual transaction frequency",
    "merchant_category_Crypto_Exchange": "High-risk crypto merchant category",
    "merchant_category_Wire_Transfer": "International wire transfer merchant",
    "merchant_category_Gambling": "High-risk gambling merchant",
    "location_International_HighRisk": "New / High-risk international location",
    "location_Online_Unverified": "Unverified online transaction source",
    "is_night_transaction": "Abnormal off-hours time pattern",
    "credit_utilization": "High credit card balance utilization",
    "debt_to_income": "Elevated debt-to-income ratio",
    "failed_login_attempts": "Multiple failed login security flags",
    "device_risk_score": "Unrecognized or high-risk device hardware signature",
    "auth_anomaly_score": "Abnormal authentication behavior pattern",
    "credit_debt_stress_score": "Severe financial credit stress indicator",
    "ae_anomaly_score": "Deep learning autoencoder transaction anomaly",
    "is_high_risk_merchant": "Flagged high-risk merchant category",
    "is_high_risk_location": "Flagged high-risk transaction location",
    "amount_to_income_ratio": "Excessive amount relative to monthly income"
}

class ShapRiskExplainer:
    """SHAP-based Explainable AI module for financial transaction risk scoring (Lazy-loaded)."""

    def __init__(self, model_path: str = "models/risk_model.pkl"):
        self.predictor = RiskPredictor(model_path=model_path)
        self.model = self.predictor.model
        self.feature_names = self.predictor.feature_names
        self.explainer = None
        self._initialized = False

    def _get_explainer(self):
        """Lazy load SHAP explainer engine on demand."""
        if not self._initialized:
            self._initialized = True
            try:
                import shap
                if hasattr(self.model, "predict_proba"):
                    self.explainer = shap.Explainer(self.model, feature_names=self.feature_names)
                else:
                    self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                print(f"Notice: SHAP explainer fallback ({e}). Using feature importance attribution.")
                self.explainer = None
        return self.explainer

    def explain_transaction(self, input_dict: Dict[str, Any], top_k: int = 4) -> Dict[str, Any]:
        """Explain why a specific transaction is risky using SHAP feature attributions."""
        prediction = self.predictor.predict_single(input_dict)
        risk_score = prediction['risk_score']
        risk_level = prediction['risk_level']
        
        # Process input to matching feature space
        df_single = pd.DataFrame([input_dict])
        df_eng = FeatureEngineer.add_engineered_features(df_single)
        X_proc = self.predictor.preprocessor.transform(df_eng)
        ae_score = compute_anomaly_scores(self.predictor.autoencoder, X_proc.values)
        X_proc['ae_anomaly_score'] = ae_score
        X_final = X_proc[self.feature_names]
        
        # Calculate SHAP values
        shap_vals_dict = {}
        explainer = self._get_explainer()
        if explainer is not None:
            try:
                shap_output = explainer(X_final)
                if len(shap_output.values.shape) == 3:  # Binary classification output shape (1, num_features, 2)
                    shap_array = shap_output.values[0, :, 1]
                else:
                    shap_array = shap_output.values[0]
                    
                for idx, feat in enumerate(self.feature_names):
                    shap_vals_dict[feat] = float(shap_array[idx])
            except Exception as e:
                print(f"SHAP calculation fallback: {e}")
                shap_vals_dict = self._fallback_feature_importance(X_final)
        else:
            shap_vals_dict = self._fallback_feature_importance(X_final)
            
        # Format top risk drivers
        sorted_features = sorted(shap_vals_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        
        important_factors = []
        for feat, val in sorted_features[:top_k]:
            direction = "+" if val > 0 else "-"
            display_name = FEATURE_DISPLAY_MAP.get(feat, feat.replace("_", " ").title())
            important_factors.append(f"{direction} {display_name}")
            
        # Human readable summary block matching prompt spec
        summary_lines = [
            f"Risk Score: {int(risk_score * 100)}%",
            f"Risk Level: {risk_level}",
            "",
            "Important Factors:"
        ] + important_factors

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "important_factors": important_factors,
            "explanation_text": "\n".join(summary_lines),
            "shap_values": {feat: round(val, 4) for feat, val in sorted_features}
        }

    def _fallback_feature_importance(self, X_final: pd.DataFrame) -> Dict[str, float]:
        """Fallback feature attribution using feature magnitudes when SHAP engine is unavailable."""
        res = {}
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            for idx, feat in enumerate(self.feature_names):
                val = float(X_final.iloc[0, idx])
                res[feat] = float(importances[idx]) * (1.0 if val > 0 else -0.5)
        else:
            for idx, feat in enumerate(self.feature_names):
                res[feat] = float(X_final.iloc[0, idx]) * 0.1
        return res
