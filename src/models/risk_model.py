import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix, roc_curve
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

class FinancialRiskClassifier:
    """Multi-algorithm risk prediction model supporting Logistic Regression, Random Forest, and XGBoost."""

    def __init__(self, model_type: str = 'xgboost', use_smote: bool = True):
        self.model_type = model_type.lower()
        self.use_smote = use_smote
        self.model = None
        self._init_model()

    def _init_model(self):
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=150, class_weight='balanced', max_depth=12, random_state=42
            )
        elif self.model_type == 'xgboost':
            if HAS_XGBOOST:
                self.model = XGBClassifier(
                    n_estimators=150, max_depth=6, learning_rate=0.05,
                    scale_pos_weight=4.0, random_state=42, eval_metric='logloss'
                )
            else:
                print("XGBoost not found, using GradientBoostingClassifier fallback.")
                self.model = GradientBoostingClassifier(
                    n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42
                )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "FinancialRiskClassifier":
        """Fit model with optional SMOTE resampling."""
        if self.use_smote and HAS_SMOTE:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train
            
        self.model.fit(X_resampled, y_resampled)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw risk probabilities."""
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            return probs[:, 1]
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-scores))
        else:
            return self.model.predict(X).astype(float)

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """Return binary risk decision based on threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.50) -> Dict[str, Any]:
        """Compute full production metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, & ROC points."""
        probs = self.predict_proba(X_test)
        preds = (probs >= threshold).astype(int)
        
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        
        try:
            roc_auc = float(roc_auc_score(y_test, probs))
        except Exception:
            roc_auc = 0.0
            
        precision_arr, recall_arr, _ = precision_recall_curve(y_test, probs)
        pr_auc = float(auc(recall_arr, precision_arr))
        
        cm = confusion_matrix(y_test, preds).tolist()
        
        fpr, tpr, _ = roc_curve(y_test, probs)
        
        # Extract feature importances if available
        feature_importances = {}
        if hasattr(self.model, "feature_importances_"):
            for feat, imp in zip(X_test.columns, self.model.feature_importances_):
                feature_importances[feat] = round(float(imp), 4)
        elif hasattr(self.model, "coef_"):
            for feat, coef in zip(X_test.columns, self.model.coef_[0]):
                feature_importances[feat] = round(float(abs(coef)), 4)

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": cm,
            "roc_curve": {
                "fpr": [round(x, 4) for x in fpr.tolist()],
                "tpr": [round(x, 4) for x in tpr.tolist()]
            },
            "feature_importances": feature_importances
        }
