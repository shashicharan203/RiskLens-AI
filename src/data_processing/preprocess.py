import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataPreprocessor:
    """Production-grade preprocessor for financial transaction data."""
    
    NUMERICAL_COLS = [
        'transaction_amount', 'transaction_frequency', 'account_age_days',
        'avg_monthly_income', 'debt_to_income', 'interest_rate',
        'credit_utilization', 'failed_login_attempts', 'device_risk_score',
        'time_pattern'
    ]
    
    CATEGORICAL_COLS = [
        'merchant_category', 'location'
    ]
    
    BINARY_COLS = [
        'is_night_transaction', 'is_weekend'
    ]

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fit ColumnTransformer on training dataset."""
        num_features = [c for c in self.NUMERICAL_COLS if c in df.columns]
        cat_features = [c for c in self.CATEGORICAL_COLS if c in df.columns]
        bin_features = [c for c in self.BINARY_COLS if c in df.columns]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
                ('bin', 'passthrough', bin_features)
            ],
            remainder='drop'
        )
        
        self.preprocessor.fit(df)
        self.is_fitted = True
        
        # Track feature names post-encoding
        cat_ohe = self.preprocessor.named_transformers_['cat']
        encoded_cat_names = list(cat_ohe.get_feature_names_out(cat_features)) if cat_features else []
        self.feature_names = num_features + encoded_cat_names + bin_features
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe using fitted pipeline."""
        if not self.is_fitted or self.preprocessor is None:
            raise ValueError("DataPreprocessor must be fitted before calling transform.")
        
        X_arr = self.preprocessor.transform(df)
        return pd.DataFrame(X_arr, columns=self.feature_names, index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def save(self, file_path: str):
        """Serialize preprocessor state."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump({'preprocessor': self.preprocessor, 'feature_names': self.feature_names}, file_path)
        print(f"DataPreprocessor saved to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "DataPreprocessor":
        """Load serialized preprocessor."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Preprocessor file not found at {file_path}")
        data = joblib.load(file_path)
        instance = cls()
        instance.preprocessor = data['preprocessor']
        instance.feature_names = data['feature_names']
        instance.is_fitted = True
        return instance
