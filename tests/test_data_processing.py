import pytest
import pandas as pd
from src.data_processing.generate_sample_data import generate_transactions
from src.data_processing.feature_engineering import FeatureEngineer
from src.data_processing.preprocess import DataPreprocessor

def test_generate_transactions():
    df = generate_transactions(num_samples=50)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert 'is_risk' in df.columns
    assert 'transaction_amount' in df.columns

def test_feature_engineering():
    df = generate_transactions(num_samples=20)
    df_eng = FeatureEngineer.add_engineered_features(df)
    assert 'amount_to_income_ratio' in df_eng.columns
    assert 'credit_debt_stress_score' in df_eng.columns

def test_preprocessor():
    df = generate_transactions(num_samples=50)
    df_eng = FeatureEngineer.add_engineered_features(df)
    
    preprocessor = DataPreprocessor()
    df_proc = preprocessor.fit_transform(df_eng)
    assert isinstance(df_proc, pd.DataFrame)
    assert df_proc.shape[0] == 50
    assert df_proc.shape[1] > 0
