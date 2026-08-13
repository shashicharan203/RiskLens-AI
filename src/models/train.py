import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_processing.generate_sample_data import main as generate_data
from src.data_processing.preprocess import DataPreprocessor
from src.data_processing.feature_engineering import FeatureEngineer
from src.models.autoencoder import train_autoencoder, compute_anomaly_scores, evaluate_autoencoder
from src.models.risk_model import FinancialRiskClassifier

def train_and_save_pipeline(
    data_path: str = "data/transactions.csv",
    output_model_path: str = "models/risk_model.pkl"
):
    """Train complete ML + PyTorch Deep Learning pipeline and serialize model."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    full_data_path = os.path.join(root_dir, data_path)
    full_output_path = os.path.join(root_dir, output_model_path)
    
    if not os.path.exists(full_data_path):
        print(f"Data file not found at {full_data_path}. Generating synthetic dataset...")
        generate_data()
        
    df = pd.read_csv(full_data_path)
    print(f"Loaded dataset shape: {df.shape}")
    
    # 1. Feature Engineering
    df_engineered = FeatureEngineer.add_engineered_features(df)
    
    X = df_engineered.drop(columns=['transaction_id', 'account_id', 'is_risk'], errors='ignore')
    y = df_engineered['is_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 2. Data Preprocessing
    preprocessor = DataPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # 3. PyTorch Deep Autoencoder Anomaly Score & Evaluation
    print("Training PyTorch Autoencoder for anomaly extraction...")
    ae_model, train_errors = train_autoencoder(X_train_proc.values, epochs=25)
    test_errors = compute_anomaly_scores(ae_model, X_test_proc.values)
    
    ae_metrics = evaluate_autoencoder(ae_model, X_test_proc.values, y_test.values)
    
    X_train_proc['ae_anomaly_score'] = train_errors
    X_test_proc['ae_anomaly_score'] = test_errors
    
    # Baseline stats for anomaly normalization
    ae_mean = float(np.mean(train_errors))
    ae_std = float(np.std(train_errors))
    
    # 4. Train Models
    algorithms = ['logistic_regression', 'random_forest', 'xgboost']
    all_metrics = {}
    fitted_classifiers = {}
    
    print("\n--- Model Evaluation ---")
    best_model_name = None
    best_roc_auc = -1.0
    
    for algo in algorithms:
        clf = FinancialRiskClassifier(model_type=algo, use_smote=True)
        clf.fit(X_train_proc, y_train)
        metrics = clf.evaluate(X_test_proc, y_test)
        all_metrics[algo] = metrics
        fitted_classifiers[algo] = clf
        print(f"[{algo.upper()}] Metrics: {metrics}")
        
        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_model_name = algo
            
    all_metrics['autoencoder'] = ae_metrics
    print(f"[AUTOENCODER] Metrics: {ae_metrics}")
    print(f"\nBest Model Selected: {best_model_name.upper()} (ROC-AUC: {best_roc_auc:.4f})")
    best_classifier = fitted_classifiers[best_model_name]
    
    # 5. Serialize Artifacts
    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
    payload = {
        'model': best_classifier.model,
        'model_type': best_model_name,
        'preprocessor': preprocessor,
        'autoencoder': ae_model,
        'ae_baseline_stats': {'mean': ae_mean, 'std': ae_std},
        'feature_names': list(X_train_proc.columns),
        'metrics': all_metrics[best_model_name],
        'all_model_metrics': all_metrics,
        'is_synthetic_data': True
    }
    
    joblib.dump(payload, full_output_path)
    print(f"Successfully saved trained model artifact to {full_output_path}")

if __name__ == "__main__":
    train_and_save_pipeline()
