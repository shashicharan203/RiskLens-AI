import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

class TransactionAutoencoder(nn.Module):
    """PyTorch Deep Autoencoder for unsupervised transaction anomaly detection."""

    def __init__(self, input_dim: int, latent_dim: int = 8):
        super(TransactionAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def train_autoencoder(
    X_train_np: np.ndarray,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3
) -> Tuple[TransactionAutoencoder, np.ndarray]:
    """Train PyTorch Autoencoder and return model and reconstruction error scores."""
    input_dim = X_train_np.shape[1]
    latent_dim = max(4, input_dim // 2)
    
    model = TransactionAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    model.train()
    
    tensor_x = torch.tensor(X_train_np, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for batch in loader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        reconstructed_all = model(tensor_x)
        mse_errors = torch.mean((tensor_x - reconstructed_all) ** 2, dim=1).numpy()
        
    return model, mse_errors

def compute_anomaly_scores(model: TransactionAutoencoder, X_np: np.ndarray) -> np.ndarray:
    """Compute MSE reconstruction error anomaly scores for unseen feature arrays."""
    model.eval()
    tensor_x = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        reconstructed = model(tensor_x)
        errors = torch.mean((tensor_x - reconstructed) ** 2, dim=1).numpy()
    return errors

def normalize_anomaly_score(raw_mse: float, baseline_mean: float = 0.5, baseline_std: float = 0.8) -> Tuple[float, str]:
    """Normalize raw PyTorch MSE error to [0.0, 1.0] scale and categorize status."""
    # Sigmoid / z-score scaling for smooth 0..1 bounding
    z = (raw_mse - baseline_mean) / max(baseline_std, 1e-5)
    normalized_score = float(1.0 / (1.0 + np.exp(-z)))
    normalized_score = round(float(np.clip(normalized_score, 0.0, 1.0)), 2)
    
    if normalized_score >= 0.70:
        status = "HIGHLY ANOMALOUS"
    elif normalized_score >= 0.40:
        status = "SUSPICIOUS"
    else:
        status = "NORMAL"
        
    return normalized_score, status

def evaluate_autoencoder(model: TransactionAutoencoder, X_test_np: np.ndarray, y_test_np: np.ndarray) -> Dict[str, Any]:
    """Evaluate Autoencoder anomaly detection accuracy metrics on test set."""
    errors = compute_anomaly_scores(model, X_test_np)
    # Threshold at 75th percentile of training MSE for binary prediction evaluation
    threshold = float(np.percentile(errors, 70))
    preds = (errors >= threshold).astype(int)
    
    acc = float(accuracy_score(y_test_np, preds))
    prec = float(precision_score(y_test_np, preds, zero_division=0))
    rec = float(recall_score(y_test_np, preds, zero_division=0))
    f1 = float(f1_score(y_test_np, preds, zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_test_np, errors))
    except Exception:
        roc_auc = 0.0
        
    cm = confusion_matrix(y_test_np, preds).tolist()
    fpr, tpr, _ = roc_curve(y_test_np, errors)
    
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": [round(x, 4) for x in fpr.tolist()],
            "tpr": [round(x, 4) for x in tpr.tolist()]
        }
    }
