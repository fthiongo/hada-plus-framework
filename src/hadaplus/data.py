from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generate_synthetic_financial(
    n_samples: int = 60000,
    n_features: int = 20,
    fraud_ratio: float = 0.02,
    abrupt_drift_at: int = 25000,
    gradual_drift_start: int = 35000,
    gradual_drift_end: int = 45000,
    recurring_drift_at: int = 52000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic financial transaction dataset with controlled drift."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    n_fraud = max(1, int(n_samples * fraud_ratio))
    fraud_idx = rng.choice(n_samples, size=n_fraud, replace=False)
    y[fraud_idx] = 1

    # Fraud shift
    X[fraud_idx, :5] += rng.normal(3.0, 0.8, size=(n_fraud, 5))

    # Abrupt drift
    X[abrupt_drift_at:, 5:10] += 1.2

    # Gradual drift
    if gradual_drift_end > gradual_drift_start:
        span = gradual_drift_end - gradual_drift_start
        ramp = np.linspace(0, 1.5, span).reshape(-1, 1)
        X[gradual_drift_start:gradual_drift_end, 10:15] += ramp

    # Recurring drift
    X[recurring_drift_at:, :5] += 0.8 * np.sin(np.linspace(0, 8 * np.pi, n_samples - recurring_drift_at)).reshape(-1, 1)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["Class"] = y
    df["Time"] = np.arange(n_samples)
    return df

def load_creditcard_csv(path: str | Path) -> pd.DataFrame:
    """Load the public Credit Card Fraud dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Credit card dataset not found at {path}. "
            "Download creditcard.csv and place it under data/public/."
        )
    return pd.read_csv(path)

def split_features_labels(df: pd.DataFrame, label_col: str = "Class"):
    """Split dataframe into feature matrix and label vector."""
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")
    y = df[label_col].astype(int).to_numpy()
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).to_numpy()
    return X, y

def fit_transform_fixed_pca(X: np.ndarray, variance: float = 0.95):
    """Standardize features and fit PCA preserving a target variance."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=variance, svd_solver="full", random_state=42)
    Z = pca.fit_transform(Xs)
    return Z, scaler, pca

def transform_fixed_pca(X: np.ndarray, scaler: StandardScaler, pca: PCA):
    """Apply fitted scaler and PCA projection."""
    return pca.transform(scaler.transform(X))
