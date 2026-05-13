from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

def score_aware_clustering(
    X: np.ndarray,
    scores: np.ndarray,
    n_clusters: int = 10,
    seed: int = 42,
):
    """Cluster selected anomalies using anomaly scores as sample weights."""
    if len(X) == 0:
        return np.array([]), np.empty((0, X.shape[1] if X.ndim == 2 else 0))
    k = min(n_clusters, len(X))
    weights = np.asarray(scores, dtype=float)
    weights = weights - weights.min() + 1e-6
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(X, sample_weight=weights)
    return labels, model.cluster_centers_

def cluster_masses(labels: np.ndarray) -> np.ndarray:
    """Return normalized cluster masses."""
    if len(labels) == 0:
        return np.array([])
    _, counts = np.unique(labels, return_counts=True)
    masses = counts.astype(float)
    return masses / masses.sum()
