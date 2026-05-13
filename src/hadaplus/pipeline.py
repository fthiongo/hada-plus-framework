from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, f1_score

try:
    import ot
except ImportError:  # pragma: no cover
    ot = None

from .clustering import score_aware_clustering, cluster_masses
from .drift import mmd_rbf
from .utils import sliding_windows

def _iforest_scores(X: np.ndarray, n_estimators: int = 100, contamination="auto", seed: int = 42):
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X)
    # Higher should mean more anomalous
    return -model.score_samples(X)

def topk_predictions(scores: np.ndarray, k: int):
    k = max(1, min(int(k), len(scores)))
    pred = np.zeros(len(scores), dtype=int)
    idx = np.argsort(scores)[-k:]
    pred[idx] = 1
    return pred, idx

def sinkhorn_transport(prev_centers, curr_centers, prev_masses, curr_masses, epsilon: float = 0.1):
    """Compute entropy-regularized OT plan between consecutive cluster centroids."""
    if prev_centers is None or len(prev_centers) == 0 or len(curr_centers) == 0:
        return None
    C = cdist(curr_centers, prev_centers, metric="sqeuclidean")
    if ot is not None:
        return ot.sinkhorn(curr_masses, prev_masses, C, reg=epsilon)
    # Fallback: softmax alignment approximation
    K = np.exp(-C / max(epsilon, 1e-8))
    K = K / (K.sum(axis=1, keepdims=True) + 1e-12)
    return K * curr_masses[:, None]

def identify_lifecycle_events(T, threshold: float = 0.05):
    """Identify approximate lifecycle events from a transport matrix."""
    if T is None:
        return {"birth": 0, "persistence": 0, "split": 0, "merge": 0, "decay": 0}
    row_links = (T > threshold).sum(axis=1)
    col_links = (T > threshold).sum(axis=0)
    return {
        "birth": int((row_links == 0).sum()),
        "persistence": int(((row_links == 1).sum() + (col_links == 1).sum()) // 2),
        "split": int((col_links > 1).sum()),
        "merge": int((row_links > 1).sum()),
        "decay": int((col_links == 0).sum()),
    }

def run_hadaplus_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 5000,
    step_fraction: float = 0.2,
    baseline_k: int = 100,
    beta: float = 0.3,
    ot_epsilon: float = 0.1,
    n_clusters: int = 10,
    iforest_estimators: int = 100,
    iforest_contamination="auto",
    seed: int = 42,
):
    """Run HADA+ over overlapping windows and return window-level metrics."""
    results = []
    prev_X = None
    prev_centers = None
    prev_masses = None

    for w_id, (start, end) in enumerate(sliding_windows(len(X), window_size, step_fraction)):
        Xw = X[start:end]
        yw = y[start:end]
        scores = _iforest_scores(Xw, iforest_estimators, iforest_contamination, seed + w_id)

        drift = mmd_rbf(Xw, prev_X) if prev_X is not None else 0.0
        k_t = int(round(baseline_k * (1.0 + beta * drift)))
        pred, idx = topk_predictions(scores, k_t)

        Xa = Xw[idx]
        sa = scores[idx]
        labels, centers = score_aware_clustering(Xa, sa, n_clusters=n_clusters, seed=seed + w_id)
        masses = cluster_masses(labels)

        T = sinkhorn_transport(prev_centers, centers, prev_masses, masses, epsilon=ot_epsilon)
        events = identify_lifecycle_events(T)

        try:
            roc = roc_auc_score(yw, scores) if len(np.unique(yw)) > 1 else np.nan
            pr = average_precision_score(yw, scores) if len(np.unique(yw)) > 1 else np.nan
        except ValueError:
            roc, pr = np.nan, np.nan

        precision_at_k = precision_score(yw, pred, zero_division=0)
        f1_at_k = f1_score(yw, pred, zero_division=0)

        ecs = np.nan
        if T is not None and T.size > 0:
            ecs = float(np.clip(np.max(T, axis=1).mean(), 0, 1))

        results.append({
            "window": w_id,
            "start": start,
            "end": end,
            "drift": drift,
            "k_t": k_t,
            "roc_auc": roc,
            "pr_auc": pr,
            "precision_at_k": precision_at_k,
            "f1_at_k": f1_at_k,
            "ecs": ecs,
            **events,
        })

        prev_X = Xw
        prev_centers = centers
        prev_masses = masses

    return results
