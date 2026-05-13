from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def _subsample(X: np.ndarray, max_samples: int = 750, seed: int = 42) -> np.ndarray:
    """Subsample rows to keep MMD computation tractable for large windows."""
    if len(X) <= max_samples:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx]

def median_heuristic_gamma(X: np.ndarray, Y: np.ndarray) -> float:
    """Estimate RBF gamma using a simple median heuristic."""
    Z = np.vstack([_subsample(X, 500), _subsample(Y, 500)])
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0])
    return 1.0 / (2.0 * med + 1e-12)

def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float | None = None, max_samples: int = 750) -> float:
    """Compute squared Maximum Mean Discrepancy using an RBF kernel.

    For scalability, large windows are subsampled before kernel evaluation.
    This preserves reproducibility while avoiding O(W^2) memory growth.
    """
    if X is None or Y is None or len(X) == 0 or len(Y) == 0:
        return 0.0

    Xs = _subsample(X, max_samples=max_samples, seed=42)
    Ys = _subsample(Y, max_samples=max_samples, seed=43)

    gamma = gamma or median_heuristic_gamma(Xs, Ys)
    Kxx = rbf_kernel(Xs, Xs, gamma=gamma).mean()
    Kyy = rbf_kernel(Ys, Ys, gamma=gamma).mean()
    Kxy = rbf_kernel(Xs, Ys, gamma=gamma).mean()
    return float(max(Kxx + Kyy - 2 * Kxy, 0.0))
