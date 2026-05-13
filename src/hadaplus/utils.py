from __future__ import annotations

import random
from pathlib import Path
import numpy as np

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def sliding_windows(n_samples: int, window_size: int, step_fraction: float):
    """Yield start and end indices for overlapping sliding windows."""
    step = max(1, int(window_size * step_fraction))
    start = 0
    while start + window_size <= n_samples:
        yield start, start + window_size
        start += step
