from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from hadaplus.data import generate_synthetic_financial, split_features_labels, fit_transform_fixed_pca
from hadaplus.pipeline import run_hadaplus_windows
from hadaplus.utils import ensure_dir, set_seed

def main():
    set_seed(42)
    ensure_dir("results")
    df = generate_synthetic_financial(
        n_samples=12000,
        n_features=12,
        fraud_ratio=0.02,
        abrupt_drift_at=5000,
        gradual_drift_start=7000,
        gradual_drift_end=9000,
        recurring_drift_at=10000,
        seed=42,
    )
    X, y = split_features_labels(df)
    Z, _, _ = fit_transform_fixed_pca(X, variance=0.95)
    rows = run_hadaplus_windows(
        Z, y,
        window_size=2000,
        step_fraction=0.5,
        baseline_k=50,
        beta=0.3,
        ot_epsilon=0.1,
        n_clusters=5,
        iforest_estimators=50,
        seed=42,
    )
    out = pd.DataFrame(rows)
    out.to_csv("results/smoke_test_results.csv", index=False)
    print(out.head())
    print("Smoke test completed successfully.")

if __name__ == "__main__":
    main()
