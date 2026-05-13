from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from hadaplus.config import load_config
from hadaplus.data import load_creditcard_csv, split_features_labels, fit_transform_fixed_pca
from hadaplus.pipeline import run_hadaplus_windows
from hadaplus.baselines import run_baselines
from hadaplus.evaluation import summarize_window_results, lifecycle_transition_summary
from hadaplus.utils import ensure_dir, set_seed

def main():
    cfg = load_config("configs/default.yaml")
    set_seed(cfg["seed"])
    ensure_dir("results")
    ensure_dir("figures")

    df = load_creditcard_csv("data/public/creditcard.csv")
    X, y = split_features_labels(df)
    Z, scaler, pca = fit_transform_fixed_pca(X, cfg["data"]["pca_variance"])

    rows = run_hadaplus_windows(
        Z, y,
        window_size=cfg["data"]["window_size"],
        step_fraction=cfg["data"]["step_fraction"],
        baseline_k=cfg["hada_plus"]["baseline_k"],
        beta=cfg["hada_plus"]["beta"],
        ot_epsilon=cfg["hada_plus"]["ot_epsilon"],
        n_clusters=cfg["hada_plus"]["n_clusters"],
        iforest_estimators=cfg["hada_plus"]["iforest_estimators"],
        iforest_contamination=cfg["hada_plus"]["iforest_contamination"],
        seed=cfg["seed"],
    )

    pd.DataFrame(rows).to_csv("results/creditcard_hadaplus_window_results.csv", index=False)
    summarize_window_results(rows).to_csv("results/creditcard_hadaplus_summary.csv", index=False)
    lifecycle_transition_summary(rows).to_csv("results/creditcard_lifecycle_statistics.csv", index=False)

    baseline_rows = run_baselines(Z, y, seed=cfg["seed"])
    pd.DataFrame(baseline_rows).to_csv("results/creditcard_baseline_results.csv", index=False)
    print("Credit card experiment complete. Results saved under results/.")

if __name__ == "__main__":
    main()
