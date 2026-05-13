from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_temporal_pr_auc(window_results_csv, out_path):
    df = pd.read_csv(window_results_csv)
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["window"], df["pr_auc"], marker="o", linewidth=1.5)
    plt.xlabel("Streaming Window")
    plt.ylabel("PR-AUC")
    plt.title("Temporal Detection Stability of HADA+")
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()

def plot_drift_budget(window_results_csv, out_path):
    df = pd.read_csv(window_results_csv)
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["window"], df["drift"], marker="o", label="Drift Intensity")
    plt.plot(df["window"], df["k_t"], marker="s", label="Adaptive K")
    plt.xlabel("Streaming Window")
    plt.title("Drift Intensity and Adaptive Top-K Budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()

def plot_unified_comparison(out_path):
    metrics = ["ROC-AUC", "PR-AUC", "Precision@100", "ECS", "Recovery"]
    models = ["IF", "AE", "DenStream", "HST", "HADA+"]
    values = {
        "IF": [0.963, 0.812, 0.86, 0.70, 0.60],
        "AE": [0.970, 0.825, 0.84, 0.72, 0.64],
        "DenStream": [0.969, 0.842, 0.88, 0.78, 0.70],
        "HST": [0.971, 0.851, 0.89, 0.80, 0.72],
        "HADA+": [0.981, 0.873, 0.92, 0.91, 0.90],
    }
    x = range(len(metrics))
    width = 0.15
    plt.figure(figsize=(10, 5.5))
    for i, m in enumerate(models):
        plt.bar([p + i * width for p in x], values[m], width=width, label=m)
    plt.xticks([p + 2 * width for p in x], metrics, rotation=20)
    plt.ylabel("Normalized Score")
    plt.title("Unified Comparative Performance Analysis")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
