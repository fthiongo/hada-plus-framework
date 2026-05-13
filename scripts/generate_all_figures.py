from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from hadaplus.figures import plot_temporal_pr_auc, plot_drift_budget, plot_unified_comparison

def main():
    Path("figures").mkdir(exist_ok=True)
    synthetic = Path("results/synthetic_hadaplus_window_results.csv")
    if synthetic.exists():
        plot_temporal_pr_auc(synthetic, "figures/figure_temporal_pr_auc.png")
        plot_drift_budget(synthetic, "figures/figure_drift_budget.png")
    plot_unified_comparison("figures/figure_unified_comparison.png")
    print("Figures saved under figures/.")

if __name__ == "__main__":
    main()
