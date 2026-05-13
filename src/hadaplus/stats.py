from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def holm_correction(p_values):
    """Return Holm-adjusted significance decisions."""
    m = len(p_values)
    order = np.argsort(p_values)
    significant = [False] * m
    for rank, idx in enumerate(order):
        alpha_i = 0.05 / (m - rank)
        if p_values[idx] <= alpha_i:
            significant[idx] = True
        else:
            break
    return significant

def paired_wilcoxon_table(reference, competitors: dict):
    rows = []
    pvals = []
    names = []
    for name, values in competitors.items():
        a = np.asarray(reference)
        b = np.asarray(values)
        n = min(len(a), len(b))
        stat, p = wilcoxon(a[:n], b[:n], zero_method="wilcox", alternative="greater")
        z_approx = abs((stat - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24))
        effect = z_approx / np.sqrt(n)
        rows.append({"Comparison": f"HADA+ vs {name}", "p-value": p, "Effect Size": effect})
        pvals.append(p); names.append(name)
    sig = holm_correction(pvals)
    for row, s in zip(rows, sig):
        row["Holm"] = "Significant" if s else "Not significant"
    return pd.DataFrame(rows)
