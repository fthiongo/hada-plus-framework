from __future__ import annotations

import pandas as pd
import numpy as np

def summarize_window_results(rows):
    df = pd.DataFrame(rows)
    numeric = df.select_dtypes(include=[np.number])
    summary = {}
    for col in numeric.columns:
        if col in {"window", "start", "end"}:
            continue
        summary[f"{col}_mean"] = numeric[col].mean()
        summary[f"{col}_std"] = numeric[col].std()
    return pd.DataFrame([summary])

def lifecycle_transition_summary(rows):
    df = pd.DataFrame(rows)
    event_cols = ["persistence", "split", "merge", "birth", "decay"]
    out = []
    for col in event_cols:
        if col in df.columns:
            freq = int(df[col].sum())
            active = df[df[col] > 0]
            avg_duration = float(active[col].mean()) if len(active) else 0.0
            stability = float(1.0 / (1.0 + df[col].std())) if df[col].std() == df[col].std() else 0.0
            out.append({
                "Event Type": "Emergence" if col == "birth" else col.capitalize(),
                "Frequency": freq,
                "Avg Duration (Windows)": round(avg_duration, 3),
                "Stability Score": round(stability, 3),
            })
    return pd.DataFrame(out)
