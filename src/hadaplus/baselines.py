from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_scores(y, scores):
    if len(np.unique(y)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan}
    return {
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
    }

def isolation_forest_scores(X, seed=42):
    model = IsolationForest(n_estimators=100, contamination="auto", random_state=seed, n_jobs=-1)
    model.fit(X)
    return -model.score_samples(X)

def lof_scores(X):
    model = LocalOutlierFactor(n_neighbors=35, novelty=False)
    pred = model.fit_predict(X)
    return -model.negative_outlier_factor_

def kmeans_distance_scores(X, n_clusters=10, seed=42):
    k = min(n_clusters, len(X))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    return np.linalg.norm(X - centers[labels], axis=1)

def dbscan_scores(X, eps=1.5, min_samples=10):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    # Noise receives highest anomaly score; non-noise score from inverse cluster size
    scores = np.zeros(len(X))
    unique, counts = np.unique(labels, return_counts=True)
    size = dict(zip(unique, counts))
    for i, lab in enumerate(labels):
        scores[i] = 1.0 if lab == -1 else 1.0 / max(size.get(lab, 1), 1)
    return scores

def autoencoder_scores(X, seed=42):
    # Lightweight sklearn autoencoder approximation
    hidden = max(4, min(32, X.shape[1] * 2))
    ae = MLPRegressor(
        hidden_layer_sizes=(hidden, max(2, hidden // 2), hidden),
        activation="relu",
        solver="adam",
        random_state=seed,
        max_iter=100,
        early_stopping=True,
    )
    ae.fit(X, X)
    Xhat = ae.predict(X)
    return ((X - Xhat) ** 2).mean(axis=1)

def run_baselines(X, y, seed=42):
    methods = {
        "Isolation Forest": isolation_forest_scores(X, seed),
        "LOF": lof_scores(X),
        "KMeans": kmeans_distance_scores(X, seed=seed),
        "DBSCAN": dbscan_scores(X),
        "Autoencoder": autoencoder_scores(X, seed),
    }
    rows = []
    for name, scores in methods.items():
        row = {"model": name}
        row.update(evaluate_scores(y, scores))
        rows.append(row)
    return rows
