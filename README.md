# Public Credit Card Fraud Dataset

This directory documents the public benchmark credit card fraud dataset used in the HADA+ experiments.

## Dataset Description

The dataset contains anonymized European credit card transactions labeled as fraudulent or legitimate and is widely used as a benchmark for anomaly detection, fraud detection, and highly imbalanced classification research.

The dataset was originally introduced by:

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi.

The transactions were transformed using Principal Component Analysis (PCA) for confidentiality preservation, resulting in anonymized features V1–V28.

## Original Dataset Source

Kaggle public repository:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Experimental Usage in HADA+

Within the HADA+ framework, this dataset is used to evaluate:

- anomaly detection accuracy,
- PR-AUC performance,
- ROC-AUC performance,
- operational alert prioritization,
- temporal stability under streaming windows,
- anomaly continuity preservation.

## Reproducibility

Experimental configurations are stored under:

```text
configs/
```

Main configuration files include:

```text
configs/default.yaml
configs/synthetic.yaml
```

The reproducibility resources associated with this work are archived at:

https://doi.org/10.5281/zenodo.20357523

## Citation

If you use this dataset, please cite the original source appropriately.

Example citation:

Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G.
"Calibrating Probability with Undersampling for Unbalanced Classification."
IEEE Symposium Series on Computational Intelligence, 2015.

## Associated Manuscript

HADA+: An Evolution-Aware Framework for Drift-Resilient Anomaly Detection with Score-Aware Clustering and Optimal Transport Evolution

## Download Link

A compressed copy of the dataset used in the experiments is accessible through the reproducibility mirror:

https://hada-plus.taskmatehub.com/creditcard.zip