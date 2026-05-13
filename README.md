# HADA+: Reproducibility Code

This repository contains code to reproduce the public and synthetic dataset experiments for:

**HADA+: A Unified Framework for Drift-Aware Anomaly Detection with Score-Aware Clustering and Optimal Transport Evolution**

The code supports:
- synthetic financial transaction generation with abrupt, gradual, and recurring drift;
- public Credit Card Fraud dataset experiments;
- preprocessing and fixed PCA feature representation;
- Isolation Forest anomaly scoring;
- drift-conditioned Top-K anomaly selection;
- score-aware clustering;
- entropy-regularized optimal transport evolution tracking;
- baseline comparisons;
- ablation studies;
- sensitivity analysis;
- statistical testing;
- figure and table generation.

> Proprietary mobile banking data are not included. The code is structured so that a private dataset can be placed under `data/private/` and loaded using the same pipeline interface.

---

## Repository Structure

```text
hadaplus_reproducibility_code/
├── configs/
│   ├── default.yaml
│   └── synthetic.yaml
├── data/
│   ├── public/
│   │   └── README.md
│   ├── private/
│   │   └── README.md
│   └── synthetic/
├── figures/
├── results/
├── scripts/
│   ├── run_synthetic_experiment.py
│   ├── run_creditcard_experiment.py
│   ├── run_ablation.py
│   ├── run_sensitivity.py
│   ├── generate_all_figures.py
│   └── run_all.py
├── src/
│   └── hadaplus/
│       ├── __init__.py
│       ├── baselines.py
│       ├── clustering.py
│       ├── config.py
│       ├── data.py
│       ├── drift.py
│       ├── evaluation.py
│       ├── figures.py
│       ├── pipeline.py
│       ├── stats.py
│       └── utils.py
├── requirements.txt
├── environment.yml
├── LICENSE
└── .gitignore
```

---


## Runtime Note

The default synthetic configuration is set to a practical reproducibility size so that the complete workflow can run on an ordinary laptop. For manuscript-scale runs, increase `n_samples`, `window_size`, `step_fraction`, and `iforest_estimators` in `configs/synthetic.yaml` and `configs/default.yaml`.


## Installation

```bash
git clone <your-github-repo-url>
cd hadaplus_reproducibility_code

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

Optional Conda setup:

```bash
conda env create -f environment.yml
conda activate hadaplus
```

---

## Datasets

### 1. Synthetic Dataset

Generate automatically:

```bash
python scripts/run_synthetic_experiment.py
```

The generated file is saved under:

```text
data/synthetic/synthetic_financial.csv
```

### 2. Credit Card Fraud Dataset

Download the public Credit Card Fraud dataset from Kaggle and place it here:

```text
data/public/creditcard.csv
```

Expected columns:
- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

Run:

```bash
python scripts/run_creditcard_experiment.py
```

---

## Run All Experiments

```bash
python scripts/run_all.py
```

Outputs are saved to:

```text
results/
figures/
```

---

## Reproducibility

Default random seed:

```text
42
```

Configuration files are stored under:

```text
configs/
```

---

## Citation

If you use this code, please cite the corresponding manuscript:

```bibtex
@article{thiongo2026hadaplus,
  title={HADA+: A Unified Framework for Drift-Aware Anomaly Detection with Score-Aware Clustering and Optimal Transport Evolution},
  author={Thiong'o, Francis K.},
  year={2026}
}
```

---

## Data Availability Statement Support

This repository supports the manuscript data availability statement:

> Code to reproduce all experiments on the public and synthetic datasets, including preprocessing steps, model configurations, random seeds, and figure-generation scripts, will be made available by the corresponding author upon reasonable request.


## Smoke Test

Run a quick validation without external datasets:

```bash
python scripts/smoke_test.py
```
