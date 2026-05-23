# Synthetic Financial Dataset

This directory contains the synthetic financial transaction dataset generated for evaluating the HADA+ framework under evolving concept drift conditions.

## Dataset Description

The synthetic dataset simulates realistic financial transaction streams with evolving behavioral distributions and anomaly structures.

The dataset was generated using a PaySim-inspired simulation methodology adapted for:

- streaming anomaly detection,
- drift-aware fraud analytics,
- anomaly continuity preservation,
- lifecycle evolution modeling,
- temporal clustering analysis.

The generated transaction stream contains:

- abrupt drift,
- gradual drift,
- recurring drift,
- evolving anomaly clusters,
- temporal behavioral transitions.

Expected compressed dataset file:

```text
Syntheticfinancial.zip
```

Expected extracted dataset file:

```text
syntheticfinancial.csv
```

## Purpose Within HADA+

The synthetic dataset supports evaluation of:

- drift-conditioned anomaly prioritization,
- score-aware clustering,
- entropy-regularized optimal transport evolution tracking,
- lifecycle transition modeling,
- anomaly continuity preservation,
- post-drift recovery behavior.

## Repository Placement

Place the extracted CSV file inside:

```text
data/synthetic/
```

Final structure:

```text
data/synthetic/
├── Syntheticfinancial.zip
├── syntheticfinancial.csv
└── README.md
```

## Running Synthetic Experiments

Run the synthetic experiment pipeline using:

```bash
python scripts/run_synthetic_experiment.py
```

Run the full experiment suite using:

```bash
python scripts/run_all.py
```

## Configuration Parameters

Synthetic dataset generation parameters are stored under:

```text
configs/synthetic.yaml
```

The configuration file controls:

- drift intensity,
- drift type,
- anomaly ratio,
- window size,
- step fraction,
- anomaly evolution behavior,
- temporal dynamics.

## Experimental Outputs

Generated outputs are saved under:

```text
results/
figures/
```

## Reproducibility

The default random seed used in experiments is:

```text
42
```

The repository includes:

- preprocessing pipelines,
- anomaly scoring modules,
- clustering pipelines,
- transport evolution modeling,
- statistical evaluation scripts,
- figure generation scripts.

## Associated Manuscript

HADA+: An Evolution-Aware Framework for Drift-Resilient Anomaly Detection with Score-Aware Clustering and Optimal Transport Evolution

## Important Note

This dataset is synthetic and does not contain real customer financial records or personally identifiable information.

## Download Link

The compressed synthetic dataset archive can be downloaded from:

https://hada-plus.taskmatehub.com/Syntheticfinancial.zip

