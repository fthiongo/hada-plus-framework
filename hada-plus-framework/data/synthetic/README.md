# Synthetic Financial Dataset

This directory documents the synthetic financial transaction dataset generated for evaluating the HADA+ framework under evolving concept drift conditions.

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

## Purpose Within HADA+

The synthetic dataset supports evaluation of:

- drift-conditioned anomaly prioritization,
- score-aware clustering,
- entropy-regularized optimal transport evolution tracking,
- lifecycle transition modeling,
- anomaly continuity preservation,
- post-drift recovery behavior.

## Configuration Parameters

Synthetic dataset generation parameters are defined under:

```text
configs/synthetic.yaml
```

The configuration settings control:

- drift intensity,
- drift type,
- anomaly ratio,
- window size,
- step fraction,
- anomaly evolution behavior,
- temporal dynamics.

## Reproducibility

The repository includes:

- preprocessing pipelines,
- anomaly scoring modules,
- clustering pipelines,
- transport evolution modeling,
- statistical evaluation scripts,
- figure generation scripts.

The reproducibility resources associated with this work are archived at:

https://doi.org/10.5281/zenodo.20357523

## Associated Manuscript

HADA+: An Evolution-Aware Framework for Drift-Resilient Anomaly Detection with Score-Aware Clustering and Optimal Transport Evolution

## Important Note

This dataset is synthetic and does not contain real customer financial records or personally identifiable information.

## Download Link

A compressed copy of the synthetic dataset used in the experiments is accessible through the reproducibility mirror:

https://hada-plus.taskmatehub.com/Syntheticfinancial.zip
