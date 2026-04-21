# PageRank and Infection Dynamics on Random Graphs

This project studies how graph structure influences diffusion and infection dynamics on random networks, and how centrality-based methods such as PageRank can be used to analyze or intervene in those processes.

## Overview

The repository contains tools for:
- generating and analyzing random graphs,
- extracting graph features,
- simulating diffusion or infection processes,
- evaluating intervention strategies,
- computing performance metrics, and
- visualizing experimental results.

The goal is to better understand how network structure affects spread dynamics and which graph-based methods are most useful for prediction and control.

## Repository Structure

```text
Directory structure:
└── sudarshand03-csci_4022_project/
    ├── README.md
    ├── main.py
    ├── pyproject.toml
    ├── requirements.txt
    ├── results/
    │   ├── raw/
    │   │   ├── graph_level_results.pkl
    │   │   ├── intervention_summary.pkl
    │   │   └── paired_comparisons.pkl
    │   └── tables/
    │       ├── graph_level_results.csv
    │       ├── intervention_summary.csv
    │       ├── paired_comparisons.csv
    │       └── static_summary.csv
    └── src/
        ├── config.py
        ├── analysis/
        │   ├── clustering.py
        │   ├── experiments.py
        │   ├── features.py
        │   ├── graph_models.py
        │   ├── metrics.py
        │   └── reduction.py
        ├── plotting/
        │   ├── __init__.py
        │   └── visualization.py
        └── simulation/
            ├── __init__.py
            ├── diffusion.py
            └── interventions.py

```

