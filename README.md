# csci_4022_project

## Setup Instructions

### 1. Clone the repository

### 2. Create a venv
python3 -m venv .venv
source .venv/bin/activate

### 3. Install Dependencies
pip install --upgrade pip

pip install -e .

*allows you to import modules directly e.g.*

``
from graph_generator import generate_erdos_renyi
``
``` text
sudarshand03-csci_4022_project/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── cleaned_random_graph_analysis.ipynb
├── scripts/
│   ├── run_full_experiment.py
│   └── make_figures.py
├── src/
│   └── random_graph_interventions/
│       ├── __init__.py
│       ├── config.py
│       ├── graph_models.py
│       ├── features.py
│       ├── reduction.py
│       ├── clustering.py
│       ├── diffusion.py
│       ├── interventions.py
│       ├── metrics.py
│       ├── visualization.py
│       └── experiments.py
├── tests/
│   ├── test_graph_models.py
│   ├── test_features.py
│   ├── test_diffusion.py
│   ├── test_interventions.py
│   └── test_metrics.py
└── results/
    ├── figures/
    ├── tables/
    └── raw/