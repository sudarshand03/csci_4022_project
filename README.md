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
Directory structure:
└── sudarshand03-csci_4022_project/
    ├── README.md
    ├── cleaned_random_graph_analysis.ipynb
    ├── pyproject.toml
    ├── requirements.txt
    ├── src/
    │   ├── config.py
    │   ├── analysis/
    │   │   ├── clustering.py
    │   │   ├── experiments.py
    │   │   ├── features.py
    │   │   ├── graph_models.py
    │   │   ├── metrics.py
    │   │   └── reduction.py
    │   ├── plotting/
    │   │   ├── __init__.py
    │   │   └── visualization.py
    │   └── simulation/
    │       ├── __init__.py
    │       ├── diffusion.py
    │       └── interventions.py
    └── tests/
        ├── test_diffusion.py
        ├── test_features.py
        ├── test_graph_models.py
        ├── test_interventions.py
        └── test_metrics.py
