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