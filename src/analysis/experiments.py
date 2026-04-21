from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import ANALYSIS_PARAMS, MODEL_ORDER, NUM_GRAPH_SEEDS
from analysis.graph_models import generate_graph
from analysis.features import FEATURE_COLS, extract_node_features
from analysis.reduction import run_pca
from analysis.clustering import fit_best_gmm
from analysis.metrics import gini, mean_ci
from simulation.interventions import run_intervention_spread_experiment