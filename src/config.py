from __future__ import annotations

RANDOM_STATE = 42

MODEL_ORDER = ["ER", "WS", "BA"]
STRATEGY_ORDER = ["none", "targeted_pagerank", "random"]

MODEL_LABELS = {
    "ER": "Erdos-Renyi",
    "WS": "Watts-Strogatz",
    "BA": "Barabasi-Albert",
}

STRATEGY_LABELS = {
    "none": "No intervention",
    "targeted_pagerank": "Top-5 PageRank removed",
    "random": "5 random nodes removed",
}

N_ANALYSIS = 500
N_GRAPH_PLOT = 120

NUM_GRAPH_SEEDS = 8
SPREAD_STEPS = 20
SPREAD_BETA = 0.20
SPREAD_REPS = 15
K_REMOVE = 50

EXAMPLE_SEED = 0


def get_default_graph_params(n: int) -> dict[str, dict]:
    """
    Return default graph parameters chosen to keep average degree
    roughly comparable across ER, WS, and BA models.
    """
    return {
        "ER": {"p": 6 / (n - 1)},
        "WS": {"k": 6, "beta": 0.20},
        "BA": {"m": 3},
    }


ANALYSIS_PARAMS = get_default_graph_params(N_ANALYSIS)
PLOT_PARAMS = get_default_graph_params(N_GRAPH_PLOT)
