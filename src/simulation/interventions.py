import numpy as np
import pandas as pd

from config import K_REMOVE, SPREAD_BETA, SPREAD_REPS, SPREAD_STEPS
from analysis.features import extract_node_features
from analysis.graph_models import generate_graph
from simulation.diffusion import area_under_curve, simulate_si_spread, time_to_threshold


def remove_nodes_by_strategy(
    G,
    df: pd.DataFrame,
    strategy: str,
    k: int = K_REMOVE,
    rng=None,
):
    """
    Remove nodes from a graph according to an intervention strategy.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    df : pd.DataFrame
        Node feature dataframe containing at least 'node' and 'pagerank'.
    strategy : str
        One of: 'none', 'targeted_pagerank', 'random'.
    k : int
        Number of nodes to remove.
    rng : np.random.Generator | None
        Random number generator for the random strategy.

    Returns
    -------
    tuple
        (G_mod, removed_nodes)
        - G_mod: modified graph after node removal
        - removed_nodes: list of removed node ids
    """
    if rng is None:
        rng = np.random.default_rng()

    G_mod = G.copy()

    if strategy == "none":
        removed_nodes = []

    elif strategy == "targeted_pagerank":
        removed_nodes = (
            df.sort_values("pagerank", ascending=False).head(k)["node"].tolist()
        )
        G_mod.remove_nodes_from(removed_nodes)

    elif strategy == "random":
        removed_nodes = rng.choice(list(G.nodes()), size=k, replace=False).tolist()
        G_mod.remove_nodes_from(removed_nodes)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return G_mod, removed_nodes


def run_intervention_spread_experiment(
    model: str,
    n: int,
    graph_params: dict,
    graph_seed: int,
    intervention_strategy: str,
    beta: float = SPREAD_BETA,
    steps: int = SPREAD_STEPS,
    k_remove: int = K_REMOVE,
    sim_reps: int = SPREAD_REPS,
) -> pd.DataFrame:
    """
    Generate a graph, apply an intervention, and run repeated SI spread simulations.

    Parameters
    ----------
    model : str
        Graph model label ('ER', 'WS', 'BA').
    n : int
        Number of nodes.
    graph_params : dict
        Model-specific graph parameters.
    graph_seed : int
        Seed used for graph generation.
    intervention_strategy : str
        One of: 'none', 'targeted_pagerank', 'random'.
    beta : float
        Infection probability.
    steps : int
        Number of SI simulation steps.
    k_remove : int
        Number of nodes to remove under intervention.
    sim_reps : int
        Number of repeated spread simulations.

    Returns
    -------
    pd.DataFrame
        One row per spread simulation replicate.
    """
    rng_graph = np.random.default_rng(graph_seed)

    G = generate_graph(model, n=n, seed=graph_seed, **graph_params)
    df = extract_node_features(G, model)

    G_mod, removed_nodes = remove_nodes_by_strategy(
        G,
        df,
        strategy=intervention_strategy,
        k=k_remove,
        rng=rng_graph,
    )

    if G_mod.number_of_nodes() == 0:
        return pd.DataFrame()

    remaining_nodes = list(G_mod.nodes())
    records = []

    for sim_rep in range(sim_reps):
        rng = np.random.default_rng(100_000 * graph_seed + sim_rep)
        initial_infected = rng.choice(remaining_nodes)

        counts, fractions = simulate_si_spread(
            G_mod,
            initial_infected=initial_infected,
            beta=beta,
            steps=steps,
            rng=rng,
        )

        records.append(
            {
                "model": model,
                "graph_seed": graph_seed,
                "strategy": intervention_strategy,
                "sim_rep": sim_rep,
                "nodes_remaining": G_mod.number_of_nodes(),
                "edges_remaining": G_mod.number_of_edges(),
                "final_fraction_infected": fractions[-1],
                "t_25": time_to_threshold(fractions, 0.25),
                "t_50": time_to_threshold(fractions, 0.50),
                "t_75": time_to_threshold(fractions, 0.75),
                "auc_infected": area_under_curve(fractions),
                "curve": fractions,
                "removed_nodes": removed_nodes,
            }
        )

    return pd.DataFrame(records)
