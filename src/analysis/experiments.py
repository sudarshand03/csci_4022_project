from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    ANALYSIS_PARAMS,
    MODEL_ORDER,
    NUM_GRAPH_SEEDS,
    N_ANALYSIS,
    RANDOM_STATE,
    STRATEGY_ORDER,
)
from analysis.graph_models import generate_graph
from analysis.features import FEATURE_COLS, extract_node_features
from analysis.reduction import run_pca
from analysis.clustering import fit_best_gmm
from analysis.metrics import mean_ci, gini
from simulation.interventions import run_intervention_spread_experiment


def analyze_graph(model: str, n: int, seed: int, params: dict) -> tuple[pd.DataFrame, dict]:
    """
    Generate one graph realization, compute node features, run PCA + GMM,
    and return both node-level results and graph-level summary metrics.

    Parameters
    ----------
    model : str
        Graph model label ('ER', 'WS', or 'BA').
    n : int
        Number of nodes.
    seed : int
        Random seed for graph generation.
    params : dict
        Graph-model-specific parameters.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        df_clustered:
            Node-level dataframe with PCA coordinates and cluster assignments.
        result:
            Graph-level summary dictionary.
    """
    G = generate_graph(model, n=n, seed=seed, **params)
    df = extract_node_features(G, model)

    df_pca, pca, _ = run_pca(df, FEATURE_COLS, n_components=2)
    df_clustered, _, best_k, _ = fit_best_gmm(df_pca, ["PC1", "PC2"], random_state=seed)

    # Important for later panel plots and joins
    df_clustered = df_clustered.copy()
    df_clustered["graph_seed"] = seed

    result = {
        "model": model,
        "graph_seed": seed,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "pagerank_std": df["pagerank"].std(),
        "pagerank_max": df["pagerank"].max(),
        "pagerank_gini": gini(df["pagerank"].values),
        "degree_std": df["degree"].std(),
        "degree_pagerank_corr": df[["degree", "pagerank"]].corr().iloc[0, 1],
        "pca_var_pc1": pca.explained_variance_ratio_[0],
        "pca_var_pc2": pca.explained_variance_ratio_[1],
        "num_clusters": best_k,
    }

    return df_clustered, result


def run_static_experiments(
    n: int = N_ANALYSIS,
    num_graph_seeds: int = NUM_GRAPH_SEEDS,
    analysis_params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run repeated static graph experiments across models and seeds.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    num_graph_seeds : int
        Number of random graph realizations per model.
    analysis_params : dict | None
        Optional override of graph parameters. Defaults to ANALYSIS_PARAMS.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        nodes_df:
            Concatenated node-level results across all runs.
        results_df:
            Graph-level summary dataframe across all runs.
    """
    if analysis_params is None:
        analysis_params = ANALYSIS_PARAMS

    all_node_results = []
    all_graph_results = []

    for graph_seed in range(num_graph_seeds):
        for model in MODEL_ORDER:
            df_clustered, result = analyze_graph(
                model=model,
                n=n,
                seed=graph_seed,
                params=analysis_params[model],
            )
            all_node_results.append(df_clustered)
            all_graph_results.append(result)

    nodes_df = pd.concat(all_node_results, ignore_index=True)
    results_df = pd.DataFrame(all_graph_results)

    return nodes_df, results_df


def add_global_pca_projection(nodes_df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """
    Fit a single PCA model across all node-level observations and add
    global PC1 / PC2 coordinates for cross-model visualization.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        Node-level dataframe from run_static_experiments.
    cols : list[str] | None
        Feature columns to use. Defaults to FEATURE_COLS.

    Returns
    -------
    pd.DataFrame
        Copy of nodes_df with global_PC1 and global_PC2 added.
    """
    if cols is None:
        cols = FEATURE_COLS

    combined_features = nodes_df[cols].values
    combined_scaled = StandardScaler().fit_transform(combined_features)

    combined_pca = PCA(n_components=2, random_state=RANDOM_STATE)
    combined_pca_values = combined_pca.fit_transform(combined_scaled)

    out = nodes_df.copy()
    out["global_PC1"] = combined_pca_values[:, 0]
    out["global_PC2"] = combined_pca_values[:, 1]

    return out


def run_intervention_experiments(
    n: int = N_ANALYSIS,
    num_graph_seeds: int = NUM_GRAPH_SEEDS,
    analysis_params: dict | None = None,
) -> pd.DataFrame:
    """
    Run repeated intervention spread experiments across graph models,
    graph realizations, and intervention strategies.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    num_graph_seeds : int
        Number of random graph realizations per model.
    analysis_params : dict | None
        Optional override of graph parameters. Defaults to ANALYSIS_PARAMS.

    Returns
    -------
    pd.DataFrame
        One row per spread replicate.
    """
    if analysis_params is None:
        analysis_params = ANALYSIS_PARAMS

    intervention_results = []

    for graph_seed in range(num_graph_seeds):
        for model in MODEL_ORDER:
            for strategy in STRATEGY_ORDER:
                intervention_results.append(
                    run_intervention_spread_experiment(
                        model=model,
                        n=n,
                        graph_params=analysis_params[model],
                        graph_seed=graph_seed,
                        intervention_strategy=strategy,
                    )
                )

    intervention_df = pd.concat(intervention_results, ignore_index=True)
    return intervention_df


def build_intervention_summary(intervention_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize intervention outcomes by model and strategy using mean and CI.

    Parameters
    ----------
    intervention_df : pd.DataFrame
        Output of run_intervention_experiments.

    Returns
    -------
    pd.DataFrame
        Summary dataframe with one row per (model, strategy, metric).
    """
    summary_rows = []

    for metric in ["final_fraction_infected", "t_25", "t_50", "t_75", "auc_infected"]:
        for (model, strategy), group in intervention_df.groupby(["model", "strategy"]):
            stats = mean_ci(group[metric])
            summary_rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "metric": metric,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "n": stats["n"],
                    "ci_lower": stats["ci_lower"],
                    "ci_upper": stats["ci_upper"],
                }
            )

    return pd.DataFrame(summary_rows)


def build_paired_comparisons(intervention_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build paired intervention comparisons within each graph realization.

    Parameters
    ----------
    intervention_df : pd.DataFrame
        Output of run_intervention_experiments.

    Returns
    -------
    pd.DataFrame
        One row per (model, graph_seed) with paired differences.
    """
    metrics = ["final_fraction_infected", "t_50", "auc_infected"]

    graph_level = (
        intervention_df.groupby(["model", "graph_seed", "strategy"])[metrics]
        .mean()
        .reset_index()
    )

    pivot = graph_level.pivot_table(
        index=["model", "graph_seed"],
        columns="strategy",
        values=metrics,
        dropna=False,
    )

    # Force all expected metric/strategy columns to exist,
    # even if some are entirely NaN.
    full_columns = pd.MultiIndex.from_product([metrics, STRATEGY_ORDER])
    pivot = pivot.reindex(columns=full_columns)

    paired_results = []

    for (model, graph_seed), row in pivot.iterrows():
        paired_results.append(
            {
                "model": model,
                "graph_seed": graph_seed,
                "delta_final_targeted_vs_random": (
                    row[("final_fraction_infected", "targeted_pagerank")]
                    - row[("final_fraction_infected", "random")]
                ),
                "delta_final_targeted_vs_none": (
                    row[("final_fraction_infected", "targeted_pagerank")]
                    - row[("final_fraction_infected", "none")]
                ),
                "delta_t50_targeted_vs_random": (
                    row[("t_50", "targeted_pagerank")]
                    - row[("t_50", "random")]
                ),
                "delta_auc_targeted_vs_random": (
                    row[("auc_infected", "targeted_pagerank")]
                    - row[("auc_infected", "random")]
                ),
            }
        )

    return pd.DataFrame(paired_results)


def summarize_static_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute grouped mean/std summaries for the main static graph metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Graph-level dataframe from run_static_experiments.

    Returns
    -------
    pd.DataFrame
        Grouped summary table indexed by model.
    """
    return (
        results_df.groupby("model")[
            [
                "pagerank_std",
                "pagerank_max",
                "pagerank_gini",
                "degree_std",
                "degree_pagerank_corr",
                "num_clusters",
            ]
        ]
        .agg(["mean", "std"])
        .reindex(MODEL_ORDER)
    )