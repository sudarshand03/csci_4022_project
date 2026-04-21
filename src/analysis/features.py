from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "degree",
    "pagerank",
    "clustering_coef",
    "betweenness",
    "closeness",
    "log_degree",
    "log_pagerank",
]


def compute_pagerank(G: nx.Graph, alpha: float = 0.85) -> dict:
    """
    Compute PageRank scores for all nodes in the graph.
    """
    return nx.pagerank(G, alpha=alpha)


def extract_node_features(G: nx.Graph, model_name: str) -> pd.DataFrame:
    """
    Compute node-level graph features and return them as a DataFrame.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    model_name : str
        Label for the graph model, e.g. 'ER', 'WS', 'BA'.

    Returns
    -------
    pd.DataFrame
        One row per node with graph-derived features.
    """
    pagerank = compute_pagerank(G)
    degree_dict = dict(G.degree())
    clustering_dict = nx.clustering(G)
    betweenness_dict = nx.betweenness_centrality(G)
    closeness_dict = nx.closeness_centrality(G)

    nodes = list(G.nodes())

    df = pd.DataFrame(
        {
            "node": nodes,
            "degree": [degree_dict[node] for node in nodes],
            "pagerank": [pagerank[node] for node in nodes],
            "clustering_coef": [clustering_dict[node] for node in nodes],
            "betweenness": [betweenness_dict[node] for node in nodes],
            "closeness": [closeness_dict[node] for node in nodes],
            "model": model_name,
        }
    )

    df["log_degree"] = np.log1p(df["degree"])
    df["log_pagerank"] = np.log1p(df["pagerank"])

    return df
