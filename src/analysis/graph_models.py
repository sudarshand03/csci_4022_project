from __future__ import annotations

import networkx as nx


def generate_erdos_renyi(n: int, p: float, seed: int | None = None) -> nx.Graph:
    return nx.erdos_renyi_graph(n=n, p=p, seed=seed)


def generate_watts_strogatz(
    n: int,
    k: int,
    beta: float,
    seed: int | None = None,
) -> nx.Graph:
    return nx.watts_strogatz_graph(n=n, k=k, p=beta, seed=seed)


def generate_barabasi_albert(
    n: int,
    m: int,
    seed: int | None = None,
) -> nx.Graph:
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)


def generate_graph(model: str, n: int, seed: int | None = None, **kwargs) -> nx.Graph:
    """
    Dispatch to the appropriate graph generator.

    Parameters
    ----------
    model : str
        One of 'ER', 'WS', or 'BA'.
    n : int
        Number of nodes.
    seed : int | None
        Random seed.
    **kwargs
        Model-specific parameters:
        - ER: p
        - WS: k, beta
        - BA: m
    """
    model = model.upper()

    if model == "ER":
        return generate_erdos_renyi(n=n, p=kwargs["p"], seed=seed)
    if model == "WS":
        return generate_watts_strogatz(
            n=n, k=kwargs["k"], beta=kwargs["beta"], seed=seed
        )
    if model == "BA":
        return generate_barabasi_albert(n=n, m=kwargs["m"], seed=seed)

    raise ValueError(f"Unknown model: {model}")
