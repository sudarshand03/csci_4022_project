import networkx as nx


def generate_erdos_renyi(n: int, p: float, seed: int | None = None) -> nx.Graph:
    return nx.erdos_renyi_graph(n=n, p=p, seed=seed)


def generate_barabasi_albert(n: int, m: int, seed: int | None = None) -> nx.Graph:
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)


def generate_watts_strogatz(
    n: int, k: int, beta: float, seed: int | None = None
) -> nx.Graph:
    return nx.watts_strogatz_graph(n=n, k=k, p=beta, seed=seed)
