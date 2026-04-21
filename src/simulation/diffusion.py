from __future__ import annotations

import networkx as nx
import numpy as np


def simulate_si_spread(
    G: nx.Graph,
    initial_infected,
    beta: float = 0.20,
    steps: int = 20,
    rng=None,
):
    """
    Simulate a simple SI (Susceptible-Infected) process on a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    initial_infected : hashable
        Initial infected node.
    beta : float
        Infection probability per infected-susceptible edge per step.
    steps : int
        Number of time steps to simulate.
    rng : np.random.Generator | None
        Random number generator. If None, a new generator is created.

    Returns
    -------
    tuple[list[int], list[float]]
        infected_counts:
            Number of infected nodes at each time step, including time 0.
        infected_fraction:
            Fraction of infected nodes at each time step, including time 0.
    """
    if rng is None:
        rng = np.random.default_rng()

    infected = {initial_infected}
    infected_counts = [1]
    infected_fraction = [1 / G.number_of_nodes()]

    for _ in range(steps):
        new_infected = set(infected)

        for node in infected:
            for neighbor in G.neighbors(node):
                if neighbor not in infected and rng.random() < beta:
                    new_infected.add(neighbor)

        infected = new_infected
        infected_counts.append(len(infected))
        infected_fraction.append(len(infected) / G.number_of_nodes())

    return infected_counts, infected_fraction


def time_to_threshold(fractions, threshold: float):
    """
    Return the first time step at which the infected fraction
    reaches or exceeds the given threshold.
    """
    for t, frac in enumerate(fractions):
        if frac >= threshold:
            return t
    return np.nan


def area_under_curve(y):
    """
    Compute the area under a 1D curve using the trapezoidal rule.
    """
    y = np.asarray(y, dtype=float)
    return np.trapezoid(y, dx=1)
