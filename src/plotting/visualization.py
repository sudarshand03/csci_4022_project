from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    EXAMPLE_SEED,
    MODEL_LABELS,
    MODEL_ORDER,
    N_ANALYSIS,
    N_GRAPH_PLOT,
    STRATEGY_LABELS,
    STRATEGY_ORDER,
    get_default_graph_params,
)
from analysis.graph_models import generate_graph
from analysis.features import extract_node_features
from analysis.metrics import summarize_metric_with_ci


# -----------------------------
# Global plotting configuration
# -----------------------------

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "ER": "#4C78A8",
    "WS": "#F58518",
    "BA": "#54A24B",
}

STRATEGY_COLORS = {
    "none": "#B0B0B0",
    "targeted_pagerank": "#4C78A8",
    "random": "#E45756",
}


def set_plot_style() -> None:
    """
    Set consistent, report-friendly matplotlib defaults.
    """
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.8),
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 2.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 10,
            "axes.grid": False,
        }
    )


set_plot_style()


def _style_axes(ax, grid_axis: str = "y") -> None:
    """
    Apply consistent axis styling.
    """
    if grid_axis is not None:
        ax.grid(axis=grid_axis, alpha=0.22, linewidth=0.8)

    ax.spines["left"].set_alpha(0.7)
    ax.spines["bottom"].set_alpha(0.7)


def save_figure(filename: str, tight: bool = True) -> None:
    """
    Save the current figure to results/figures as both PDF and PNG.
    """
    if tight:
        plt.tight_layout()

    plt.savefig(FIG_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{filename}.png", bbox_inches="tight")


# -----------------------------
# Static graph visualizations
# -----------------------------

def plot_example_graphs(
    n: int = N_GRAPH_PLOT,
    seed: int = EXAMPLE_SEED,
    filename: str = "random_graph_examples",
) -> None:
    """
    Plot one example graph from each random graph family.
    """
    params = get_default_graph_params(n)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for ax, model in zip(axes, MODEL_ORDER):
        G = generate_graph(model, n=n, seed=seed, **params[model])

        # Use networkx spring layout for visual clarity
        pos = None
        try:
            import networkx as nx

            pos = nx.spring_layout(G, seed=seed)
            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=18,
                node_color=MODEL_COLORS[model],
                alpha=0.9,
                ax=ax,
            )
            nx.draw_networkx_edges(
                G,
                pos,
                width=0.4,
                alpha=0.22,
                edge_color="#666666",
                ax=ax,
            )
        except Exception:
            # Fallback: if plotting fails for any reason, leave blank but safe
            pass

        ax.set_title(MODEL_LABELS[model])
        ax.axis("off")

    save_figure(filename)
    plt.show()


def plot_pagerank_distributions(
    example_seed: int = EXAMPLE_SEED,
    n: int = N_ANALYSIS,
    filename: str = "pagerank_distributions",
) -> None:
    """
    Plot PageRank histograms for one example realization of each graph model.
    """
    params = get_default_graph_params(n)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

    for ax, model in zip(axes, MODEL_ORDER):
        G = generate_graph(model, n=n, seed=example_seed, **params[model])
        df = extract_node_features(G, model)

        ax.hist(
            df["pagerank"],
            bins=30,
            color=MODEL_COLORS[model],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("PageRank")
        _style_axes(ax, grid_axis="y")

    axes[0].set_ylabel("Node count")
    save_figure(filename)
    plt.show()


def plot_model_metric_ci(
    results_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    """
    Plot model-level means with 95% confidence intervals.
    """
    summary = summarize_metric_with_ci(results_df, metric, group_col="model")
    summary["order"] = summary["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    summary = summary.sort_values("order")

    x = np.arange(len(summary))
    means = summary["mean"].to_numpy()
    lower = means - summary["ci_lower"].to_numpy()
    upper = summary["ci_upper"].to_numpy() - means

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.errorbar(
        x,
        means,
        yerr=np.array([lower, upper]),
        fmt="o",
        markersize=7,
        capsize=5,
        color="#2F2F2F",
    )

    for i, row in summary.iterrows():
        ax.scatter(
            row["order"],
            row["mean"],
            s=75,
            color=MODEL_COLORS[row["model"]],
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in summary["model"]])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style_axes(ax, grid_axis="y")

    save_figure(filename)
    plt.show()


# -----------------------------
# PCA / clustering visualizations
# -----------------------------

def plot_global_pca_clean(
    nodes_df: pd.DataFrame,
    filename: str = "combined_pca_projection",
) -> None:
    """
    Plot combined PCA projection across all models.
    Expects columns global_PC1 and global_PC2.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    for model in MODEL_ORDER:
        subset = nodes_df[nodes_df["model"] == model]
        ax.scatter(
            subset["global_PC1"],
            subset["global_PC2"],
            alpha=0.23,
            s=18,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Combined PCA projection of node features")
    ax.legend(frameon=False)
    _style_axes(ax, grid_axis="both")

    save_figure(filename)
    plt.show()


def plot_clustered_pca_panel(
    nodes_df: pd.DataFrame,
    example_seed: int = EXAMPLE_SEED,
    filename: str = "clustered_pca_panels",
) -> None:
    """
    Plot PCA cluster panels for one graph realization per model.
    Expects columns: model, graph_seed, PC1, PC2, cluster.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=False, sharey=False)

    for ax, model in zip(axes, MODEL_ORDER):
        subset = nodes_df[
            (nodes_df["model"] == model) & (nodes_df["graph_seed"] == example_seed)
        ]

        if subset.empty:
            ax.set_title(f"{MODEL_LABELS[model]} (no data)")
            ax.set_xlabel("PC1")
            _style_axes(ax, grid_axis="both")
            continue

        clusters = sorted(subset["cluster"].dropna().unique())
        cluster_colors = plt.cm.Set2(np.linspace(0, 1, max(len(clusters), 1)))

        for color, cluster in zip(cluster_colors, clusters):
            cluster_df = subset[subset["cluster"] == cluster]
            ax.scatter(
                cluster_df["PC1"],
                cluster_df["PC2"],
                s=24,
                alpha=0.65,
                color=color,
                label=f"Cluster {cluster}",
            )

        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("PC1")
        _style_axes(ax, grid_axis="both")

    axes[0].set_ylabel("PC2")
    axes[-1].legend(frameon=False, loc="best")

    save_figure(filename)
    plt.show()


# -----------------------------
# Intervention visualizations
# -----------------------------

def plot_metric_with_ci_clean(
    summary_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    """
    Plot intervention outcomes by graph model and strategy with 95% CIs.
    Expects build_intervention_summary-style dataframe.
    """
    subset = summary_df[summary_df["metric"] == metric].copy()

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    width = 0.22
    x = np.arange(len(MODEL_ORDER))

    for j, strategy in enumerate(STRATEGY_ORDER):
        means = []
        lower_err = []
        upper_err = []

        for model in MODEL_ORDER:
            row = subset[
                (subset["model"] == model) & (subset["strategy"] == strategy)
            ].iloc[0]

            means.append(row["mean"])
            lower_err.append(row["mean"] - row["ci_lower"])
            upper_err.append(row["ci_upper"] - row["mean"])

        offset = (j - 1) * width

        ax.bar(
            x + offset,
            means,
            width=width,
            label=STRATEGY_LABELS[strategy],
            color=STRATEGY_COLORS[strategy],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=np.array([lower_err, upper_err]),
            fmt="none",
            ecolor="#2F2F2F",
            capsize=4,
            linewidth=1.2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    _style_axes(ax, grid_axis="y")

    save_figure(filename)
    plt.show()


def plot_paired_difference_boxplot(
    paired_df: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    """
    Plot paired intervention differences by model as boxplots.
    """
    fig, ax = plt.subplots(figsize=(7.8, 5.1))

    data = [
        paired_df[paired_df["model"] == model][column].dropna()
        for model in MODEL_ORDER
    ]

    bp = ax.boxplot(
        data,
        labels=[MODEL_LABELS[m] for m in MODEL_ORDER],
        widths=0.58,
        patch_artist=True,
    )

    for patch, model in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.45)
        patch.set_edgecolor("#2F2F2F")

    for median in bp["medians"]:
        median.set_color("#2F2F2F")
        median.set_linewidth(2)

    ax.axhline(0, linestyle="--", linewidth=1.5, color="#444444")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    _style_axes(ax, grid_axis="y")

    save_figure(filename)
    plt.show()


def plot_mean_spread_curves_by_model(
    intervention_df: pd.DataFrame,
    filename: str = "mean_spread_curves_by_model",
) -> None:
    """
    Plot mean spread curves with percentile bands for each intervention strategy,
    split across graph models.
    Expects intervention_df to contain columns: model, strategy, curve.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)

    for ax, model in zip(axes, MODEL_ORDER):
        model_df = intervention_df[intervention_df["model"] == model]

        for strategy in STRATEGY_ORDER:
            subset = model_df[model_df["strategy"] == strategy]

            if subset.empty:
                continue

            curves = np.vstack(subset["curve"].to_numpy())
            mean_curve = curves.mean(axis=0)
            lower = np.percentile(curves, 2.5, axis=0)
            upper = np.percentile(curves, 97.5, axis=0)
            t = np.arange(curves.shape[1])

            ax.plot(
                t,
                mean_curve,
                label=STRATEGY_LABELS[strategy],
                color=STRATEGY_COLORS[strategy],
            )
            ax.fill_between(
                t,
                lower,
                upper,
                alpha=0.18,
                color=STRATEGY_COLORS[strategy],
            )

        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Time step")
        _style_axes(ax, grid_axis="y")

    axes[0].set_ylabel("Fraction infected")
    axes[-1].legend(frameon=False, loc="lower right")

    save_figure(filename)
    plt.show()