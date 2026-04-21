from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.experiments import (
    add_global_pca_projection,
    build_intervention_summary,
    build_paired_comparisons,
    run_intervention_experiments,
    run_static_experiments,
    summarize_static_results,
)
from plotting.visualization import (
    plot_clustered_pca_panel,
    plot_example_graphs,
    plot_global_pca_clean,
    plot_mean_spread_curves_by_model,
    plot_metric_with_ci_clean,
    plot_model_metric_ci,
    plot_pagerank_distributions,
    plot_paired_difference_boxplot,
)


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
RAW_DIR = RESULTS_DIR / "raw"


def ensure_directories() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def save_tables(
    nodes_df: pd.DataFrame,
    results_df: pd.DataFrame,
    static_summary_df: pd.DataFrame,
    intervention_df: pd.DataFrame,
    intervention_summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
) -> None:
    """
    Save tabular outputs to results/tables and richer raw outputs to results/raw.
    """
    # Flat CSV-friendly outputs
    nodes_df.to_csv(TABLES_DIR / "node_level_results.csv", index=False)
    results_df.to_csv(TABLES_DIR / "graph_level_results.csv", index=False)
    static_summary_df.to_csv(TABLES_DIR / "static_summary.csv")

    intervention_summary_df.to_csv(TABLES_DIR / "intervention_summary.csv", index=False)
    paired_df.to_csv(TABLES_DIR / "paired_comparisons.csv", index=False)

    # Intervention dataframe contains list-like columns, so save both:
    # 1) a flat version for quick inspection
    intervention_flat = intervention_df.drop(columns=["curve", "removed_nodes"], errors="ignore")
    intervention_flat.to_csv(TABLES_DIR / "intervention_results.csv", index=False)

    # 2) full rich versions in raw/
    nodes_df.to_pickle(RAW_DIR / "node_level_results.pkl")
    results_df.to_pickle(RAW_DIR / "graph_level_results.pkl")
    intervention_df.to_pickle(RAW_DIR / "intervention_results.pkl")
    intervention_summary_df.to_pickle(RAW_DIR / "intervention_summary.pkl")
    paired_df.to_pickle(RAW_DIR / "paired_comparisons.pkl")


def make_figures(
    nodes_df: pd.DataFrame,
    results_df: pd.DataFrame,
    intervention_df: pd.DataFrame,
    intervention_summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
) -> None:
    """
    Generate all report-ready figures.
    """
    plot_example_graphs()
    plot_pagerank_distributions()

    plot_model_metric_ci(
        results_df,
        metric="pagerank_gini",
        title="PageRank inequality by graph model",
        ylabel="PageRank Gini coefficient",
        filename="pagerank_gini_by_model",
    )

    plot_model_metric_ci(
        results_df,
        metric="degree_std",
        title="Degree variability by graph model",
        ylabel="Standard deviation of degree",
        filename="degree_std_by_model",
    )

    plot_model_metric_ci(
        results_df,
        metric="num_clusters",
        title="Selected GMM clusters by graph model",
        ylabel="Number of selected clusters",
        filename="num_clusters_by_model",
    )

    plot_global_pca_clean(
        nodes_df,
        filename="combined_pca_projection",
    )

    plot_clustered_pca_panel(
        nodes_df,
        filename="clustered_pca_panels",
    )

    plot_metric_with_ci_clean(
        intervention_summary_df,
        metric="final_fraction_infected",
        title="Final outbreak size by graph model and intervention strategy",
        ylabel="Final fraction infected",
        filename="final_fraction_infected_ci",
    )

    plot_metric_with_ci_clean(
        intervention_summary_df,
        metric="t_50",
        title="Time to 50% infection by graph model and intervention strategy",
        ylabel="Time steps to 50% infection",
        filename="t50_ci",
    )

    plot_metric_with_ci_clean(
        intervention_summary_df,
        metric="auc_infected",
        title="Cumulative infection burden by graph model and intervention strategy",
        ylabel="Area under infection curve",
        filename="auc_ci",
    )

    plot_paired_difference_boxplot(
        paired_df,
        column="delta_final_targeted_vs_random",
        title="Paired effect of targeted vs random intervention",
        ylabel="Difference in final infected fraction\n(targeted - random)",
        filename="paired_targeted_vs_random_final_fraction",
    )

    plot_mean_spread_curves_by_model(
        intervention_df,
        filename="mean_spread_curves_by_model",
    )


def main() -> None:
    ensure_directories()

    print("Running static graph experiments...")
    nodes_df, results_df = run_static_experiments()

    print("Adding combined PCA projection...")
    nodes_df = add_global_pca_projection(nodes_df)

    print("Summarizing static results...")
    static_summary_df = summarize_static_results(results_df)

    print("Running intervention experiments...")
    intervention_df = run_intervention_experiments()

    print("Building intervention summaries...")
    intervention_summary_df = build_intervention_summary(intervention_df)

    print("Building paired comparisons...")
    paired_df = build_paired_comparisons(intervention_df)

    print("Saving tables...")
    save_tables(
        nodes_df=nodes_df,
        results_df=results_df,
        static_summary_df=static_summary_df,
        intervention_df=intervention_df,
        intervention_summary_df=intervention_summary_df,
        paired_df=paired_df,
    )

    print("Generating figures...")
    make_figures(
        nodes_df=nodes_df,
        results_df=results_df,
        intervention_df=intervention_df,
        intervention_summary_df=intervention_summary_df,
        paired_df=paired_df,
    )

    print("Done.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Raw outputs saved to: {RAW_DIR}")


if __name__ == "__main__":
    main()