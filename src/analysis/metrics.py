from __future__ import annotations

import numpy as np
import pandas as pd


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)

    if np.any(x < 0):
        raise ValueError("Gini coefficient is not defined for negative values.")

    if np.allclose(x.sum(), 0):
        return 0.0

    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)

    return (2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n


def mean_ci(series, z: float = 1.96) -> pd.Series:
    """
    Compute mean, standard deviation, sample size, and an approximate
    normal-based confidence interval.s
    """
    x = pd.Series(series).dropna()
    n = len(x)

    if n == 0:
        return pd.Series(
            {
                "mean": np.nan,
                "std": np.nan,
                "n": 0,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            }
        )

    mean = x.mean()
    std = x.std(ddof=1) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0

    return pd.Series(
        {
            "mean": mean,
            "std": std,
            "n": n,
            "ci_lower": mean - z * se,
            "ci_upper": mean + z * se,
        }
    )


def summarize_metric_with_ci(
    df: pd.DataFrame, metric: str, group_col: str = "model"
) -> pd.DataFrame:
    """
    Group a dataframe by `group_col` and summarize one metric using mean_ci.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    metric : str
        Column name to summarize.
    group_col : str
        Column to group by.

    Returns
    -------
    pd.DataFrame
        Summary dataframe with one row per group.
    """
    rows = []

    for group_value, group_df in df.groupby(group_col):
        stats = mean_ci(group_df[metric])
        rows.append(
            {
                group_col: group_value,
                "mean": stats["mean"],
                "std": stats["std"],
                "n": stats["n"],
                "ci_lower": stats["ci_lower"],
                "ci_upper": stats["ci_upper"],
            }
        )

    return pd.DataFrame(rows)
