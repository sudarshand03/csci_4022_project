from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from config import RANDOM_STATE


def fit_best_gmm(
    df: pd.DataFrame,
    cols: list[str],
    k_range=range(2, 7),
    random_state: int | None = None,
):
    """
    Fit Gaussian Mixture Models for multiple values of k and select the one
    with the lowest BIC.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the columns to cluster on.
    cols : list[str]
        Column names to use for clustering.
    k_range : iterable
        Candidate numbers of clusters to evaluate.
    random_state : int | None
        Random state for reproducibility. Defaults to RANDOM_STATE from config.

    Returns
    -------
    tuple
        (output_df, best_model, best_k, best_bic)
        - output_df: copy of df with a 'cluster' column added
        - best_model: fitted GaussianMixture model
        - best_k: selected number of clusters
        - best_bic: BIC score of the selected model
    """
    if random_state is None:
        random_state = RANDOM_STATE

    X = df[cols].values

    best_model = None
    best_k = None
    best_bic = np.inf

    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=random_state)
        gmm.fit(X)
        bic = gmm.bic(X)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_model = gmm

    out = df.copy()
    out["cluster"] = best_model.predict(X)

    return out, best_model, best_k, best_bic
