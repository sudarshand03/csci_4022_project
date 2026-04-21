from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE


def run_pca(
    df: pd.DataFrame,
    cols: list[str],
    n_components: int = 2,
):
    """
    Standardize selected columns and run PCA.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing feature columns.
    cols : list[str]
        Columns to include in PCA.
    n_components : int
        Number of principal components to compute.

    Returns
    -------
    tuple
        (output_df, pca_model, scaler)
        - output_df: copy of df with PC1, PC2, ... columns added
        - pca_model: fitted sklearn PCA object
        - scaler: fitted StandardScaler object
    """
    X = df[cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    out = df.copy()
    for i in range(n_components):
        out[f"PC{i + 1}"] = X_pca[:, i]

    return out, pca, scaler
