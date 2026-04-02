from __future__ import annotations

import numpy as np
import pandas as pd

from src.causal.estimands import survival_difference


def naive_pooled_estimate(
    df: pd.DataFrame,
    t0: float,
) -> dict:
    """
    Pool all data ignoring trial structure.
    """
    return survival_difference(df, t0=t0)


def trial_weighted_estimate(
    df: pd.DataFrame,
    t0: float,
    trial_col: str = "trial_id",
) -> pd.DataFrame:
    """
    Compute trial-level effects and weight by sample size.
    """
    rows = []

    for trial_id, g in df.groupby(trial_col):
        est = survival_difference(g, t0=t0)
        rows.append({
            "trial_id": trial_id,
            "n": len(g),
            **est
        })

    df_trials = pd.DataFrame(rows)

    weights = df_trials["n"] / df_trials["n"].sum()

    delta_weighted = (weights * df_trials["Delta"]).sum()

    return {
        "Delta_weighted": float(delta_weighted),
        "trial_table": df_trials
    }


def compute_covariate_means(
    df: pd.DataFrame,
    covariates: list[str],
    weights: np.ndarray | None = None,
) -> pd.Series:
    """
    Compute (possibly weighted) covariate means.
    """
    X = df[covariates].to_numpy()

    if weights is None:
        return pd.Series(X.mean(axis=0), index=covariates)

    w = weights / weights.sum()
    return pd.Series((w[:, None] * X).sum(axis=0), index=covariates)


def reweight_to_target(
    df: pd.DataFrame,
    covariates: list[str],
    target_means: pd.Series,
) -> np.ndarray:
    """
    Simple exponential tilting weights (prototype).

    This is NOT full entropy balancing yet,
    but a stepping stone toward it.
    """
    X = df[covariates].to_numpy()

    # solve via simple least squares proxy (temporary)
    theta = np.linalg.lstsq(X, target_means.values, rcond=None)[0]

    logits = X @ theta
    logits = np.clip(logits, -50, 50)  # avoid overflow

    w = np.exp(logits)
    w /= w.sum()

    return w


def weighted_survival_difference(
    df: pd.DataFrame,
    weights: np.ndarray,
    t0: float,
) -> dict:
    """
    Compute weighted survival difference.
    """
    df = df.copy()
    df["w"] = weights

    # resample approximation (simple version)
    df_rep = df.sample(
        n=len(df),
        replace=True,
        weights=df["w"],
        random_state=0,
    )

    return survival_difference(df_rep, t0=t0)