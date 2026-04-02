from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
) -> dict:
    """
    Compute trial-level effects and a sample-size-weighted pooled contrast.
    """
    rows = []

    for trial_id, g in df.groupby(trial_col):
        est = survival_difference(g, t0=t0)
        rows.append(
            {
                "trial_id": trial_id,
                "n": len(g),
                **est,
            }
        )

    df_trials = pd.DataFrame(rows)
    weights = df_trials["n"] / df_trials["n"].sum()
    delta_weighted = float((weights * df_trials["Delta"]).sum())

    return {
        "Delta_weighted": delta_weighted,
        "trial_table": df_trials,
    }


def compute_covariate_means(
    df: pd.DataFrame,
    covariates: list[str],
    weights: np.ndarray | None = None,
) -> pd.Series:
    """
    Compute unweighted or weighted covariate means.
    """
    X = df[covariates].to_numpy(dtype=float)

    if weights is None:
        return pd.Series(X.mean(axis=0), index=covariates)

    w = np.asarray(weights, dtype=float)
    if len(w) != len(df):
        raise ValueError("weights must have the same length as df.")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative.")
    if w.sum() <= 0:
        raise ValueError("weights must sum to a positive value.")

    w = w / w.sum()
    return pd.Series((w[:, None] * X).sum(axis=0), index=covariates)


def reweight_to_target(
    df: pd.DataFrame,
    covariates: list[str],
    target_means: pd.Series | dict,
    standardize: bool = True,
    clip_logits: tuple[float, float] = (-50.0, 50.0),
    method: str = "BFGS",
) -> np.ndarray:
    """
    Construct weights by exponential tilting so weighted covariate means
    match target means as closely as possible.

    Parameters
    ----------
    df : pd.DataFrame
        Input pooled dataset.
    covariates : list[str]
        Covariates to balance.
    target_means : pd.Series or dict
        Target means keyed by covariate name.
    standardize : bool, default True
        Standardize covariates before optimization for numerical stability.
    clip_logits : tuple[float, float], default (-50, 50)
        Bounds for linear predictor before exponentiation.
    method : str, default "BFGS"
        scipy.optimize.minimize method.

    Returns
    -------
    np.ndarray
        Normalized nonnegative weights summing to 1.
    """
    X_raw = df[covariates].to_numpy(dtype=float)

    if isinstance(target_means, dict):
        target_series = pd.Series(target_means)
    else:
        target_series = target_means.copy()

    target_series = target_series.loc[covariates].astype(float)
    target_raw = target_series.to_numpy()

    if standardize:
        X_mean = X_raw.mean(axis=0)
        X_std = X_raw.std(axis=0, ddof=0)
        X_std[X_std == 0] = 1.0

        X = (X_raw - X_mean) / X_std
        target = (target_raw - X_mean) / X_std
    else:
        X = X_raw
        target = target_raw

    def compute_weights(theta: np.ndarray) -> np.ndarray:
        logits = X @ theta #softmax stabilization
        logits = np.nan_to_num(logits, neginf=clip_logits[0], posinf=clip_logits[1])
        logits = logits - np.max(logits)
        logits = np.clip(logits, clip_logits[0], clip_logits[1])
        w = np.exp(logits)
        w /= w.sum()
        return w

    def objective(theta: np.ndarray) -> float:
        w = compute_weights(theta)
        moments = w @ X
        moment_loss = np.sum((moments - target) ** 2)
        ridge = 1e-4 * np.sum(theta ** 2)
        return float(moment_loss + ridge)

    theta0 = np.zeros(X.shape[1], dtype=float)
    bounds = [(-10, 10)] * X.shape[1]
    res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds)

    if not res.success:
        raise RuntimeError(f"Weight optimization failed: {res.message}")

    return compute_weights(res.x)


def weighted_survival_difference(
    df: pd.DataFrame,
    weights: np.ndarray,
    t0: float,
    sample_size: int | None = None,
    random_state: int = 0,
) -> dict:
    """
    Approximate a weighted survival contrast by weighted bootstrap resampling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    weights : np.ndarray
        Nonnegative weights of length len(df).
    t0 : float
        Fixed evaluation time.
    sample_size : int or None
        Size of weighted resample. Defaults to len(df).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Output from survival_difference(...)
    """
    w = np.asarray(weights, dtype=float)

    if len(w) != len(df):
        raise ValueError("weights must have the same length as df.")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative.")
    if w.sum() <= 0:
        raise ValueError("weights must sum to a positive value.")

    w = w / w.sum()

    df_rep = df.sample(
        n=len(df) if sample_size is None else int(sample_size),
        replace=True,
        weights=w,
        random_state=random_state,
    )

    return survival_difference(df_rep, t0=t0)