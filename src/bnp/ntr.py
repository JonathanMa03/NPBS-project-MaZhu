from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from .ntr_utils import (
    make_time_grid,
    compute_piecewise_exposure,
    compute_piecewise_events,
    posterior_survival_from_hazard_draws,
)


@dataclass
class NTRFit:
    """
    Piecewise-constant hazard model with Gamma priors on interval hazards.

    This is an NTR / hazard-process-inspired Bayesian survival model:
    survival is represented through
        S(t) = exp{-Lambda(t)}
    with a random cumulative hazard built from interval-specific hazards.

    For each treatment arm a in {0,1}:
        lambda_{a,j} ~ Gamma(shape_j, rate_j),   j = 1,...,J
        T_i | A_i=a follows piecewise-exponential likelihood
        with right censoring handled through exposure contributions.

    Posterior conjugacy:
        lambda_{a,j} | data ~ Gamma(shape_j + d_{a,j}, rate_j + Y_{a,j})

    where:
        d_{a,j} = number of events in interval j
        Y_{a,j} = total exposure time in interval j
    """

    hazard_draws: Dict[int, np.ndarray]
    breaks: np.ndarray
    t0_default: float | None
    metadata: Dict[str, Any]


def _as_binary_treatment(x: np.ndarray) -> np.ndarray:
    vals = np.unique(x)
    if not set(vals).issubset({0, 1}):
        raise ValueError("treatment column must contain only 0/1.")
    return x.astype(int)


def summarize_draws(draws: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(draws)),
        "sd": float(np.std(draws, ddof=0)),
        "q025": float(np.quantile(draws, 0.025)),
        "q975": float(np.quantile(draws, 0.975)),
    }


def fit_ntr_piecewise(
    df: pd.DataFrame,
    time_col: str = "time",
    event_col: str = "event",
    treatment_col: str = "treatment",
    n_intervals: int = 10,
    grid_strategy: str = "quantile",
    prior_shape: float = 0.5,
    prior_rate: float = 0.5,
    n_draws: int = 2000,
    random_state: int = 733,
    t0_default: float | None = None,
) -> NTRFit:
    """
    Fit an NTR-inspired piecewise hazard model separately by treatment arm.
    """
    required = {time_col, event_col, treatment_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rng = np.random.default_rng(random_state)

    times = df[time_col].to_numpy(dtype=float)
    events = df[event_col].to_numpy(dtype=int)
    treatment = _as_binary_treatment(df[treatment_col].to_numpy())

    if np.any(times <= 0):
        times = np.maximum(times, 1e-8)

    breaks = make_time_grid(times, n_intervals=n_intervals, strategy=grid_strategy)
    J = len(breaks) - 1

    hazard_draws: Dict[int, np.ndarray] = {}

    for a in [0, 1]:
        mask = treatment == a
        if mask.sum() == 0:
            raise ValueError(f"No observations found for treatment={a}.")

        t_a = times[mask]
        e_a = events[mask]

        exposure = compute_piecewise_exposure(t_a, breaks)
        counts = compute_piecewise_events(t_a, e_a, breaks)

        post_shape = prior_shape + counts
        post_rate = prior_rate + exposure

        draws = np.column_stack([
            rng.gamma(shape=post_shape[j], scale=1.0 / post_rate[j], size=n_draws)
            for j in range(J)
        ])

        hazard_draws[a] = draws

    metadata = {
        "n_intervals": J,
        "grid_strategy": grid_strategy,
        "prior_shape": prior_shape,
        "prior_rate": prior_rate,
        "n_draws": n_draws,
        "random_state": random_state,
        "time_col": time_col,
        "event_col": event_col,
        "treatment_col": treatment_col,
        "n_obs": len(df),
        "n_treat0": int(np.sum(treatment == 0)),
        "n_treat1": int(np.sum(treatment == 1)),
    }

    return NTRFit(
        hazard_draws=hazard_draws,
        breaks=breaks,
        t0_default=t0_default,
        metadata=metadata,
    )


def posterior_survival_draws(
    fit: NTRFit,
    t0: float,
) -> Dict[str, np.ndarray]:
    """
    Compute posterior draws of S0(t0), S1(t0), and Delta(t0).
    """
    h0 = fit.hazard_draws[0]
    h1 = fit.hazard_draws[1]

    n = min(h0.shape[0], h1.shape[0])

    s0 = posterior_survival_from_hazard_draws(h0[:n], fit.breaks, t0)
    s1 = posterior_survival_from_hazard_draws(h1[:n], fit.breaks, t0)
    delta = s1 - s0

    return {
        "S0": s0,
        "S1": s1,
        "Delta": delta,
    }


def compute_delta_posterior(
    fit: NTRFit,
    t0: float | None = None,
) -> Dict[str, Any]:
    """
    Summarize posterior distribution of Delta(t0).
    """
    if t0 is None:
        if fit.t0_default is None:
            raise ValueError("t0 must be supplied if no default was set in fit_ntr_piecewise.")
        t0 = fit.t0_default

    post = posterior_survival_draws(fit, t0=t0)

    delta_summary = summarize_draws(post["Delta"])
    s0_summary = summarize_draws(post["S0"])
    s1_summary = summarize_draws(post["S1"])

    return {
        "t0": float(t0),
        "S0_mean": s0_summary["mean"],
        "S1_mean": s1_summary["mean"],
        "mean": delta_summary["mean"],
        "sd": delta_summary["sd"],
        "q025": delta_summary["q025"],
        "q975": delta_summary["q975"],
        "delta_draws": post["Delta"],
        "s0_draws": post["S0"],
        "s1_draws": post["S1"],
    }