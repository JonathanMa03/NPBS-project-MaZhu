from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd


@dataclass
class DPSMFit:
    """
    Finite-mixture approximation to a Dirichlet Process Survival Model.

    We fit separate treatment-specific survival distributions using a
    Bayesian mixture of exponential kernels with a Dirichlet prior on
    mixture weights. This is a practical truncated approximation to a DP
    mixture, and it handles right censoring through the exponential
    survival likelihood.

    For treatment arm a in {0,1}:
        z_i | pi_a ~ Categorical(pi_a)
        T_i | z_i=k, lambda_{a,k} ~ Exponential(rate=lambda_{a,k})
        pi_a ~ Dirichlet(alpha / K, ..., alpha / K)
        lambda_{a,k} ~ Gamma(rate_shape, rate_scale)

    Censoring:
        If event_i = 1:
            p(t_i | lambda_k) = lambda_k * exp(-lambda_k * t_i)
        If event_i = 0:
            p(t_i | lambda_k) = exp(-lambda_k * t_i)

    Notes
    -----
    - This is model-based BNP and is more explicit than the Dirichlet
      weighting / Bayesian bootstrap layer used elsewhere in the project.
    - It is intentionally lightweight and stable enough for a course project.
    """

    treatment_draws: Dict[int, Dict[str, np.ndarray]]
    t0_default: float | None
    metadata: Dict[str, Any]


def _as_binary_treatment(x: np.ndarray) -> np.ndarray:
    vals = np.unique(x)
    if not set(vals).issubset({0, 1}):
        raise ValueError("treatment column must contain only 0/1.")
    return x.astype(int)


def _survival_from_draw(weights: np.ndarray, rates: np.ndarray, t: float) -> float:
    return float(np.sum(weights * np.exp(-rates * t)))


def _sample_allocations(
    times: np.ndarray,
    events: np.ndarray,
    weights: np.ndarray,
    rates: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample latent component allocations for one treatment arm.
    """
    n = len(times)
    k = len(weights)

    # log p(z_i = j | ...) ∝ log w_j + δ_i log λ_j - λ_j t_i
    logp = np.log(weights + 1e-300)[None, :] + events[:, None] * np.log(rates + 1e-300)[None, :] - times[:, None] * rates[None, :]
    logp = logp - logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    p = p / p.sum(axis=1, keepdims=True)

    z = np.empty(n, dtype=int)
    for i in range(n):
        z[i] = rng.choice(k, p=p[i])
    return z


def _gibbs_one_arm(
    times: np.ndarray,
    events: np.ndarray,
    n_components: int,
    alpha: float,
    rate_shape: float,
    rate_scale: float,
    n_samples: int,
    burn_in: int,
    thin: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Gibbs sampler for one treatment arm.
    """
    n = len(times)
    k = n_components

    # Initialize
    weights = np.ones(k) / k
    # spread rates across a reasonable range based on observed times
    mean_time = max(np.mean(times), 1e-3)
    base_rate = 1.0 / mean_time
    rates = rng.gamma(shape=rate_shape, scale=1.0 / rate_scale, size=k) + np.linspace(0.25, 2.0, k) * base_rate

    kept_weights = []
    kept_rates = []

    total_iters = burn_in + n_samples * thin
    dirichlet_base = np.full(k, alpha / k)

    for it in range(total_iters):
        # Sample allocations
        z = _sample_allocations(times, events, weights, rates, rng)

        # Sample component-specific rates
        for j in range(k):
            mask = z == j
            d_j = np.sum(events[mask])
            y_j = np.sum(times[mask])

            post_shape = rate_shape + d_j
            post_rate = rate_scale + y_j  # Gamma in rate parameterization

            # numpy uses scale = 1 / rate
            rates[j] = rng.gamma(shape=post_shape, scale=1.0 / post_rate)

        # Sample weights
        counts = np.bincount(z, minlength=k)
        weights = rng.dirichlet(dirichlet_base + counts)

        # Store
        if it >= burn_in and ((it - burn_in) % thin == 0):
            kept_weights.append(weights.copy())
            kept_rates.append(rates.copy())

    return {
        "weights": np.asarray(kept_weights),
        "rates": np.asarray(kept_rates),
    }


def fit_dpsm(
    df: pd.DataFrame,
    time_col: str = "time",
    event_col: str = "event",
    treatment_col: str = "treatment",
    n_components: int = 8,
    alpha: float = 1.0,
    rate_shape: float = 1.0,
    rate_scale: float = 1.0,
    n_samples: int = 500,
    burn_in: int = 500,
    thin: int = 2,
    random_state: int = 733,
    t0_default: float | None = None,
) -> DPSMFit:
    """
    Fit a truncated DP-style survival mixture separately by treatment arm.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns for time, event, and treatment (0/1).
    time_col, event_col, treatment_col : str
        Column names.
    n_components : int
        Truncation level for DP approximation.
    alpha : float
        Concentration parameter for Dirichlet prior on mixture weights.
    rate_shape, rate_scale : float
        Hyperparameters for Gamma prior on exponential rates.
        Prior is Gamma(shape=rate_shape, rate=rate_scale).
    n_samples : int
        Number of posterior draws to keep.
    burn_in : int
        Burn-in iterations.
    thin : int
        Thinning factor.
    random_state : int
        RNG seed.
    t0_default : float | None
        Optional default evaluation time.

    Returns
    -------
    DPSMFit
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
        # Exponential likelihood is defined at 0, but tiny positive jitter is safer numerically
        times = np.maximum(times, 1e-8)

    treatment_draws: Dict[int, Dict[str, np.ndarray]] = {}

    for a in [0, 1]:
        mask = treatment == a
        if mask.sum() == 0:
            raise ValueError(f"No observations found for treatment={a}.")

        arm_draws = _gibbs_one_arm(
            times=times[mask],
            events=events[mask],
            n_components=n_components,
            alpha=alpha,
            rate_shape=rate_shape,
            rate_scale=rate_scale,
            n_samples=n_samples,
            burn_in=burn_in,
            thin=thin,
            rng=rng,
        )
        treatment_draws[a] = arm_draws

    metadata = {
        "n_components": n_components,
        "alpha": alpha,
        "rate_shape": rate_shape,
        "rate_scale": rate_scale,
        "n_samples": n_samples,
        "burn_in": burn_in,
        "thin": thin,
        "random_state": random_state,
        "time_col": time_col,
        "event_col": event_col,
        "treatment_col": treatment_col,
        "n_obs": len(df),
        "n_treat0": int(np.sum(treatment == 0)),
        "n_treat1": int(np.sum(treatment == 1)),
    }

    return DPSMFit(
        treatment_draws=treatment_draws,
        t0_default=t0_default,
        metadata=metadata,
    )


def posterior_survival_draws(
    fit: DPSMFit,
    t0: float,
) -> Dict[str, np.ndarray]:
    """
    Compute posterior draws of S0(t0), S1(t0), and Delta(t0).
    """
    w0 = fit.treatment_draws[0]["weights"]
    r0 = fit.treatment_draws[0]["rates"]

    w1 = fit.treatment_draws[1]["weights"]
    r1 = fit.treatment_draws[1]["rates"]

    n0 = w0.shape[0]
    n1 = w1.shape[0]
    n = min(n0, n1)

    s0 = np.array([_survival_from_draw(w0[i], r0[i], t0) for i in range(n)])
    s1 = np.array([_survival_from_draw(w1[i], r1[i], t0) for i in range(n)])
    delta = s1 - s0

    return {
        "S0": s0,
        "S1": s1,
        "Delta": delta,
    }


def summarize_draws(draws: np.ndarray) -> Dict[str, float]:
    """
    Posterior summary helper.
    """
    return {
        "mean": float(np.mean(draws)),
        "sd": float(np.std(draws, ddof=0)),
        "q025": float(np.quantile(draws, 0.025)),
        "q975": float(np.quantile(draws, 0.975)),
    }


def compute_delta_posterior(
    fit: DPSMFit,
    t0: float | None = None,
) -> Dict[str, Any]:
    """
    Summarize the posterior distribution of Delta(t0).

    Returns
    -------
    dict
        {
            "t0": ...,
            "S0_mean": ...,
            "S1_mean": ...,
            "mean": ...,
            "sd": ...,
            "q025": ...,
            "q975": ...,
            "delta_draws": np.ndarray,
            "s0_draws": np.ndarray,
            "s1_draws": np.ndarray,
        }
    """
    if t0 is None:
        if fit.t0_default is None:
            raise ValueError("t0 must be supplied if no default was set in fit_dpsm.")
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