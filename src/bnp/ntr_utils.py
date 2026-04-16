from __future__ import annotations

from typing import Tuple

import numpy as np


def make_time_grid(
    times: np.ndarray,
    n_intervals: int = 10,
    strategy: str = "quantile",
) -> np.ndarray:
    """
    Create a monotone grid of interval endpoints for a piecewise-constant hazard model.

    Parameters
    ----------
    times : np.ndarray
        Observed follow-up times.
    n_intervals : int
        Number of intervals.
    strategy : str
        "quantile" or "equal".

    Returns
    -------
    np.ndarray
        Breakpoints of length n_intervals + 1, starting at 0.
    """
    times = np.asarray(times, dtype=float)
    if np.any(times < 0):
        raise ValueError("times must be nonnegative.")

    t_max = float(np.max(times))
    if t_max <= 0:
        raise ValueError("times must contain at least one positive value.")

    if strategy == "quantile":
        probs = np.linspace(0, 1, n_intervals + 1)
        breaks = np.quantile(times, probs)
        breaks[0] = 0.0
        breaks[-1] = max(breaks[-1], t_max)
    elif strategy == "equal":
        breaks = np.linspace(0.0, t_max, n_intervals + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'equal'.")

    breaks = np.unique(breaks)
    if len(breaks) < 2:
        raise ValueError("Failed to construct a valid time grid.")

    if breaks[0] > 0:
        breaks = np.insert(breaks, 0, 0.0)

    return breaks.astype(float)


def interval_lengths(breaks: np.ndarray) -> np.ndarray:
    """
    Lengths of intervals induced by breaks.
    """
    breaks = np.asarray(breaks, dtype=float)
    if np.any(np.diff(breaks) <= 0):
        raise ValueError("breaks must be strictly increasing.")
    return np.diff(breaks)


def compute_piecewise_exposure(
    times: np.ndarray,
    breaks: np.ndarray,
) -> np.ndarray:
    """
    Compute total exposure time in each interval.

    For subject i with observed time t_i, contribution to interval j is:
        max(0, min(t_i, tau_j) - tau_{j-1})

    Parameters
    ----------
    times : np.ndarray
        Observed follow-up times.
    breaks : np.ndarray
        Monotone interval boundaries.

    Returns
    -------
    np.ndarray
        Exposure totals per interval, shape (J,).
    """
    times = np.asarray(times, dtype=float)
    breaks = np.asarray(breaks, dtype=float)

    J = len(breaks) - 1
    exposure = np.zeros(J, dtype=float)

    for j in range(J):
        left, right = breaks[j], breaks[j + 1]
        contrib = np.clip(
            np.minimum(times, right) - left,
            a_min=0.0,
            a_max=right - left,
        )
        exposure[j] = np.sum(contrib)

    return exposure


def compute_piecewise_events(
    times: np.ndarray,
    events: np.ndarray,
    breaks: np.ndarray,
) -> np.ndarray:
    """
    Count observed events falling in each interval.

    Event at time t_i is assigned to the interval (tau_{j-1}, tau_j].

    Parameters
    ----------
    times : np.ndarray
        Observed follow-up times.
    events : np.ndarray
        Event indicators (1=event, 0=censored).
    breaks : np.ndarray
        Monotone interval boundaries.

    Returns
    -------
    np.ndarray
        Event counts per interval, shape (J,).
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    breaks = np.asarray(breaks, dtype=float)

    if len(times) != len(events):
        raise ValueError("times and events must have the same length.")

    J = len(breaks) - 1
    counts = np.zeros(J, dtype=int)

    event_times = times[events == 1]
    # indices in 0,...,J-1
    idx = np.searchsorted(breaks[1:], event_times, side="left")
    idx = np.clip(idx, 0, J - 1)

    for j in idx:
        counts[j] += 1

    return counts


def cumulative_hazard_from_draw(
    hazards: np.ndarray,
    breaks: np.ndarray,
    t: float,
) -> float:
    """
    Evaluate cumulative hazard at time t for one hazard draw.
    """
    hazards = np.asarray(hazards, dtype=float)
    breaks = np.asarray(breaks, dtype=float)

    if t <= 0:
        return 0.0

    J = len(hazards)
    total = 0.0

    for j in range(J):
        left, right = breaks[j], breaks[j + 1]
        if t <= left:
            break
        dt = min(t, right) - left
        if dt > 0:
            total += hazards[j] * dt
        if t <= right:
            break

    return float(total)


def survival_from_hazard_draw(
    hazards: np.ndarray,
    breaks: np.ndarray,
    t: float,
) -> float:
    """
    Survival function at time t from one piecewise-constant hazard draw.
    """
    return float(np.exp(-cumulative_hazard_from_draw(hazards, breaks, t)))


def posterior_survival_from_hazard_draws(
    hazard_draws: np.ndarray,
    breaks: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Vectorized survival draws at time t from hazard draws.

    Parameters
    ----------
    hazard_draws : np.ndarray
        Shape (n_draws, J)
    breaks : np.ndarray
        Shape (J+1,)
    t : float
        Evaluation time

    Returns
    -------
    np.ndarray
        Shape (n_draws,)
    """
    return np.array(
        [survival_from_hazard_draw(h, breaks, t) for h in hazard_draws],
        dtype=float,
    )