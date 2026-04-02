from __future__ import annotations

import numpy as np
import pandas as pd


def km_step_update(n_risk: int, d: int) -> float:
    """
    One KM step:

    S(t) = S(t-) * (1 - d / n_risk)
    """
    if n_risk <= 0:
        return 1.0
    return 1.0 * (1 - d / n_risk)


def survival_from_events(
    event_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute survival curve from event table.

    event_table must contain:
    - n_risk
    - n_events

    Returns
    -------
    DataFrame with survival column
    """
    S = 1.0
    survival = []

    for _, row in event_table.iterrows():
        n_risk = row["n_risk"]
        d = row["n_events"]

        if n_risk > 0:
            S *= (1 - d / n_risk)

        survival.append(S)

    out = event_table.copy()
    out["survival"] = survival
    return out


def estimate_events_from_survival_drop(
    S_prev: float,
    S_next: float,
    n_risk: int,
) -> int:
    """
    Invert KM step to estimate number of events:

    S_next = S_prev * (1 - d / n_risk)

    ⇒ d ≈ n_risk * (1 - S_next / S_prev)
    """
    if S_prev <= 0 or n_risk <= 0:
        return 0

    frac = 1 - (S_next / S_prev)
    d = n_risk * frac

    return max(0, int(round(d)))


def compute_risk_set(
    n_initial: int,
    events: list[int],
    censors: list[int],
) -> list[int]:
    """
    Compute risk set over time:

    n_{t+1} = n_t - d_t - c_t
    """
    n_risk = [n_initial]

    for d, c in zip(events, censors):
        n_next = n_risk[-1] - d - c
        n_risk.append(max(n_next, 0))

    return n_risk[:-1]


def reconstruct_event_table_from_curve(
    curve_points: pd.DataFrame,
    n_initial: int,
) -> pd.DataFrame:
    """
    Build event table from KM curve (no censoring yet).

    Parameters
    ----------
    curve_points : DataFrame
        Columns: time, survival
    n_initial : int
        Initial sample size

    Returns
    -------
    DataFrame with:
        time, survival, n_risk, n_events
    """
    times = curve_points["time"].values
    surv = curve_points["survival"].values

    n = len(times)

    n_risk = [n_initial]
    n_events = []

    for i in range(1, n):
        S_prev = surv[i - 1]
        S_next = surv[i]

        d = estimate_events_from_survival_drop(
            S_prev=S_prev,
            S_next=S_next,
            n_risk=n_risk[-1],
        )

        n_events.append(d)

        n_next = n_risk[-1] - d
        n_risk.append(max(n_next, 0))

    n_events = [0] + n_events  # first time has no event

    return pd.DataFrame(
        {
            "time": times,
            "survival": surv,
            "n_risk": n_risk,
            "n_events": n_events,
        }
    )