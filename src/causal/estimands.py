from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def km_fit(
    df: pd.DataFrame,
    time_col: str = "time",
    event_col: str = "event",
) -> KaplanMeierFitter:
    """
    Fit a Kaplan–Meier model to one dataset.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], event_observed=df[event_col])
    return kmf


def survival_probability(
    df: pd.DataFrame,
    t0: float,
    time_col: str = "time",
    event_col: str = "event",
) -> float:
    """
    Estimate S(t0) = P(T > t0) via Kaplan–Meier.
    """
    kmf = km_fit(df, time_col=time_col, event_col=event_col)
    return float(kmf.predict(t0))


def treatment_specific_survival(
    df: pd.DataFrame,
    t0: float,
    treatment_value: int,
    treatment_col: str = "treatment",
    time_col: str = "time",
    event_col: str = "event",
) -> float:
    """
    Estimate survival probability at t0 within one treatment arm.
    """
    sub = df.loc[df[treatment_col] == treatment_value].copy()
    if sub.empty:
        raise ValueError(f"No rows found for treatment_value={treatment_value}.")
    return survival_probability(sub, t0=t0, time_col=time_col, event_col=event_col)


def survival_difference(
    df: pd.DataFrame,
    t0: float,
    treatment_col: str = "treatment",
    time_col: str = "time",
    event_col: str = "event",
) -> dict[str, float]:
    """
    Compute:
      S1(t0), S0(t0), Delta(t0)=S1(t0)-S0(t0)
    """
    s0 = treatment_specific_survival(
        df,
        t0=t0,
        treatment_value=0,
        treatment_col=treatment_col,
        time_col=time_col,
        event_col=event_col,
    )
    s1 = treatment_specific_survival(
        df,
        t0=t0,
        treatment_value=1,
        treatment_col=treatment_col,
        time_col=time_col,
        event_col=event_col,
    )

    return {
        "t0": float(t0),
        "S0": float(s0),
        "S1": float(s1),
        "Delta": float(s1 - s0),
    }


def trial_level_survival_difference(
    df: pd.DataFrame,
    t0: float,
    trial_col: str = "trial_id",
    treatment_col: str = "treatment",
    time_col: str = "time",
    event_col: str = "event",
) -> pd.DataFrame:
    """
    Compute fixed-time survival differences separately by trial.
    """
    rows = []

    for trial_id, g in df.groupby(trial_col):
        est = survival_difference(
            g,
            t0=t0,
            treatment_col=treatment_col,
            time_col=time_col,
            event_col=event_col,
        )
        rows.append({"trial_id": trial_id, **est})

    return pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True)


def pooled_survival_difference(
    df: pd.DataFrame,
    t0: float,
    treatment_col: str = "treatment",
    time_col: str = "time",
    event_col: str = "event",
) -> pd.DataFrame:
    """
    Compute one pooled fixed-time survival difference across all rows.
    """
    est = survival_difference(
        df,
        t0=t0,
        treatment_col=treatment_col,
        time_col=time_col,
        event_col=event_col,
    )
    return pd.DataFrame([est])