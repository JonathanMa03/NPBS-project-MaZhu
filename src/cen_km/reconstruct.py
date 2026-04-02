from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.cen_km.km_math import estimate_events_from_survival_drop


@dataclass
class ReconstructionResult:
    event_table: pd.DataFrame
    ipd: pd.DataFrame


def _safe_int(x: float | int) -> int:
    return int(max(0, round(float(x))))


def _survival_after_interval(n_risk: int, d: int, S_prev: float) -> float:
    """
    KM survival update for a single event time.
    """
    if n_risk <= 0:
        return S_prev
    return S_prev * (1.0 - d / n_risk)


def _expand_ipd_rows(
    event_times: list[float],
    censor_times: list[float],
) -> pd.DataFrame:
    """
    Build patient-level IPD from lists of event and censor times.
    """
    event_df = pd.DataFrame(
        {
            "time": event_times,
            "event": np.ones(len(event_times), dtype=int),
        }
    )
    censor_df = pd.DataFrame(
        {
            "time": censor_times,
            "event": np.zeros(len(censor_times), dtype=int),
        }
    )

    ipd = pd.concat([event_df, censor_df], ignore_index=True)
    if not ipd.empty:
        ipd = ipd.sort_values(["time", "event"], ascending=[True, False]).reset_index(drop=True)
    return ipd


def reconstruct_ipd_basic(
    curve_points: pd.DataFrame,
    n_initial: int,
) -> ReconstructionResult:
    """
    Basic KM reconstruction without explicit censor marks.

    This is a simple baseline:
    - invert KM drops into event counts
    - assume no censoring
    - place all events exactly at the recorded step times

    Parameters
    ----------
    curve_points : pd.DataFrame
        Must contain columns: time, survival
    n_initial : int
        Number at risk at baseline

    Returns
    -------
    ReconstructionResult
    """
    required = {"time", "survival"}
    missing = required - set(curve_points.columns)
    if missing:
        raise ValueError(f"curve_points missing required columns: {sorted(missing)}")

    curve = curve_points.sort_values("time").reset_index(drop=True).copy()

    times = curve["time"].to_list()
    surv = curve["survival"].to_list()

    n_risk = n_initial
    S_prev = surv[0]

    event_rows: list[dict[str, Any]] = []
    event_times: list[float] = []
    censor_times: list[float] = []

    # first row is baseline / starting point
    event_rows.append(
        {
            "time": times[0],
            "survival_target": surv[0],
            "survival_reconstructed": surv[0],
            "n_risk": n_risk,
            "n_events": 0,
            "n_censored": 0,
        }
    )

    for i in range(1, len(times)):
        t = times[i]
        S_target = surv[i]

        d = estimate_events_from_survival_drop(
            S_prev=S_prev,
            S_next=S_target,
            n_risk=n_risk,
        )

        S_recon = _survival_after_interval(n_risk=n_risk, d=d, S_prev=S_prev)

        event_rows.append(
            {
                "time": t,
                "survival_target": S_target,
                "survival_reconstructed": S_recon,
                "n_risk": n_risk,
                "n_events": d,
                "n_censored": 0,
            }
        )

        event_times.extend([t] * d)

        n_risk = max(n_risk - d, 0)
        S_prev = S_recon

    event_table = pd.DataFrame(event_rows)
    ipd = _expand_ipd_rows(event_times=event_times, censor_times=censor_times)

    return ReconstructionResult(event_table=event_table, ipd=ipd)


def reconstruct_ipd_with_censoring(
    curve_points: pd.DataFrame,
    censor_points: pd.DataFrame,
    n_initial: int,
    max_additional_censors_per_interval: int = 5,
) -> ReconstructionResult:
    """
    Censor-aware reconstruction prototype.

    Uses:
    - KM step coordinates
    - explicit censor times
    - simple branching over event counts near the KM-implied count
    - interval-level matching to the target survival

    This is a first implementation of the CEN-KM idea, not yet risk-table aligned.

    Parameters
    ----------
    curve_points : pd.DataFrame
        Columns: time, survival
    censor_points : pd.DataFrame
        Columns: time, survival, optionally n_censored
    n_initial : int
        Initial sample size
    max_additional_censors_per_interval : int
        Max extra censor multiplicity to consider beyond observed marks

    Returns
    -------
    ReconstructionResult
    """
    required_curve = {"time", "survival"}
    missing_curve = required_curve - set(curve_points.columns)
    if missing_curve:
        raise ValueError(f"curve_points missing required columns: {sorted(missing_curve)}")

    required_censor = {"time"}
    missing_censor = required_censor - set(censor_points.columns)
    if missing_censor:
        raise ValueError(f"censor_points missing required columns: {sorted(missing_censor)}")

    curve = curve_points.sort_values("time").reset_index(drop=True).copy()
    censors = censor_points.sort_values("time").reset_index(drop=True).copy()

    if "n_censored" not in censors.columns:
        censors["n_censored"] = 1

    times = curve["time"].to_list()
    surv = curve["survival"].to_list()

    n_risk = n_initial
    S_prev = surv[0]

    event_rows: list[dict[str, Any]] = []
    event_times: list[float] = []
    censor_times_out: list[float] = []

    event_rows.append(
        {
            "time": times[0],
            "survival_target": surv[0],
            "survival_reconstructed": surv[0],
            "n_risk": n_risk,
            "n_events": 0,
            "n_censored": 0,
        }
    )

    for i in range(1, len(times)):
        t_prev = times[i - 1]
        t_curr = times[i]
        S_target = surv[i]

        # explicit censors in this interval
        interval_censors = censors.loc[
            (censors["time"] > t_prev) & (censors["time"] <= t_curr)
        ].copy()

        base_censor_count = _safe_int(interval_censors["n_censored"].sum())
        base_censor_times = interval_censors["time"].tolist()

        # KM-implied event count
        d_hat = estimate_events_from_survival_drop(
            S_prev=S_prev,
            S_next=S_target,
            n_risk=n_risk,
        )

        candidate_ds = sorted(set([max(d_hat - 1, 0), d_hat, d_hat + 1]))

        best = None

        for d in candidate_ds:
            for extra_c in range(0, max_additional_censors_per_interval + 1):
                total_c = base_censor_count + extra_c

                if d + total_c > n_risk:
                    continue

                S_recon = _survival_after_interval(n_risk=n_risk, d=d, S_prev=S_prev)
                err = abs(S_recon - S_target)

                cand = {
                    "d": d,
                    "n_censored": total_c,
                    "survival_reconstructed": S_recon,
                    "error": err,
                }

                if best is None or cand["error"] < best["error"]:
                    best = cand

        if best is None:
            best = {
                "d": 0,
                "n_censored": 0,
                "survival_reconstructed": S_prev,
                "error": abs(S_prev - S_target),
            }

        d_star = int(best["d"])
        c_star = int(best["n_censored"])
        S_recon = float(best["survival_reconstructed"])

        # assign event times at the step time
        event_times.extend([t_curr] * d_star)

        # keep observed censor times and, if needed, duplicate the last observed time
        # or place extras just before the event time if no observed censor exists
        if base_censor_times:
            censor_times_interval = base_censor_times.copy()
            while len(censor_times_interval) < c_star:
                censor_times_interval.append(base_censor_times[-1])
        else:
            censor_times_interval = [max(t_prev + 1e-8, t_curr - 1e-8)] * c_star

        censor_times_out.extend(censor_times_interval)

        event_rows.append(
            {
                "time": t_curr,
                "survival_target": S_target,
                "survival_reconstructed": S_recon,
                "n_risk": n_risk,
                "n_events": d_star,
                "n_censored": c_star,
                "interval_error": abs(S_recon - S_target),
            }
        )

        n_risk = max(n_risk - d_star - c_star, 0)
        S_prev = S_recon

    event_table = pd.DataFrame(event_rows)
    ipd = _expand_ipd_rows(event_times=event_times, censor_times=censor_times_out)

    return ReconstructionResult(event_table=event_table, ipd=ipd)


def align_to_risk_table(
    event_table: pd.DataFrame,
    risk_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Placeholder for future risk-table alignment.

    Expected future behavior:
    - compare reconstructed n_risk to published risk table
    - adjust censor counts locally while preserving event times

    For now, returns event_table unchanged.
    """
    return event_table.copy()