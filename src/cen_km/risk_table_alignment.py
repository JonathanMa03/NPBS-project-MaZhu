from __future__ import annotations

import numpy as np
import pandas as pd


def compute_n_at_risk(ipd: pd.DataFrame, times: list[float] | np.ndarray) -> pd.DataFrame:
    """
    Compute number at risk just prior to each requested time.

    Parameters
    ----------
    ipd : pd.DataFrame
        Must contain columns: time, event
    times : list or array
        Risk-table times

    Returns
    -------
    pd.DataFrame
        Columns: time, n_risk
    """
    if not {"time", "event"}.issubset(ipd.columns):
        raise ValueError("ipd must contain columns ['time', 'event'].")

    t = ipd["time"].to_numpy()
    out = []

    for tau in times:
        n_risk = int(np.sum(t >= tau))
        out.append({"time": float(tau), "n_risk": n_risk})

    return pd.DataFrame(out)


def align_ipd_to_risk_table(
    ipd: pd.DataFrame,
    risk_table: pd.DataFrame,
    time_col: str = "time",
    n_risk_col: str = "n_risk",
) -> pd.DataFrame:
    """
    Align reconstructed IPD to a published risk table by modifying only censoring times.

    Strategy
    --------
    For each boundary time tau:
      - if current n_risk(tau) is too small, move some earlier censor times to tau
      - if current n_risk(tau) is too large, move some later censor times down to just before tau

    This preserves event times and only adjusts censoring locations.

    Parameters
    ----------
    ipd : pd.DataFrame
        Must contain columns: time, event
    risk_table : pd.DataFrame
        Must contain columns: time_col, n_risk_col

    Returns
    -------
    pd.DataFrame
        Adjusted IPD
    """
    if not {"time", "event"}.issubset(ipd.columns):
        raise ValueError("ipd must contain columns ['time', 'event'].")
    if not {time_col, n_risk_col}.issubset(risk_table.columns):
        raise ValueError(f"risk_table must contain columns ['{time_col}', '{n_risk_col}'].")

    out = ipd.copy().sort_values(["time", "event"], ascending=[True, False]).reset_index(drop=True)
    rt = risk_table.sort_values(time_col).reset_index(drop=True)

    censor_mask = out["event"] == 0

    for _, row in rt.iterrows():
        tau = float(row[time_col])
        target = int(row[n_risk_col])

        current = int((out["time"] >= tau).sum())
        diff = target - current

        if diff == 0:
            continue

        # Need MORE at risk at tau: move some censorings from before tau to tau
        if diff > 0:
            candidates = out.index[censor_mask & (out["time"] < tau)].tolist()
            if len(candidates) == 0:
                continue

            move_n = min(diff, len(candidates))
            # move the latest pre-tau censorings first
            candidates = sorted(candidates, key=lambda i: out.loc[i, "time"], reverse=True)[:move_n]
            out.loc[candidates, "time"] = tau

        # Need FEWER at risk at tau: move some post-tau censorings to just before tau
        else:
            diff = abs(diff)
            candidates = out.index[censor_mask & (out["time"] >= tau)].tolist()
            if len(candidates) == 0:
                continue

            move_n = min(diff, len(candidates))
            # move the earliest post-tau censorings first
            candidates = sorted(candidates, key=lambda i: out.loc[i, "time"])[:move_n]
            out.loc[candidates, "time"] = np.nextafter(tau, -np.inf)

        out = out.sort_values(["time", "event"], ascending=[True, False]).reset_index(drop=True)

    return out


def risk_table_error(
    ipd: pd.DataFrame,
    risk_table: pd.DataFrame,
    time_col: str = "time",
    n_risk_col: str = "n_risk",
) -> pd.DataFrame:
    """
    Compare reconstructed n_at_risk to target risk table.
    """
    comp = compute_n_at_risk(ipd, risk_table[time_col].to_list())
    comp = comp.rename(columns={"n_risk": "n_risk_reconstructed"})
    comp["n_risk_target"] = risk_table[n_risk_col].to_numpy()
    comp["error"] = comp["n_risk_reconstructed"] - comp["n_risk_target"]
    return comp