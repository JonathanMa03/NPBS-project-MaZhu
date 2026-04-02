from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _project_root(start: str | Path = ".") -> Path:
    start = Path(start).resolve()
    if start.is_file():
        start = start.parent

    current = start
    while current != current.parent:
        if (current / "src").exists() and (current / "external").exists():
            return current
        current = current.parent

    raise FileNotFoundError("Could not locate project root.")


def resolve_ipd_root(start: str | Path = ".") -> Path:
    root = _project_root(start)
    path = root / "external" / "resolve_ipd"
    if not path.exists():
        raise FileNotFoundError(f"RESOLVE-IPD submodule not found at: {path}")
    return path


def load_cen_km_module(start: str | Path = "."):
    """
    Dynamically load external/resolve_ipd/CEN_KM.py as a Python module.
    """
    root = resolve_ipd_root(start)
    cen_path = root / "CEN_KM.py"

    if not cen_path.exists():
        raise FileNotFoundError(f"CEN_KM.py not found at: {cen_path}")

    spec = importlib.util.spec_from_file_location("resolve_cen_km", cen_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {cen_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruct_ipd_with_resolve(
    n: int,
    t: Iterable[float],
    S: Iterable[float],
    cens_t: Optional[Iterable[float]] = None,
    match_tol: float = 5e-3,
    max_extra_censors_per_bin: int = 100,
    random_state: Optional[int] = None,
    risk_table: Optional[pd.DataFrame] = None,
    debug: bool = False,
    start: str | Path = ".",
) -> pd.DataFrame:
    """
    Wrapper around RESOLVE-IPD's get_ipd(...) function.

    Parameters
    ----------
    n : int
        Initial sample size.
    t : iterable of float
        Event/drop times from digitized KM curve.
    S : iterable of float
        Survival probabilities at drop times.
    cens_t : iterable of float, optional
        Observed censor times.
    risk_table : pd.DataFrame, optional
        Must contain columns ['time', 'n_at_risk'] if provided.

    Returns
    -------
    pd.DataFrame
        Reconstructed IPD with columns ['time', 'event'].
    """
    mod = load_cen_km_module(start=start)

    if risk_table is not None:
        expected = {"time", "n_at_risk"}
        missing = expected - set(risk_table.columns)
        if missing:
            raise ValueError(f"risk_table missing required columns: {sorted(missing)}")

    ipd = mod.get_ipd(
        n=n,
        t=t,
        S=S,
        cens_t=cens_t,
        match_tol=match_tol,
        max_extra_censors_per_bin=max_extra_censors_per_bin,
        random_state=random_state,
        risk_table=risk_table,
        debug=debug,
    )

    if not isinstance(ipd, pd.DataFrame):
        raise TypeError("Expected RESOLVE-IPD get_ipd(...) to return a pandas DataFrame.")

    required_cols = {"time", "event"}
    missing = required_cols - set(ipd.columns)
    if missing:
        raise ValueError(f"Returned IPD missing required columns: {sorted(missing)}")

    return ipd.copy()


def resolve_available_objects(start: str | Path = ".") -> list[str]:
    """
    List exposed names from external CEN_KM.py for debugging.
    """
    mod = load_cen_km_module(start=start)
    return sorted([name for name in dir(mod) if not name.startswith("__")])