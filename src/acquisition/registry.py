from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REQUIRED_COLUMNS = [
    "trial_id",
    "study_name",
    "endpoint",
    "source_pdf",
    "source_svg",
    "overall_km_available",
    "subgroup_km_available",
    "risk_table_available",
    "subgroup_summary_available",
    "notes",
]


def initialize_source_registry(path: str | Path) -> pd.DataFrame:
    """
    Create an empty source registry with the required columns and save it.

    Parameters
    ----------
    path : str or Path
        Output CSV path.

    Returns
    -------
    pd.DataFrame
        Empty registry DataFrame.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    df.to_csv(path, index=False)
    return df


def load_source_registry(path: str | Path) -> pd.DataFrame:
    """
    Load a source registry CSV and validate its schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")

    df = pd.read_csv(path)
    validate_source_registry(df)
    return df


def save_source_registry(df: pd.DataFrame, path: str | Path) -> None:
    """
    Validate and save a source registry CSV.
    """
    validate_source_registry(df)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def validate_source_registry(df: pd.DataFrame) -> None:
    """
    Ensure required columns exist and trial IDs are unique.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Registry missing required columns: {missing}")

    if df["trial_id"].duplicated().any():
        dups = df.loc[df["trial_id"].duplicated(), "trial_id"].tolist()
        raise ValueError(f"Duplicate trial_id values found: {dups}")


def add_source_entry(df: pd.DataFrame, entry: dict[str, Any]) -> pd.DataFrame:
    """
    Add one trial/source entry to the registry.

    Missing required fields are filled with defaults if possible.
    """
    trial_id = entry.get("trial_id")
    if not trial_id:
        raise ValueError("Entry must include a non-empty 'trial_id'.")

    if trial_id in df["trial_id"].astype(str).tolist():
        raise ValueError(f"trial_id '{trial_id}' already exists in registry.")

    row = {col: entry.get(col, _default_value_for_column(col)) for col in REQUIRED_COLUMNS}
    out = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    validate_source_registry(out)
    return out


def update_source_entry(
    df: pd.DataFrame,
    trial_id: str,
    updates: dict[str, Any],
) -> pd.DataFrame:
    """
    Update an existing registry row by trial_id.
    """
    mask = df["trial_id"].astype(str) == str(trial_id)
    if not mask.any():
        raise KeyError(f"trial_id '{trial_id}' not found in registry.")

    out = df.copy()
    for key, value in updates.items():
        if key not in out.columns:
            raise KeyError(f"Column '{key}' is not in the registry schema.")
        out.loc[mask, key] = value

    validate_source_registry(out)
    return out


def get_trial_record(df: pd.DataFrame, trial_id: str) -> pd.Series:
    """
    Return a single trial record as a pandas Series.
    """
    mask = df["trial_id"].astype(str) == str(trial_id)
    if not mask.any():
        raise KeyError(f"trial_id '{trial_id}' not found in registry.")
    return df.loc[mask].iloc[0]


def list_trials(df: pd.DataFrame) -> list[str]:
    """
    Return all trial IDs in registry order.
    """
    return df["trial_id"].astype(str).tolist()


def filter_trials(
    df: pd.DataFrame,
    *,
    endpoint: str | None = None,
    overall_km_available: bool | None = None,
    subgroup_km_available: bool | None = None,
    risk_table_available: bool | None = None,
    subgroup_summary_available: bool | None = None,
) -> pd.DataFrame:
    """
    Filter registry rows by common study metadata.
    """
    out = df.copy()

    if endpoint is not None:
        out = out.loc[out["endpoint"].astype(str).str.lower() == endpoint.lower()]

    for column, value in [
        ("overall_km_available", overall_km_available),
        ("subgroup_km_available", subgroup_km_available),
        ("risk_table_available", risk_table_available),
        ("subgroup_summary_available", subgroup_summary_available),
    ]:
        if value is not None:
            out = out.loc[_as_bool_series(out[column]) == value]

    return out.reset_index(drop=True)


def registry_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a compact summary view useful in notebooks.
    """
    cols = [
        "trial_id",
        "study_name",
        "endpoint",
        "overall_km_available",
        "subgroup_km_available",
        "risk_table_available",
        "subgroup_summary_available",
    ]
    return df[cols].copy()


def ensure_registry(
    path: str | Path,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Load an existing registry or create a new empty one.

    Parameters
    ----------
    path : str or Path
        CSV path for the registry.
    overwrite : bool, default False
        If True, create a fresh empty registry even if file exists.
    """
    path = Path(path)
    if overwrite or not path.exists():
        return initialize_source_registry(path)
    return load_source_registry(path)


def add_many_source_entries(
    df: pd.DataFrame,
    entries: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    """
    Add multiple entries sequentially.
    """
    out = df.copy()
    for entry in entries:
        out = add_source_entry(out, entry)
    return out


def _default_value_for_column(column: str) -> Any:
    if column in {
        "overall_km_available",
        "subgroup_km_available",
        "risk_table_available",
        "subgroup_summary_available",
    }:
        return False
    return ""


def _as_bool_series(series: pd.Series) -> pd.Series:
    """
    Robustly coerce a registry column to boolean values.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    lowered = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y"}
    return lowered.isin(truthy)