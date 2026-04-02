from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    """
    Ensure the parent directory of a file path exists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(src: str | Path, dst: str | Path, overwrite: bool = False) -> Path:
    """
    Copy a file to a destination path.

    Parameters
    ----------
    src : str or Path
        Source file path.
    dst : str or Path
        Destination file path.
    overwrite : bool, default False
        Whether to overwrite the destination if it exists.

    Returns
    -------
    Path
        Destination path.
    """
    src = Path(src)
    dst = ensure_parent(dst)

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    if dst.exists() and not overwrite:
        return dst

    shutil.copy2(src, dst)
    return dst


def read_json(path: str | Path) -> dict[str, Any]:
    """
    Read a JSON file into a dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: dict[str, Any], path: str | Path, indent: int = 2) -> Path:
    """
    Write a dictionary to JSON.
    """
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    return path


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Read a CSV into a DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, index: bool = False, **kwargs: Any) -> Path:
    """
    Write a DataFrame to CSV.
    """
    path = ensure_parent(path)
    df.to_csv(path, index=index, **kwargs)
    return path


def append_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """
    Append a DataFrame to an existing CSV, or create it if absent.
    """
    path = ensure_parent(path)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=index)
    return path


def path_exists(path: str | Path) -> bool:
    """
    Check if a path exists.
    """
    return Path(path).exists()


def file_size_bytes(path: str | Path) -> int:
    """
    Return file size in bytes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.stat().st_size


def list_files(directory: str | Path, pattern: str = "*") -> list[Path]:
    """
    List files in a directory matching a glob pattern.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted([p for p in directory.glob(pattern) if p.is_file()])


def list_subdirectories(directory: str | Path) -> list[Path]:
    """
    List subdirectories in a directory.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted([p for p in directory.iterdir() if p.is_dir()])


def make_text_snapshot(text: str, path: str | Path) -> Path:
    """
    Save raw text content to a file.
    """
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
    return path