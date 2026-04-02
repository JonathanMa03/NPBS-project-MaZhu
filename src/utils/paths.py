from __future__ import annotations

from pathlib import Path


def project_root(start: str | Path = ".") -> Path:
    """
    Infer the project root by walking upward until a directory containing
    both 'data' and 'notebooks' is found.

    Parameters
    ----------
    start : str or Path, default "."
        Starting path for the upward search.

    Returns
    -------
    Path
        Project root path.

    Raises
    ------
    FileNotFoundError
        If no valid project root is found.
    """
    start = Path(start).resolve()

    if start.is_file():
        start = start.parent

    current = start
    while current != current.parent:
        if (current / "data").exists() and (current / "notebooks").exists():
            return current
        current = current.parent

    raise FileNotFoundError(
        "Could not infer project root. Expected a directory containing both "
        "'data' and 'notebooks'."
    )


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : str or Path
        Directory path.

    Returns
    -------
    Path
        Resolved directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir(root: str | Path) -> Path:
    return Path(root) / "data"


def raw_dir(root: str | Path) -> Path:
    return data_dir(root) / "raw"


def interim_dir(root: str | Path) -> Path:
    return data_dir(root) / "interim"


def processed_dir(root: str | Path) -> Path:
    return data_dir(root) / "processed"


def outputs_dir(root: str | Path) -> Path:
    return data_dir(root) / "outputs"


def reports_dir(root: str | Path) -> Path:
    return Path(root) / "reports"


def figures_dir(root: str | Path) -> Path:
    return reports_dir(root) / "figures"


def tables_dir(root: str | Path) -> Path:
    return reports_dir(root) / "tables"


def notebooks_dir(root: str | Path) -> Path:
    return Path(root) / "notebooks"


def src_dir(root: str | Path) -> Path:
    return Path(root) / "src"


def registry_path(root: str | Path) -> Path:
    return raw_dir(root) / "source_registry.csv"


def trial_raw_dir(root: str | Path, trial_id: str) -> Path:
    return raw_dir(root) / trial_id


def trial_interim_dir(root: str | Path, trial_id: str) -> Path:
    return interim_dir(root) / trial_id


def trial_processed_dir(root: str | Path, trial_id: str) -> Path:
    return processed_dir(root) / trial_id


def trial_outputs_dir(root: str | Path, trial_id: str) -> Path:
    return outputs_dir(root) / trial_id


def vec_km_dir(root: str | Path, trial_id: str) -> Path:
    return trial_interim_dir(root, trial_id) / "vec_km"


def cen_km_dir(root: str | Path, trial_id: str) -> Path:
    return trial_interim_dir(root, trial_id) / "cen_km"


def maple_dir(root: str | Path, trial_id: str) -> Path:
    return trial_interim_dir(root, trial_id) / "maple"


def reconstructed_ipd_path(root: str | Path, trial_id: str, endpoint: str = "os") -> Path:
    """
    Path for reconstructed overall IPD CSV.
    """
    return trial_processed_dir(root, trial_id) / f"{endpoint.lower()}_overall_ipd.csv"


def vec_km_points_path(root: str | Path, trial_id: str, curve_name: str = "overall") -> Path:
    return vec_km_dir(root, trial_id) / f"{curve_name}_curve_points.csv"


def vec_km_censors_path(root: str | Path, trial_id: str, curve_name: str = "overall") -> Path:
    return vec_km_dir(root, trial_id) / f"{curve_name}_censor_points.csv"


def vec_km_axes_path(root: str | Path, trial_id: str, curve_name: str = "overall") -> Path:
    return vec_km_dir(root, trial_id) / f"{curve_name}_axes.json"


def maple_labelings_dir(root: str | Path, trial_id: str) -> Path:
    return maple_dir(root, trial_id) / "labelings"


def maple_summary_path(root: str | Path, trial_id: str) -> Path:
    return maple_dir(root, trial_id) / "maple_summary.csv"


def trial_figure_path(root: str | Path, trial_id: str, figure_name: str) -> Path:
    return figures_dir(root) / trial_id / figure_name


def ensure_project_dirs(root: str | Path) -> None:
    """
    Ensure all high-level project directories exist.
    """
    for path in [
        data_dir(root),
        raw_dir(root),
        interim_dir(root),
        processed_dir(root),
        outputs_dir(root),
        reports_dir(root),
        figures_dir(root),
        tables_dir(root),
    ]:
        ensure_dir(path)


def ensure_trial_dirs(root: str | Path, trial_id: str) -> None:
    """
    Ensure all standard per-trial directories exist.
    """
    for path in [
        trial_raw_dir(root, trial_id),
        trial_interim_dir(root, trial_id),
        trial_processed_dir(root, trial_id),
        trial_outputs_dir(root, trial_id),
        vec_km_dir(root, trial_id),
        cen_km_dir(root, trial_id),
        maple_dir(root, trial_id),
        maple_labelings_dir(root, trial_id),
        figures_dir(root) / trial_id,
    ]:
        ensure_dir(path)