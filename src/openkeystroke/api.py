"""Stable public Python API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentConfig, SyntheticConfig, load_config, parse_config
from .data import Dataset, ValidationReport
from .data import generate_synthetic as _generate
from .data import ingest_consented_recordings as ingest_consented_recordings
from .data import validate_dataset as _validate
from .experiments.runner import create_experiment, validate_experiment
from .experiments.runner import run_experiment as _run
from .presets import preset_config
from .registry import Workspace

__all__ = [
    "ExperimentConfig",
    "Workspace",
    "create_experiment",
    "export_results",
    "generate_synthetic_dataset",
    "get_experiment",
    "get_summary",
    "ingest_consented_recordings",
    "list_experiments",
    "load_config",
    "load_dataset",
    "parse_config",
    "preset_config",
    "run_experiment",
    "validate_dataset",
    "validate_experiment",
]


def generate_synthetic_dataset(
    config: SyntheticConfig | None = None, out: str | Path | None = None
) -> Dataset:
    ds = _generate(config or SyntheticConfig())
    if out is not None:
        ds.save(out)
    return ds


def load_dataset(path: str | Path, *, verify: bool = True) -> Dataset:
    return Dataset.load(path, verify=verify)


def validate_dataset(ds: Dataset | str | Path) -> ValidationReport:
    return _validate(Dataset.load(ds) if isinstance(ds, (str, Path)) else ds)


def run_experiment(config: ExperimentConfig | str, workspace: Workspace | None = None) -> tuple[str, str]:
    """Create (or reuse by id), validate and run an experiment in-process. Returns (experiment_id, final_status)."""
    ws = workspace or Workspace()
    eid = config if isinstance(config, str) else create_experiment(ws, config)
    return eid, _run(ws, eid)


def get_experiment(experiment_id: str, workspace: Workspace | None = None) -> dict[str, Any]:
    return (workspace or Workspace()).store.get(experiment_id)


def list_experiments(workspace: Workspace | None = None, **filters: Any) -> list[dict[str, Any]]:
    return (workspace or Workspace()).store.list_experiments(**filters)


def get_summary(experiment_id: str, workspace: Workspace | None = None) -> dict[str, Any]:
    return (workspace or Workspace()).store.get_json(experiment_id, "summary.json")


def export_results(
    experiment_ids: list[str], path: str | Path, fmt: str = "csv", workspace: Workspace | None = None
) -> Path:
    return (workspace or Workspace()).store.export_metrics(experiment_ids, Path(path), fmt)
