"""OpenKeystroke: a research platform for authorized acoustic side-channel experiments and defenses."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

_API = (
    "Workspace",
    "generate_synthetic_dataset",
    "load_dataset",
    "validate_dataset",
    "ingest_consented_recordings",
    "run_experiment",
    "create_experiment",
    "get_experiment",
    "list_experiments",
    "get_summary",
    "export_results",
    "preset_config",
    "ExperimentConfig",
)


def __getattr__(name: str) -> Any:  # lazy: keep `import openkeystroke` cheap (torch is heavy)
    if name in _API:
        from . import api

        return getattr(api, name)
    raise AttributeError(name)


__all__ = ["__version__", *_API]
