"""Dashboard data access. Everything here reads the real registry / artifact store / dataset directory."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ExperimentConfig, SyntheticConfig
from ..data import Dataset, validate_dataset
from ..errors import NotFoundError
from ..experiments import LocalExecutor, create_experiment
from ..registry import Workspace


def get_workspace(home: str | None = None) -> Workspace:
    return Workspace(home or os.environ.get("OPENKEYSTROKE_HOME"))


# ---------------------------------------------------------------------------- datasets
def list_datasets(ws: Workspace) -> list[dict[str, Any]]:
    import json

    out = []
    for d in sorted(ws.datasets_dir.iterdir()) if ws.datasets_dir.exists() else []:
        mf = d / "manifest.json"
        if d.is_dir() and mf.is_file():
            try:
                m = json.loads(mf.read_text())
                out.append(
                    {
                        "path": str(d),
                        "name": d.name,
                        "dataset_id": m["dataset_id"],
                        "version": m["version"],
                        "n_samples": m["n_samples"],
                        "is_synthetic": m["is_synthetic"],
                        "fingerprint": m["fingerprint"],
                        "mtime": mf.stat().st_mtime,
                    }
                )
            except (ValueError, KeyError):
                continue
    return out


def generate_demo_dataset(ws: Workspace, cfg: SyntheticConfig | None = None) -> Path:
    from ..config import config_hash
    from ..data import generate_synthetic

    cfg = cfg or SyntheticConfig()
    path = ws.datasets_dir / f"synthetic-{config_hash(cfg, 10)}"
    if not path.exists():
        generate_synthetic(cfg).save(path)
    return path


def dataset_quality(ds: Dataset) -> dict[str, Any]:
    return validate_dataset(ds).to_dict()


# ---------------------------------------------------------------------------- experiments
def experiment_overview(ws: Workspace) -> dict[str, Any]:
    ws.store.reap_stale()
    return {"counts": ws.store.counts(), "recent": ws.store.list_experiments(limit=8)}


def completed_with_summary(
    ws: Workspace, kind: str | None = None, *, with_defenses: bool = False
) -> list[dict[str, Any]]:
    out = []
    for r in ws.store.list_experiments(status="COMPLETED", kind=kind, limit=500):
        if not ws.store.has_artifact(r["id"], "summary.json"):
            continue
        if with_defenses and not load_summary(ws, r["id"])["defense_points"]:
            continue
        out.append(r)
    return out


def artifact_mtime(ws: Workspace, eid: str, name: str) -> float:
    try:
        return ws.store.artifact_path(eid, name).stat().st_mtime
    except (NotFoundError, OSError):
        return 0.0


def normalize_summary(s: dict[str, Any]) -> dict[str, Any]:
    """Make summaries written by older code versions look current (adds the `eligible` flag / eligible-only frontier)."""
    from ..defense.metrics import pareto_frontier

    for t in s.get("table", []):
        for k in ("accuracy", "balanced_accuracy", "f1_macro", "val_accuracy"):
            if k in t and "ci_lo" in t[k]:
                t[k]["ci_lo"], t[k]["ci_hi"] = max(t[k]["ci_lo"], 0.0), min(t[k]["ci_hi"], 1.0)
    for p in s.get("defense_points", []):
        if "eligible" not in p:
            ev = p.get("evidence", {})
            p["eligible"] = bool(
                p.get("privacy_gain") is not None
                and p["privacy_gain"] > 0
                and "no_significant_reduction" not in ev.values()
            )
    for fr in s.get("pareto", []):
        if "n_eligible" not in fr:
            pts = [
                p
                for p in s["defense_points"]
                if (p["variant"], p["protocol"], p["model"]) == (fr["variant"], fr["protocol"], fr["model"])
                and p["eligible"]
            ]
            fr["frontier"], fr["n_eligible"] = (
                pareto_frontier(pts, "privacy_gain", "log_spectral_distance_db"),
                len(pts),
            )
    return s


def load_summary(ws: Workspace, eid: str) -> dict[str, Any]:
    return normalize_summary(ws.store.get_json(eid, "summary.json"))


def load_results(ws: Workspace, eid: str) -> dict[str, Any]:
    return ws.store.get_json(eid, "results.json")


def model_leaderboard(ws: Workspace) -> list[dict[str, Any]]:
    """Best clean/static held-out accuracy per model across completed experiments (real stored metrics)."""
    rows = []
    for r in completed_with_summary(ws):
        try:
            s = load_summary(ws, r["id"])
        except (NotFoundError, ValueError):
            continue
        for t in s["table"]:
            if t["condition"] == "clean" and t["attacker"] == "static":
                rows.append(
                    {
                        "experiment": r["id"],
                        "name": r["name"],
                        "protocol": t["protocol"],
                        "model": t["model"],
                        "accuracy": t["accuracy"]["mean"],
                        "std": t["accuracy"]["std"],
                        "n_seeds": t["accuracy"]["n"],
                        "synthetic": bool(r["dataset_is_synthetic"]),
                    }
                )
    return rows


def defense_overview(ws: Workspace) -> list[dict[str, Any]]:
    out = []
    for r in completed_with_summary(ws, "defense_comparison"):
        s = load_summary(ws, r["id"])
        for p in s["defense_points"]:
            out.append(
                {
                    "experiment": r["id"],
                    **{
                        k: p[k]
                        for k in ("name", "model", "privacy_gain", "log_spectral_distance_db", "eligible")
                    },
                }
            )
    return out


def config_from_form(base: ExperimentConfig, **updates: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**base.model_dump(mode="json"), **updates})


def launch(ws: Workspace, cfg: ExperimentConfig) -> str:
    """Create + validate + start in a bounded worker subprocess. Validation errors are raised, not hidden."""
    from ..experiments import validate_experiment

    eid = create_experiment(ws, cfg)
    validate_experiment(ws, eid)
    LocalExecutor(ws, cfg.resources.max_concurrent_experiments).submit(eid)
    return eid


def relaunch(ws: Workspace, eid: str) -> str:
    return launch(ws, ws.store.get_config(eid))


def active_runs(ws: Workspace) -> list[dict[str, Any]]:
    ws.store.reap_stale()
    return [r for s in ("RUNNING", "VALIDATED", "CREATED") for r in ws.store.list_experiments(status=s)]


# ---------------------------------------------------------------------------- system
def system_status(ws: Workspace) -> dict[str, Any]:
    import torch

    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines()[:5]:
            k, v = line.split(":")
            mem[k] = int(v.split()[0]) / 1024
    except OSError:
        pass
    du = shutil.disk_usage(ws.root)
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    return {
        "cpu_count": os.cpu_count(),
        "load_1m": load,
        "mem_total_mb": mem.get("MemTotal"),
        "mem_available_mb": mem.get("MemAvailable"),
        "disk_free_gb": du.free / 1e9,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "workspace": str(ws.root),
        "registry_schema": ws.store.schema_version(),
        "time": time.strftime("%H:%M:%S"),
    }


def sample_view(ds: Dataset, idx: int) -> dict[str, Any]:
    """Raw waveform + metadata for one sample."""
    r = ds.records[idx]
    return {
        "x": np.asarray(ds.waveforms[idx], dtype=np.float32),
        "sr": ds.sample_rate,
        "meta": r.model_dump(mode="json"),
    }
