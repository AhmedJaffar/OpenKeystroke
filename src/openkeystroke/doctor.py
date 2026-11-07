"""Installation and workspace diagnostics."""

from __future__ import annotations

import importlib
import sys
import tempfile
from typing import Any

from .registry import Workspace


def run_doctor(ws: Workspace, self_test: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        out.append({"name": name, "ok": bool(ok), "detail": detail})

    add("python", sys.version_info >= (3, 10), sys.version.split()[0])
    for mod in (
        "numpy",
        "scipy",
        "sklearn",
        "torch",
        "pydantic",
        "soundfile",
        "pandas",
        "plotly",
        "matplotlib",
        "streamlit",
        "click",
        "yaml",
    ):
        try:
            m = importlib.import_module(mod)
            add(f"import {mod}", True, getattr(m, "__version__", "?"))
        except Exception as exc:  # noqa: BLE001
            add(f"import {mod}", False, str(exc))
    try:
        import torch

        add("torch device", True, "cuda" if torch.cuda.is_available() else "cpu (CUDA not available)")
    except Exception as exc:  # noqa: BLE001
        add("torch device", False, str(exc))
    try:
        probe = ws.root / ".write_test"
        probe.write_text("x")
        probe.unlink()
        add("workspace writable", True, str(ws.root))
        add("registry schema", True, f"v{ws.store.schema_version()} ({ws.store.db_path.name})")
        stale = ws.store.reap_stale()
        add("stale runs", True, f"{len(stale)} reaped")
    except Exception as exc:  # noqa: BLE001
        add("workspace", False, str(exc))
    if self_test:
        try:
            from .config import SyntheticConfig, TrainConfig
            from .experiments import create_experiment, run_experiment
            from .presets import preset_config

            tmp = Workspace(tempfile.mkdtemp(prefix="oks-doctor-"))
            cfg = preset_config(
                "single",
                synthetic=SyntheticConfig(
                    n_classes=3, n_devices=3, sessions_per_device=1, samples_per_class_per_session=6
                ),
            )
            eid = create_experiment(tmp, cfg)
            st = run_experiment(tmp, eid)
            acc = (
                tmp.store.get_json(eid, "summary.json")["table"][0]["accuracy"]["mean"]
                if st == "COMPLETED"
                else None
            )
            add("end-to-end self-test", st == "COMPLETED", f"{st}; held-out accuracy={acc}")
            del TrainConfig
        except Exception as exc:  # noqa: BLE001
            add("end-to-end self-test", False, f"{type(exc).__name__}: {exc}")
    return out
