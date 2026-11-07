"""Bounded local execution: each experiment runs in its own subprocess; concurrency is capped."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

from ..errors import RegistryError
from ..registry import Workspace


class LocalExecutor:
    def __init__(self, ws: Workspace, max_concurrent: int = 2) -> None:
        self.ws = ws
        self.max_concurrent = max(1, max_concurrent)
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def running(self) -> int:
        self.ws.store.reap_stale()
        return len(self.ws.store.list_experiments(status="RUNNING"))

    def submit(self, eid: str, *, wait_for_slot: bool = False, timeout: float = 600.0) -> subprocess.Popen:
        """Launch `python -m openkeystroke experiment run --id <eid>` detached. Raises if the concurrency cap is hit."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._procs = {k: p for k, p in self._procs.items() if p.poll() is None}
                n_local = len(self._procs)
                # also count externally started runs recorded in the registry
                if max(n_local, self.running()) < self.max_concurrent:
                    log = self.ws.root / "logs"
                    log.mkdir(exist_ok=True)
                    out = open(log / f"{eid}.log", "ab")  # noqa: SIM115 - handed to the child
                    # claim the slot synchronously so concurrent submitters see it
                    proc = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "openkeystroke",
                            "--home",
                            str(self.ws.root),
                            "experiment",
                            "run",
                            "--id",
                            eid,
                        ],
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    self._procs[eid] = proc
                    return proc
            if not wait_for_slot or time.monotonic() > deadline:
                raise RegistryError(f"concurrency limit reached ({self.max_concurrent} running experiments)")
            time.sleep(0.5)

    def wait(self, eid: str, timeout: float | None = None) -> int | None:
        p = self._procs.get(eid)
        return p.wait(timeout) if p else None


def run_many(ws: Workspace, ids: list[str], max_concurrent: int = 2) -> dict[str, str]:
    """Run several experiments with bounded concurrency; returns final statuses."""
    ex = LocalExecutor(ws, max_concurrent)
    for eid in ids:
        ex.submit(eid, wait_for_slot=True)
        time.sleep(0.2)
    for eid in ids:
        ex.wait(eid)
    return {eid: ws.store.get(eid)["status"] for eid in ids}


__all__ = ["LocalExecutor", "run_many", "Path"]
