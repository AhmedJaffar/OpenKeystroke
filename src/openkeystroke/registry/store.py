"""Persistent local experiment registry: SQLite metadata (WAL, migrations, atomic transitions) + filesystem artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ExperimentConfig
from ..errors import LifecycleError, NotFoundError, RegistryError

STATUSES = ("CREATED", "VALIDATED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED")
TRANSITIONS: dict[str, set[str]] = {
    "CREATED": {"VALIDATED", "FAILED", "CANCELLED"},
    "VALIDATED": {"RUNNING", "FAILED", "CANCELLED"},
    "RUNNING": {"COMPLETED", "FAILED", "CANCELLED"},
    "COMPLETED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}

MIGRATIONS: list[str] = [
    # v1: core tables
    """
    CREATE TABLE experiments (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('CREATED','VALIDATED','RUNNING','COMPLETED','FAILED','CANCELLED')),
        config_json TEXT NOT NULL, config_hash TEXT NOT NULL,
        dataset_id TEXT, dataset_version TEXT, dataset_fingerprint TEXT, dataset_is_synthetic INTEGER,
        code_revision TEXT, code_dirty INTEGER, environment_json TEXT,
        created_at TEXT NOT NULL, validated_at TEXT, started_at TEXT, finished_at TEXT,
        pid INTEGER, progress_done INTEGER NOT NULL DEFAULT 0, progress_total INTEGER NOT NULL DEFAULT 0,
        progress_message TEXT, error TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0, parent_id TEXT
    );
    CREATE TRIGGER experiments_config_immutable BEFORE UPDATE OF config_json, config_hash ON experiments
    BEGIN SELECT RAISE(ABORT, 'experiment configuration is immutable'); END;
    CREATE TABLE metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
        variant TEXT NOT NULL, protocol TEXT NOT NULL, model TEXT NOT NULL, condition_name TEXT NOT NULL,
        attacker TEXT NOT NULL, seed INTEGER NOT NULL, split TEXT NOT NULL, name TEXT NOT NULL, value REAL NOT NULL
    );
    CREATE INDEX idx_metrics_exp ON metrics(experiment_id);
    CREATE TABLE artifacts (
        experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
        name TEXT NOT NULL, kind TEXT NOT NULL, relpath TEXT NOT NULL, sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL, created_at TEXT NOT NULL, extra_json TEXT,
        PRIMARY KEY (experiment_id, name)
    );
    """,
    # v2: execution event log + query indexes
    """
    CREATE TABLE events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
        ts TEXT NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL
    );
    CREATE INDEX idx_events_exp ON events(experiment_id);
    CREATE INDEX idx_experiments_status ON experiments(status, created_at);
    CREATE INDEX idx_metrics_lookup ON metrics(experiment_id, model, protocol, condition_name, name);
    """,
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_revision() -> tuple[str, bool]:
    try:
        root = Path(__file__).resolve().parents[3]
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5
        )
        if rev.returncode != 0:
            raise OSError
        dirty = (
            subprocess.run(
                ["git", "status", "--porcelain", "--", "src"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            != ""
        )
        return rev.stdout.strip(), dirty
    except (OSError, subprocess.SubprocessError):
        from .. import __version__

        return f"pkg-{__version__}", False  # installed (non-git) package: record the release


def environment_metadata() -> dict[str, Any]:
    import scipy
    import sklearn
    import torch

    from .. import __version__

    return {
        "openkeystroke": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_count": os.cpu_count(),
    }


class Workspace:
    """Directory layout: registry.sqlite, artifacts/, datasets/, cache/."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.environ.get("OPENKEYSTROKE_HOME") or ".openkeystroke").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.root / "artifacts"
        self.datasets_dir = self.root / "datasets"
        self.cache_dir = self.root / "cache"
        for d in (self.artifacts_dir, self.datasets_dir, self.cache_dir):
            d.mkdir(exist_ok=True)
        self.store = ExperimentStore(self.root / "registry.sqlite", self.artifacts_dir)


class ExperimentStore:
    def __init__(self, db_path: Path, artifacts_dir: Path) -> None:
        self.db_path = db_path
        self.artifacts_dir = artifacts_dir
        self._migrate()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA foreign_keys=ON")
            con.execute("PRAGMA busy_timeout=30000")
            yield con
        finally:
            con.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._conn() as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                yield con
            except BaseException:
                con.execute("ROLLBACK")
                raise
            else:
                con.execute("COMMIT")

    def _migrate(self) -> None:
        with self._conn() as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                ver = con.execute("PRAGMA user_version").fetchone()[0]
                if ver > len(MIGRATIONS):
                    raise RegistryError(
                        f"registry schema v{ver} is newer than this code (v{len(MIGRATIONS)})"
                    )
                for i in range(ver, len(MIGRATIONS)):
                    for stmt in _split_sql(MIGRATIONS[i]):
                        con.execute(stmt)
                    con.execute(f"PRAGMA user_version={i + 1}")
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise

    def schema_version(self) -> int:
        with self._conn() as con:
            return int(con.execute("PRAGMA user_version").fetchone()[0])

    # ------------------------------------------------------------------ lifecycle
    def create(
        self, cfg: ExperimentConfig, *, parent_id: str | None = None, experiment_id: str | None = None
    ) -> str:
        eid = experiment_id or f"exp_{datetime.now(timezone.utc):%Y%m%d-%H%M%S}_{uuid.uuid4().hex[:8]}"
        rev, dirty = code_revision()
        try:
            with self._tx() as con:
                con.execute(
                    "INSERT INTO experiments (id,name,kind,status,config_json,config_hash,code_revision,code_dirty,environment_json,created_at,parent_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        eid,
                        cfg.name,
                        cfg.kind,
                        "CREATED",
                        cfg.snapshot(),
                        cfg.hash(),
                        rev,
                        int(dirty),
                        json.dumps(environment_metadata()),
                        utcnow(),
                        parent_id,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise RegistryError(f"experiment id already exists: {eid}") from exc
        self.log(eid, "info", "created")
        return eid

    def transition(self, eid: str, new: str, *, error: str | None = None, **fields: Any) -> None:
        """Atomic compare-and-set status change; illegal transitions raise LifecycleError."""
        if new not in STATUSES:
            raise LifecycleError(f"unknown status {new}")
        with self._tx() as con:
            row = con.execute("SELECT status FROM experiments WHERE id=?", (eid,)).fetchone()
            if row is None:
                raise NotFoundError(f"experiment not found: {eid}")
            if new not in TRANSITIONS[row["status"]]:
                raise LifecycleError(f"illegal transition {row['status']} -> {new} for {eid}")
            sets: dict[str, Any] = {"status": new}
            now = utcnow()
            if new == "VALIDATED":
                sets["validated_at"] = now
            elif new == "RUNNING":
                sets.update(started_at=now, pid=os.getpid())
            elif new in ("COMPLETED", "FAILED", "CANCELLED"):
                sets.update(finished_at=now, pid=None)
            if error is not None:
                sets["error"] = error[:20000]
            sets.update(fields)
            cols = ", ".join(f"{k}=?" for k in sets)
            con.execute(
                f"UPDATE experiments SET {cols} WHERE id=? AND status=?", (*sets.values(), eid, row["status"])
            )
        self.log(
            eid, "error" if new == "FAILED" else "info", f"-> {new}" + (f": {error[:500]}" if error else "")
        )

    def update_progress(self, eid: str, done: int, total: int, message: str = "") -> None:
        with self._tx() as con:
            con.execute(
                "UPDATE experiments SET progress_done=?, progress_total=?, progress_message=? WHERE id=?",
                (done, total, message[:300], eid),
            )

    def set_dataset(
        self, eid: str, dataset_id: str, version: str, fingerprint: str, is_synthetic: bool
    ) -> None:
        with self._tx() as con:
            con.execute(
                "UPDATE experiments SET dataset_id=?, dataset_version=?, dataset_fingerprint=?, dataset_is_synthetic=? WHERE id=?",
                (dataset_id, version, fingerprint, int(is_synthetic), eid),
            )

    def request_cancel(self, eid: str) -> str:
        with self._tx() as con:
            row = con.execute("SELECT status FROM experiments WHERE id=?", (eid,)).fetchone()
            if row is None:
                raise NotFoundError(f"experiment not found: {eid}")
            st = row["status"]
            if st in ("CREATED", "VALIDATED"):
                con.execute(
                    "UPDATE experiments SET status='CANCELLED', finished_at=? WHERE id=?", (utcnow(), eid)
                )
                return "CANCELLED"
            if st == "RUNNING":
                con.execute("UPDATE experiments SET cancel_requested=1 WHERE id=?", (eid,))
                return "CANCEL_REQUESTED"
            raise LifecycleError(f"cannot cancel experiment in state {st}")

    def is_cancel_requested(self, eid: str) -> bool:
        with self._conn() as con:
            row = con.execute("SELECT cancel_requested FROM experiments WHERE id=?", (eid,)).fetchone()
        return bool(row and row[0])

    def reap_stale(self) -> list[str]:
        """Mark RUNNING experiments whose process no longer exists as FAILED."""
        reaped = []
        for r in self.list_experiments(status="RUNNING"):
            pid = r.get("pid")
            if pid and not _pid_alive(int(pid)):
                self.transition(
                    r["id"], "FAILED", error="worker process died before completing the experiment"
                )
                reaped.append(r["id"])
        return reaped

    # ------------------------------------------------------------------ queries
    def get(self, eid: str) -> dict[str, Any]:
        with self._conn() as con:
            row = con.execute("SELECT * FROM experiments WHERE id=?", (eid,)).fetchone()
        if row is None:
            raise NotFoundError(f"experiment not found: {eid}")
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json"))
        d["environment"] = json.loads(d.pop("environment_json") or "{}")
        return d

    def get_config(self, eid: str) -> ExperimentConfig:
        return ExperimentConfig.model_validate(self.get(eid)["config"])

    def list_experiments(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        dataset_id: str | None = None,
        model: str | None = None,
        search: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM experiments WHERE 1=1"
        args: list[Any] = []
        if status:
            q += " AND status=?"
            args.append(status)
        if kind:
            q += " AND kind=?"
            args.append(kind)
        if dataset_id:
            q += " AND dataset_id=?"
            args.append(dataset_id)
        if search:
            q += " AND (name LIKE ? OR id LIKE ?)"
            args += [f"%{search}%", f"%{search}%"]
        if model:
            q += " AND config_json LIKE ?"
            args.append(f'%"name":"{model}"%')
        q += " ORDER BY created_at DESC, id DESC LIMIT ?"
        args.append(limit)
        with self._conn() as con:
            rows = con.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d.pop("config_json", None)
            d.pop("environment_json", None)
            out.append(d)
        return out

    def counts(self) -> dict[str, int]:
        with self._conn() as con:
            rows = con.execute("SELECT status, COUNT(*) c FROM experiments GROUP BY status").fetchall()
        return {r["status"]: r["c"] for r in rows}

    # ------------------------------------------------------------------ events
    def log(self, eid: str, level: str, message: str) -> None:
        with self._tx() as con:
            con.execute(
                "INSERT INTO events (experiment_id, ts, level, message) VALUES (?,?,?,?)",
                (eid, utcnow(), level, message[:5000]),
            )

    def events(self, eid: str) -> list[dict[str, Any]]:
        with self._conn() as con:
            return [
                dict(r)
                for r in con.execute(
                    "SELECT ts, level, message FROM events WHERE experiment_id=? ORDER BY id", (eid,)
                )
            ]

    # ------------------------------------------------------------------ metrics
    def add_metrics(self, eid: str, rows: list[dict[str, Any]]) -> None:
        with self._tx() as con:
            con.executemany(
                "INSERT INTO metrics (experiment_id,variant,protocol,model,condition_name,attacker,seed,split,name,value) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        eid,
                        r["variant"],
                        r["protocol"],
                        r["model"],
                        r["condition"],
                        r["attacker"],
                        int(r["seed"]),
                        r["split"],
                        r["name"],
                        float(r["value"]),
                    )
                    for r in rows
                ],
            )

    def metrics(self, eid: str, *, name: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
        q, args = (
            "SELECT variant,protocol,model,condition_name AS condition,attacker,seed,split,name,value FROM metrics WHERE experiment_id=?",
            [eid],
        )
        if name:
            q += " AND name=?"
            args.append(name)
        if split:
            q += " AND split=?"
            args.append(split)
        with self._conn() as con:
            return [dict(r) for r in con.execute(q, args)]

    # ------------------------------------------------------------------ artifacts
    def _artifact_path(self, eid: str, name: str) -> Path:
        if not name or name.startswith("/") or ".." in Path(name).parts:
            raise RegistryError(f"invalid artifact name: {name!r}")
        return self.artifacts_dir / eid / name

    def put_bytes(
        self, eid: str, name: str, data: bytes, kind: str = "file", extra: dict[str, Any] | None = None
    ) -> Path:
        """Atomic write (tmp file + fsync + rename), then register with sha256."""
        p = self._artifact_path(eid, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".partial")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        self._register(eid, name, kind, p, hashlib.sha256(data).hexdigest(), len(data), extra)
        return p

    def put_json(self, eid: str, name: str, obj: Any, kind: str = "json") -> Path:
        return self.put_bytes(eid, name, json.dumps(obj, indent=1, default=_json_default).encode(), kind)

    def put_npz(self, eid: str, name: str, **arrays: np.ndarray) -> Path:
        buf = io.BytesIO()
        payload: dict[str, Any] = dict(arrays)
        np.savez_compressed(buf, **payload)
        return self.put_bytes(eid, name, buf.getvalue(), "npz")

    def register_dir(self, eid: str, name: str, kind: str, hashes: dict[str, str]) -> None:
        p = self._artifact_path(eid, name)
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        self._register(
            eid,
            name,
            kind,
            p,
            hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            size,
            {"file_hashes": hashes},
        )

    def artifact_dir(self, eid: str, name: str) -> Path:
        p = self._artifact_path(eid, name)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _register(
        self, eid: str, name: str, kind: str, path: Path, sha: str, size: int, extra: dict[str, Any] | None
    ) -> None:
        rel = str(path.relative_to(self.artifacts_dir))
        with self._tx() as con:
            con.execute(
                "INSERT OR REPLACE INTO artifacts (experiment_id,name,kind,relpath,sha256,size_bytes,created_at,extra_json) VALUES (?,?,?,?,?,?,?,?)",
                (eid, name, kind, rel, sha, size, utcnow(), json.dumps(extra or {})),
            )

    def list_artifacts(self, eid: str) -> list[dict[str, Any]]:
        with self._conn() as con:
            rows = [
                dict(r)
                for r in con.execute("SELECT * FROM artifacts WHERE experiment_id=? ORDER BY name", (eid,))
            ]
        for r in rows:
            r["extra"] = json.loads(r.pop("extra_json") or "{}")
        return rows

    def artifact_path(self, eid: str, name: str) -> Path:
        with self._conn() as con:
            row = con.execute(
                "SELECT relpath FROM artifacts WHERE experiment_id=? AND name=?", (eid, name)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"artifact '{name}' not found for {eid}")
        return self.artifacts_dir / row["relpath"]

    def get_json(self, eid: str, name: str) -> Any:
        p = self.artifact_path(eid, name)
        if not p.is_file():
            raise NotFoundError(f"artifact file missing on disk: {p}")
        return json.loads(p.read_text())

    def has_artifact(self, eid: str, name: str) -> bool:
        try:
            return self.artifact_path(eid, name).exists()
        except NotFoundError:
            return False

    def load_npz(self, eid: str, name: str) -> dict[str, np.ndarray]:
        """Lazy-friendly loader: returns an NpzFile-backed dict materialised on access."""
        with np.load(self.artifact_path(eid, name), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def verify_artifacts(self, eid: str) -> list[str]:
        """Return names of artifacts whose file is missing or whose sha256 no longer matches."""
        bad = []
        for a in self.list_artifacts(eid):
            p = self.artifacts_dir / a["relpath"]
            if a["kind"].endswith("_dir"):
                ok = p.is_dir()
            elif not p.is_file():
                ok = False
            else:
                ok = hashlib.sha256(p.read_bytes()).hexdigest() == a["sha256"]
            if not ok:
                bad.append(a["name"])
        return bad

    # ------------------------------------------------------------------ export
    def export_metrics(self, eids: list[str], path: Path, fmt: str = "csv") -> Path:
        rows = []
        for e in eids:
            for m in self.metrics(e):
                rows.append({"experiment_id": e, **m})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            path.write_text(json.dumps(rows, indent=1))
        elif fmt == "csv":
            cols = [
                "experiment_id",
                "variant",
                "protocol",
                "model",
                "condition",
                "attacker",
                "seed",
                "split",
                "name",
                "value",
            ]
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
        else:
            raise RegistryError(f"unsupported export format '{fmt}'")
        return path

    def export_experiment(self, eid: str, out_dir: Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rec = self.get(eid)
        (out / "experiment.json").write_text(json.dumps(rec, indent=1, default=_json_default))
        self.export_metrics([eid], out / "metrics.csv", "csv")
        self.export_metrics([eid], out / "metrics.json", "json")
        for a in self.list_artifacts(eid):
            p = self.artifacts_dir / a["relpath"]
            if p.is_file() and a["kind"] in ("json", "png", "csv"):
                (out / "artifacts").mkdir(exist_ok=True)
                (out / "artifacts" / Path(a["name"]).name).write_bytes(p.read_bytes())
        return out


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _split_sql(script: str) -> list[str]:
    """Split a migration script into statements (triggers contain ';' inside BEGIN..END)."""
    stmts, buf, in_trigger = [], [], False
    for line in script.splitlines():
        s = line.strip()
        if not s:
            continue
        buf.append(line)
        if s.upper().startswith("CREATE TRIGGER"):
            in_trigger = True
        if (in_trigger and s.upper().endswith("END;")) or (not in_trigger and s.endswith(";")):
            stmts.append("\n".join(buf))
            buf, in_trigger = [], False
    if buf:
        stmts.append("\n".join(buf))
    return stmts


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
