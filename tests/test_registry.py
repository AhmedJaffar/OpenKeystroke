import json
import sqlite3
import subprocess
import sys

import pytest

from conftest import make_cfg
from openkeystroke.errors import LifecycleError, NotFoundError, RegistryError
from openkeystroke.registry import Workspace
from openkeystroke.registry.store import MIGRATIONS


def test_lifecycle_valid_and_illegal_transitions(ws):
    s = ws.store
    e = s.create(make_cfg())
    assert s.get(e)["status"] == "CREATED"
    for bad in ("RUNNING", "COMPLETED"):
        with pytest.raises(LifecycleError):
            s.transition(e, bad)
    s.transition(e, "VALIDATED")
    s.transition(e, "RUNNING")
    assert s.get(e)["pid"] and s.get(e)["started_at"]
    s.transition(e, "COMPLETED")
    assert s.get(e)["finished_at"] and s.get(e)["pid"] is None
    for new in ("RUNNING", "FAILED", "CREATED"):
        with pytest.raises(LifecycleError):  # terminal states are final
            s.transition(e, new)
    with pytest.raises(NotFoundError):
        s.transition("nope", "VALIDATED")


def test_duplicate_ids_and_config_immutability(ws):
    s = ws.store
    cfg = make_cfg()
    e = s.create(cfg, experiment_id="fixed-id")
    with pytest.raises(RegistryError, match="already exists"):
        s.create(cfg, experiment_id="fixed-id")
    with pytest.raises(sqlite3.DatabaseError, match="immutable"), s._tx() as con:
        con.execute("UPDATE experiments SET config_json='{}' WHERE id=?", (e,))
    assert s.get_config(e) == cfg and s.get(e)["config_hash"] == cfg.hash()


def test_provenance_recorded(ws):
    r = ws.store.get(ws.store.create(make_cfg()))
    assert r["code_revision"] and "python" in r["environment"] and "torch" in r["environment"]


def test_persistence_across_reopen_and_processes(tmp_path):
    w1 = Workspace(tmp_path / "h")
    e = w1.store.create(make_cfg())
    w1.store.put_json(e, "x/y.json", {"a": 1})
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from openkeystroke.registry import Workspace; w=Workspace(r'{tmp_path / 'h'}'); print(w.store.get('{e}')['status'], w.store.get_json('{e}','x/y.json'))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "CREATED {'a': 1}" in out.stdout
    assert Workspace(tmp_path / "h").store.get(e)["name"] == make_cfg().name


def test_migrations_versioned_and_idempotent(tmp_path):
    w = Workspace(tmp_path / "h")
    assert w.store.schema_version() == len(MIGRATIONS)
    Workspace(tmp_path / "h")  # reopening does not re-run migrations
    con = sqlite3.connect(w.store.db_path)
    con.execute(f"PRAGMA user_version={len(MIGRATIONS) + 5}")
    con.close()
    with pytest.raises(RegistryError, match="newer"):
        Workspace(tmp_path / "h")


def test_migrate_from_v1_database(tmp_path):
    db = tmp_path / "h" / "registry.sqlite"
    (tmp_path / "h").mkdir()
    con = sqlite3.connect(db)
    con.executescript(MIGRATIONS[0])
    con.execute("PRAGMA user_version=1")
    con.close()
    w = Workspace(tmp_path / "h")
    assert w.store.schema_version() == 2
    e = w.store.create(make_cfg())
    assert w.store.events(e)  # v2 table exists


def test_artifacts_atomic_checksum_and_corruption_detection(ws):
    s = ws.store
    e = s.create(make_cfg())
    p = s.put_json(e, "r.json", {"v": 1})
    assert not list(p.parent.glob("*.partial")) and s.verify_artifacts(e) == []
    p.write_text('{"v": 2}')
    assert s.verify_artifacts(e) == ["r.json"]
    (p.parent / "r.json").unlink()
    assert s.verify_artifacts(e) == ["r.json"]
    with pytest.raises(RegistryError):
        s.put_bytes(e, "../escape", b"x")
    with pytest.raises(NotFoundError):
        s.artifact_path(e, "missing")


def test_failed_write_leaves_no_partial_artifact(ws, monkeypatch):
    import os

    s = ws.store
    e = s.create(make_cfg())
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        s.put_bytes(e, "a.bin", b"data")
    monkeypatch.undo()
    assert not list((ws.artifacts_dir / e).rglob("*")) or all(
        f.is_dir() for f in (ws.artifacts_dir / e).rglob("*")
    )
    assert s.list_artifacts(e) == []


def test_metrics_filter_export_csv_json(ws, tmp_path):
    s = ws.store
    cfg = make_cfg()
    a, b = s.create(cfg), s.create(make_cfg(models=None, seeds=[1]).model_copy(update={"name": "other"}))
    row = {
        "variant": "base",
        "protocol": "session_disjoint",
        "model": "logreg",
        "condition": "clean",
        "attacker": "static",
        "seed": 0,
        "split": "test",
        "name": "accuracy",
        "value": 0.75,
    }
    s.add_metrics(a, [row, {**row, "name": "f1_macro", "value": 0.7}])
    assert len(s.metrics(a)) == 2 and len(s.metrics(a, name="accuracy")) == 1 and s.metrics(b) == []
    csv_p = s.export_metrics([a], tmp_path / "m.csv", "csv")
    assert (
        csv_p.read_text().splitlines()[0].startswith("experiment_id,variant")
        and len(csv_p.read_text().splitlines()) == 3
    )
    assert json.loads(s.export_metrics([a], tmp_path / "m.json", "json").read_text())[0]["value"] == 0.75
    with pytest.raises(RegistryError):
        s.export_metrics([a], tmp_path / "m.x", "xml")
    assert (
        [r["id"] for r in s.list_experiments(search="other")] == [b]
        and len(s.list_experiments(model="logreg")) == 2
        and s.list_experiments(status="RUNNING") == []
    )


def test_cancel_semantics_and_stale_reaping(ws):
    s = ws.store
    e = s.create(make_cfg())
    assert s.request_cancel(e) == "CANCELLED" and s.get(e)["status"] == "CANCELLED"
    with pytest.raises(LifecycleError):
        s.request_cancel(e)
    e2 = s.create(make_cfg())
    s.transition(e2, "VALIDATED")
    s.transition(e2, "RUNNING", pid=999_999_999)  # no such process
    assert s.request_cancel(e2) == "CANCEL_REQUESTED" and s.is_cancel_requested(e2)
    assert s.reap_stale() == [e2] and s.get(e2)["status"] == "FAILED" and "died" in s.get(e2)["error"]
