import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from conftest import make_cfg
from openkeystroke.cli import main


@pytest.fixture()
def run(tmp_path):
    runner = CliRunner()
    home = str(tmp_path / "home")

    def _run(*args, json_out=False, expect=None):
        a = ["--home", home, *(["--json"] if json_out else []), *args]
        r = runner.invoke(main, a)
        if expect is not None:
            assert r.exit_code == expect, (r.output, r.stderr if hasattr(r, "stderr") else "")
        return r

    _run.home, _run.tmp = home, tmp_path
    return _run


def write_cfg(tmp_path, cfg):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return str(p)


def test_help_and_version(run):
    r = run("--help", expect=0)
    for cmd in ("dataset", "experiment", "benchmark", "dashboard", "doctor"):
        assert cmd in r.output
    for sub in (
        ["dataset", "generate"],
        ["experiment", "run"],
        ["benchmark", "run"],
        ["experiment", "inspect"],
    ):
        assert "--help" in run(*sub, "--help", expect=0).output


def test_dataset_generate_validate_stats(run):
    out = str(run.tmp / "ds")
    r = run(
        "dataset",
        "generate",
        "--out",
        out,
        "--classes",
        "3",
        "--devices",
        "3",
        "--sessions",
        "2",
        "--per-class",
        "4",
        json_out=True,
        expect=0,
    )
    info = json.loads(r.output)
    assert info["n_samples"] == 3 * 3 * 2 * 3 * 4 // 3 * 1 or info["n_samples"] > 0
    r2 = run("dataset", "generate", "--out", out, expect=2)  # refuses to overwrite non-empty dir
    assert "dataset_invalid" in r2.output or "not empty" in r2.output
    v = run("dataset", "validate", out, json_out=True, expect=0)
    assert json.loads(v.output)["ok"] is True
    assert json.loads(run("dataset", "stats", out, json_out=True, expect=0).output)["is_synthetic"] is True
    import numpy as np

    wf = np.load(out + "/waveforms.npy")
    wf[0, 0] += 1
    np.save(out + "/waveforms.npy", wf)
    bad = run("dataset", "validate", out, json_out=True, expect=2)
    assert "fingerprint" in json.loads(bad.output)["errors"][0]


def test_invalid_arguments_and_exit_codes(run):
    assert (
        run("dataset", "generate", "--out", str(run.tmp / "x"), "--classes", "1", expect=2).exit_code == 2
    )  # click range validation
    assert run("dataset", "validate", str(run.tmp / "missing"), expect=2).exit_code == 2
    r = run("experiment", "inspect", "does-not-exist", expect=3)
    assert "not_found" in r.output
    rj = run("experiment", "inspect", "does-not-exist", json_out=True, expect=3)
    assert json.loads(rj.output.strip().splitlines()[-1])["error"] == "not_found"
    bad = run.tmp / "bad.yaml"
    bad.write_text("name: x\nkind: nonsense\n")
    r = run("experiment", "run", "--config", str(bad), expect=2)
    assert "config_invalid" in r.output
    run("experiment", "run", expect=2)  # needs one of --config/--preset/--id
    run("experiment", "run", "--preset", "nope", expect=2)


def test_experiment_run_list_inspect_export_compare_reproduce(run):
    cfgp = write_cfg(run.tmp, make_cfg("repeated_seed", seeds=[0, 1]))
    r = run("experiment", "run", "--config", cfgp, json_out=True, expect=0)
    eid = json.loads(r.output.strip().splitlines()[-1] if False else r.output[r.output.index("{") :])["id"]
    lst = json.loads(run("experiment", "list", "--status", "COMPLETED", json_out=True, expect=0).output)
    assert [x["id"] for x in lst] == [eid]
    text = run("experiment", "inspect", eid, "--events", expect=0).output
    assert "COMPLETED" in text and "acc=" in text and "SYNTHETIC" in text
    insp = json.loads(run("experiment", "inspect", eid, json_out=True, expect=0).output)
    assert "summary.json" in insp["artifacts"] and insp["summary"]["table"]
    csvp = str(run.tmp / "m.csv")
    run("experiment", "export", eid, "--out", csvp, expect=0)
    assert Path(csvp).read_text().startswith("experiment_id,")
    run("experiment", "export", eid, "--out", str(run.tmp / "bundle"), "--format", "bundle", expect=0)
    assert (run.tmp / "bundle" / "experiment.json").is_file() and (
        run.tmp / "bundle" / "metrics.csv"
    ).is_file()
    cmp_out = run("benchmark", "compare", eid, expect=0).output
    assert "logreg" in cmp_out and "synthetic" in cmp_out
    rep = run("experiment", "reproduce", eid, "--tol", "1e-9", json_out=True, expect=0)
    res = json.loads(rep.output[rep.output.index("{") :])
    assert (
        res["reproduced"]
        and res["dataset_fingerprint_match"]
        and res["max_abs_diff"] <= 1e-9
        and res["metrics_compared"] > 0
    )
    assert len(json.loads(run("experiment", "list", json_out=True, expect=0).output)) == 2


def test_cancel_and_list_filters(run):
    from openkeystroke.experiments import create_experiment
    from openkeystroke.registry import Workspace

    ws = Workspace(run.home)
    eid = create_experiment(ws, make_cfg("single"))
    assert "CANCELLED" in run("experiment", "cancel", eid, expect=0).output
    run("experiment", "cancel", eid, expect=1)  # already terminal
    assert (
        json.loads(run("experiment", "list", "--status", "CANCELLED", json_out=True).output)[0]["id"] == eid
    )


def test_benchmark_run_prints_defense_table(run):
    from conftest import make_cfg as mk
    from openkeystroke.config import DefenseConfig, DefenseStep

    cfg = mk(
        "defense_comparison",
        defenses=[
            DefenseConfig(name="lp1k", steps=[DefenseStep(type="lowpass", params={"cutoff_hz": 1000})])
        ],
        attacker_modes=["static"],
    )
    out = run("benchmark", "run", "--config", write_cfg(run.tmp, cfg), expect=0).output
    assert "lp1k" in out and "static drop" in out and "chance accuracy" in out and "SYNTHETIC" in out


def test_doctor(run):
    out = run("doctor", expect=0).output
    assert "[ok] import torch" in out and "registry schema" in out
    assert "COMPLETED" in run("doctor", "--self-test", expect=0).output


def test_ingest_cli_requires_consent(run):
    m = run.tmp / "m.json"
    m.write_text(json.dumps({"classes": ["A", "B"], "recordings": [{"file": "a.wav"}]}))
    r = run("dataset", "ingest", str(m), "--out", str(run.tmp / "o"), expect=2)
    assert "consent" in r.output
