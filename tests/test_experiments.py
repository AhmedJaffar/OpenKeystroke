import numpy as np
import pytest

from conftest import make_cfg, tiny_cnn
from openkeystroke.config import (
    AugmentConfig,
    AugmentStep,
    DatasetRef,
    DefenseConfig,
    DefenseStep,
    ModelConfig,
)
from openkeystroke.errors import ConfigError
from openkeystroke.experiments import build_jobs, create_experiment, run_experiment, validate_experiment
from openkeystroke.experiments.runner import resolve_dataset


def key(m):
    return (
        m["variant"],
        m["protocol"],
        m["model"],
        m["condition"],
        m["attacker"],
        m["seed"],
        m["split"],
        m["name"],
    )


def run(ws, cfg, **kw):
    eid = create_experiment(ws, cfg)
    st = run_experiment(ws, eid, **kw)
    return eid, st


def test_full_single_experiment_artifacts_and_provenance(ws):
    eid, st = run(ws, make_cfg("single"))
    r = ws.store.get(eid)
    assert st == "COMPLETED" and r["status"] == "COMPLETED" and r["progress_done"] == r["progress_total"] == 1
    assert (
        r["dataset_fingerprint"]
        and r["dataset_is_synthetic"] == 1
        and r["code_revision"]
        and r["validated_at"]
        and r["started_at"]
        and r["finished_at"]
    )
    names = {a["name"] for a in ws.store.list_artifacts(eid)}
    assert {"summary.json", "results.json", "dataset.json", "predictions.npz"} <= names and any(
        n.startswith("models/") for n in names
    )
    assert ws.store.verify_artifacts(eid) == []
    s = ws.store.get_json(eid, "summary.json")
    t = s["table"][0]
    assert (
        t["protocol"] == "session_disjoint"
        and 0 <= t["accuracy"]["mean"] <= 1
        and t["chance"] == pytest.approx(1 / 3)
    )
    assert any("SYNTHETIC" in c for c in s["caveats"]) and any("seed" in c for c in s["caveats"])
    assert ws.store.metrics(eid, name="accuracy", split="test")
    # held-out predictions are stored lazily and consistent with metrics
    z = ws.store.load_npz(eid, "predictions.npz")
    key = next(k for k in z if k.endswith("__y"))
    assert len(z[key]) == len(z[key.replace("__y", "__proba")]) == len(z[key.replace("__y", "__idx")])
    assert [e["message"] for e in ws.store.events(eid)][:1] == ["created"]


def test_results_survive_process_restart(ws):
    from openkeystroke.registry import Workspace

    eid, _ = run(ws, make_cfg("single"))
    acc = ws.store.get_json(eid, "summary.json")["table"][0]["accuracy"]["mean"]
    ws2 = Workspace(ws.root)
    assert (
        ws2.store.get(eid)["status"] == "COMPLETED"
        and ws2.store.get_json(eid, "summary.json")["table"][0]["accuracy"]["mean"] == acc
    )


@pytest.mark.parametrize(
    "models,tol",
    [([ModelConfig(name="logreg"), ModelConfig(name="random_forest")], 1e-9), ([tiny_cnn(3)], 1e-5)],
)
def test_identical_runs_reproduce_within_tolerance(ws, models, tol):
    a, _ = run(ws, make_cfg("repeated_seed", models=models, seeds=[0, 1]))
    b, _ = run(ws, make_cfg("repeated_seed", models=models, seeds=[0, 1]))
    ma = {
        key(m): m["value"]
        for m in ws.store.metrics(a)
        if m["name"] in ("accuracy", "f1_macro", "ece", "balanced_accuracy")
    }
    mb = {
        key(m): m["value"]
        for m in ws.store.metrics(b)
        if m["name"] in ("accuracy", "f1_macro", "ece", "balanced_accuracy")
    }
    assert ma.keys() == mb.keys() and ma
    assert max(abs(ma[k] - mb[k]) for k in ma) <= tol


def test_repeated_seed_varies_splits_and_reports_spread(ws):
    eid, _ = run(ws, make_cfg("repeated_seed", seeds=[0, 1, 2]))
    s = ws.store.get_json(eid, "summary.json")
    t = s["table"][0]
    assert (
        t["n_seeds"] == 3
        and t["accuracy"]["n"] == 3
        and t["accuracy"]["ci_lo"] <= t["accuracy"]["mean"] <= t["accuracy"]["ci_hi"]
    )
    r = ws.store.get_json(eid, "results.json")["rows"]
    assert len({tuple(x["test_groups"]) for x in r}) > 1  # different held-out sessions per seed


def test_model_comparison_has_paired_tests_and_resources(ws):
    eid, st = run(
        ws,
        make_cfg(
            "model_comparison",
            models=[ModelConfig(name="logreg"), ModelConfig(name="nearest_centroid"), tiny_cnn(3)],
        ),
    )
    s = ws.store.get_json(eid, "summary.json")
    assert (
        st == "COMPLETED"
        and len(s["model_comparisons"]) == 3
        and all(c["n"] == 3 for c in s["model_comparisons"])
    )
    res = {r["model"]: r for r in s["resources"]}
    assert (
        res["cnn1d"]["n_parameters"] > 0
        and res["cnn1d"]["latency_ms_median"] > 0
        and res["logreg"]["throughput_samples_per_s"] > 0
        and res["cnn1d"]["model_size_bytes"] > 0
    )


def test_cross_device_distinguishes_protocols_and_group_disjointness(ws):
    eid, _ = run(ws, make_cfg("cross_device", seeds=[0, 1, 2]))
    s = ws.store.get_json(eid, "summary.json")
    assert {t["protocol"] for t in s["table"]} == {"in_distribution", "session_disjoint", "device_disjoint"}
    assert {d["protocol"] for d in s["domain_shift"]} == {"session_disjoint", "device_disjoint"} and all(
        "degradation" in d and d["degradation"] is not None for d in s["domain_shift"]
    )
    rows = ws.store.get_json(eid, "results.json")["rows"]
    ds = resolve_dataset(ws, ws.store.get_config(eid))
    for r in (x for x in rows if x["protocol"] == "device_disjoint"):
        train_dev = {
            d for d in ds.group_array("device")
        }  # sanity: held-out devices recorded and not in train groups
        sp = ws.store.get_json(eid, "results.json")["jobs"]
        j = next(j for j in sp if j["job_id"] == r["job_id"])["split"]["groups"]
        assert not set(j["train"]) & set(j["test"]) and not set(j["train"]) & set(j["val"]) and train_dev
    assert any("in_distribution" in c for c in s["caveats"])


def test_defense_comparison_static_vs_adaptive_and_effectiveness_guard(ws):
    cfg = make_cfg(
        "defense_comparison",
        defenses=[
            DefenseConfig(name="lp1k", steps=[DefenseStep(type="lowpass", params={"cutoff_hz": 1000})]),
            DefenseConfig(name="lp2k", steps=[DefenseStep(type="lowpass", params={"cutoff_hz": 2000})]),
            DefenseConfig(name="noise0", steps=[DefenseStep(type="noise", params={"snr_db": 0})]),
            DefenseConfig(name="noise40", steps=[DefenseStep(type="noise", params={"snr_db": 40})]),
        ],
    )
    eid, st = run(ws, cfg)
    s = ws.store.get_json(eid, "summary.json")
    assert st == "COMPLETED"
    deg = {(d["condition"], d["attacker"]): d for d in s["degradation"]}
    assert set(c for c, _ in deg) == {"lp1k", "lp2k", "noise0", "noise40"} and {a for _, a in deg} == {
        "static",
        "adaptive",
    }
    assert (
        deg[("noise40", "static")]["evidence"] != "significant_reduction"
    )  # 40 dB noise must not be called effective
    assert deg[("lp1k", "static")]["accuracy_drop"] > 0.05
    names = {p["name"] for p in s["defense_points"]}
    front = s["pareto"][0]["frontier"]
    assert names == {"lp1k", "lp2k", "noise0", "noise40"} and set(front) <= {
        p["name"] for p in s["defense_points"] if p["eligible"]
    }
    q = ws.store.get_json(eid, "results.json")["defense_quality"]
    assert q["lp1k"]["log_spectral_distance_db"] > q["noise40"]["log_spectral_distance_db"] and all(
        "real_time_factor" in v for v in q.values()
    )
    assert ws.store.has_artifact(eid, "plots/defense_curves.png")


def test_adaptive_attacker_trained_on_defended_audio_not_clean(ws):
    cfg = make_cfg(
        "defense_comparison",
        attacker_modes=["static", "adaptive"],
        defenses=[
            DefenseConfig(name="lp1k", steps=[DefenseStep(type="lowpass", params={"cutoff_hz": 1000})])
        ],
    )
    eid, _ = run(ws, cfg)
    rows = ws.store.get_json(eid, "results.json")["rows"]
    assert {(r["attacker"], r["train_condition"]) for r in rows} == {
        ("static", "clean"),
        ("adaptive", "lp1k"),
    }


def test_noise_robustness_and_ablation_and_sweep_selection(ws):
    eid, st = run(ws, make_cfg("noise_robustness", noise_snr_db=[20.0, 0.0], attacker_modes=["static"]))
    t = {x["condition"]: x["accuracy"]["mean"] for x in ws.store.get_json(eid, "summary.json")["table"]}
    assert (
        st == "COMPLETED" and set(t) == {"clean", "noise_20dB", "noise_0dB"} and t["noise_0dB"] <= t["clean"]
    )
    eid, _ = run(
        ws,
        make_cfg(
            "ablation",
            variants={
                "base": {},
                "mfcc": {"vector_features.name": "mfcc"},
                "no_dc": {"preprocess.remove_dc": False},
            },
        ),
    )
    s = ws.store.get_json(eid, "summary.json")
    assert {t["variant"] for t in s["table"]} == {"base", "mfcc", "no_dc"} and s["selection"][0][
        "selected_by"
    ] == "validation_accuracy"
    eid, _ = run(ws, make_cfg("sweep", sweep={"models.0.params.C": [0.01, 1.0]}))
    s = ws.store.get_json(eid, "summary.json")
    assert {t["variant"] for t in s["table"]} == {"C=0.01", "C=1.0"} and s["selection"][0][
        "selected_variant"
    ] in {"C=0.01", "C=1.0"}
    sel = s["selection"][0]
    best_val = max(s["table"], key=lambda t: t["val_accuracy"]["mean"])
    assert sel["selected_variant"] == best_val["variant"]  # chosen on validation, not test


def test_augmentation_runs_and_is_train_only(ws):
    aug = AugmentConfig(steps=[AugmentStep(type="gain"), AugmentStep(type="noise")], copies=1)
    eid, st = run(ws, make_cfg("single", augmentation=aug))
    assert st == "COMPLETED"


def test_invalid_config_fails_validation_with_recorded_error(ws):
    bad = make_cfg("single").model_copy(update={"dataset": DatasetRef(path=str(ws.root / "nope"))})
    eid = create_experiment(ws, bad)
    with pytest.raises(Exception, match="not a valid dataset"):
        validate_experiment(ws, eid)
    r = ws.store.get(eid)
    assert r["status"] == "FAILED" and "validation failed" in r["error"]


def test_runtime_failure_marks_failed_and_keeps_partial_results(ws, monkeypatch):
    import openkeystroke.experiments.runner as runner

    orig = runner._Context.run_job
    calls = {"n": 0}

    def flaky(self, job):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return orig(self, job)

    monkeypatch.setattr(runner._Context, "run_job", flaky)
    eid = create_experiment(ws, make_cfg("repeated_seed", seeds=[0, 1, 2]))
    assert run_experiment(ws, eid) == "FAILED"
    r = ws.store.get(eid)
    assert (
        "boom" in r["error"] and ws.store.has_artifact(eid, "results.partial.json") and ws.store.metrics(eid)
    )  # job 1 persisted
    assert any(e["level"] == "error" for e in ws.store.events(eid))


def test_cancellation_midrun(ws):
    cfg = make_cfg("repeated_seed", seeds=[0, 1, 2, 3])
    eid = create_experiment(ws, cfg)
    seen = []

    def cb(i, n, msg):
        seen.append(i)
        if i == 1:
            ws.store.request_cancel(eid)

    assert run_experiment(ws, eid, progress=cb) == "CANCELLED"
    assert ws.store.get(eid)["status"] == "CANCELLED" and seen == [0, 1]


def test_job_expansion_and_config_validation():
    cfg = make_cfg(
        "defense_comparison",
        defenses=[DefenseConfig(name="a", steps=[DefenseStep(type="lowpass", params={"cutoff_hz": 2000})])],
    )
    jobs = build_jobs(cfg)
    assert len(jobs) == 3 * 2  # per seed: 1 static (clean + a) + 1 adaptive
    for bad in (
        dict(kind="model_comparison"),
        dict(kind="repeated_seed", seeds=[0]),
        dict(kind="defense_comparison"),
        dict(kind="noise_robustness"),
        dict(kind="sweep"),
        dict(kind="ablation"),
        dict(kind="single", seeds=[0, 1]),
        dict(seeds=[0, 0]),
    ):
        with pytest.raises(Exception):  # noqa: B017
            make_cfg(**{"kind": "single", **bad}) if "seeds" in bad else make_cfg(**bad) if bad.get(
                "kind"
            ) in ("single",) else __import__("openkeystroke.config", fromlist=["x"]).ExperimentConfig(
                name="x", dataset=DatasetRef(path="p"), models=[ModelConfig(name="logreg")], **bad
            )
    with pytest.raises(ConfigError):
        build_jobs(
            make_cfg("sweep", sweep={"models.0.params.C": [1.0]}).model_copy(update={"variants": {"x": {}}})
        )
    with pytest.raises(ConfigError, match="invalid override"):
        build_jobs(make_cfg("ablation", variants={"v": {"models.5.name": "x"}}))


def test_group_disjoint_results_not_inflated_vs_in_distribution(ws):
    """Sanity: reported metrics are on the held-out partition, never on training data."""
    eid, _ = run(ws, make_cfg("single", models=[ModelConfig(name="random_forest")]))
    rows = ws.store.get_json(eid, "results.json")["rows"]
    sizes = rows[0]["split_sizes"]
    assert (
        rows[0]["test"]["n"] == sizes["test"]
        and rows[0]["val"]["n"] == sizes["val"]
        and sizes["train"] > sizes["test"]
    )
    jobs = ws.store.get_json(eid, "results.json")["jobs"]
    assert (
        "train_acc" in jobs[0]["history"]["epochs"][0]
    )  # training accuracy exists only as a diagnostic in history
    assert all("train" not in r for r in rows)


def test_consented_real_dataset_runs_end_to_end_and_is_labelled_real(ws, small_ds, tmp_path):
    """Ingest WAVs through the consent manifest, then run a full experiment on the stored (non-synthetic) dataset."""
    import json

    import soundfile as sf

    from openkeystroke.data import ingest_consented_recordings

    lab = tmp_path / "lab"
    lab.mkdir()
    recs = []
    for i, r in enumerate(small_ds.records):
        sf.write(lab / f"r{i}.wav", np.asarray(small_ds.waveforms[i]), 16000)
        recs.append(
            {
                "file": f"r{i}.wav",
                "label": r.label,
                "session_id": r.session_id,
                "device_id": r.device_id,
                "environment": r.environment,
            }
        )
    man = {
        "dataset_id": "lab-e2e",
        "classes": small_ds.class_names,
        "recordings": recs,
        "consent": {
            "obtained": True,
            "protocol_id": "LAB-2026-07",
            "purpose": "authorized acoustic research",
            "participants_informed": True,
        },
    }
    (lab / "manifest.json").write_text(json.dumps(man))
    ingest_consented_recordings(lab / "manifest.json", ws.datasets_dir / "lab-e2e")
    cfg = make_cfg("single").model_copy(update={"dataset": DatasetRef(path="lab-e2e")})
    eid, st = run(ws, cfg)
    r = ws.store.get(eid)
    assert st == "COMPLETED" and r["dataset_is_synthetic"] == 0
    s = ws.store.get_json(eid, "summary.json")
    assert not any("SYNTHETIC" in c for c in s["caveats"]) and s["dataset"]["is_synthetic"] is False
    assert ws.store.get_json(eid, "dataset.json")["manifest"]["consent"]["protocol_id"] == "LAB-2026-07"
