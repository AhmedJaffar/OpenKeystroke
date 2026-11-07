"""Experiment lifecycle: CREATED -> VALIDATED -> RUNNING -> COMPLETED | FAILED | CANCELLED."""

from __future__ import annotations

import contextlib
import resource
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import ExperimentConfig, canonical_json
from ..data import Dataset, assert_no_leakage, augment, generate_synthetic, make_split, validate_dataset
from ..defense import DefenseChain, defense_quality
from ..dsp import PreprocessPipeline
from ..errors import CancelledError, DatasetError, OpenKeystrokeError
from ..evaluation import SCALAR_KEYS, bootstrap_ci, compute_metrics, measure_inference
from ..evaluation.summary import summarize_experiment
from ..features import FeatureCache, make_extractor
from ..models import create_model, input_kind
from ..registry import Workspace
from . import plots
from .plan import CLEAN, Condition, Job, build_jobs

ProgressCb = Callable[[int, int, str], None]


# ----------------------------------------------------------------------------- dataset resolution
def resolve_dataset(ws: Workspace, cfg: ExperimentConfig) -> Dataset:
    ref = cfg.dataset
    if ref.synthetic is not None:
        from ..config import config_hash

        path = ws.datasets_dir / f"synthetic-{config_hash(ref.synthetic, 10)}"
        if path.is_dir():
            try:
                return Dataset.load(path, verify=True)
            except DatasetError:
                pass  # corrupted cache of a deterministic dataset: regenerate
        ds = generate_synthetic(ref.synthetic)
        ds.save(path, overwrite=True)
        return Dataset.load(path, verify=True)
    assert ref.path is not None
    p = Path(ref.path)
    if not p.is_absolute() and not p.exists():
        p = ws.datasets_dir / ref.path
    ds = Dataset.load(p, verify=True)
    rep = validate_dataset(ds, check_fingerprint=False)
    if not rep.ok:
        raise DatasetError("dataset failed validation: " + "; ".join(rep.errors), details=rep.to_dict())
    return ds


# ----------------------------------------------------------------------------- lifecycle
def create_experiment(ws: Workspace, cfg: ExperimentConfig, *, parent_id: str | None = None) -> str:
    return ws.store.create(cfg, parent_id=parent_id)


def validate_experiment(ws: Workspace, eid: str) -> list[str]:
    """CREATED -> VALIDATED (or FAILED). Returns warnings."""
    cfg = ws.store.get_config(eid)
    try:
        warnings = _validate(ws, cfg, eid)
    except OpenKeystrokeError as exc:
        ws.store.transition(eid, "FAILED", error=f"validation failed: {exc.message}")
        raise
    ws.store.transition(eid, "VALIDATED")
    return warnings


def _validate(ws: Workspace, cfg: ExperimentConfig, eid: str) -> list[str]:
    jobs = build_jobs(cfg)  # raises ConfigError on bad overrides / collisions
    for j in jobs:
        input_kind(j.cfg.models[j.model_idx].name)  # raises ModelError for unknown model names
    ds = resolve_dataset(ws, cfg)
    ws.store.set_dataset(eid, ds.manifest.dataset_id, ds.manifest.version, ds.fingerprint, ds.is_synthetic)
    rep = validate_dataset(ds)
    if not rep.ok:
        raise DatasetError("dataset failed validation: " + "; ".join(rep.errors))
    for proto in {j.protocol for j in jobs}:
        for seed in cfg.seeds:
            sp = make_split(ds, proto, cfg.split, seed)
            assert_no_leakage(ds, sp)
    return rep.warnings


def run_experiment(ws: Workspace, eid: str, *, progress: ProgressCb | None = None) -> str:
    """Execute an experiment end to end. Returns the final status."""
    store = ws.store
    if store.get(eid)["status"] == "CREATED":
        validate_experiment(ws, eid)
    cfg = store.get_config(eid)
    store.transition(eid, "RUNNING")
    rows: list[dict[str, Any]] = []
    jobs_out: list[dict[str, Any]] = []
    preds: dict[str, np.ndarray] = {}
    quality: dict[str, Any] = {}
    ds_info: dict[str, Any] = {}
    try:
        torch.set_num_threads(cfg.resources.max_threads)
        ds = resolve_dataset(ws, cfg)
        ds_info = {
            "dataset_id": ds.manifest.dataset_id,
            "version": ds.manifest.version,
            "fingerprint": ds.fingerprint,
            "is_synthetic": ds.is_synthetic,
            "class_names": ds.class_names,
        }
        store.put_json(
            eid, "dataset.json", {"manifest": ds.manifest.model_dump(mode="json"), "stats": ds.stats()}
        )
        jobs = build_jobs(cfg)
        total = len(jobs)
        store.update_progress(eid, 0, total, "starting")
        ctx = _Context(ws, eid, ds, FeatureCache(ws.cache_dir))
        for i, job in enumerate(jobs):
            if store.is_cancel_requested(eid):
                raise CancelledError("cancelled by user")
            store.update_progress(eid, i, total, job.job_id)
            if progress:
                progress(i, total, job.job_id)
            job_rows, job_info, job_preds = ctx.run_job(job)
            rows += job_rows
            jobs_out.append(job_info)
            preds.update(job_preds)
            store.add_metrics(eid, _metric_rows(job_rows))  # incremental persistence
        # defense / noise conditions: distortion + latency measured on held-out-agnostic raw audio sample
        for c in _all_conditions(jobs):
            if c.defense is not None:
                quality[c.name] = _quality(ds, c, cfg)
        store.update_progress(eid, total, total, "summarising")
        summary = summarize_experiment(rows, cfg, quality, ds_info)
        store.put_json(eid, "summary.json", summary, "json")
        _plots(store, eid, rows, jobs_out, summary, ds_info)
        store.put_json(
            eid,
            "results.json",
            {"rows": rows, "jobs": jobs_out, "defense_quality": quality, "dataset": ds_info},
            "json",
        )
        if preds:
            store.put_npz(eid, "predictions.npz", **preds)
        store.transition(eid, "COMPLETED")
        return "COMPLETED"
    except CancelledError as exc:
        _save_partial(store, eid, rows, jobs_out, quality, ds_info)
        store.transition(eid, "CANCELLED", error=exc.message)
        return "CANCELLED"
    except BaseException as exc:
        _save_partial(store, eid, rows, jobs_out, quality, ds_info)
        msg = exc.message if isinstance(exc, OpenKeystrokeError) else f"{type(exc).__name__}: {exc}"
        store.log(eid, "error", traceback.format_exc()[-4000:])
        store.transition(eid, "FAILED", error=msg)
        if isinstance(exc, KeyboardInterrupt):
            raise
        return "FAILED"


def _save_partial(store, eid, rows, jobs_out, quality, ds_info) -> None:
    with contextlib.suppress(Exception):  # never mask the original failure
        store.put_json(
            eid,
            "results.partial.json",
            {"rows": rows, "jobs": jobs_out, "defense_quality": quality, "dataset": ds_info},
        )


def _all_conditions(jobs: list[Job]) -> list[Condition]:
    seen: dict[str, Condition] = {}
    for j in jobs:
        for c in (j.train_condition, *j.test_conditions):
            seen.setdefault(c.name, c)
    return list(seen.values())


def _quality(ds: Dataset, cond: Condition, cfg: ExperimentConfig) -> dict[str, Any]:
    n = min(cfg.evaluation.defense_quality_samples, len(ds))
    idx = np.random.default_rng([cfg.seeds[0], 5]).choice(len(ds), n, replace=False)
    x = np.asarray(ds.waveforms)[np.sort(idx)]
    assert cond.defense is not None
    return defense_quality(DefenseChain(cond.defense), x, ds.sample_rate, cfg.seeds[0])


def _metric_rows(job_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in job_rows:
        base = {k: r[k] for k in ("variant", "protocol", "model", "condition", "attacker", "seed")}
        for split in ("val", "test"):
            for k in (*SCALAR_KEYS, "n"):
                if k in r[split]:
                    out.append({**base, "split": split, "name": k, "value": r[split][k]})
        for k, v in (r.get("test_ci") or {}).items():
            out += [
                {**base, "split": "test", "name": f"{k}_ci_lo", "value": v["lo"]},
                {**base, "split": "test", "name": f"{k}_ci_hi", "value": v["hi"]},
            ]
        for k, v in (r.get("resource") or {}).items():
            out.append({**base, "split": "test", "name": k, "value": v})
    return out


def _plots(store, eid, rows, jobs_out, summary, ds_info) -> None:
    classes = ds_info["class_names"]
    clean = [r for r in rows if r["condition"] == "clean" and r["attacker"] == "static"]
    if clean:
        best = max(clean, key=lambda r: r["test"]["accuracy"])
        store.put_bytes(
            eid,
            "plots/confusion_best.png",
            plots.confusion_png(
                best["test"]["confusion_matrix"],
                classes,
                f"{best['model']} / {best['protocol']} / seed {best['seed']}",
            ),
            "png",
        )
    deep = [j for j in jobs_out if len(j["history"]["epochs"]) > 1]
    if deep:
        j = deep[0]
        store.put_bytes(eid, "plots/training_curve.png", plots.training_png(j["history"], j["job_id"]), "png")
    if summary["effectiveness_curves"]:
        store.put_bytes(
            eid,
            "plots/defense_curves.png",
            plots.defense_png(summary["effectiveness_curves"], summary["chance_accuracy"]),
            "png",
        )


# ----------------------------------------------------------------------------- job execution
class _Context:
    def __init__(self, ws: Workspace, eid: str, ds: Dataset, cache: FeatureCache) -> None:
        self.ws, self.eid, self.ds, self.cache = ws, eid, ds, cache
        self.raw = np.asarray(ds.waveforms)
        self._defended: dict[tuple, np.ndarray] = {}
        self._proc: dict[tuple, np.ndarray] = {}
        self._base: dict[tuple, np.ndarray] = {}
        self._models_saved: set[tuple] = set()

    # waveform stages -------------------------------------------------------
    def defended(self, cond: Condition, seed: int) -> np.ndarray:
        if cond.defense is None:
            return self.raw
        chain = DefenseChain(cond.defense)
        key = (cond.name, seed if chain.stochastic else 0)
        if key not in self._defended:
            self._defended[key] = chain.apply(self.raw, self.ds.sample_rate, seed)
        return self._defended[key]

    def processed(self, cfg: ExperimentConfig, cond: Condition, seed: int) -> tuple[np.ndarray, int]:
        stoch = cond.defense is not None and DefenseChain(cond.defense).stochastic
        key = (cfg.preprocess.model_dump_json(), cond.name, seed if stoch else 0)
        if key not in self._proc:
            self._proc[key] = PreprocessPipeline(cfg.preprocess)(
                self.defended(cond, seed), self.ds.sample_rate
            ).waveforms
        return self._proc[key], cfg.preprocess.target_sample_rate or self.ds.sample_rate

    def base_features(self, cfg: ExperimentConfig, kind: str, cond: Condition, seed: int, ex) -> np.ndarray:
        wf, _ = self.processed(cfg, cond, seed)
        if kind == "waveform":
            return wf
        stoch = cond.defense is not None and DefenseChain(cond.defense).stochastic
        wave_key = {
            "preprocess": cfg.preprocess.model_dump(mode="json"),
            "defense": cond.defense.model_dump(mode="json") if cond.defense else None,
            "seed": seed if stoch else 0,
        }
        mkey = (canonical_json(wave_key), ex.cache_id())
        if mkey not in self._base:
            ck = FeatureCache.key(self.ds.fingerprint, wave_key, ex.cache_id())
            self._base[mkey] = np.asarray(self.cache.get_or_compute(ck, lambda: ex.batched(wf)))
        return self._base[mkey]

    # one job ----------------------------------------------------------------
    def run_job(self, job: Job) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
        cfg, store, eid = job.cfg, self.ws.store, self.eid
        mcfg = cfg.models[job.model_idx]
        kind = input_kind(mcfg.name)
        split = make_split(self.ds, job.protocol, cfg.split, job.seed)
        assert_no_leakage(self.ds, split)
        y = self.ds.labels
        _, sr = self.processed(cfg, job.train_condition, job.seed)
        fcfg = {"vector": cfg.vector_features, "spectrogram": cfg.tensor_features}.get(
            kind, cfg.tensor_features
        )
        ex = (
            make_extractor(fcfg, sr)
            if kind != "waveform"
            else make_extractor(fcfg.model_copy(update={"name": "waveform"}), sr)
        )

        # training features (training partition only is used for every fit)
        base_tr = self.base_features(cfg, kind, job.train_condition, job.seed, ex)[split.train]
        y_tr = y[split.train]
        if cfg.augmentation is not None:
            aug_raw = augment(
                self.defended(job.train_condition, job.seed)[split.train],
                cfg.augmentation,
                job.seed,
                self.ds.sample_rate,
            )
            aug_wf = PreprocessPipeline(cfg.preprocess)(aug_raw, self.ds.sample_rate).waveforms
            base_aug = aug_wf if kind == "waveform" else ex.batched(aug_wf)
            base_tr = np.concatenate([base_tr, base_aug])
            y_tr = np.concatenate([y_tr, np.tile(y_tr[: len(split.train)], cfg.augmentation.copies)])
        if ex.requires_fit:
            ex.fit(base_tr)  # fit on training partition only
        X_tr = ex.finish(base_tr)
        base_va = self.base_features(
            cfg, kind, job.train_condition if job.attacker == "adaptive" else CLEAN, job.seed, ex
        )[split.val]
        X_va = ex.finish(base_va)
        model = create_model(mcfg, len(self.ds.class_names), X_tr.shape[1:], job.seed)
        ckpt = (
            store.artifact_dir(eid, f"checkpoints/{job.slug}")
            if model.family == "deep" and mcfg.training.checkpoint
            else None
        )
        cancel = lambda: store.is_cancel_requested(eid)  # noqa: E731
        history = model.fit(X_tr, y_tr, X_va, y[split.val], should_cancel=cancel, checkpoint_dir=ckpt)
        if ckpt is not None:
            store.register_dir(
                eid,
                f"checkpoints/{job.slug}",
                "checkpoint_dir",
                {f.name: str(f.stat().st_size) for f in ckpt.iterdir()},
            )

        # evaluation on val + test under each requested test condition
        rows, preds = [], {}
        groups = self.ds.group_array("session")[split.test]
        for cond in job.test_conditions:
            B = self.base_features(cfg, kind, cond, job.seed, ex)
            Xte, Xva = ex.finish(B[split.test]), ex.finish(B[split.val])
            p_te, p_va = model.predict_proba(Xte), model.predict_proba(Xva)
            m_te = compute_metrics(
                y[split.test], p_te, self.ds.class_names, calibration_bins=cfg.evaluation.calibration_bins
            )
            m_va = compute_metrics(
                y[split.val], p_va, self.ds.class_names, calibration_bins=cfg.evaluation.calibration_bins
            )
            ci = bootstrap_ci(
                y[split.test],
                p_te.argmax(1),
                len(self.ds.class_names),
                groups=groups,
                n_boot=cfg.evaluation.bootstrap_samples,
                confidence=cfg.evaluation.confidence,
                seed=job.seed,
            )
            res: dict[str, float] = {}
            if cond is CLEAN or cond.name == "clean":
                res = {
                    "n_parameters": float(model.num_parameters()),
                    "model_size_bytes": float(model.size_bytes()),
                    "train_seconds": history.seconds,
                    "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                }
                if cfg.evaluation.measure_latency:
                    res.update(measure_inference(model.predict_proba, Xte))
            rows.append(
                {
                    "job_id": job.job_id,
                    "variant": job.variant,
                    "protocol": job.protocol,
                    "model": job.model_label,
                    "model_name": mcfg.name,
                    "seed": job.seed,
                    "attacker": job.attacker,
                    "train_condition": job.train_condition.name,
                    "condition": cond.name,
                    "split_sizes": split.sizes(),
                    "test_groups": split.groups.get("test", []),
                    "group_kind": split.group_kind,
                    "val": m_va,
                    "test": m_te,
                    "test_ci": ci,
                    "resource": res,
                }
            )
            tag = f"{job.slug}__{cond.name}"
            preds[f"{tag}__y"], preds[f"{tag}__proba"], preds[f"{tag}__idx"] = (
                y[split.test],
                p_te.astype(np.float32),
                split.test,
            )
        # persist a trained clean/static model once per (variant, protocol, model) for later inspection
        mk = (job.variant, job.protocol, job.model_label)
        if cfg.save_models and job.attacker == "static" and mk not in self._models_saved:
            self._models_saved.add(mk)
            d = store.artifact_dir(eid, f"models/{job.slug}")
            hashes = model.save(d)
            ex.save(d / "extractor")
            store.register_dir(eid, f"models/{job.slug}", "model_dir", hashes)
        info = {
            "job_id": job.job_id,
            "history": history.to_dict(),
            "feature": ex.config_dict(),
            "model": {"name": mcfg.name, "params": mcfg.params, "family": model.family},
            "split": {"protocol": job.protocol, "sizes": split.sizes(), "groups": split.groups},
        }
        return rows, info, preds
