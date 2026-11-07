"""Measured (not assumed) performance of each pipeline stage."""

from __future__ import annotations

import resource
import tempfile
import time
from typing import Any

import numpy as np

from .config import FeatureConfig, ModelConfig, PreprocessConfig, SplitConfig, SyntheticConfig, TrainConfig
from .data import Dataset, generate_synthetic, make_split
from .dsp import PreprocessPipeline
from .features import FeatureCache, make_extractor
from .models import create_model
from .registry import Workspace


def _t(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


def profile_pipeline(ws: Workspace, samples_per_class: int = 20) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cfg = SyntheticConfig(samples_per_class_per_session=samples_per_class)
    ds, dt = _t(lambda: generate_synthetic(cfg))
    out["samples"] = len(ds)
    out["synthetic generation (samples/s)"] = round(len(ds) / dt)
    with tempfile.TemporaryDirectory() as d:
        _, dt = _t(lambda: ds.save(d + "/ds", overwrite=True))
        out["dataset save (MB/s)"] = round(np.asarray(ds.waveforms).nbytes / 1e6 / dt, 1)
        loaded, dt = _t(lambda: Dataset.load(d + "/ds", verify=True))
        out["dataset load+verify (samples/s)"] = round(len(ds) / dt)
        wf, dt = _t(
            lambda: (
                PreprocessPipeline(PreprocessConfig())(
                    np.asarray(loaded.waveforms), loaded.sample_rate
                ).waveforms
            )
        )
        out["preprocessing (samples/s)"] = round(len(ds) / dt)
        cache = FeatureCache(d + "/cache")
        for name in ("combined_stats", "mfcc", "log_mel"):
            ex = make_extractor(FeatureConfig(name=name), 16000)
            key = FeatureCache.key(ds.fingerprint, "profile", ex.cache_id())
            _, cold = _t(lambda ex=ex, key=key: cache.get_or_compute(key, lambda: ex.batched(wf)))
            _, warm = _t(lambda key=key, ex=ex: cache.get_or_compute(key, lambda: ex.batched(wf)))
            out[f"features {name} cold (samples/s)"] = round(len(ds) / cold)
            out[f"features {name} cached (samples/s)"] = round(len(ds) / max(warm, 1e-9))
        sp = make_split(ds, "session_disjoint", SplitConfig(), 0)
        y = ds.labels
        for mname, kind, shape_name in (
            ("logreg", "vector", "combined_stats"),
            ("cnn1d", "waveform", "waveform"),
            ("cnn2d", "spectrogram", "log_mel"),
        ):
            X = (
                wf
                if kind == "waveform"
                else make_extractor(FeatureConfig(name=shape_name), 16000).batched(wf)
            )
            m = create_model(
                ModelConfig(name=mname, training=TrainConfig(epochs=3, patience=3)),
                len(ds.class_names),
                X.shape[1:],
                0,
            )
            h, dt = _t(lambda m=m, X=X: m.fit(X[sp.train], y[sp.train], X[sp.val], y[sp.val]))
            out[f"train {mname} (samples*epochs/s)"] = round(len(sp.train) * len(h.epochs) / dt)
            _, dt = _t(lambda m=m, X=X: m.predict_proba(X[sp.test]))
            out[f"infer {mname} (samples/s)"] = round(len(sp.test) / max(dt, 1e-9))
        _, dt = _t(lambda: [ws.store.list_experiments(limit=200) for _ in range(20)])
        out["registry list query (ms)"] = round(dt / 20 * 1000, 2)
    out["process peak RSS (MB)"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return out
