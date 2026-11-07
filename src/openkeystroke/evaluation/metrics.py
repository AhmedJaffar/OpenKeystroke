"""Evaluation metrics, calibration, bootstrap confidence intervals, and cross-seed statistics."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats as sstats
from sklearn import metrics as skm

from ..errors import OpenKeystrokeError


def _bal_acc(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    recalls = [(p[y == c] == c).mean() for c in range(n_classes) if (y == c).any()]
    return float(np.mean(recalls))


def calibration(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> dict[str, Any]:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    rel = []
    ece = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            rel.append(
                {
                    "bin": b,
                    "lo": float(edges[b]),
                    "hi": float(edges[b + 1]),
                    "count": int(m.sum()),
                    "confidence": float(conf[m].mean()),
                    "accuracy": float(correct[m].mean()),
                }
            )
    onehot = np.eye(proba.shape[1])[y]
    nll = float(-np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1)).mean())
    return {
        "ece": float(ece),
        "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()),
        "nll": nll,
        "mean_confidence": float(conf.mean()),
        "reliability": rel,
    }


def compute_metrics(
    y: np.ndarray, proba: np.ndarray, class_names: list[str], *, calibration_bins: int = 10
) -> dict[str, Any]:
    """Scalar metrics + confusion matrix + per-class + calibration for one evaluation set."""
    y = np.asarray(y)
    n_classes = len(class_names)
    if proba.shape != (len(y), n_classes):
        raise OpenKeystrokeError(f"probability matrix shape {proba.shape} != ({len(y)}, {n_classes})")
    if (
        not np.isfinite(proba).all()
        or (proba < -1e-9).any()
        or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)
    ):
        raise OpenKeystrokeError("predicted probabilities must be finite, non-negative and sum to 1 per row")
    pred = proba.argmax(axis=1)
    labels = list(range(n_classes))
    p, r, f, s = skm.precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    pw, rw, fw, _ = skm.precision_recall_fscore_support(
        y, pred, labels=labels, average="weighted", zero_division=0
    )
    out: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": _bal_acc(y, pred, n_classes),
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": float(f.mean()),
        "precision_weighted": float(pw),
        "recall_weighted": float(rw),
        "f1_weighted": float(fw),
        "confusion_matrix": skm.confusion_matrix(y, pred, labels=labels).tolist(),
        "per_class": {
            class_names[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in labels
        },
    }
    cal = calibration(y, proba, calibration_bins)
    out.update({k: cal[k] for k in ("ece", "brier", "nll", "mean_confidence")})
    out["reliability"] = cal["reliability"]
    present = np.unique(y)
    if n_classes == 2 and len(present) == 2:
        out["roc_auc"] = float(skm.roc_auc_score(y, proba[:, 1]))
        out["pr_auc"] = float(skm.average_precision_score(y, proba[:, 1]))
    elif n_classes > 2 and len(present) == n_classes:
        out["roc_auc_ovr_macro"] = float(
            skm.roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=labels)
        )
    return out


SCALAR_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "ece",
    "brier",
    "nll",
)


def bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    n_classes: int,
    *,
    groups: np.ndarray | None = None,
    n_boot: int = 300,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Percentile bootstrap CIs for accuracy / balanced accuracy / macro-F1.

    When `groups` is given the bootstrap resamples whole groups (cluster bootstrap) because samples from the same
    session or device are not independent; the plain bootstrap would give over-confident intervals.
    """
    if n_boot <= 0 or len(y) < 2:
        return {}
    rng = np.random.default_rng([seed, 7])
    n = len(y)
    if groups is not None:
        ug = np.unique(groups)
        members = [np.flatnonzero(groups == g) for g in ug]
    vals: dict[str, list[float]] = {"accuracy": [], "balanced_accuracy": [], "f1_macro": []}
    for _ in range(n_boot):
        if groups is not None:
            pick = rng.integers(0, len(ug), len(ug))
            idx = np.concatenate([members[i] for i in pick])
        else:
            idx = rng.integers(0, n, n)
        yb, pb = y[idx], pred[idx]
        vals["accuracy"].append((yb == pb).mean())
        vals["balanced_accuracy"].append(_bal_acc(yb, pb, n_classes))
        vals["f1_macro"].append(
            skm.f1_score(yb, pb, average="macro", labels=list(range(n_classes)), zero_division=0)
        )
    a = (1 - confidence) / 2
    return {
        k: {
            "lo": float(np.quantile(v, a)),
            "hi": float(np.quantile(v, 1 - a)),
            "se": float(np.std(v, ddof=1)),
            "method": "cluster_bootstrap" if groups is not None else "bootstrap",
            "n_clusters": int(len(ug)) if groups is not None else int(n),
        }
        for k, v in vals.items()
    }


def summarize_seeds(
    values: list[float], confidence: float = 0.95, bounds: tuple[float, float] | None = (0.0, 1.0)
) -> dict[str, Any]:
    """Mean/std over seeds with a t-interval, clipped to the metric's valid range (bounds=None for unbounded metrics)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    out: dict[str, Any] = {
        "n": n,
        "mean": float(v.mean()) if n else float("nan"),
        "std": float(v.std(ddof=1)) if n > 1 else 0.0,
    }
    if n > 1:
        h = sstats.t.ppf(0.5 + confidence / 2, n - 1) * out["std"] / np.sqrt(n)
        out["ci_lo"], out["ci_hi"] = out["mean"] - float(h), out["mean"] + float(h)
        if bounds:
            out["ci_lo"], out["ci_hi"] = max(out["ci_lo"], bounds[0]), min(out["ci_hi"], bounds[1])
    else:
        out["ci_lo"] = out["ci_hi"] = out["mean"]
    return out


def paired_comparison(a: list[float], b: list[float]) -> dict[str, Any]:
    """Paired comparison of per-seed scores (same seeds / splits): mean difference, paired t-test, Cohen's d_z."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if len(x) != len(y) or len(x) < 2:
        return {
            "n": int(len(x)),
            "mean_diff": float((x - y).mean()) if len(x) else float("nan"),
            "p_value": None,
            "note": "need >= 2 paired seeds for inference",
        }
    d = x - y
    sd = d.std(ddof=1)
    if sd == 0:
        return {
            "n": len(d),
            "mean_diff": float(d.mean()),
            "p_value": 0.0 if d.mean() != 0 else 1.0,
            "cohen_dz": None,
            "note": "zero variance in paired differences",
        }
    t, p = sstats.ttest_rel(x, y)
    return {"n": len(d), "mean_diff": float(d.mean()), "p_value": float(p), "cohen_dz": float(d.mean() / sd)}


def measure_inference(
    predict: Callable[[np.ndarray], Any], X: np.ndarray, *, repeats: int = 20
) -> dict[str, float]:
    """Single-sample latency (median/p95) and batch throughput."""
    n1 = min(repeats, len(X))
    predict(X[:1])  # warm-up
    lat = []
    for i in range(n1):
        t = time.perf_counter()
        predict(X[i : i + 1])
        lat.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    predict(X)
    dt = max(time.perf_counter() - t, 1e-9)
    return {
        "latency_ms_median": float(np.median(lat)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "throughput_samples_per_s": float(len(X) / dt),
    }
