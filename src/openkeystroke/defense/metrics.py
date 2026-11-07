"""Quality/distortion metrics for defenses, plus Pareto-frontier analysis."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..dsp import ops
from .transforms import DefenseChain


def distortion_metrics(clean: np.ndarray, processed: np.ndarray, sr: int) -> dict[str, float]:
    x, y = clean.astype(np.float64), processed.astype(np.float64)
    err = ((x - y) ** 2).sum(axis=1)
    sig = (x**2).sum(axis=1)
    snr = 10 * np.log10(np.maximum(sig, 1e-20) / np.maximum(err, 1e-20))
    blind_c, blind_p = ops.estimate_snr_db(x, sr), ops.estimate_snr_db(y, sr)
    lc, lp = (
        ops.log_magnitude(ops.stft(x), eps=1e-5),
        ops.log_magnitude(ops.stft(y), eps=1e-5),
    )  # -100 dB floor
    lsd = np.sqrt(((lc - lp) ** 2).mean(axis=(1, 2)))
    xc, yc = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    corr = (xc * yc).sum(axis=1) / np.maximum(np.sqrt((xc**2).sum(axis=1) * (yc**2).sum(axis=1)), 1e-12)
    return {
        "waveform_snr_db": float(np.clip(snr, -60, 100).mean()),
        "blind_snr_change_db": float((blind_p - blind_c).mean()),
        "log_spectral_distance_db": float(lsd.mean()),
        "waveform_correlation": float(corr.mean()),
        "rms_ratio": float((ops.rms_energy(y) / np.maximum(ops.rms_energy(x), 1e-12)).mean()),
    }


def defense_quality(
    chain: DefenseChain, clean: np.ndarray, sr: int, seed: int = 0, repeats: int = 3
) -> dict[str, Any]:
    """Distortion + processing latency (real-time factor = processing seconds per second of audio)."""
    out = chain.apply(clean, sr, seed)
    m = distortion_metrics(clean, out, sr)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        chain.apply(clean, sr, seed)
        times.append(time.perf_counter() - t)
    audio_s = clean.shape[0] * clean.shape[1] / sr
    best = min(times)
    m["latency_ms_per_sample"] = best / clean.shape[0] * 1000
    m["real_time_factor"] = best / audio_s
    return m


def pareto_frontier(points: list[dict[str, Any]], maximize: str, minimize: str) -> list[str]:
    """Names of non-dominated points: higher `maximize` (privacy gain) and lower `minimize` (distortion) are better."""
    usable = [p for p in points if p.get(maximize) is not None and p.get(minimize) is not None]
    front = []
    for p in usable:
        dominated = any(
            (q[maximize] >= p[maximize] and q[minimize] <= p[minimize])
            and (q[maximize] > p[maximize] or q[minimize] < p[minimize])
            for q in usable
            if q is not p
        )
        if not dominated:
            front.append(p["name"])
    return front
