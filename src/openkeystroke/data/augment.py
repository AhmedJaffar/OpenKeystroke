"""Configurable, seeded waveform augmentation (training partition only)."""

from __future__ import annotations

import numpy as np

from ..config import AugmentConfig
from ..errors import ConfigError


def _gain(x: np.ndarray, rng: np.random.Generator, db: float = 3.0) -> np.ndarray:
    return x * (10 ** (rng.uniform(-db, db, (x.shape[0], 1)) / 20))


def _shift(x: np.ndarray, rng: np.random.Generator, max_ms: float = 5.0, sr: float = 16000) -> np.ndarray:
    m = int(max_ms * sr / 1000)
    out = np.zeros_like(x)
    shifts: list[int] = np.asarray(rng.integers(-m, m + 1, size=x.shape[0])).tolist()
    for i, s in enumerate(shifts):
        if s >= 0:
            out[i, s:] = x[i, : x.shape[1] - s]
        else:
            out[i, : x.shape[1] + s] = x[i, -s:]
    return out


def _noise(
    x: np.ndarray, rng: np.random.Generator, snr_db_min: float = 20.0, snr_db_max: float = 35.0
) -> np.ndarray:
    snr = rng.uniform(snr_db_min, snr_db_max, (x.shape[0], 1))
    p = (x.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-20
    return x + rng.standard_normal(x.shape) * np.sqrt(p / 10 ** (snr / 10))


def augment(x: np.ndarray, cfg: AugmentConfig, seed: int, sample_rate: int) -> np.ndarray:
    """Return `cfg.copies` augmented copies of x stacked along axis 0 (originals not included)."""
    outs = []
    for c in range(cfg.copies):
        rng = np.random.default_rng([seed, 99, c])
        y = x.astype(np.float64).copy()
        for st in cfg.steps:
            if st.type == "gain":
                y = _gain(y, rng, **st.params)
            elif st.type == "time_shift":
                y = _shift(y, rng, sr=sample_rate, **st.params)
            elif st.type == "noise":
                y = _noise(y, rng, **st.params)
            elif st.type == "polarity":
                y = y * rng.choice([-1.0, 1.0], (y.shape[0], 1))
            else:  # pragma: no cover - guarded by config literal
                raise ConfigError(f"unknown augmentation '{st.type}'")
        outs.append(np.clip(y, -1, 1).astype(np.float32))
    return np.concatenate(outs, axis=0)
