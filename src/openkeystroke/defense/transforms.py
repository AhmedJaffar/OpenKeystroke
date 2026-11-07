"""Acoustic privacy defenses: signal transformations evaluated for leakage reduction vs audio quality.

All defenses are deterministic given the supplied RNG. They are research instruments for measuring how much class
information survives a transformation; they are not tuned to evade monitoring.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..config import DefenseConfig, DefenseStep
from ..dsp import ops
from ..errors import ConfigError


class Defense:
    type = "base"
    stochastic = False

    def __init__(self, **params: float) -> None:
        self.params = params

    def apply(
        self, x: np.ndarray, sr: int, rng: np.random.Generator
    ) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


def _req(p: dict[str, float], key: str, default: float | None = None) -> float:
    if key in p:
        return float(p[key])
    if default is None:
        raise ConfigError(f"defense parameter '{key}' is required")
    return default


class LowPass(Defense):
    type = "lowpass"

    def apply(self, x, sr, rng):
        return ops.lowpass(x, sr, _req(self.params, "cutoff_hz"), int(_req(self.params, "order", 4)))


class BandPass(Defense):
    type = "bandpass"

    def apply(self, x, sr, rng):
        return ops.bandpass(
            x,
            sr,
            _req(self.params, "low_hz"),
            _req(self.params, "high_hz"),
            int(_req(self.params, "order", 4)),
        )


class NoiseInjection(Defense):
    type = "noise"
    stochastic = True

    def apply(self, x, sr, rng):
        x = ops.as_batch(x).astype(np.float64)
        snr = _req(self.params, "snr_db")
        alpha = _req(self.params, "alpha", 0.0)  # 0 = white, 1 = pink
        w = rng.standard_normal(x.shape)
        if alpha:
            spec = np.fft.rfft(w, axis=1)
            f = np.fft.rfftfreq(x.shape[1])
            f[0] = f[1]
            w = np.fft.irfft(spec * f ** (-alpha / 2), x.shape[1], axis=1)
        w /= np.maximum(w.std(axis=1, keepdims=True), 1e-12)
        p = (x**2).mean(axis=1, keepdims=True)
        return (x + w * np.sqrt(p / 10 ** (snr / 10))).astype(np.float32)


class TransientAttenuation(Defense):
    type = "transient_attenuation"

    def apply(self, x, sr, rng):
        x = ops.as_batch(x)
        att = 10 ** (-_req(self.params, "attenuation_db") / 20)
        n_win = max(int(_req(self.params, "window_ms", 20.0) * sr / 1000), 4)
        onset = ops.detect_transient(x, sr)
        out = x.astype(np.float64).copy()
        ramp = max(n_win // 8, 1)
        env = np.ones(n_win)
        env[:] = att
        env[:ramp] = np.linspace(1, att, ramp)
        env[-ramp:] = np.linspace(att, 1, ramp)
        for i, o in enumerate(onset):
            if o < 0:
                continue
            seg = slice(o, min(o + n_win, x.shape[1]))
            out[i, seg] *= env[: seg.stop - seg.start]
        return out.astype(np.float32)


class SpectralSmoothing(Defense):
    type = "spectral_smoothing"

    def apply(self, x, sr, rng):
        x = ops.as_batch(x).astype(np.float64)
        n_fft, hop = int(_req(self.params, "n_fft", 256)), int(_req(self.params, "hop", 64))
        width = max(int(_req(self.params, "width_bins")), 1)
        s = ops.stft(x, n_fft, hop)
        mag = ndimage.uniform_filter1d(
            np.abs(s), size=width, axis=1, mode="nearest"
        )  # smooth magnitude along frequency
        phase = np.exp(1j * np.angle(s))
        return ops.istft(mag * phase, n_fft, hop, length=x.shape[1]).astype(np.float32)


class TFMask(Defense):
    type = "tf_mask"
    stochastic = True

    def apply(self, x, sr, rng):
        x = ops.as_batch(x).astype(np.float64)
        n_fft, hop = int(_req(self.params, "n_fft", 256)), int(_req(self.params, "hop", 64))
        s = ops.stft(x, n_fft, hop)
        n, f, t = s.shape
        n_masks = int(_req(self.params, "n_masks", 2))
        fw = max(int(_req(self.params, "freq_frac", 0.15) * f), 1)
        tw = max(int(_req(self.params, "time_frac", 0.15) * t), 1)
        mask = np.ones((n, f, t))
        for i in range(n):
            for _ in range(n_masks):
                f0, t0 = rng.integers(0, max(f - fw, 1)), rng.integers(0, max(t - tw, 1))
                mask[i, f0 : f0 + fw, :] = 0.0
                mask[i, :, t0 : t0 + tw] *= 0.0
        return ops.istft(s * mask, n_fft, hop, length=x.shape[1]).astype(np.float32)


DEFENSES: dict[str, type[Defense]] = {
    c.type: c for c in (LowPass, BandPass, NoiseInjection, TransientAttenuation, SpectralSmoothing, TFMask)
}


class DefenseChain:
    """Sequential composition of defenses built from a DefenseConfig."""

    def __init__(self, cfg: DefenseConfig) -> None:
        self.cfg = cfg
        self.steps = [self._make(s) for s in cfg.steps]

    @staticmethod
    def _make(step: DefenseStep) -> Defense:
        if step.type not in DEFENSES:
            raise ConfigError(f"unknown defense '{step.type}'")
        return DEFENSES[step.type](**step.params)

    @property
    def stochastic(self) -> bool:
        return any(s.stochastic for s in self.steps)

    def apply(self, x: np.ndarray, sr: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng([seed, 31])
        y = ops.as_batch(x)
        for s in self.steps:
            y = s.apply(y, sr, rng)
        if not np.isfinite(y).all():
            raise ConfigError(f"defense '{self.cfg.name}' produced non-finite output")
        return np.asarray(y, dtype=np.float32)
