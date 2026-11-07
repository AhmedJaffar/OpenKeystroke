"""Feature extractors with a common interface.

`compute()` is stateless and cacheable. `fit()` (only for learned embeddings) is called on *training* data only;
`finish()` applies fitted state to computed base features. `transform = finish(compute(x))`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from scipy import stats as sstats

from ..config import FeatureConfig, config_hash
from ..dsp import ops
from ..errors import ConfigError, DSPError


class FeatureExtractor:
    name: ClassVar[str] = "base"
    version: ClassVar[str] = "1"
    kind: ClassVar[str] = "vector"  # vector | tensor | waveform
    requires_fit: ClassVar[bool] = False

    def __init__(self, cfg: FeatureConfig, sample_rate: int) -> None:
        if cfg.n_fft // 2 + 1 < 8:
            raise ConfigError("n_fft too small")
        self.cfg = cfg
        self.sample_rate = ops.check_sr(sample_rate)
        self._fitted = not self.requires_fit

    # -- subclass hooks
    def compute(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, base: np.ndarray) -> FeatureExtractor:
        return self

    def finish(self, base: np.ndarray) -> np.ndarray:
        return base

    # -- public API
    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ConfigError(f"extractor '{self.name}' must be fit on training data before transform")
        return self.finish(self.compute(x))

    def batched(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Memory-bounded `compute` over large arrays (works on memory-mapped inputs)."""
        parts = [self.compute(np.asarray(x[i : i + batch_size])) for i in range(0, len(x), batch_size)]
        # always C-contiguous: memory layout must never influence downstream numerics (see tests)
        return np.ascontiguousarray(np.concatenate(parts, axis=0))

    def output_shape(self, n_waveform_samples: int) -> tuple[int, ...]:
        probe = np.random.default_rng(0).standard_normal((2, n_waveform_samples)).astype(np.float32)
        if self.requires_fit and not self._fitted:
            return tuple(self.compute(probe).shape[1:])
        return tuple(self.transform(probe).shape[1:])

    def config_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "sample_rate": self.sample_rate,
            "cfg": self.cfg.model_dump(mode="json"),
        }

    def cache_id(self) -> str:
        """Identifies the stateless `compute` output (excludes fitted state)."""
        return config_hash(self.config_dict(), 20)

    def get_state(self) -> dict[str, np.ndarray]:
        return {}

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self._fitted = True

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "extractor.json").write_text(json.dumps(self.config_dict(), indent=2))
        state: dict[str, Any] = dict(self.get_state())
        np.savez(p / "state.npz", **state)

    @staticmethod
    def load(path: str | Path) -> FeatureExtractor:
        p = Path(path)
        meta = json.loads((p / "extractor.json").read_text())
        ex = make_extractor(FeatureConfig.model_validate(meta["cfg"]), meta["sample_rate"])
        if ex.version != meta["version"]:
            raise ConfigError(f"extractor version mismatch: stored {meta['version']} vs code {ex.version}")
        with np.load(p / "state.npz", allow_pickle=False) as z:
            ex.set_state({k: z[k] for k in z.files})
        return ex

    # -- shared helpers
    def _mag(self, x: np.ndarray) -> np.ndarray:
        return np.abs(ops.stft(x, self.cfg.n_fft, self.cfg.hop, self.cfg.window))

    def _freqs(self) -> np.ndarray:
        return np.linspace(0, self.sample_rate / 2, self.cfg.n_fft // 2 + 1)

    def _mel_fb(self) -> np.ndarray:
        return ops.mel_filterbank(
            self.sample_rate, self.cfg.n_fft, self.cfg.n_mels, self.cfg.f_min, self.cfg.f_max
        )


def _safe(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


class WaveformExtractor(FeatureExtractor):
    name = "waveform"
    kind = "waveform"

    def compute(self, x: np.ndarray) -> np.ndarray:
        return ops.as_batch(x).astype(np.float32)


class TimeStats(FeatureExtractor):
    name = "time_stats"
    N_SEG = 8

    def compute(self, x: np.ndarray) -> np.ndarray:
        x = ops.as_batch(x).astype(np.float64)
        n, t = x.shape
        env = np.abs(x)
        peak = env.max(axis=1)
        rms = ops.rms_energy(x)
        peak_pos = env.argmax(axis=1) / t
        # decay: samples after the peak until the smoothed envelope falls below 10% of the peak
        k = max(int(0.001 * self.sample_rate), 2)
        sm = np.lib.stride_tricks.sliding_window_view(env, k, axis=-1).mean(axis=-1)
        pi = sm.argmax(axis=1)
        below = sm < 0.1 * sm.max(axis=1, keepdims=True)
        after = below & (np.arange(sm.shape[1])[None, :] > pi[:, None])
        decay = np.where(after.any(axis=1), after.argmax(axis=1) - pi, sm.shape[1] - pi) / self.sample_rate
        seg = np.array_split(np.arange(t), self.N_SEG)
        seg_e = np.stack([np.log10((x[:, s] ** 2).mean(axis=1) + 1e-12) for s in seg], axis=1)
        feats = np.column_stack(
            [
                np.log10(rms + 1e-9),
                np.log10(peak + 1e-9),
                peak / np.maximum(rms, 1e-9),
                ops.zero_crossing_rate(x),
                peak_pos,
                decay * 100,
                _safe(sstats.skew(x, axis=1)),
                _safe(sstats.kurtosis(x, axis=1)),
                env.mean(axis=1),
                env.std(axis=1),
                seg_e,
            ]
        )
        return feats.astype(np.float32)


class FreqStats(FeatureExtractor):
    name = "freq_stats"
    N_BANDS = 16

    def compute(self, x: np.ndarray) -> np.ndarray:
        mag = self._mag(ops.as_batch(x).astype(np.float64))
        f = self._freqs()
        cols = []
        for d in (
            ops.spectral_centroid(mag, f),
            ops.spectral_bandwidth(mag, f),
            ops.spectral_rolloff(mag, f),
            ops.spectral_flux(mag),
        ):
            cols += [d.mean(axis=1), d.std(axis=1)]
        pw = (mag**2).mean(axis=2)  # (N, F)
        edges = np.linspace(0, pw.shape[1], self.N_BANDS + 1).astype(int)
        bands = np.stack(
            [np.log10(pw[:, a:b].sum(axis=1) + 1e-12) for a, b in zip(edges[:-1], edges[1:], strict=True)],
            axis=1,
        )
        return np.column_stack([*cols, bands]).astype(np.float32)


class CombinedStats(FeatureExtractor):
    name = "combined_stats"

    def __init__(self, cfg: FeatureConfig, sample_rate: int) -> None:
        super().__init__(cfg, sample_rate)
        self._t, self._f = TimeStats(cfg, sample_rate), FreqStats(cfg, sample_rate)

    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([self._t.compute(x), self._f.compute(x)], axis=1)


class MFCC(FeatureExtractor):
    name = "mfcc"

    def compute(self, x: np.ndarray) -> np.ndarray:
        mag = self._mag(ops.as_batch(x).astype(np.float64))
        mel = ops.mel_spectrogram(mag**2, self._mel_fb())
        c = ops.mfcc_from_logmel(10 * np.log10(mel + 1e-10), self.cfg.n_mfcc)  # (N, n_mfcc, T)
        return np.concatenate([c.mean(axis=2), c.std(axis=2)], axis=1).astype(np.float32)


class Spectrogram(FeatureExtractor):
    name = "spectrogram"
    kind = "tensor"

    def compute(self, x: np.ndarray) -> np.ndarray:
        return ops.log_magnitude(
            ops.stft(ops.as_batch(x), self.cfg.n_fft, self.cfg.hop, self.cfg.window)
        ).astype(np.float32)


class Mel(FeatureExtractor):
    name = "mel"
    kind = "tensor"

    def compute(self, x: np.ndarray) -> np.ndarray:
        p = ops.power_spectrogram(ops.stft(ops.as_batch(x), self.cfg.n_fft, self.cfg.hop, self.cfg.window))
        return ops.mel_spectrogram(p, self._mel_fb()).astype(np.float32)


class LogMel(Mel):
    name = "log_mel"

    def compute(self, x: np.ndarray) -> np.ndarray:
        return (10.0 * np.log10(super().compute(x) + 1e-10)).astype(np.float32)


class PCAEmbedding(FeatureExtractor):
    """Learned linear embedding of the flattened log-mel spectrogram (PCA via SVD, fit on training data only)."""

    name = "pca_embedding"
    requires_fit = True

    def __init__(self, cfg: FeatureConfig, sample_rate: int) -> None:
        super().__init__(cfg, sample_rate)
        self._lm = LogMel(cfg, sample_rate)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    def compute(self, x: np.ndarray) -> np.ndarray:
        z = self._lm.compute(x)
        return z.reshape(len(z), -1)

    def fit(self, base: np.ndarray) -> PCAEmbedding:
        k = min(self.cfg.pca_components, base.shape[0] - 1, base.shape[1])
        if k < 1:
            raise DSPError("not enough training samples to fit PCA")
        self.mean_ = base.mean(axis=0)
        _, _, vt = np.linalg.svd(base - self.mean_, full_matrices=False)
        comps = vt[:k]
        # deterministic sign convention
        signs = np.sign(comps[np.arange(k), np.abs(comps).argmax(axis=1)])
        self.components_ = (comps * signs[:, None]).astype(np.float32)
        self._fitted = True
        return self

    def finish(self, base: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.components_ is not None
        return ((base - self.mean_) @ self.components_.T).astype(np.float32)

    def get_state(self) -> dict[str, np.ndarray]:
        if self.mean_ is None or self.components_ is None:
            return {}
        return {"mean": self.mean_, "components": self.components_}

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        if state:
            self.mean_, self.components_ = state["mean"], state["components"]
            self._fitted = True


EXTRACTORS: dict[str, type[FeatureExtractor]] = {
    c.name: c
    for c in (
        WaveformExtractor,
        TimeStats,
        FreqStats,
        CombinedStats,
        MFCC,
        Spectrogram,
        Mel,
        LogMel,
        PCAEmbedding,
    )
}


def make_extractor(cfg: FeatureConfig, sample_rate: int) -> FeatureExtractor:
    if cfg.name not in EXTRACTORS:
        raise ConfigError(f"unknown feature extractor '{cfg.name}'. Available: {sorted(EXTRACTORS)}")
    return EXTRACTORS[cfg.name](cfg, sample_rate)
