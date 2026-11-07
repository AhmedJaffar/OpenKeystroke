"""Deterministic synthetic acoustic-transient generator.

Classes are abstract acoustic source categories: each has its own resonant modes (damped sinusoids) and a
broadband impact "click". Devices apply a smooth frequency response, sessions apply gain / shift / environment.
Every sample has its own RNG derived from (seed, index), so output is independent of generation order.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from scipy import signal as ssig

from ..config import SyntheticConfig, config_hash
from .dataset import Dataset
from .schema import DatasetManifest, SampleRecord

GENERATOR_VERSION = "synthetic-transient-1"
# name -> (noise spectral exponent alpha for 1/f^alpha, snr offset dB, reverb RT60 s)
ENVIRONMENTS: dict[str, tuple[float, float, float]] = {
    "quiet": (0.0, 4.0, 0.0),
    "office": (1.0, 0.0, 0.03),
    "cafe": (1.6, -4.0, 0.08),
    "outdoor": (0.5, -2.0, 0.0),
}
KEYBOARD_CATEGORIES = ("membrane", "mechanical_linear", "mechanical_clicky", "laptop_scissor")
MIC_CATEGORIES = ("mems", "condenser", "headset", "contact")


def _class_params(cfg: SyntheticConfig) -> list[dict]:
    rng = np.random.default_rng([cfg.seed, 1])
    n = cfg.n_classes
    spread = np.log(4.3) * cfg.class_separation
    params = []
    for k in range(n):
        centre = (k - (n - 1) / 2) / max(n - 1, 1)
        f0 = 1500.0 * np.exp(centre * spread)
        ratios = np.array([1.0, rng.uniform(1.5, 2.2), rng.uniform(2.3, 3.2)])
        params.append(
            {
                "freqs": f0 * ratios,
                "amps": np.array([1.0, rng.uniform(0.3, 0.8), rng.uniform(0.1, 0.5)]),
                "taus": rng.uniform(0.004, 0.015, 3),
                "click_cut": rng.uniform(2500, 6000) * (0.6 + 0.4 * cfg.class_separation),
            }
        )
    return params


def _device_response(cfg: SyntheticConfig, dev: int, n_bins: int, sr: int) -> np.ndarray:
    rng = np.random.default_rng([cfg.seed, 3, dev])
    f = np.linspace(0, sr / 2, n_bins)
    logf = np.log(np.maximum(f, 50.0) / 1000.0)
    gain_db = rng.normal(0, 1.5 * cfg.device_variation) * logf
    for _ in range(3):
        c, w, a = rng.uniform(-0.5, 2.0), rng.uniform(0.3, 0.8), rng.normal(0, 5.0 * cfg.device_variation)
        gain_db += a * np.exp(-((logf - c) ** 2) / (2 * w**2))
    return 10 ** (gain_db / 20)


def _pink_like_noise(rng: np.random.Generator, n: int, alpha: float) -> np.ndarray:
    w = rng.standard_normal(n)
    if alpha == 0:
        return w
    spec = np.fft.rfft(w)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec *= f ** (-alpha / 2)
    out = np.fft.irfft(spec, n)
    return out / (out.std() + 1e-12)


def generate_synthetic(cfg: SyntheticConfig) -> Dataset:
    sr, n = cfg.sample_rate, int(round(cfg.sample_rate * cfg.duration_s))
    classes = _class_params(cfg)
    class_names = [f"cat_{k:02d}" for k in range(cfg.n_classes)]
    t = np.arange(n) / sr
    records: list[SampleRecord] = []
    wave = np.zeros(
        (cfg.n_devices * cfg.sessions_per_device * cfg.n_classes * cfg.samples_per_class_per_session, n),
        dtype=np.float32,
    )
    dataset_id = f"synthetic-{config_hash(cfg, 10)}"
    idx = 0
    session_counter = 0
    for d in range(cfg.n_devices):
        resp = _device_response(cfg, d, n // 2 + 1, sr)
        kb, mic = (
            KEYBOARD_CATEGORIES[d % len(KEYBOARD_CATEGORIES)],
            MIC_CATEGORIES[(d + d // 4) % len(MIC_CATEGORIES)],
        )
        for s in range(cfg.sessions_per_device):
            srng = np.random.default_rng([cfg.seed, 4, d, s])
            env_name = cfg.environments[session_counter % len(cfg.environments)]
            alpha, snr_off, rt60 = ENVIRONMENTS.get(env_name, ENVIRONMENTS["office"])
            session_counter += 1
            gain = 10 ** (srng.normal(0, 2.0 * cfg.session_variation) / 20)
            fshift = 1.0 + srng.normal(0, 0.03 * cfg.session_variation)
            ir = None
            if rt60 > 0:
                m = int(rt60 * sr)
                ir = np.exp(-np.arange(m) / (rt60 * sr / 6.9)) * srng.standard_normal(m) * 0.15
                ir[0] = 1.0
            for c in range(cfg.n_classes):
                cp = classes[c]
                for _ in range(cfg.samples_per_class_per_session):
                    rng = np.random.default_rng([cfg.seed, 2, idx])
                    onset = rng.uniform(0.008, 0.025)
                    amp = np.exp(rng.normal(0, 0.2))
                    fj = np.exp(rng.normal(0, 0.03)) * fshift
                    tt = np.maximum(t - onset, 0.0)
                    gate = (t >= onset).astype(float)
                    x: np.ndarray = np.zeros(n)
                    for fr, a, tau in zip(cp["freqs"], cp["amps"], cp["taus"], strict=True):
                        ph = rng.uniform(0, 2 * np.pi)
                        x += (
                            a
                            * np.exp(-tt / (tau * np.exp(rng.normal(0, 0.1))))
                            * np.sin(2 * np.pi * fr * fj * tt + ph)
                            * gate
                        )
                    # impact click: short low-passed noise burst
                    clk_n = int(0.0015 * sr)
                    sos = ssig.butter(2, min(cp["click_cut"], sr / 2 * 0.9), fs=sr, output="sos")
                    click = ssig.sosfilt(sos, rng.standard_normal(clk_n)) * np.hanning(clk_n)
                    o = int(onset * sr)
                    x[o : o + clk_n] += 2.0 * click[: max(0, min(clk_n, n - o))]
                    x *= 0.12 * amp * gain
                    x = np.fft.irfft(np.fft.rfft(x) * resp, n)  # device frequency response
                    if ir is not None:
                        x = ssig.fftconvolve(x, ir)[:n]
                    snr_db = rng.uniform(*cfg.snr_db_range) + snr_off
                    p_sig = float((x**2).mean()) + 1e-20
                    noise = _pink_like_noise(rng, n, alpha) * np.sqrt(p_sig / 10 ** (snr_db / 10))
                    y = np.clip(x + noise, -1.0, 1.0)
                    wave[idx] = y
                    records.append(
                        SampleRecord(
                            sample_id=f"{dataset_id}-{idx:06d}",
                            dataset_id=dataset_id,
                            dataset_version=cfg.version,
                            waveform_index=idx,
                            sample_rate=sr,
                            duration_s=n / sr,
                            label=class_names[c],
                            session_id=f"dev{d}-ses{s}",
                            device_id=f"dev{d}",
                            source_category=kb,
                            mic_category=mic,
                            environment={"name": env_name, "noise_alpha": alpha, "rt60_s": rt60},
                            noise={"snr_db": round(float(snr_db), 3), "type": f"1/f^{alpha}"},
                            provenance=f"{GENERATOR_VERSION}; seed={cfg.seed}",
                            lineage=[f"generate:{GENERATOR_VERSION}"],
                        )
                    )
                    idx += 1
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        version=cfg.version,
        name=f"synthetic seed={cfg.seed}",
        is_synthetic=True,
        sample_rate=sr,
        n_samples=len(records),
        n_waveform_samples=n,
        class_names=class_names,
        provenance={
            "generator": GENERATOR_VERSION,
            "config": cfg.model_dump(mode="json"),
            "note": "SYNTHETIC data; results do not transfer to real-world recordings",
        },
        consent=None,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    ds = Dataset(manifest, records, wave)
    ds.manifest = manifest.model_copy(update={"fingerprint": ds.compute_fingerprint()})
    return ds
