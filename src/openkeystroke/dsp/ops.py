"""Batched, vectorized signal-processing primitives (NumPy/SciPy). All inputs are (N, T) arrays."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy import fft as sfft
from scipy import signal as ssig

from ..errors import DSPError

EPS = 1e-10


def as_batch(x: np.ndarray, *, name: str = "waveforms") -> np.ndarray:
    """Validate and return a finite float array of shape (N, T). 1-D input is treated as one waveform."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise DSPError(f"{name} must have shape (N, T) mono audio; got shape {x.shape}")
    if x.shape[1] < 2:
        raise DSPError(f"{name} too short: {x.shape[1]} samples")
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise DSPError(f"{name} contain NaN or infinite values")
    return x


def check_sr(sr: int, *, min_hz: int = 1000) -> int:
    if not isinstance(sr, (int, np.integer)) or sr < min_hz:
        raise DSPError(f"invalid sample rate: {sr!r}")
    return int(sr)


def remove_dc(x: np.ndarray) -> np.ndarray:
    x = as_batch(x)
    return x - x.mean(axis=-1, keepdims=True)


def normalize(x: np.ndarray, mode: str = "peak") -> np.ndarray:
    x = as_batch(x)
    if mode == "none":
        return x
    if mode == "peak":
        scale = np.abs(x).max(axis=-1, keepdims=True)
    elif mode == "rms":
        scale = np.sqrt((x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True))
    else:
        raise DSPError(f"unknown normalization mode '{mode}'")
    return (x / np.maximum(scale, EPS)).astype(x.dtype)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    x = as_batch(x)
    sr_in, sr_out = check_sr(sr_in), check_sr(sr_out)
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return ssig.resample_poly(x, sr_out // g, sr_in // g, axis=-1).astype(x.dtype)


def _check_cutoff(sr: int, *freqs: float) -> None:
    nyq = sr / 2
    for f in freqs:
        if not (0 < f < nyq):
            raise DSPError(f"cutoff {f} Hz must be in (0, {nyq}) for sample rate {sr}")


def _padlen(x: np.ndarray, sr: int, f_low: float) -> int:
    return int(min(x.shape[1] - 1, max(15, 6 * sr / f_low)))


def bandpass(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    x = as_batch(x)
    _check_cutoff(sr, lo, hi)
    if lo >= hi:
        raise DSPError("bandpass requires lo < hi")
    sos = ssig.butter(order, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return ssig.sosfiltfilt(sos, x, axis=-1, padlen=_padlen(x, sr, lo)).astype(x.dtype)


def lowpass(x: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    x = as_batch(x)
    _check_cutoff(sr, cutoff)
    sos = ssig.butter(order, cutoff, btype="lowpass", fs=sr, output="sos")
    return ssig.sosfiltfilt(sos, x, axis=-1, padlen=_padlen(x, sr, cutoff)).astype(x.dtype)


def highpass(x: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    x = as_batch(x)
    _check_cutoff(sr, cutoff)
    sos = ssig.butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    return ssig.sosfiltfilt(sos, x, axis=-1, padlen=_padlen(x, sr, cutoff)).astype(x.dtype)


def get_window(name: str, n: int) -> np.ndarray:
    try:
        return ssig.get_window(name, n, fftbins=True)
    except ValueError as exc:
        raise DSPError(f"unknown window '{name}'") from exc


def stft(
    x: np.ndarray, n_fft: int = 256, hop: int = 64, window: str = "hann", center: bool = True
) -> np.ndarray:
    """Complex STFT, shape (N, n_fft//2+1, frames). Reflect-padded when center=True."""
    x = as_batch(x)
    if hop <= 0 or hop > n_fft:
        raise DSPError("hop must be in (0, n_fft]")
    if center:
        if x.shape[1] <= n_fft // 2:
            raise DSPError("signal shorter than n_fft/2; cannot center-pad")
        extra = (-x.shape[1]) % hop  # so the final frame centre is >= the last sample (exact istft coverage)
        if x.shape[1] + extra <= n_fft // 2 + extra and extra >= x.shape[1]:
            raise DSPError("signal too short for centered STFT with this hop")
        x = np.pad(x, ((0, 0), (n_fft // 2, n_fft // 2 + extra)), mode="reflect")
    if x.shape[1] < n_fft:
        raise DSPError(f"signal ({x.shape[1]}) shorter than n_fft ({n_fft})")
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft, axis=-1)[:, ::hop, :]
    win = get_window(window, n_fft).astype(x.dtype)
    spec = sfft.rfft(frames * win, axis=-1)
    return np.ascontiguousarray(spec.transpose(0, 2, 1))


def istft(
    spec: np.ndarray,
    n_fft: int = 256,
    hop: int = 64,
    window: str = "hann",
    length: int | None = None,
    center: bool = True,
) -> np.ndarray:
    """Inverse STFT via weighted overlap-add (least-squares window normalisation)."""
    if spec.ndim != 3 or spec.shape[1] != n_fft // 2 + 1:
        raise DSPError(f"spec must have shape (N, {n_fft // 2 + 1}, frames); got {spec.shape}")
    n, _, t = spec.shape
    win = get_window(window, n_fft)
    frames = sfft.irfft(spec.transpose(0, 2, 1), n=n_fft, axis=-1) * win
    out_len = n_fft + hop * (t - 1)
    out = np.zeros((n, out_len))
    norm = np.zeros(out_len)
    for i in range(t):
        s = i * hop
        out[:, s : s + n_fft] += frames[:, i]
        norm[s : s + n_fft] += win**2
    out /= np.maximum(norm, 1e-8)
    if center:
        out = out[:, n_fft // 2 : out_len - n_fft // 2]
    if length is not None:
        out = out[:, :length]
        if out.shape[1] < length:
            out = np.pad(out, ((0, 0), (0, length - out.shape[1])))
    return out.astype(np.float32 if spec.dtype == np.complex64 else np.float64)


def power_spectrogram(spec: np.ndarray) -> np.ndarray:
    return (spec.real**2 + spec.imag**2).astype(spec.real.dtype)


def log_magnitude(spec: np.ndarray, eps: float = EPS) -> np.ndarray:
    """20*log10(|S|) with a floor (dB)."""
    return (20.0 * np.log10(np.maximum(np.abs(spec), eps))).astype(spec.real.dtype)


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    return np.asarray(2595.0 * np.log10(1.0 + np.asarray(f) / 700.0))


def _mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    return np.asarray(700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0))


def mel_filterbank(
    sr: int, n_fft: int, n_mels: int, f_min: float = 0.0, f_max: float | None = None
) -> np.ndarray:
    """Triangular HTK-mel filterbank, area-normalised. Shape (n_mels, n_fft//2+1)."""
    f_max = sr / 2 if f_max is None else f_max
    if not (0 <= f_min < f_max <= sr / 2):
        raise DSPError(f"invalid mel range [{f_min}, {f_max}] for sr={sr}")
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2))
    fb = np.zeros((n_mels, freqs.size))
    for i in range(n_mels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / max(ce - lo, EPS)
        down = (hi - freqs) / max(hi - ce, EPS)
        fb[i] = np.maximum(0, np.minimum(up, down))
        fb[i] *= 2.0 / max(hi - lo, EPS)
    return fb.astype(np.float32)


def mel_spectrogram(power: np.ndarray, fb: np.ndarray) -> np.ndarray:
    if power.shape[1] != fb.shape[1]:
        raise DSPError(f"power has {power.shape[1]} bins but filterbank expects {fb.shape[1]}")
    return np.einsum("mf,nft->nmt", fb, power, optimize=True).astype(power.dtype)


def mfcc_from_logmel(log_mel_db: np.ndarray, n_mfcc: int) -> np.ndarray:
    return sfft.dct(log_mel_db, type=2, axis=1, norm="ortho")[:, :n_mfcc]


# ---- frame-level spectral descriptors; magnitude spec (N, F, T), freqs (F,)
def spectral_centroid(mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    tot = np.maximum(mag.sum(axis=1), EPS)
    return (freqs[None, :, None] * mag).sum(axis=1) / tot


def spectral_bandwidth(mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    c = spectral_centroid(mag, freqs)
    tot = np.maximum(mag.sum(axis=1), EPS)
    var = (((freqs[None, :, None] - c[:, None, :]) ** 2) * mag).sum(axis=1) / tot
    return np.sqrt(np.maximum(var, 0))


def spectral_rolloff(mag: np.ndarray, freqs: np.ndarray, pct: float = 0.85) -> np.ndarray:
    if not 0 < pct < 1:
        raise DSPError("rolloff percentage must be in (0, 1)")
    cum = np.cumsum(mag, axis=1)
    thresh = pct * cum[:, -1:, :]
    idx = (cum < thresh).sum(axis=1).clip(0, len(freqs) - 1)
    return freqs[idx]


def spectral_flux(mag: np.ndarray) -> np.ndarray:
    """Half-wave-rectified L2 flux between consecutive frames; first frame is 0. Shape (N, T)."""
    d = np.diff(mag, axis=2, prepend=mag[:, :, :1])
    return np.sqrt((np.maximum(d, 0) ** 2).sum(axis=1))


def zero_crossing_rate(x: np.ndarray) -> np.ndarray:
    """Fraction of adjacent sample pairs with a sign change. Shape (N,)."""
    x = as_batch(x)
    s = np.signbit(x)
    return (s[:, 1:] != s[:, :-1]).mean(axis=1)


def rms_energy(x: np.ndarray) -> np.ndarray:
    x = as_batch(x)
    return np.sqrt((x.astype(np.float64) ** 2).mean(axis=1))


def frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    x = as_batch(x)
    if x.shape[1] < frame:
        raise DSPError("signal shorter than frame length")
    fr = np.lib.stride_tricks.sliding_window_view(x.astype(np.float64), frame, axis=-1)[:, ::hop, :]
    return np.sqrt((fr**2).mean(axis=-1))


# ---- laboratory transient detection / noise & SNR estimation
def _floor(env: np.ndarray, pct: float = 20.0) -> np.ndarray:
    """Low percentile of frame RMS, ignoring digital silence (exact zeros from padding)."""
    e = np.where(env > 1e-9, env, np.nan)
    with np.errstate(all="ignore"):
        f = np.nanpercentile(e, pct, axis=1)
    return np.maximum(np.nan_to_num(f, nan=EPS), EPS)


def noise_floor(x: np.ndarray, sr: int, frame_ms: float = 2.0, pct: float = 20.0) -> np.ndarray:
    """Noise-floor RMS estimate: low percentile of short-frame RMS. Shape (N,)."""
    fl = max(int(sr * frame_ms / 1000), 4)
    env = frame_rms(x, fl, max(fl // 2, 1))
    return _floor(env, pct)


def detect_transient(x: np.ndarray, sr: int, frame_ms: float = 1.0, k: float = 6.0) -> np.ndarray:
    """Index (samples) of the first frame whose RMS exceeds k * noise floor; -1 if none. Shape (N,)."""
    x = as_batch(x)
    fl = max(int(sr * frame_ms / 1000), 2)
    hop = max(fl // 2, 1)
    env = frame_rms(x, fl, hop)
    floor = _floor(env)
    above = env > (k * floor)[:, None]
    first = np.where(above.any(axis=1), above.argmax(axis=1), -1)
    return np.where(first >= 0, first * hop, -1)


def estimate_snr_db(x: np.ndarray, sr: int, frame_ms: float = 2.0) -> np.ndarray:
    """Blind SNR (dB): power of the loudest frames vs the noise-floor power. Shape (N,)."""
    fl = max(int(sr * frame_ms / 1000), 4)
    env = frame_rms(x, fl, max(fl // 2, 1))
    floor = _floor(env)
    peak = np.maximum(env.max(axis=1), EPS)
    sig_p = np.maximum(peak**2 - floor**2, EPS)
    return 10.0 * np.log10(sig_p / floor**2)


def align_to_transient(x: np.ndarray, sr: int, pre_s: float) -> np.ndarray:
    """Shift each waveform (zero-padded) so the detected transient sits at `pre_s` seconds."""
    x = as_batch(x)
    onset = detect_transient(x, sr)
    target = int(pre_s * sr)
    out = np.zeros_like(x)
    t = x.shape[1]
    for i, o in enumerate(onset):
        shift = 0 if o < 0 else target - int(o)
        if shift >= 0:
            out[i, shift:] = x[i, : t - shift]
        else:
            out[i, : t + shift] = x[i, -shift:]
    return out
