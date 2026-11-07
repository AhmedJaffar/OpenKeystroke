import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from openkeystroke.config import PreprocessConfig
from openkeystroke.dsp import PreprocessPipeline, ops
from openkeystroke.errors import DSPError

SR = 16000


def tone(f, n=SR // 4, sr=SR, amp=1.0):
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * f * t))[None, :].astype(np.float32)


def test_stft_istft_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 1600)).astype(np.float64)
    s = ops.stft(x, 256, 64)
    y = ops.istft(s, 256, 64, length=1600)
    assert s.shape == (3, 129, 26)
    assert np.max(np.abs(x - y)) < 1e-9


def test_stft_peak_at_tone_frequency():
    s = ops.stft(tone(2000), 512, 128)
    mag = np.abs(s).mean(axis=2)[0]
    freqs = np.linspace(0, SR / 2, 257)
    assert abs(freqs[mag.argmax()] - 2000) <= SR / 512


def test_parseval_energy():
    x = np.random.default_rng(1).standard_normal((1, 1024))
    X = np.fft.rfft(x)
    e_t = (x**2).sum()
    e_f = (np.abs(X[0, 0]) ** 2 + 2 * (np.abs(X[0, 1:-1]) ** 2).sum() + np.abs(X[0, -1]) ** 2) / 1024
    assert e_t == pytest.approx(e_f, rel=1e-9)


def test_lowpass_attenuates_high_tone_keeps_low():
    lo, hi = tone(300), tone(5000)
    inner = slice(400, -400)  # filtfilt end effects on an arbitrarily-truncated tone are inherent
    assert ops.rms_energy(ops.lowpass(lo, SR, 1000)[:, inner])[0] > 0.99 * ops.rms_energy(lo[:, inner])[0]
    assert ops.rms_energy(ops.lowpass(hi, SR, 1000)[:, inner])[0] < 1e-4 * ops.rms_energy(hi[:, inner])[0]


def test_bandpass_selects_band():
    x = tone(500) + tone(3000)
    y = ops.bandpass(x, SR, 2500, 3500)
    s = np.abs(np.fft.rfft(y[0]))
    f = np.fft.rfftfreq(y.shape[1], 1 / SR)
    assert abs(f[s.argmax()] - 3000) < 20
    assert s[np.argmin(abs(f - 500))] < 0.01 * s.max()


def test_resample_preserves_tone_frequency():
    y = ops.resample(tone(1000), SR, 8000)
    assert y.shape[1] == SR // 4 // 2
    s = np.abs(np.fft.rfft(y[0]))
    f = np.fft.rfftfreq(y.shape[1], 1 / 8000)
    assert abs(f[s.argmax()] - 1000) < 10


def test_normalize_and_dc():
    x = tone(440, amp=0.3) + 0.5
    assert abs(ops.remove_dc(x).mean()) < 1e-6
    assert np.abs(ops.normalize(x, "peak")).max() == pytest.approx(1.0, rel=1e-5)
    assert ops.rms_energy(ops.normalize(x, "rms"))[0] == pytest.approx(1.0, rel=1e-4)


def test_mel_filterbank_properties():
    fb = ops.mel_filterbank(SR, 256, 40, 50, None)
    assert fb.shape == (40, 129)
    assert (fb >= 0).all() and (fb.sum(axis=1) > 0).all()
    centers = fb.argmax(axis=1)
    assert (np.diff(centers) >= 0).all()


def test_spectral_descriptors_known_values():
    freqs = np.linspace(0, SR / 2, 129)
    mag = np.zeros((1, 129, 1))
    mag[0, 32, 0] = 1.0
    assert ops.spectral_centroid(mag, freqs)[0, 0] == pytest.approx(freqs[32])
    assert ops.spectral_bandwidth(mag, freqs)[0, 0] == pytest.approx(0, abs=1e-9)
    assert ops.spectral_rolloff(mag, freqs)[0, 0] == pytest.approx(freqs[32])
    flat = np.ones((1, 129, 5))
    assert ops.spectral_flux(flat).sum() == 0


def test_zcr_and_rms():
    assert ops.zero_crossing_rate(tone(1000))[0] == pytest.approx(2 * 1000 / SR, rel=0.05)
    assert ops.rms_energy(tone(1000, amp=2.0))[0] == pytest.approx(2 / np.sqrt(2), rel=1e-3)


def test_transient_detection_and_snr():
    rng = np.random.default_rng(3)
    x = 0.001 * rng.standard_normal((1, 3200))
    x[0, 1000:1100] += 0.5 * rng.standard_normal(100)
    onset = ops.detect_transient(x, SR)[0]
    assert 900 <= onset <= 1010
    assert ops.detect_transient(0.001 * rng.standard_normal((1, 3200)), SR)[0] == -1
    assert ops.estimate_snr_db(x, SR)[0] > 20
    assert ops.noise_floor(x, SR)[0] == pytest.approx(0.001, rel=0.5)


def test_align_to_transient_moves_onset():
    rng = np.random.default_rng(4)
    x = 0.001 * rng.standard_normal((2, 3200))
    x[0, 1500:1600] += 0.5
    x[1, 400:500] += 0.5
    y = ops.align_to_transient(x, SR, 0.05)
    assert np.abs(ops.detect_transient(y, SR) - 800).max() <= 16


def test_invalid_inputs_fail_loudly():
    with pytest.raises(DSPError):
        ops.as_batch(np.zeros((2, 3, 100)))
    with pytest.raises(DSPError):
        ops.as_batch(np.array([[np.nan] * 10]))
    with pytest.raises(DSPError):
        ops.lowpass(tone(100), SR, 9000)
    with pytest.raises(DSPError):
        ops.bandpass(tone(100), SR, 3000, 1000)
    with pytest.raises(DSPError):
        ops.mel_spectrogram(np.zeros((1, 100, 3)), ops.mel_filterbank(SR, 256, 10))
    with pytest.raises(DSPError):
        ops.resample(tone(10), SR, 0)


def test_pipeline_lineage_and_resample():
    cfg = PreprocessConfig(target_sample_rate=8000, lowpass_hz=3000, normalize="peak")
    out = PreprocessPipeline(cfg)(tone(500), SR)
    assert out.sample_rate == 8000
    assert [s["op"] for s in out.lineage] == ["input", "resample", "remove_dc", "lowpass", "normalize"]
    assert out.waveforms.dtype == np.float32


@settings(max_examples=25, deadline=None)
@given(hnp.arrays(np.float64, (2, 600), elements=st.floats(-1, 1, allow_nan=False, width=32)))
def test_istft_inverts_stft_property(x):
    y = ops.istft(ops.stft(x, 128, 32), 128, 32, length=600)
    assert np.max(np.abs(x - y)) < 1e-8


@settings(max_examples=25, deadline=None)
@given(hnp.arrays(np.float64, (3, 200), elements=st.floats(-5, 5, allow_nan=False, width=32)))
def test_peak_normalize_bounds_property(x):
    assert np.abs(ops.normalize(x, "peak")).max() <= 1.0 + 1e-9
