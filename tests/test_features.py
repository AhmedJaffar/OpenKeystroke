import numpy as np
import pytest

from openkeystroke.config import FeatureConfig, PreprocessConfig
from openkeystroke.dsp import PreprocessPipeline
from openkeystroke.errors import ConfigError
from openkeystroke.features import EXTRACTORS, FeatureCache, FeatureExtractor, make_extractor


@pytest.fixture(scope="module")
def wf(small_ds):
    return PreprocessPipeline(PreprocessConfig())(
        np.asarray(small_ds.waveforms), small_ds.sample_rate
    ).waveforms


@pytest.mark.parametrize("name", list(EXTRACTORS))
def test_extractor_shapes_finite_deterministic_batched(name, wf):
    ex = make_extractor(FeatureConfig(name=name), 16000)
    base = ex.compute(wf[:20])
    assert np.isfinite(base).all()
    assert np.array_equal(base, ex.compute(wf[:20]))
    assert np.allclose(ex.batched(wf[:20], batch_size=7), base, atol=1e-6)  # batching does not change values
    if ex.requires_fit:
        ex.fit(ex.compute(wf))
    out = ex.transform(wf[:20])
    assert out.shape[1:] == ex.output_shape(wf.shape[1])
    assert out.dtype == np.float32


def test_unfitted_extractor_refuses_transform(wf):
    with pytest.raises(ConfigError):
        make_extractor(FeatureConfig(name="pca_embedding"), 16000).transform(wf[:4])


def test_pca_fit_uses_only_training_data(wf):
    cfg = FeatureConfig(name="pca_embedding", pca_components=5)
    a, b = make_extractor(cfg, 16000), make_extractor(cfg, 16000)
    train = a.compute(wf[:40])
    a.fit(train)
    b.fit(train)
    # transforming different "test" data must not alter fitted state
    _ = a.transform(wf[40:60])
    assert np.array_equal(a.components_, b.components_) and np.array_equal(a.mean_, b.mean_)
    c = make_extractor(cfg, 16000).fit(make_extractor(cfg, 16000).compute(wf[:20]))
    assert not np.array_equal(c.components_, a.components_)


def test_extractor_serialization_roundtrip(wf, tmp_path):
    ex = make_extractor(FeatureConfig(name="pca_embedding", pca_components=4), 16000)
    ex.fit(ex.compute(wf[:30]))
    ex.save(tmp_path / "ex")
    ex2 = FeatureExtractor.load(tmp_path / "ex")
    assert np.allclose(ex.transform(wf[30:40]), ex2.transform(wf[30:40]))


def test_log_mel_matches_independent_computation(wf):
    from openkeystroke.dsp import ops

    ex = make_extractor(FeatureConfig(name="log_mel"), 16000)
    s = (
        np.abs(
            np.fft.rfft(
                np.pad(wf[0], (128, 128), mode="reflect")[None, 0:256] * np.hanning(257)[:-1], axis=-1
            )
        )
        ** 2
    )
    ref = ops.mel_filterbank(16000, 256, 40, 50.0) @ s[0]
    got = 10 ** (ex.compute(wf[:1])[0, :, 0] / 10)
    assert np.allclose(got, ref + 1e-10, rtol=1e-2, atol=1e-6)


def test_feature_cache_hits_and_atomic(wf, tmp_path):
    cache = FeatureCache(tmp_path / "c")
    ex = make_extractor(FeatureConfig(name="mfcc"), 16000)
    k = FeatureCache.key("fp1", {"p": 1}, ex.cache_id())
    calls = []
    f = lambda: (calls.append(1), ex.compute(wf[:10]))[1]  # noqa: E731
    a = np.array(cache.get_or_compute(k, f))
    b = np.array(cache.get_or_compute(k, f))
    assert len(calls) == 1 and cache.hits == 1 and cache.misses == 1 and np.array_equal(a, b)
    assert FeatureCache.key("fp2", {"p": 1}, ex.cache_id()) != k  # keyed by dataset fingerprint
    assert FeatureCache.key("fp1", {"p": 2}, ex.cache_id()) != k  # ... and preprocessing config
    other = make_extractor(FeatureConfig(name="mfcc", n_mfcc=10), 16000).cache_id()
    assert FeatureCache.key("fp1", {"p": 1}, other) != k  # ... and extractor config
    # corrupt cache file is recomputed, not trusted
    path = next((tmp_path / "c").rglob("*.npy"))
    path.write_bytes(b"garbage")
    cache.get_or_compute(k, f)
    assert len(calls) == 2
    assert not list((tmp_path / "c").rglob("*.tmp"))


def test_cold_and_warm_cache_return_identical_arrays(wf, tmp_path):
    """Regression: a cold compute and a warm read must be bitwise equal AND identically laid out in memory."""
    ex = make_extractor(FeatureConfig(name="log_mel"), 16000)
    cache = FeatureCache(tmp_path / "c")
    k = FeatureCache.key("fp", {"p": 1}, ex.cache_id())
    cold = cache.get_or_compute(k, lambda: ex.batched(wf[:20]))
    warm = cache.get_or_compute(k, lambda: ex.batched(wf[:20]))
    assert (
        np.array_equal(cold, warm)
        and cold.strides == warm.strides
        and cold.flags["C_CONTIGUOUS"]
        and warm.flags["C_CONTIGUOUS"]
    )
    assert ex.batched(wf[:20]).flags["C_CONTIGUOUS"]
