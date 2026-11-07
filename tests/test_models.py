import numpy as np
import pytest

from openkeystroke.config import FeatureConfig, ModelConfig, PreprocessConfig, SplitConfig, TrainConfig
from openkeystroke.data import make_split
from openkeystroke.dsp import PreprocessPipeline
from openkeystroke.errors import ModelError, TrainingError
from openkeystroke.features import make_extractor
from openkeystroke.models import MODEL_REGISTRY, create_model, input_kind, load_model


@pytest.fixture(scope="module")
def data(small_ds):
    wf = PreprocessPipeline(PreprocessConfig())(np.asarray(small_ds.waveforms), 16000).waveforms
    feats = {
        "vector": make_extractor(FeatureConfig(name="combined_stats"), 16000).batched(wf),
        "spectrogram": make_extractor(FeatureConfig(name="log_mel"), 16000).batched(wf),
        "waveform": wf,
    }
    sp = make_split(small_ds, "session_disjoint", SplitConfig(), 0)
    return feats, small_ds.labels, sp


def _fit(name, data, epochs=3, seed=0, **train):
    feats, y, sp = data
    X = feats[input_kind(name)]
    m = create_model(
        ModelConfig(name=name, training=TrainConfig(epochs=epochs, patience=5, batch_size=32, **train)),
        3,
        X.shape[1:],
        seed,
    )
    h = m.fit(X[sp.train], y[sp.train], X[sp.val], y[sp.val])
    return m, h, X, y, sp


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_every_model_trains_and_outputs_valid_probabilities(name, data):
    m, h, X, y, sp = _fit(name, data)
    p = m.predict_proba(X[sp.test])
    assert p.shape == (len(sp.test), 3) and np.allclose(p.sum(1), 1, atol=1e-5) and (p >= 0).all()
    assert h.n_train == len(sp.train) and h.epochs
    assert m.size_bytes() > 0


@pytest.mark.parametrize(
    "name",
    ["nearest_centroid", "logreg", "random_forest", "svm", "cnn1d", "cnn2d", "resnet2d", "audio_transformer"],
)
def test_save_load_roundtrip(name, data, tmp_path):
    m, _, X, y, sp = _fit(name, data, epochs=2)
    hashes = m.save(tmp_path / "m")
    m2 = load_model(name, tmp_path / "m", hashes)
    assert np.allclose(m.predict_proba(X[sp.test]), m2.predict_proba(X[sp.test]), atol=1e-5)


def test_pickled_estimator_refused_without_registry_hash(data, tmp_path):
    m, *_ = _fit("random_forest", data)
    hashes = m.save(tmp_path / "m")
    with pytest.raises(ModelError, match="refusing to unpickle"):
        load_model("random_forest", tmp_path / "m", None)
    (tmp_path / "m" / "estimator.joblib").write_bytes(b"tampered")
    with pytest.raises(ModelError, match="refusing to unpickle"):
        load_model("random_forest", tmp_path / "m", hashes)


def test_training_is_deterministic_for_same_seed(data):
    for name in ("cnn1d", "logreg", "random_forest"):
        a, *_ = _fit(name, data, epochs=3, seed=5)
        b, _, X, _, sp = _fit(name, data, epochs=3, seed=5)
        assert np.allclose(a.predict_proba(X[sp.test]), b.predict_proba(X[sp.test]), atol=1e-6), name
    c, _, X, _, sp = _fit("cnn1d", data, epochs=3, seed=6)
    a, *_ = _fit("cnn1d", data, epochs=3, seed=5)
    assert not np.allclose(a.predict_proba(X[sp.test]), c.predict_proba(X[sp.test]), atol=1e-4)


def test_normalisation_uses_training_statistics_only(data):
    feats, y, sp = data
    X = feats["spectrogram"]
    m = create_model(ModelConfig(name="cnn2d", training=TrainConfig(epochs=1)), 3, X.shape[1:], 0)
    m.fit(X[sp.train], y[sp.train])
    assert np.allclose(m.module.mean.numpy()[:, 0], X[sp.train].mean(axis=(0, 2)), atol=1e-4)
    cm = create_model(ModelConfig(name="logreg"), 3, feats["vector"].shape[1:], 0)
    cm.fit(feats["vector"][sp.train], y[sp.train])
    assert np.allclose(cm.scaler.mean, feats["vector"][sp.train].mean(0))


def test_checkpoint_and_resume(data, tmp_path):
    feats, y, sp = data
    X = feats["waveform"]
    ck = tmp_path / "ck"
    cfg2 = ModelConfig(
        name="cnn1d", training=TrainConfig(epochs=2, patience=9, scheduler="none", batch_size=32)
    )
    m = create_model(cfg2, 3, X.shape[1:], 0)
    m.fit(X[sp.train], y[sp.train], X[sp.val], y[sp.val], checkpoint_dir=ck)
    assert (ck / "last.pt").is_file()
    cfg4 = ModelConfig(
        name="cnn1d", training=TrainConfig(epochs=4, patience=9, scheduler="none", batch_size=32)
    )
    m2 = create_model(cfg4, 3, X.shape[1:], 0)
    h = m2.fit(X[sp.train], y[sp.train], X[sp.val], y[sp.val], checkpoint_dir=ck, resume=True)
    assert (
        h.resumed_from_epoch == 2
        and len(h.epochs) == 4
        and [e["epoch"] for e in h.epochs] == [1.0, 2.0, 3.0, 4.0]
    )


def test_nan_loss_recovery_then_failure(data):
    feats, y, sp = data
    X = feats["waveform"]
    bad = ModelConfig(
        name="cnn1d",
        training=TrainConfig(
            epochs=3, optimizer="sgd", lr=1e30, grad_clip=None, max_nan_retries=1, scheduler="none"
        ),
    )
    m = create_model(bad, 3, X.shape[1:], 0)
    with pytest.raises(TrainingError, match="non-finite"):
        m.fit(X[sp.train], y[sp.train], X[sp.val], y[sp.val])


def test_early_stopping_and_schedulers(data):
    for sched in ("none", "cosine", "step", "plateau"):
        m, h, *_ = _fit("cnn1d", data, epochs=4, scheduler=sched)
        assert h.epochs[0]["lr"] > 0
    feats, y, sp = data
    X = feats["waveform"]
    m = create_model(
        ModelConfig(name="cnn1d", training=TrainConfig(epochs=60, patience=2, lr=1e-2)), 3, X.shape[1:], 0
    )
    h = m.fit(X[sp.train], y[sp.train], X[sp.val], y[sp.val])
    assert h.stopped_early and len(h.epochs) < 60


def test_cancellation_stops_training(data):
    from openkeystroke.errors import CancelledError

    feats, y, sp = data
    X = feats["waveform"]
    m = create_model(ModelConfig(name="cnn1d", training=TrainConfig(epochs=10)), 3, X.shape[1:], 0)
    with pytest.raises(CancelledError):
        m.fit(X[sp.train], y[sp.train], should_cancel=lambda: True)


def test_shape_mismatch_and_unknown_model_raise(data):
    feats, y, sp = data
    m = create_model(ModelConfig(name="logreg"), 3, (10,), 0)
    with pytest.raises(ModelError):
        m.fit(feats["vector"][sp.train], y[sp.train])
    with pytest.raises(ModelError):
        create_model(ModelConfig(name="nope"), 3, (4,), 0)
    with pytest.raises(ModelError):
        create_model(ModelConfig(name="logreg"), 3, (4, 4), 0)


def test_cuda_request_without_cuda_fails_clearly(data):
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    with pytest.raises(ModelError, match="CUDA"):
        create_model(ModelConfig(name="cnn1d", training=TrainConfig(device="cuda")), 3, (1600,), 0)


def test_results_independent_of_input_memory_layout(data):
    """Regression: a non-contiguous (transposed-view) feature array must give the same model as a contiguous copy."""
    feats, y, sp = data
    X = feats["spectrogram"]
    Xf = np.asfortranarray(X)  # same values, different memory layout
    assert np.array_equal(X, Xf) and not Xf.flags["C_CONTIGUOUS"]
    outs = []
    for arr in (X, Xf):
        m = create_model(
            ModelConfig(name="cnn2d", training=TrainConfig(epochs=4, patience=9, batch_size=32)),
            3,
            arr.shape[1:],
            0,
        )
        m.fit(arr[sp.train], y[sp.train], arr[sp.val], y[sp.val])
        outs.append(m.predict_proba(arr[sp.test]))
    assert np.array_equal(outs[0], outs[1])
