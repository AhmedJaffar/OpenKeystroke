import pytest

from openkeystroke.config import DatasetRef, ExperimentConfig, ModelConfig, SyntheticConfig, TrainConfig
from openkeystroke.data import generate_synthetic
from openkeystroke.registry import Workspace

SMALL = SyntheticConfig(
    n_classes=3, n_devices=3, sessions_per_device=2, samples_per_class_per_session=5, seed=3
)


@pytest.fixture(scope="session")
def small_cfg():
    return SMALL


@pytest.fixture(scope="session")
def small_ds():
    return generate_synthetic(SMALL)


@pytest.fixture()
def ws(tmp_path):
    return Workspace(tmp_path / "home")


def make_cfg(kind="single", models=None, seeds=None, **kw):
    return ExperimentConfig(
        name=f"test-{kind}",
        kind=kind,
        dataset=DatasetRef(synthetic=SMALL),
        models=models or [ModelConfig(name="logreg")],
        seeds=seeds or ([0] if kind == "single" else [0, 1, 2]),
        **kw,
    )


def tiny_cnn(epochs=3):
    return ModelConfig(name="cnn1d", training=TrainConfig(epochs=epochs, patience=3, batch_size=32))
