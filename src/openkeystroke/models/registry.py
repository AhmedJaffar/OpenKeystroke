"""Model registry: name -> class, input kind, family."""

from __future__ import annotations

from typing import Any

from ..config import ModelConfig
from ..errors import ModelError
from .base import BaseClassifier
from .classical import SVM, LogReg, NearestCentroid, RandomForest
from .deep import CNN1DClassifier, CNN2DClassifier, ResNet2DClassifier, TorchClassifier, TransformerClassifier

MODEL_REGISTRY: dict[str, type[BaseClassifier]] = {
    "nearest_centroid": NearestCentroid,
    "logreg": LogReg,
    "random_forest": RandomForest,
    "svm": SVM,
    "cnn1d": CNN1DClassifier,
    "cnn2d": CNN2DClassifier,
    "resnet2d": ResNet2DClassifier,
    "audio_transformer": TransformerClassifier,
}


def _cls(name: str) -> type[BaseClassifier]:
    if name not in MODEL_REGISTRY:
        raise ModelError(f"unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def input_kind(name: str) -> str:
    return _cls(name).input_kind


def create_model(cfg: ModelConfig, n_classes: int, input_shape: tuple[int, ...], seed: int) -> BaseClassifier:
    cls: Any = _cls(cfg.name)
    if issubclass(cls, TorchClassifier):
        return cls(n_classes, input_shape, cfg.params, seed, cfg.training)
    return cls(n_classes, input_shape, cfg.params, seed)


def load_model(name: str, directory, expected_sha256: dict[str, str] | None = None) -> BaseClassifier:
    return _cls(name).load(directory, expected_sha256)
