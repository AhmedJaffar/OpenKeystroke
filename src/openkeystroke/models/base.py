"""Shared classifier protocol so classical and deep models are interchangeable in the experiment engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrainingHistory:
    epochs: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int | None = None
    stopped_early: bool = False
    resumed_from_epoch: int | None = None
    nan_recoveries: int = 0
    device: str = "cpu"
    seconds: float = 0.0
    n_train: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressFn = Callable[[int, int, dict[str, float]], None]


class BaseClassifier:
    family = "classical"
    input_kind = "vector"

    def __init__(
        self, n_classes: int, input_shape: tuple[int, ...], params: dict[str, Any], seed: int
    ) -> None:
        self.n_classes = n_classes
        self.input_shape = tuple(input_shape)
        self.params = dict(params)
        self.seed = seed
        self.name = type(self).__name__

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        *,
        progress: ProgressFn | None = None,
        should_cancel: Callable[[], bool] | None = None,
        checkpoint_dir: Path | None = None,
        resume: bool = False,
    ) -> TrainingHistory:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, directory: Path) -> dict[str, str]:  # pragma: no cover - abstract
        raise NotImplementedError

    @classmethod
    def load(
        cls, directory: Path, expected_sha256: dict[str, str] | None = None
    ) -> BaseClassifier:  # pragma: no cover
        raise NotImplementedError

    def num_parameters(self) -> int:
        return 0

    def size_bytes(self) -> int:
        return 0
