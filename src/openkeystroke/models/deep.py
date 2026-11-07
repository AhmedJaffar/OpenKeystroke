"""Deep classifiers wrapping the networks + trainer behind the shared protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ..config import TrainConfig
from ..errors import ModelError
from .base import BaseClassifier, TrainingHistory
from .nets import CNN1D, CNN2D, AudioTransformer, Normalized, ResCNN2D
from .trainer import fit_module, predict_logits, resolve_device


class TorchClassifier(BaseClassifier):
    family = "deep"
    kind = "base"

    def __init__(
        self,
        n_classes: int,
        input_shape: tuple[int, ...],
        params: dict[str, Any],
        seed: int,
        train_cfg: TrainConfig | None = None,
    ) -> None:
        super().__init__(n_classes, input_shape, params, seed)
        self.train_cfg = train_cfg or TrainConfig()
        torch.manual_seed(seed)
        net = self._build_net()
        self.module = Normalized(net, self._stat_shape())
        self.device = resolve_device(self.train_cfg.device)

    def _build_net(self) -> torch.nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    def _stat_shape(self) -> tuple[int, ...]:
        return (1,) * len(self.input_shape)

    def _fit_stats(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        *,
        progress=None,
        should_cancel=None,
        checkpoint_dir=None,
        resume=False,
    ) -> TrainingHistory:
        X_train = np.ascontiguousarray(X_train, dtype=np.float32)
        if X_train.shape[1:] != self.input_shape:
            raise ModelError(f"{self.kind}: expected input (N, {self.input_shape}); got {X_train.shape}")
        if not (resume and checkpoint_dir and (checkpoint_dir / "last.pt").is_file()):
            self._fit_stats(X_train)  # training-only normalisation statistics
        return fit_module(
            self.module,
            X_train,
            y_train,
            X_val,
            y_val,
            self.train_cfg,
            self.seed,
            self.n_classes,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            progress=progress,
            should_cancel=should_cancel,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.shape[1:] != self.input_shape:
            raise ModelError(f"{self.kind}: expected input (N, {self.input_shape}); got {X.shape}")
        lg = predict_logits(self.module, torch.as_tensor(X), self.device)
        return torch.softmax(lg.double(), dim=1).numpy()

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.module.net.parameters()))

    def size_bytes(self) -> int:
        return int(
            sum(p.numel() * p.element_size() for p in self.module.parameters())
            + sum(b.numel() * b.element_size() for b in self.module.buffers())
        )

    def save(self, directory: Path) -> dict[str, str]:
        import hashlib

        directory.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in self.module.state_dict().items()}, directory / "model.pt")
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "kind": self.kind,
                    "n_classes": self.n_classes,
                    "input_shape": list(self.input_shape),
                    "params": self.params,
                    "seed": self.seed,
                    "train": self.train_cfg.model_dump(mode="json"),
                }
            )
        )
        return {"model.pt": hashlib.sha256((directory / "model.pt").read_bytes()).hexdigest()}

    @classmethod
    def load(cls, directory: Path, expected_sha256: dict[str, str] | None = None) -> TorchClassifier:
        meta = json.loads((directory / "meta.json").read_text())
        obj = cls(
            meta["n_classes"],
            tuple(meta["input_shape"]),
            meta["params"],
            meta["seed"],
            TrainConfig.model_validate(meta["train"]),
        )
        sd = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        obj.module.load_state_dict(sd)
        obj.module.to(obj.device)
        return obj


class _WaveMixin:
    def _fit_stats(self, X: np.ndarray) -> None:
        self.module.mean.fill_(float(X.mean()))  # type: ignore[attr-defined]
        self.module.std.fill_(float(max(X.std(), 1e-8)))  # type: ignore[attr-defined]


class _SpecMixin:
    def _stat_shape(self) -> tuple[int, ...]:
        return (self.input_shape[0], 1)  # type: ignore[attr-defined]

    def _fit_stats(self, X: np.ndarray) -> None:
        self.module.mean.copy_(torch.as_tensor(X.mean(axis=(0, 2))[:, None]))  # type: ignore[attr-defined]
        self.module.std.copy_(torch.as_tensor(np.maximum(X.std(axis=(0, 2))[:, None], 1e-6)))  # type: ignore[attr-defined]


class CNN1DClassifier(_WaveMixin, TorchClassifier):
    kind, input_kind = "cnn1d", "waveform"

    def _build_net(self):
        p = self.params
        return CNN1D(
            self.n_classes,
            self.input_shape[0],
            tuple(p.get("channels", (16, 32, 64))),
            int(p.get("kernel", 9)),
            float(p.get("dropout", 0.2)),
        )


class CNN2DClassifier(_SpecMixin, TorchClassifier):
    kind, input_kind = "cnn2d", "spectrogram"

    def _build_net(self):
        p = self.params
        return CNN2D(
            self.n_classes,
            cast(tuple[int, int], self.input_shape),
            tuple(p.get("channels", (16, 32))),
            float(p.get("dropout", 0.2)),
        )


class ResNet2DClassifier(_SpecMixin, TorchClassifier):
    kind, input_kind = "resnet2d", "spectrogram"

    def _build_net(self):
        p = self.params
        return ResCNN2D(
            self.n_classes,
            cast(tuple[int, int], self.input_shape),
            int(p.get("width", 16)),
            int(p.get("blocks", 3)),
            float(p.get("dropout", 0.2)),
        )


class TransformerClassifier(_SpecMixin, TorchClassifier):
    kind, input_kind = "audio_transformer", "spectrogram"

    def _build_net(self):
        p = self.params
        return AudioTransformer(
            self.n_classes,
            cast(tuple[int, int], self.input_shape),
            int(p.get("d_model", 48)),
            int(p.get("heads", 4)),
            int(p.get("layers", 2)),
            float(p.get("dropout", 0.1)),
        )
