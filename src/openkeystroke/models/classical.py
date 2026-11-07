"""Classical baselines (scikit-learn / NumPy). Feature standardisation is fit on the training partition only."""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ..errors import ModelError
from .base import BaseClassifier, TrainingHistory


class _Scaler:
    def __init__(self) -> None:
        self.mean: np.ndarray = np.zeros(1)
        self.scale: np.ndarray = np.ones(1)

    def fit(self, X: np.ndarray) -> _Scaler:
        self.mean = X.mean(axis=0)
        s = X.std(axis=0)
        self.scale = np.where(s < 1e-8, 1.0, s)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.scale


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _acc(model: BaseClassifier, X: np.ndarray, y: np.ndarray) -> float:
    return float((model.predict(X) == y).mean())


class _Classical(BaseClassifier):
    family = "classical"
    input_kind = "vector"
    kind = "base"

    def __init__(
        self, n_classes: int, input_shape: tuple[int, ...], params: dict[str, Any], seed: int
    ) -> None:
        super().__init__(n_classes, input_shape, params, seed)
        if len(input_shape) != 1:
            raise ModelError(f"{self.kind} expects vector features; got input shape {input_shape}")
        self.scaler = _Scaler()
        self.est: Any = None
        self.class_weight = params.get("class_weight", "balanced")

    def _fit_est(self, Xs: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover - abstract
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
        t0 = time.perf_counter()
        X_train = np.asarray(X_train, dtype=np.float64)
        if X_train.ndim != 2 or X_train.shape[1:] != self.input_shape:
            raise ModelError(f"expected X of shape (N, {self.input_shape[0]}); got {X_train.shape}")
        if len(np.unique(y_train)) < 2:
            raise ModelError("training labels contain fewer than 2 classes")
        self.scaler.fit(X_train)  # training-only statistics
        self._fit_est(self.scaler(X_train), np.asarray(y_train))
        h = TrainingHistory(device="cpu", n_train=len(y_train), seconds=time.perf_counter() - t0)
        rec = {"epoch": 1.0, "train_acc": _acc(self, X_train, y_train)}
        if X_val is not None and y_val is not None and len(y_val):
            rec["val_acc"] = _acc(self, X_val, y_val)
        h.epochs.append(rec)
        h.best_epoch = 1
        return h

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._proba_scaled(self.scaler(np.asarray(X)))
        out = np.zeros((p.shape[0], self.n_classes))
        out[:, self._classes()] = p
        return out

    def _classes(self) -> np.ndarray:
        return np.asarray(self.est.classes_ if hasattr(self.est, "classes_") else np.arange(self.n_classes))

    def _proba_scaled(self, Xs: np.ndarray) -> np.ndarray:
        return self.est.predict_proba(Xs)

    def num_parameters(self) -> int:
        return 0

    def size_bytes(self) -> int:
        buf = io.BytesIO()
        joblib.dump(self.est, buf)
        return buf.getbuffer().nbytes

    def _meta(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_classes": self.n_classes,
            "input_shape": list(self.input_shape),
            "params": self.params,
            "seed": self.seed,
        }

    def save(self, directory: Path) -> dict[str, str]:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "scaler.npz", mean=self.scaler.mean, scale=self.scaler.scale)
        (directory / "meta.json").write_text(json.dumps(self._meta()))
        return {"scaler.npz": _sha(directory / "scaler.npz"), **self._save_est(directory)}

    def _save_est(self, directory: Path) -> dict[str, str]:
        joblib.dump(self.est, directory / "estimator.joblib")  # integrity-checked on load; see load()
        return {"estimator.joblib": _sha(directory / "estimator.joblib")}

    @classmethod
    def load(cls, directory: Path, expected_sha256: dict[str, str] | None = None) -> _Classical:
        meta = json.loads((directory / "meta.json").read_text())
        obj = cls(meta["n_classes"], tuple(meta["input_shape"]), meta["params"], meta["seed"])
        with np.load(directory / "scaler.npz", allow_pickle=False) as z:
            obj.scaler.mean, obj.scaler.scale = z["mean"], z["scale"]
        obj._load_est(directory, expected_sha256)
        return obj

    def _load_est(self, directory: Path, expected: dict[str, str] | None) -> None:
        p = directory / "estimator.joblib"
        # joblib uses pickle: only load files whose hash was recorded by this platform's registry.
        if not expected or expected.get("estimator.joblib") != _sha(p):
            raise ModelError(
                "refusing to unpickle estimator: missing or mismatching registry sha256 (untrusted artifact?)"
            )
        self.est = joblib.load(p)


class NearestCentroid(_Classical):
    """Shared-isotropic-Gaussian nearest centroid; probabilities from softmax(-d^2 / 2s^2)."""

    kind = "nearest_centroid"

    def _fit_est(self, Xs: np.ndarray, y: np.ndarray) -> None:
        classes = np.unique(y)
        self.centroids = np.stack([Xs[y == c].mean(axis=0) for c in classes])
        resid = np.concatenate([Xs[y == c] - self.centroids[i] for i, c in enumerate(classes)])
        self.s2 = float(max((resid**2).mean(), 1e-6))
        self.classes_ = classes
        self.est = self

    def _proba_scaled(self, Xs: np.ndarray) -> np.ndarray:
        d2 = ((Xs[:, None, :] - self.centroids[None]) ** 2).sum(axis=2)
        z = -d2 / (2 * self.s2)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def size_bytes(self) -> int:
        return int(self.centroids.nbytes)

    def num_parameters(self) -> int:
        return int(self.centroids.size)

    def _save_est(self, directory: Path) -> dict[str, str]:
        np.savez(directory / "centroid.npz", centroids=self.centroids, s2=self.s2, classes=self.classes_)
        return {"centroid.npz": _sha(directory / "centroid.npz")}

    def _load_est(self, directory: Path, expected: dict[str, str] | None) -> None:
        with np.load(directory / "centroid.npz", allow_pickle=False) as z:
            self.centroids, self.s2, self.classes_ = z["centroids"], float(z["s2"]), z["classes"]
        self.est = self


class LogReg(_Classical):
    kind = "logreg"

    def _fit_est(self, Xs: np.ndarray, y: np.ndarray) -> None:
        self.est = LogisticRegression(
            C=float(self.params.get("C", 1.0)),
            max_iter=int(self.params.get("max_iter", 2000)),
            class_weight=self.class_weight if self.class_weight != "none" else None,
        )
        self.est.fit(Xs, y)

    def num_parameters(self) -> int:
        return int(self.est.coef_.size + self.est.intercept_.size)

    def size_bytes(self) -> int:
        return int(self.est.coef_.nbytes + self.est.intercept_.nbytes)

    def _save_est(self, directory: Path) -> dict[str, str]:  # plain arrays, no pickle
        np.savez(
            directory / "logreg.npz",
            coef=self.est.coef_,
            intercept=self.est.intercept_,
            classes=self.est.classes_,
        )
        return {"logreg.npz": _sha(directory / "logreg.npz")}

    def _load_est(self, directory: Path, expected: dict[str, str] | None) -> None:
        with np.load(directory / "logreg.npz", allow_pickle=False) as z:
            est = LogisticRegression()
            est.coef_, est.intercept_, est.classes_ = z["coef"], z["intercept"], z["classes"]
        self.est = est


class RandomForest(_Classical):
    kind = "random_forest"

    def _fit_est(self, Xs: np.ndarray, y: np.ndarray) -> None:
        self.est = RandomForestClassifier(
            n_estimators=int(self.params.get("n_estimators", 150)),
            max_depth=self.params.get("max_depth"),
            min_samples_leaf=int(self.params.get("min_samples_leaf", 1)),
            class_weight=self.class_weight if self.class_weight != "none" else None,
            random_state=self.seed,
            n_jobs=1,
        )
        self.est.fit(Xs, y)


class SVM(_Classical):
    kind = "svm"

    def _fit_est(self, Xs: np.ndarray, y: np.ndarray) -> None:
        base = SVC(
            C=float(self.params.get("C", 1.0)),
            kernel=str(self.params.get("kernel", "rbf")),
            gamma=self.params.get("gamma", "scale"),
            class_weight=self.class_weight if self.class_weight != "none" else None,
            random_state=self.seed,
        )
        # Platt-style calibrated probabilities (cross-fitted on the training partition only)
        self.est = CalibratedClassifierCV(base, method="sigmoid", cv=3, ensemble=False)
        self.est.fit(Xs, y)
