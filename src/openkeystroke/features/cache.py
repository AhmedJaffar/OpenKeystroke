"""On-disk feature cache keyed by dataset fingerprint + waveform-processing config + extractor id/version."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..config import canonical_json


class FeatureCache:
    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root) if root else None
        self.hits = 0
        self.misses = 0
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(dataset_fingerprint: str, waveform_key: object, extractor_id: str) -> str:
        return hashlib.sha256(
            canonical_json({"ds": dataset_fingerprint, "wave": waveform_key, "ex": extractor_id}).encode()
        ).hexdigest()

    def _path(self, key: str) -> Path:
        assert self.root is not None
        return self.root / key[:2] / f"{key}.npy"

    def get_or_compute(self, key: str, fn: Callable[[], np.ndarray]) -> np.ndarray:
        if self.root is None:
            self.misses += 1
            return fn()
        p = self._path(key)
        if p.is_file():
            try:
                arr = np.load(p, mmap_mode="r", allow_pickle=False)
                self.hits += 1
                return arr
            except (ValueError, OSError):
                p.unlink(missing_ok=True)  # corrupted partial file: recompute
        self.misses += 1
        arr = np.ascontiguousarray(fn())  # return exactly what a warm read would give
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.save(f, np.ascontiguousarray(arr), allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return arr
