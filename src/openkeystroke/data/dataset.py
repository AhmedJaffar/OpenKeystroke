"""Dataset container: memory-mapped waveforms + typed metadata + content fingerprint."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..config import canonical_json
from ..errors import DatasetError
from .schema import DatasetManifest, SampleRecord

SPLIT_NAMES = ("train", "val", "test")


class Dataset:
    def __init__(
        self,
        manifest: DatasetManifest,
        records: list[SampleRecord],
        waveforms: np.ndarray,
        root: Path | None = None,
    ) -> None:
        if waveforms.ndim != 2 or waveforms.shape[0] != len(records):
            raise DatasetError(f"waveforms shape {waveforms.shape} does not match {len(records)} records")
        self.manifest = manifest
        self.records = records
        self.waveforms = waveforms
        self.root = root
        self._label_to_idx = {c: i for i, c in enumerate(manifest.class_names)}

    # ---- basic accessors
    def __len__(self) -> int:
        return len(self.records)

    @property
    def sample_rate(self) -> int:
        return self.manifest.sample_rate

    @property
    def class_names(self) -> list[str]:
        return self.manifest.class_names

    @property
    def fingerprint(self) -> str:
        return self.manifest.fingerprint

    @property
    def is_synthetic(self) -> bool:
        return self.manifest.is_synthetic

    @property
    def labels(self) -> np.ndarray:
        return np.array([self._label_to_idx[r.label] for r in self.records], dtype=np.int64)

    def group_array(self, kind: str) -> np.ndarray:
        if kind == "session":
            vals = [r.session_id for r in self.records]
        elif kind == "device":
            vals = [r.device_id for r in self.records]
        elif kind == "environment":
            vals = [r.environment_name for r in self.records]
        else:
            raise DatasetError(f"unknown group kind '{kind}'")
        return np.array(vals)

    # ---- fingerprint
    def compute_fingerprint(self) -> str:
        h = hashlib.sha256()
        m = self.manifest
        h.update(
            canonical_json(
                {
                    "id": m.dataset_id,
                    "version": m.version,
                    "sr": m.sample_rate,
                    "classes": m.class_names,
                    "synthetic": m.is_synthetic,
                }
            ).encode()
        )
        for r in self.records:
            d = r.model_dump(mode="json", exclude={"partition"})
            h.update(canonical_json(d).encode())
        for i in range(0, len(self), 256):
            h.update(np.ascontiguousarray(self.waveforms[i : i + 256], dtype=np.float32).tobytes())
        return h.hexdigest()

    def verify(self) -> bool:
        return bool(self.manifest.fingerprint) and self.compute_fingerprint() == self.manifest.fingerprint

    # ---- subsetting / partitions
    def subset(self, idx: np.ndarray) -> Dataset:
        idx = np.asarray(idx)
        recs = [self.records[int(i)].model_copy(update={"waveform_index": j}) for j, i in enumerate(idx)]
        m = self.manifest.model_copy(update={"n_samples": len(recs)})
        return Dataset(m, recs, np.asarray(self.waveforms)[idx], None)

    def with_partitions(self, split: Any) -> Dataset:
        part = np.full(len(self), "unassigned", dtype=object)
        for name in SPLIT_NAMES:
            part[getattr(split, name)] = name
        recs = [r.model_copy(update={"partition": str(p)}) for r, p in zip(self.records, part, strict=True)]
        return Dataset(self.manifest, recs, self.waveforms, self.root)

    # ---- statistics
    def stats(self) -> dict[str, Any]:
        recs = self.records
        wf = np.asarray(self.waveforms)
        rms = np.sqrt((wf.astype(np.float64) ** 2).mean(axis=1))
        snr = [float(r.noise["snr_db"]) for r in recs if r.noise.get("snr_db") is not None]
        return {
            "dataset_id": self.manifest.dataset_id,
            "version": self.manifest.version,
            "fingerprint": self.fingerprint,
            "is_synthetic": self.is_synthetic,
            "n_samples": len(self),
            "sample_rate": self.sample_rate,
            "duration_s": float(wf.shape[1] / self.sample_rate),
            "classes": dict(Counter(r.label for r in recs)),
            "devices": dict(Counter(r.device_id for r in recs)),
            "sessions": dict(Counter(r.session_id for r in recs)),
            "environments": dict(Counter(r.environment_name for r in recs)),
            "source_categories": dict(Counter(r.source_category for r in recs)),
            "mic_categories": dict(Counter(r.mic_category for r in recs)),
            "rms": {
                "mean": float(rms.mean()),
                "std": float(rms.std()),
                "min": float(rms.min()),
                "max": float(rms.max()),
            },
            "snr_db": {
                "mean": float(np.mean(snr)),
                "std": float(np.std(snr)),
                "min": float(np.min(snr)),
                "max": float(np.max(snr)),
            }
            if snr
            else None,
        }

    # ---- persistence
    def save(self, directory: str | Path, *, overwrite: bool = False) -> Path:
        out = Path(directory)
        if out.exists() and any(out.iterdir()) and not overwrite:
            raise DatasetError(f"dataset directory not empty: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(
            tempfile.mkdtemp(prefix=out.name + ".tmp-", dir=out.parent)
        )  # atomic: build aside, then rename
        try:
            m = self.manifest.model_copy(
                update={"fingerprint": self.compute_fingerprint(), "n_samples": len(self)}
            )
            np.save(
                tmp / "waveforms.npy",
                np.ascontiguousarray(self.waveforms, dtype=np.float32),
                allow_pickle=False,
            )
            (tmp / "samples.json").write_text(json.dumps([r.model_dump(mode="json") for r in self.records]))
            (tmp / "manifest.json").write_text(json.dumps(m.model_dump(mode="json"), indent=2))
            if out.exists():
                import shutil

                shutil.rmtree(out)
            os.replace(tmp, out)
        except BaseException:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self.manifest = m
        self.root = out
        return out

    @classmethod
    def load(cls, directory: str | Path, *, verify: bool = False, mmap: bool = True) -> Dataset:
        d = Path(directory)
        mf, sf, wf = d / "manifest.json", d / "samples.json", d / "waveforms.npy"
        for f in (mf, sf, wf):
            if not f.is_file():
                raise DatasetError(f"not a valid dataset directory (missing {f.name}): {d}")
        try:
            manifest = DatasetManifest.model_validate_json(mf.read_text())
            records = [SampleRecord.model_validate(r) for r in json.loads(sf.read_text())]
            waveforms = np.load(wf, mmap_mode="r" if mmap else None, allow_pickle=False)
        except Exception as exc:
            raise DatasetError(f"cannot read dataset at {d}: {exc}") from exc
        ds = cls(manifest, records, waveforms, d)
        if verify and not ds.verify():
            raise DatasetError(f"fingerprint mismatch — dataset at {d} was modified or corrupted")
        return ds


def sanitize_id(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)[:80]
