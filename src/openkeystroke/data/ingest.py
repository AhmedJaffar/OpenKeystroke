"""Explicit, auditable, consent-based ingestion of lab recordings. Nothing is downloaded or discovered automatically."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..dsp import ops
from ..errors import ConsentError, DatasetError
from .dataset import Dataset
from .schema import DatasetManifest, SampleRecord
from .validate import REQUIRED_CONSENT

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_FILES = 200_000
ALLOWED_SUFFIX = {".wav", ".flac"}
_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_PII = re.compile(r"[@/\\]|\b\d{3}[-. ]\d{3}[-. ]\d{4}\b")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID.match(value):
        raise DatasetError(
            f"{field}={value!r}: must match [A-Za-z0-9_.-]{{1,64}} (no personal identifiers or paths)"
        )
    return value


def _normalise_audio(path: Path, target_sr: int, window_s: float) -> np.ndarray:
    """Read -> mono float32 -> resample -> fixed window anchored on the detected transient."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise DatasetError(f"cannot decode audio '{path.name}': {exc}") from exc
    x = data.mean(axis=1)[None, :]
    if not np.isfinite(x).all():
        raise DatasetError(f"'{path.name}' contains non-finite samples")
    x = ops.resample(x, sr, target_sr)
    n = int(round(window_s * target_sr))
    onset = int(ops.detect_transient(x, target_sr)[0])
    start = max(0, (onset if onset >= 0 else 0) - int(0.01 * target_sr))
    seg = x[0, start : start + n]
    return np.pad(seg, (0, n - seg.size)).astype(np.float32)


def ingest_consented_recordings(
    manifest_path: str | Path, out_dir: str | Path, *, target_sample_rate: int = 16000, window_s: float = 0.1
) -> Dataset:
    """Build a dataset from a consent manifest (JSON). Recording paths must be relative and stay inside the manifest's directory."""
    mp = Path(manifest_path)
    if not mp.is_file():
        raise DatasetError(f"consent manifest not found: {mp}")
    if mp.stat().st_size > 50_000_000:
        raise DatasetError("consent manifest too large")
    try:
        man = json.loads(mp.read_text())
    except json.JSONDecodeError as exc:
        raise DatasetError(f"consent manifest is not valid JSON: {exc}") from exc
    consent = man.get("consent")
    if not isinstance(consent, dict):
        raise ConsentError(
            "manifest has no 'consent' record; refusing to ingest recordings without documented consent"
        )
    missing = [k for k in REQUIRED_CONSENT if not consent.get(k)]
    if missing:
        raise ConsentError(f"consent record missing or false for: {missing}")
    for k in ("protocol_id", "purpose"):
        if _PII.search(str(consent[k])):
            raise ConsentError(f"consent.{k} appears to contain personal identifiers/paths")
    recs = man.get("recordings")
    classes = man.get("classes")
    if not isinstance(recs, list) or not recs or not isinstance(classes, list) or len(classes) < 2:
        raise DatasetError("manifest needs a non-empty 'recordings' list and >= 2 'classes'")
    if len(recs) > MAX_FILES:
        raise DatasetError(f"too many recordings (> {MAX_FILES})")
    class_names = [_check_id(c, "class") for c in classes]
    base = mp.parent.resolve()
    dataset_id = _check_id(man.get("dataset_id", "lab-dataset"), "dataset_id")
    version = _check_id(man.get("version", "1.0.0"), "version")
    waves, records = [], []
    for i, r in enumerate(recs):
        rel = r.get("file")
        if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise DatasetError(f"recording {i}: path must be relative and inside the manifest directory")
        p = (base / rel).resolve()
        if not p.is_relative_to(base):
            raise DatasetError(f"recording {i}: path escapes the manifest directory")
        if p.suffix.lower() not in ALLOWED_SUFFIX or not p.is_file():
            raise DatasetError(f"recording {i}: '{rel}' must be an existing .wav/.flac file")
        if p.stat().st_size > MAX_FILE_BYTES:
            raise DatasetError(f"recording {i}: file exceeds {MAX_FILE_BYTES} bytes")
        if r.get("sha256") and _sha256(p) != r["sha256"]:
            raise DatasetError(f"recording {i}: sha256 mismatch for '{rel}'")
        label = _check_id(r.get("label"), "label")
        if label not in class_names:
            raise DatasetError(f"recording {i}: label '{label}' not in declared classes")
        env = r.get("environment", {"name": "unknown"})
        if not isinstance(env, dict) or _PII.search(json.dumps(env)):
            raise DatasetError(f"recording {i}: invalid environment metadata")
        waves.append(_normalise_audio(p, target_sample_rate, window_s))
        records.append(
            SampleRecord(
                sample_id=f"{dataset_id}-{i:06d}",
                dataset_id=dataset_id,
                dataset_version=version,
                waveform_index=i,
                sample_rate=target_sample_rate,
                duration_s=window_s,
                label=label,
                session_id=_check_id(r.get("session_id"), "session_id"),
                device_id=_check_id(r.get("device_id"), "device_id"),
                source_category=_check_id(r.get("source_category", "unknown"), "source_category"),
                mic_category=_check_id(r.get("mic_category", "unknown"), "mic_category"),
                environment=env,
                noise={},
                provenance=f"consented-lab-recording; protocol={consent['protocol_id']}",
                lineage=[
                    "ingest:mono-downmix",
                    f"ingest:resample->{target_sample_rate}",
                    f"ingest:window={window_s}s",
                ],
            )
        )
    wave = np.stack(waves)
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        version=version,
        name=str(man.get("name", dataset_id))[:120],
        is_synthetic=False,
        sample_rate=target_sample_rate,
        n_samples=len(records),
        n_waveform_samples=wave.shape[1],
        class_names=class_names,
        provenance={
            "source_manifest_sha256": _sha256(mp),
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_files": len(recs),
        },
        consent=consent,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    ds = Dataset(manifest, records, wave)
    ds.manifest = manifest.model_copy(update={"fingerprint": ds.compute_fingerprint()})
    ds.save(out_dir)
    return ds
