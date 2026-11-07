"""Dataset validation and integrity checking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dataset import Dataset


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "info": self.info}


REQUIRED_CONSENT = ("obtained", "protocol_id", "purpose", "participants_informed")


def validate_dataset(ds: Dataset, *, check_fingerprint: bool = True) -> ValidationReport:
    rep = ValidationReport()
    m = ds.manifest
    wf = np.asarray(ds.waveforms)
    if check_fingerprint:
        if not m.fingerprint:
            rep.errors.append("manifest has no fingerprint")
        elif not ds.verify():
            rep.errors.append("fingerprint mismatch: dataset content differs from its manifest")
    if m.n_samples != len(ds):
        rep.errors.append(f"manifest n_samples={m.n_samples} but {len(ds)} records present")
    if wf.shape[1] != m.n_waveform_samples:
        rep.errors.append(f"waveform length {wf.shape[1]} != manifest {m.n_waveform_samples}")
    ids = [r.sample_id for r in ds.records]
    if len(set(ids)) != len(ids):
        rep.errors.append("duplicate sample_id values")
    if sorted(r.waveform_index for r in ds.records) != list(range(len(ds))):
        rep.errors.append("waveform_index values are not a permutation of 0..N-1")
    if not np.isfinite(wf).all():
        rep.errors.append("waveforms contain NaN/Inf")
    bad_sr = {r.sample_rate for r in ds.records} - {m.sample_rate}
    if bad_sr:
        rep.errors.append(
            f"records with sample rates {sorted(bad_sr)} differ from dataset sample rate {m.sample_rate}"
        )
    unknown = {r.label for r in ds.records} - set(m.class_names)
    if unknown:
        rep.errors.append(f"labels not declared in class_names: {sorted(unknown)}")
    if any(abs(r.duration_s - wf.shape[1] / m.sample_rate) > 1e-6 for r in ds.records):
        rep.errors.append("record duration_s inconsistent with waveform length")
    if not m.is_synthetic:
        c = m.consent
        if not c:
            rep.errors.append("non-synthetic dataset has no consent record")
        else:
            missing = [k for k in REQUIRED_CONSENT if not c.get(k)]
            if missing:
                rep.errors.append(f"consent record incomplete/false: {missing}")
    peak = np.abs(wf).max(axis=1)
    silent = int((peak < 1e-6).sum())
    clipped = int((peak >= 0.999).sum())
    if silent:
        rep.warnings.append(f"{silent} silent samples")
    if clipped > 0.05 * len(ds):
        rep.warnings.append(f"{clipped} samples ({clipped / len(ds):.0%}) are clipped")
    counts = np.bincount(ds.labels, minlength=len(m.class_names))
    cmin, cmax = int(np.min(counts)), int(np.max(counts))
    if cmin == 0:
        rep.warnings.append("some declared classes have no samples")
    elif cmax > 3 * cmin:
        rep.warnings.append(f"class imbalance: max/min = {cmax / cmin:.1f}")
    for kind in ("session", "device"):
        n = len(set(ds.group_array(kind).tolist()))
        if n < 3:
            rep.warnings.append(f"only {n} {kind} group(s): {kind}-disjoint evaluation unavailable")
    if m.is_synthetic:
        rep.warnings.append("SYNTHETIC dataset: results are not evidence of real-world performance")
    rep.info = {
        "n_samples": len(ds),
        "n_classes": len(m.class_names),
        "fingerprint": m.fingerprint,
        "is_synthetic": m.is_synthetic,
    }
    return rep
