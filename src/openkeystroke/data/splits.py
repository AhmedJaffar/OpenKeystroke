"""Reproducible, stratified and group-aware train/val/test splits plus leakage detection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import train_test_split

from ..config import SplitConfig
from ..errors import DatasetError, LeakageError
from .dataset import Dataset

GROUP_FOR_PROTOCOL = {
    "session_disjoint": "session",
    "device_disjoint": "device",
    "environment_disjoint": "environment",
}


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    protocol: str
    seed: int
    group_kind: str | None = None
    groups: dict[str, list[str]] = field(default_factory=dict)

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def make_split(ds: Dataset, protocol: str, cfg: SplitConfig, seed: int) -> Split:
    y = ds.labels
    n = len(ds)
    if protocol == "in_distribution":
        idx = np.arange(n)
        try:
            tv, te = train_test_split(idx, test_size=cfg.test_fraction, stratify=y, random_state=seed)
            tr, va = train_test_split(
                tv, test_size=cfg.val_fraction / (1 - cfg.test_fraction), stratify=y[tv], random_state=seed
            )
        except ValueError as exc:
            raise DatasetError(f"cannot create stratified split: {exc}") from exc
        return Split(np.sort(tr), np.sort(va), np.sort(te), protocol, seed)
    if protocol not in GROUP_FOR_PROTOCOL:
        raise DatasetError(f"unknown evaluation protocol '{protocol}'")
    kind = GROUP_FOR_PROTOCOL[protocol]
    g = ds.group_array(kind)
    uniq = np.array(sorted(set(g.tolist())))
    if len(uniq) < 3:
        raise DatasetError(
            f"{protocol} needs >= 3 distinct {kind} groups (train/val/test); dataset has {len(uniq)}"
        )
    rng = np.random.default_rng([seed, 17])
    perm = uniq[rng.permutation(len(uniq))]
    n_test = max(1, round(cfg.test_fraction * len(uniq)))
    n_val = max(1, round(cfg.val_fraction * len(uniq)))
    if len(uniq) - n_test - n_val < 1:
        n_val = max(1, len(uniq) - n_test - 1)
    test_g, val_g, train_g = perm[:n_test], perm[n_test : n_test + n_val], perm[n_test + n_val :]
    tr, va, te = (np.flatnonzero(np.isin(g, x)) for x in (train_g, val_g, test_g))
    train_classes = set(y[tr].tolist())
    if train_classes != set(range(len(ds.class_names))):
        raise DatasetError(f"{protocol}: some classes are missing from the training groups")
    return Split(
        tr,
        va,
        te,
        protocol,
        seed,
        kind,
        {"train": sorted(train_g.tolist()), "val": sorted(val_g.tolist()), "test": sorted(test_g.tolist())},
    )


@dataclass
class LeakageReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _hashes(ds: Dataset, idx: np.ndarray) -> set[str]:
    wf = np.asarray(ds.waveforms)
    return {hashlib.blake2b(np.ascontiguousarray(wf[i]).tobytes(), digest_size=16).hexdigest() for i in idx}


def detect_leakage(ds: Dataset, split: Split) -> LeakageReport:
    rep = LeakageReport()
    parts = {"train": split.train, "val": split.val, "test": split.test}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ov = np.intersect1d(parts[a], parts[b])
            if ov.size:
                rep.errors.append(f"{ov.size} sample indices appear in both {a} and {b}")
    total = sum(len(v) for v in parts.values())
    if total != len(np.unique(np.concatenate(list(parts.values())))):
        rep.errors.append("duplicate indices within the split")
    if split.group_kind:
        g = ds.group_array(split.group_kind)
        sets = {k: set(g[v].tolist()) for k, v in parts.items()}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                shared = sets[a] & sets[b]
                if shared:
                    rep.errors.append(
                        f"{split.group_kind} groups shared between {a} and {b}: {sorted(shared)[:5]}"
                    )
    else:
        rep.warnings.append(
            "in-distribution split: sessions/devices overlap across partitions; do not read this as cross-domain generalisation"
        )
    th, eh = _hashes(ds, split.train), _hashes(ds, split.test)
    vh = _hashes(ds, split.val)
    if th & eh or th & vh or vh & eh:
        rep.errors.append("identical waveforms appear in different partitions (content leakage)")
    return rep


def assert_no_leakage(ds: Dataset, split: Split) -> LeakageReport:
    rep = detect_leakage(ds, split)
    if not rep.ok:
        raise LeakageError("data leakage detected: " + "; ".join(rep.errors), details={"errors": rep.errors})
    return rep
