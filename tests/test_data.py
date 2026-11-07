import json

import numpy as np
import pytest
import soundfile as sf
from hypothesis import given, settings
from hypothesis import strategies as st

from openkeystroke.config import AugmentConfig, AugmentStep, SplitConfig, SyntheticConfig
from openkeystroke.data import (
    Dataset,
    augment,
    detect_leakage,
    generate_synthetic,
    ingest_consented_recordings,
    make_split,
    validate_dataset,
)
from openkeystroke.data.splits import Split, assert_no_leakage
from openkeystroke.errors import ConsentError, DatasetError, LeakageError


def test_synthetic_is_deterministic_and_fingerprint_stable(small_cfg):
    a, b = generate_synthetic(small_cfg), generate_synthetic(small_cfg)
    assert a.fingerprint == b.fingerprint
    assert np.array_equal(a.waveforms, b.waveforms)
    assert generate_synthetic(small_cfg.model_copy(update={"seed": 4})).fingerprint != a.fingerprint


def test_fingerprint_detects_content_and_metadata_change(small_ds):
    assert small_ds.verify()
    wf = np.array(small_ds.waveforms)
    wf[0, 10] += 1e-3
    tampered = Dataset(small_ds.manifest, small_ds.records, wf)
    assert not tampered.verify()
    recs = [r.model_copy() for r in small_ds.records]
    other = next(c for c in small_ds.class_names if c != recs[0].label)
    recs[0] = recs[0].model_copy(update={"label": other})
    assert not Dataset(small_ds.manifest, recs, np.asarray(small_ds.waveforms)).verify()


def test_partition_not_part_of_fingerprint(small_ds):
    sp = make_split(small_ds, "session_disjoint", SplitConfig(), 0)
    assert small_ds.with_partitions(sp).compute_fingerprint() == small_ds.fingerprint


def test_save_load_roundtrip_and_corruption(small_ds, tmp_path):
    p = small_ds.save(tmp_path / "d")
    loaded = Dataset.load(p, verify=True)
    assert loaded.fingerprint == small_ds.fingerprint and len(loaded) == len(small_ds)
    with pytest.raises(DatasetError):
        small_ds.save(p)  # refuses to overwrite
    arr = np.load(p / "waveforms.npy")
    arr[0, 0] += 0.1
    np.save(p / "waveforms.npy", arr)
    with pytest.raises(DatasetError, match="fingerprint"):
        Dataset.load(p, verify=True)
    assert not validate_dataset(Dataset.load(p)).ok
    with pytest.raises(DatasetError):
        Dataset.load(tmp_path / "missing")


def test_validation_flags_problems(small_ds):
    assert validate_dataset(small_ds).ok
    wf = np.array(small_ds.waveforms)
    wf[2, 5] = np.nan
    rep = validate_dataset(Dataset(small_ds.manifest, small_ds.records, wf))
    assert not rep.ok and any("NaN" in e for e in rep.errors)
    real = small_ds.manifest.model_copy(update={"is_synthetic": False})
    rep = validate_dataset(
        Dataset(real, small_ds.records, np.asarray(small_ds.waveforms)), check_fingerprint=False
    )
    assert any("consent" in e for e in rep.errors)


@pytest.mark.parametrize(
    "protocol,kind",
    [("session_disjoint", "session"), ("device_disjoint", "device"), ("environment_disjoint", "environment")],
)
def test_group_splits_are_disjoint(small_ds, protocol, kind):
    for seed in range(5):
        sp = make_split(small_ds, protocol, SplitConfig(), seed)
        g = small_ds.group_array(kind)
        sets = [set(g[i]) for i in (sp.train, sp.val, sp.test)]
        assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
        assert len(np.concatenate([sp.train, sp.val, sp.test])) == len(small_ds)
        assert detect_leakage(small_ds, sp).ok


def test_split_reproducible_and_seed_varies(small_ds):
    a, b = (
        make_split(small_ds, "session_disjoint", SplitConfig(), 1),
        make_split(small_ds, "session_disjoint", SplitConfig(), 1),
    )
    assert np.array_equal(a.test, b.test)
    assert any(
        not np.array_equal(a.test, make_split(small_ds, "session_disjoint", SplitConfig(), s).test)
        for s in range(2, 8)
    )
    d1, d2 = (
        make_split(small_ds, "in_distribution", SplitConfig(), 0),
        make_split(small_ds, "in_distribution", SplitConfig(), 0),
    )
    assert np.array_equal(d1.train, d2.train)
    # stratified: all classes present in every partition
    y = small_ds.labels
    assert all(set(y[p]) == set(range(3)) for p in (d1.train, d1.val, d1.test))


def test_too_few_groups_rejected():
    ds = generate_synthetic(
        SyntheticConfig(n_classes=2, n_devices=2, sessions_per_device=1, samples_per_class_per_session=4)
    )
    with pytest.raises(DatasetError, match=">= 3"):
        make_split(ds, "device_disjoint", SplitConfig(), 0)


def test_leakage_detection_catches_overlap_and_duplicates(small_ds):
    sp = make_split(small_ds, "device_disjoint", SplitConfig(), 0)
    bad = Split(
        sp.train, sp.val, np.concatenate([sp.test, sp.train[:3]]), sp.protocol, 0, "device", sp.groups
    )
    rep = detect_leakage(small_ds, bad)
    assert (
        not rep.ok
        and any("both train and test" in e for e in rep.errors)
        and any("device groups shared" in e for e in rep.errors)
    )
    with pytest.raises(LeakageError):
        assert_no_leakage(small_ds, bad)
    # identical waveform in two partitions under different sample ids
    wf = np.array(small_ds.waveforms)
    wf[sp.test[0]] = wf[sp.train[0]]
    dup = Dataset(small_ds.manifest, small_ds.records, wf)
    assert any("identical waveforms" in e for e in detect_leakage(dup, sp).errors)
    assert detect_leakage(small_ds, make_split(small_ds, "in_distribution", SplitConfig(), 0)).warnings


@settings(max_examples=20, deadline=None)
@given(
    st.integers(0, 10_000),
    st.sampled_from(["in_distribution", "session_disjoint", "device_disjoint", "environment_disjoint"]),
)
def test_split_property_never_leaks(small_ds_prop, seed, proto):
    sp = make_split(small_ds_prop, proto, SplitConfig(), seed)
    assert detect_leakage(small_ds_prop, sp).ok


@pytest.fixture(scope="module")
def small_ds_prop():
    return generate_synthetic(
        SyntheticConfig(
            n_classes=3, n_devices=3, sessions_per_device=2, samples_per_class_per_session=5, seed=3
        )
    )


# ---------------------------------------------------------------- consent-based ingestion
def _write_lab(tmp_path, consent=True, extra=None):
    rng = np.random.default_rng(0)
    recs = []
    for i in range(8):
        x = 0.001 * rng.standard_normal(8000)
        x[3000:3200] += 0.3 * rng.standard_normal(200)
        sf.write(tmp_path / f"r{i}.wav", x, 16000)
        recs.append(
            {
                "file": f"r{i}.wav",
                "label": "A" if i % 2 else "B",
                "session_id": f"s{i % 4}",
                "device_id": f"d{i % 2}",
            }
        )
    man = {"dataset_id": "lab-1", "classes": ["A", "B"], "recordings": recs}
    if consent:
        man["consent"] = {
            "obtained": True,
            "protocol_id": "LAB-2026-01",
            "purpose": "authorized acoustic research",
            "participants_informed": True,
        }
    man.update(extra or {})
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    return tmp_path / "manifest.json"


def test_ingest_requires_consent(tmp_path):
    with pytest.raises(ConsentError):
        ingest_consented_recordings(_write_lab(tmp_path, consent=False), tmp_path / "out")


def test_ingest_valid_recordings(tmp_path):
    ds = ingest_consented_recordings(_write_lab(tmp_path), tmp_path / "out")
    assert len(ds) == 8 and not ds.is_synthetic and ds.manifest.consent["obtained"]
    assert validate_dataset(Dataset.load(tmp_path / "out")).ok
    assert all("/" not in r.provenance.split("protocol=")[0] for r in ds.records)


def test_ingest_rejects_traversal_pii_and_bad_files(tmp_path):
    m = _write_lab(tmp_path)
    man = json.loads(m.read_text())
    man["recordings"][0]["file"] = "../../etc/passwd"
    m.write_text(json.dumps(man))
    with pytest.raises(DatasetError, match="relative"):
        ingest_consented_recordings(m, tmp_path / "o1")
    man = json.loads(_write_lab(tmp_path).read_text())
    man["recordings"][0]["device_id"] = "alice@example.com"
    m.write_text(json.dumps(man))
    with pytest.raises(DatasetError, match="personal"):
        ingest_consented_recordings(m, tmp_path / "o2")
    man = json.loads(_write_lab(tmp_path).read_text())
    (tmp_path / "bad.wav").write_bytes(b"not audio")
    man["recordings"][0]["file"] = "bad.wav"
    m.write_text(json.dumps(man))
    with pytest.raises(DatasetError, match="decode"):
        ingest_consented_recordings(m, tmp_path / "o3")
    man = json.loads(_write_lab(tmp_path).read_text())
    man["recordings"][0]["sha256"] = "0" * 64
    m.write_text(json.dumps(man))
    with pytest.raises(DatasetError, match="sha256"):
        ingest_consented_recordings(m, tmp_path / "o4")


def test_augmentation_deterministic_and_shape(small_ds):
    cfg = AugmentConfig(
        steps=[AugmentStep(type="gain"), AugmentStep(type="time_shift"), AugmentStep(type="noise")], copies=2
    )
    x = np.asarray(small_ds.waveforms[:10])
    a, b = augment(x, cfg, 1, 16000), augment(x, cfg, 1, 16000)
    assert a.shape == (20, x.shape[1]) and np.array_equal(a, b) and not np.array_equal(a[:10], x)
    assert not np.array_equal(a, augment(x, cfg, 2, 16000))
