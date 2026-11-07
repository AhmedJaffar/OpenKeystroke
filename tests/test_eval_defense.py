import numpy as np
import pytest

from openkeystroke.config import DefenseConfig, DefenseStep
from openkeystroke.defense import DefenseChain, defense_quality, distortion_metrics, pareto_frontier
from openkeystroke.errors import ConfigError, DSPError
from openkeystroke.evaluation import bootstrap_ci, compute_metrics, paired_comparison, summarize_seeds
from openkeystroke.evaluation.metrics import calibration

CN = ["a", "b", "c"]


def test_metrics_against_hand_computation():
    y = np.array([0, 0, 1, 1, 2, 2])
    proba = np.eye(3)[[0, 1, 1, 1, 2, 0]] * 0.9 + 0.1 / 3
    m = compute_metrics(y, proba, CN)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert m["per_class"]["b"]["precision"] == pytest.approx(2 / 3) and m["per_class"]["b"][
        "recall"
    ] == pytest.approx(1.0)
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 1 + 0.5) / 3)
    assert m["f1_macro"] == pytest.approx(np.mean([0.5, 0.8, 2 / 3]))
    assert m["f1_weighted"] == pytest.approx(m["f1_macro"])  # balanced supports


def test_metrics_reject_non_probabilities():
    with pytest.raises(Exception, match="sum to 1"):
        compute_metrics(np.array([0, 1]), np.array([[2.0, 0.0], [0.0, 1.0]]), ["a", "b"])


def test_calibration_perfect_and_overconfident():
    y = np.array([0, 1, 0, 1])
    perfect = np.array([[1.0, 0], [0, 1.0], [1.0, 0], [0, 1.0]])
    assert calibration(y, perfect)["ece"] == pytest.approx(0.0)
    wrong = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    c = calibration(y, wrong)
    assert c["ece"] == pytest.approx(1.0) and c["nll"] > 10
    uncertain = np.full((4, 2), 0.5)
    assert calibration(y, uncertain)["brier"] == pytest.approx(0.5)


def test_binary_auc_metrics():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]]), ["n", "p"])
    assert m["roc_auc"] == pytest.approx(1.0) and m["pr_auc"] == pytest.approx(1.0)
    assert "roc_auc_ovr_macro" not in m and "roc_auc" not in compute_metrics(
        np.zeros(4, int), np.full((4, 2), 0.5), ["n", "p"]
    )


def test_bootstrap_ci_contains_estimate_and_cluster_is_wider():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 20)
    y = rng.integers(0, 3, 200)
    pred = np.where(rng.random(200) < 0.5 + 0.3 * (groups % 2), y, (y + 1) % 3)  # accuracy varies by cluster
    plain = bootstrap_ci(y, pred, 3, n_boot=400, seed=1)["accuracy"]
    clus = bootstrap_ci(y, pred, 3, groups=groups, n_boot=400, seed=1)["accuracy"]
    acc = (y == pred).mean()
    assert plain["lo"] <= acc <= plain["hi"] and clus["lo"] <= acc <= clus["hi"]
    assert clus["hi"] - clus["lo"] > plain["hi"] - plain["lo"] and clus["method"] == "cluster_bootstrap"
    assert bootstrap_ci(y, pred, 3, n_boot=400, seed=1) == bootstrap_ci(
        y, pred, 3, n_boot=400, seed=1
    )  # deterministic


def test_seed_summary_and_paired_comparison():
    s = summarize_seeds([0.8, 0.9, 1.0])
    assert (
        s["mean"] == pytest.approx(0.9)
        and s["std"] == pytest.approx(0.1)
        and s["ci_lo"] < 0.9 < s["ci_hi"]
        and s["ci_hi"] <= 1.0
    )
    assert summarize_seeds([0.7])["std"] == 0.0
    c = paired_comparison([0.9, 0.92, 0.95, 0.91], [0.8, 0.83, 0.86, 0.8])
    assert c["mean_diff"] > 0.09 and c["p_value"] < 0.01
    assert paired_comparison([0.8, 0.9], [0.8, 0.9])["p_value"] == 1.0
    assert paired_comparison([0.8], [0.7])["p_value"] is None


def chain(*steps):
    return DefenseChain(DefenseConfig(name="t", steps=[DefenseStep(type=t, params=p) for t, p in steps]))


def test_defenses_deterministic_and_shape_preserving(small_ds):
    x = np.asarray(small_ds.waveforms[:16])
    for steps in [
        [("lowpass", {"cutoff_hz": 2000})],
        [("bandpass", {"low_hz": 300, "high_hz": 3000})],
        [("noise", {"snr_db": 10})],
        [("transient_attenuation", {"attenuation_db": 12})],
        [("spectral_smoothing", {"width_bins": 7})],
        [("tf_mask", {})],
        [("lowpass", {"cutoff_hz": 3000}), ("noise", {"snr_db": 15})],
    ]:
        c = chain(*steps)
        a, b = c.apply(x, 16000, 3), c.apply(x, 16000, 3)
        assert a.shape == x.shape and np.isfinite(a).all() and np.array_equal(a, b)
        if c.stochastic:
            assert not np.array_equal(a, c.apply(x, 16000, 4))


def test_noise_defense_hits_target_snr(small_ds):
    x = np.asarray(small_ds.waveforms[:32])
    y = chain(("noise", {"snr_db": 10})).apply(x, 16000, 0)
    snr = 10 * np.log10((x**2).sum(1) / ((y - x) ** 2).sum(1))
    assert abs(snr.mean() - 10) < 0.5


def test_lowpass_defense_removes_high_band_energy(small_ds):
    x = np.asarray(small_ds.waveforms[:16])
    y = chain(("lowpass", {"cutoff_hz": 1500})).apply(x, 16000, 0)
    hi = lambda a: np.abs(np.fft.rfft(a, axis=1))[:, np.fft.rfftfreq(a.shape[1], 1 / 16000) > 3000].sum()  # noqa: E731
    assert hi(y) < 0.05 * hi(x)


def test_transient_attenuation_reduces_energy_after_onset(small_ds):
    x = np.asarray(small_ds.waveforms[:16])
    y = chain(("transient_attenuation", {"attenuation_db": 20, "window_ms": 20})).apply(x, 16000, 0)
    assert (y**2).sum() < 0.5 * (x**2).sum()


def test_defense_metrics_identity_and_degradation(small_ds):
    x = np.asarray(small_ds.waveforms[:16])
    m0 = distortion_metrics(x, x.copy(), 16000)
    assert (
        m0["log_spectral_distance_db"] == pytest.approx(0)
        and m0["waveform_correlation"] == pytest.approx(1)
        and m0["waveform_snr_db"] >= 99
    )
    q = defense_quality(chain(("lowpass", {"cutoff_hz": 1000})), x, 16000)
    assert q["log_spectral_distance_db"] > 5 and q["real_time_factor"] < 1 and q["latency_ms_per_sample"] > 0


def test_bad_defense_parameters_fail_safely(small_ds):
    x = np.asarray(small_ds.waveforms[:4])
    with pytest.raises(DSPError):
        chain(("lowpass", {"cutoff_hz": 20000})).apply(x, 16000, 0)
    with pytest.raises(ConfigError):
        chain(("lowpass", {})).apply(x, 16000, 0)
    with pytest.raises(Exception):  # noqa: B017
        DefenseConfig(name="x", steps=[])


def test_pareto_frontier():
    pts = [
        {"name": "a", "g": 0.5, "d": 10},
        {"name": "b", "g": 0.4, "d": 5},
        {"name": "c", "g": 0.3, "d": 8},
        {"name": "d", "g": 0.5, "d": 12},
        {"name": "e", "g": None, "d": 1},
    ]
    assert set(pareto_frontier(pts, "g", "d")) == {"a", "b"}
