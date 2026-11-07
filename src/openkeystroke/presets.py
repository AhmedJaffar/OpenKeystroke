"""Ready-made experiment configurations (same schema as user configs; used by CLI, API and dashboard)."""

from __future__ import annotations

from .config import (
    DatasetRef,
    DefenseConfig,
    DefenseStep,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    PreprocessConfig,
    SyntheticConfig,
    TrainConfig,
)
from .errors import ConfigError

PRESET_KINDS = (
    "single",
    "repeated_seed",
    "model_comparison",
    "sweep",
    "defense_comparison",
    "cross_device",
    "cross_session",
    "noise_robustness",
    "ablation",
)


def _d(name: str, *steps: tuple[str, dict]) -> DefenseConfig:
    return DefenseConfig(name=name, steps=[DefenseStep(type=t, params=p) for t, p in steps])


def default_defenses() -> list[DefenseConfig]:
    return [
        _d("lowpass_4k", ("lowpass", {"cutoff_hz": 4000})),
        _d("lowpass_2k", ("lowpass", {"cutoff_hz": 2000})),
        _d("lowpass_1k", ("lowpass", {"cutoff_hz": 1000})),
        _d("bandpass_500_3k", ("bandpass", {"low_hz": 500, "high_hz": 3000})),
        _d("noise_20dB", ("noise", {"snr_db": 20})),
        _d("noise_10dB", ("noise", {"snr_db": 10})),
        _d("noise_0dB", ("noise", {"snr_db": 0})),
        _d("transient_att_12dB", ("transient_attenuation", {"attenuation_db": 12})),
        _d("transient_att_24dB", ("transient_attenuation", {"attenuation_db": 24})),
        _d("smooth_9", ("spectral_smoothing", {"width_bins": 9})),
        _d("smooth_25", ("spectral_smoothing", {"width_bins": 25})),
        _d("tf_mask", ("tf_mask", {"n_masks": 3, "freq_frac": 0.15, "time_frac": 0.2})),
        _d("lowpass_2k+noise_10dB", ("lowpass", {"cutoff_hz": 2000}), ("noise", {"snr_db": 10})),
    ]


def preset_config(
    kind: str = "single",
    *,
    name: str | None = None,
    synthetic: SyntheticConfig | None = None,
    seeds: list[int] | None = None,
) -> ExperimentConfig:
    if kind not in PRESET_KINDS:
        raise ConfigError(f"unknown preset '{kind}'. Available: {list(PRESET_KINDS)}")
    deep = ModelConfig(name="cnn1d", training=TrainConfig(epochs=12, patience=5))
    kw: dict = dict(
        name=name or f"{kind} (preset)",
        kind=kind,
        dataset=DatasetRef(synthetic=synthetic or SyntheticConfig()),
        models=[ModelConfig(name="logreg")],
        seeds=seeds or ([0] if kind == "single" else [0, 1, 2]),
        hypothesis="",
    )
    if kind == "single":
        kw["hypothesis"] = (
            "A linear model separates the synthetic acoustic categories on unseen sessions above chance."
        )
    elif kind == "repeated_seed":
        kw["hypothesis"] = "Session-disjoint accuracy is stable across random group assignments."
    elif kind == "model_comparison":
        kw["models"] = [
            ModelConfig(name="nearest_centroid"),
            ModelConfig(name="logreg"),
            ModelConfig(name="random_forest"),
            ModelConfig(name="svm"),
            deep,
            ModelConfig(
                name="cnn2d",
                params={"channels": [32, 64]},
                training=TrainConfig(epochs=40, lr=5e-3, patience=8),
            ),
        ]
        kw["hypothesis"] = (
            "Learned waveform/spectrogram models outperform hand-crafted-feature baselines on held-out sessions."
        )
    elif kind == "sweep":
        kw["sweep"] = {"models.0.params.C": [0.01, 0.1, 1.0, 10.0]}
        kw["hypothesis"] = (
            "Regularisation strength affects held-out accuracy; the best C is selected on validation, not test."
        )
    elif kind == "defense_comparison":
        kw["models"] = [ModelConfig(name="random_forest"), deep]
        kw["defenses"] = default_defenses()
        kw["hypothesis"] = (
            "Signal transformations reduce class distinguishability for static and adaptive attackers at acceptable distortion."
        )
    elif kind == "cross_device":
        kw["models"] = [ModelConfig(name="logreg"), deep]
        kw["hypothesis"] = "Accuracy degrades on unseen devices relative to in-distribution evaluation."
    elif kind == "cross_session":
        kw["models"] = [ModelConfig(name="logreg"), deep]
        kw["hypothesis"] = "Accuracy degrades on unseen sessions relative to in-distribution evaluation."
    elif kind == "noise_robustness":
        kw["models"] = [ModelConfig(name="logreg"), deep]
        kw["noise_snr_db"] = [30.0, 20.0, 10.0, 5.0, 0.0]
        kw["hypothesis"] = "Accuracy decreases monotonically as test-time SNR falls."
    elif kind == "ablation":
        kw["models"] = [ModelConfig(name="logreg")]
        kw["variants"] = {
            "baseline": {},
            "no_dc_removal": {"preprocess.remove_dc": False},
            "bandpass_300_6000": {"preprocess.bandpass_hz": [300.0, 6000.0]},
            "mfcc_features": {"vector_features.name": "mfcc"},
            "time_stats_only": {"vector_features.name": "time_stats"},
            "freq_stats_only": {"vector_features.name": "freq_stats"},
            "transient_aligned": {"preprocess.align_transient": True},
        }
        kw["hypothesis"] = "Each preprocessing / feature component contributes to held-out accuracy."
    return ExperimentConfig(**kw)


__all__ = ["PRESET_KINDS", "default_defenses", "preset_config", "FeatureConfig", "PreprocessConfig"]
