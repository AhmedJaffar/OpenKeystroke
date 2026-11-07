"""Single typed configuration schema shared by the Python API, the CLI and the dashboard."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from .errors import ConfigError

EvalProtocol = Literal["in_distribution", "session_disjoint", "device_disjoint", "environment_disjoint"]
ExperimentKind = Literal[
    "single",
    "repeated_seed",
    "model_comparison",
    "sweep",
    "defense_comparison",
    "cross_device",
    "cross_session",
    "noise_robustness",
    "ablation",
]
VECTOR_FEATURES = ("time_stats", "freq_stats", "combined_stats", "mfcc", "pca_embedding")
TENSOR_FEATURES = ("spectrogram", "mel", "log_mel")
INPUT_KINDS = ("vector", "spectrogram", "waveform")


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def canonical_json(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(obj: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:length]


# ----------------------------------------------------------------------------- data
class SyntheticConfig(_Base):
    """Controlled synthetic acoustic-transient dataset. Classes are abstract acoustic source categories."""

    seed: int = 0
    n_classes: int = Field(6, ge=2, le=26)
    n_devices: int = Field(4, ge=1, le=16)
    sessions_per_device: int = Field(3, ge=1, le=16)
    samples_per_class_per_session: int = Field(8, ge=1, le=500)
    sample_rate: int = Field(16000, ge=4000, le=96000)
    duration_s: float = Field(0.1, ge=0.03, le=1.0)
    snr_db_range: tuple[float, float] = (12.0, 30.0)
    class_separation: float = Field(0.35, gt=0, le=3.0)
    device_variation: float = Field(0.8, ge=0, le=3.0)
    session_variation: float = Field(0.5, ge=0, le=3.0)
    environments: tuple[str, ...] = ("quiet", "office", "cafe")
    version: str = "1.0.0"

    @field_validator("snr_db_range")
    @classmethod
    def _snr(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] > v[1]:
            raise ValueError("snr_db_range must be (low, high)")
        return v


class DatasetRef(_Base):
    """Either a path to a stored dataset directory or a synthetic spec (generated deterministically)."""

    path: str | None = None
    synthetic: SyntheticConfig | None = None

    @model_validator(mode="after")
    def _one(self) -> DatasetRef:
        if (self.path is None) == (self.synthetic is None):
            raise ValueError("exactly one of 'path' or 'synthetic' must be set")
        return self


class SplitConfig(_Base):
    val_fraction: float = Field(0.2, gt=0, lt=0.5)
    test_fraction: float = Field(0.2, gt=0, lt=0.5)


class AugmentStep(_Base):
    type: Literal["gain", "time_shift", "noise", "polarity"]
    params: dict[str, float] = Field(default_factory=dict)


class AugmentConfig(_Base):
    """Applied to training waveforms only. Appends `copies` augmented copies of the training set."""

    steps: list[AugmentStep]
    copies: int = Field(1, ge=1, le=10)


# ----------------------------------------------------------------------------- dsp / features
class PreprocessConfig(_Base):
    target_sample_rate: int | None = 16000
    remove_dc: bool = True
    bandpass_hz: tuple[float, float] | None = None
    lowpass_hz: float | None = None
    normalize: Literal["none", "peak", "rms"] = "peak"
    align_transient: bool = False
    align_pre_s: float = Field(0.01, ge=0, le=0.5)


class FeatureConfig(_Base):
    name: str = "combined_stats"
    n_fft: int = Field(256, ge=32, le=8192)
    hop: int = Field(64, ge=8, le=4096)
    window: str = "hann"
    n_mels: int = Field(40, ge=4, le=256)
    n_mfcc: int = Field(20, ge=2, le=64)
    f_min: float = 50.0
    f_max: float | None = None
    pca_components: int = Field(16, ge=2, le=256)

    @model_validator(mode="after")
    def _chk(self) -> FeatureConfig:
        if self.hop > self.n_fft:
            raise ValueError("hop must be <= n_fft")
        if self.n_mfcc > self.n_mels:
            raise ValueError("n_mfcc must be <= n_mels")
        return self


# ----------------------------------------------------------------------------- models
class TrainConfig(_Base):
    epochs: int = Field(30, ge=1, le=1000)
    batch_size: int = Field(64, ge=1, le=8192)
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = Field(2e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    momentum: float = 0.9
    scheduler: Literal["none", "cosine", "step", "plateau"] = "cosine"
    patience: int = Field(8, ge=1)
    grad_clip: float | None = 1.0
    amp: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"
    class_weighting: Literal["none", "balanced"] = "balanced"
    max_nan_retries: int = Field(2, ge=0, le=10)
    checkpoint: bool = True


class ModelConfig(_Base):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    training: TrainConfig = Field(default_factory=TrainConfig)


# ----------------------------------------------------------------------------- defenses
class DefenseStep(_Base):
    type: Literal["lowpass", "bandpass", "noise", "transient_attenuation", "spectral_smoothing", "tf_mask"]
    params: dict[str, float] = Field(default_factory=dict)


class DefenseConfig(_Base):
    name: str
    steps: list[DefenseStep] = Field(min_length=1)


class EvalConfig(_Base):
    bootstrap_samples: int = Field(300, ge=0, le=20000)
    confidence: float = Field(0.95, gt=0.5, lt=1)
    calibration_bins: int = Field(10, ge=2, le=50)
    measure_latency: bool = True
    defense_quality_samples: int = Field(64, ge=4, le=2000)


class ResourceConfig(_Base):
    max_threads: int = Field(4, ge=1, le=256)
    max_concurrent_experiments: int = Field(2, ge=1, le=64)


# ----------------------------------------------------------------------------- experiment
_DEFAULT_PROTOCOLS: dict[str, list[str]] = {
    "cross_device": ["in_distribution", "session_disjoint", "device_disjoint"],
    "cross_session": ["in_distribution", "session_disjoint"],
}


def _default_attackers() -> list[Literal["static", "adaptive"]]:
    return ["static", "adaptive"]


class ExperimentConfig(_Base):
    name: str = Field(min_length=1, max_length=120)
    hypothesis: str = ""
    kind: ExperimentKind = "single"
    dataset: DatasetRef
    split: SplitConfig = Field(default_factory=SplitConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    vector_features: FeatureConfig = Field(default_factory=lambda: FeatureConfig(name="combined_stats"))
    tensor_features: FeatureConfig = Field(default_factory=lambda: FeatureConfig(name="log_mel"))
    models: list[ModelConfig] = Field(min_length=1)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    augmentation: AugmentConfig | None = None
    defenses: list[DefenseConfig] = Field(default_factory=list)
    attacker_modes: list[Literal["static", "adaptive"]] = Field(default_factory=_default_attackers)
    noise_snr_db: list[float] = Field(default_factory=list)
    protocols: list[EvalProtocol] | None = None
    variants: dict[str, dict[str, Any]] = Field(default_factory=dict)
    sweep: dict[str, list[Any]] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    save_models: bool = True

    @field_validator("vector_features")
    @classmethod
    def _vec(cls, v: FeatureConfig) -> FeatureConfig:
        if v.name not in VECTOR_FEATURES:
            raise ValueError(f"vector_features.name must be one of {VECTOR_FEATURES}")
        return v

    @field_validator("tensor_features")
    @classmethod
    def _ten(cls, v: FeatureConfig) -> FeatureConfig:
        if v.name not in TENSOR_FEATURES:
            raise ValueError(f"tensor_features.name must be one of {TENSOR_FEATURES}")
        return v

    @field_validator("seeds")
    @classmethod
    def _seeds(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError("seeds must be unique")
        return v

    @model_validator(mode="after")
    def _kind_rules(self, info: ValidationInfo) -> ExperimentConfig:
        k = self.kind
        leaf = bool(
            info.context and info.context.get("leaf")
        )  # an expanded sweep/ablation variant: grid already applied
        if k == "model_comparison" and len(self.models) < 2:
            raise ValueError("model_comparison requires at least two models")
        if k == "repeated_seed" and len(self.seeds) < 2:
            raise ValueError("repeated_seed requires at least two seeds")
        if k == "defense_comparison" and not self.defenses:
            raise ValueError("defense_comparison requires at least one defense")
        if k == "noise_robustness" and not self.noise_snr_db:
            raise ValueError("noise_robustness requires noise_snr_db levels")
        if k == "sweep" and not self.sweep and not leaf:
            raise ValueError("sweep requires a non-empty 'sweep' grid")
        if k == "ablation" and not self.variants and not leaf:
            raise ValueError("ablation requires 'variants'")
        if k == "single" and len(self.seeds) != 1:
            raise ValueError("single experiments use exactly one seed")
        names = [d.name for d in self.defenses]
        if len(set(names)) != len(names):
            raise ValueError("defense names must be unique")
        return self

    def effective_protocols(self) -> list[str]:
        return list(self.protocols or _DEFAULT_PROTOCOLS.get(self.kind, ["session_disjoint"]))

    def snapshot(self) -> str:
        return canonical_json(self)

    def hash(self) -> str:
        return config_hash(self)


# ----------------------------------------------------------------------------- helpers
def apply_overrides(cfg: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """Apply dotted-path overrides (list indices allowed: 'models.0.params.C') and re-validate."""
    data = cfg.model_dump(mode="json")
    for path, value in overrides.items():
        _set_path(data, path, copy.deepcopy(value))
    data["variants"], data["sweep"] = {}, {}
    try:
        return ExperimentConfig.model_validate(data, context={"leaf": True})
    except Exception as exc:
        raise ConfigError(f"override produced an invalid configuration: {exc}") from exc


def _set_path(root: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = root
    for i, p in enumerate(parts):
        last = i == len(parts) - 1
        try:
            if isinstance(cur, list):
                idx = int(p)
                if last:
                    cur[idx] = value
                else:
                    cur = cur[idx]
            else:
                if last:
                    cur[p] = value
                else:
                    cur = cur.setdefault(p, {}) if isinstance(cur, dict) else cur[p]
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            raise ConfigError(f"invalid override path '{path}': {exc}") from exc


def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    if p.stat().st_size > 1_000_000:
        raise ConfigError("config file too large (>1MB)")
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse {p}: {exc}") from exc
    return parse_config(raw)


def parse_config(raw: Any) -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError -> structured error
        raise ConfigError(f"invalid experiment config: {exc}") from exc
