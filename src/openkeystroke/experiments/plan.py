"""Expand one ExperimentConfig into variants x protocols x models x seeds x attacker jobs."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from ..config import DefenseConfig, DefenseStep, ExperimentConfig, apply_overrides
from ..errors import ConfigError


@dataclass(frozen=True)
class Condition:
    name: str
    defense: DefenseConfig | None = None


CLEAN = Condition("clean", None)


@dataclass(frozen=True)
class Job:
    variant: str
    cfg: ExperimentConfig
    protocol: str
    model_idx: int
    model_label: str
    seed: int
    attacker: str  # static: trained on clean audio; adaptive: trained on defended audio
    train_condition: Condition
    test_conditions: tuple[Condition, ...]

    @property
    def job_id(self) -> str:
        return f"{self.variant}|{self.protocol}|{self.model_label}|s{self.seed}|{self.attacker}|{self.train_condition.name}"

    @property
    def slug(self) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in self.job_id)


def expand_variants(cfg: ExperimentConfig) -> dict[str, ExperimentConfig]:
    if cfg.sweep and cfg.variants:
        raise ConfigError("use either 'sweep' or 'variants', not both")
    if cfg.sweep:
        keys = sorted(cfg.sweep)
        out = {}
        for combo in itertools.product(*(cfg.sweep[k] for k in keys)):
            name = ",".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo, strict=True))
            out[name] = apply_overrides(cfg, dict(zip(keys, combo, strict=True)))
        return out
    if cfg.variants:
        return {name: apply_overrides(cfg, ov) for name, ov in cfg.variants.items()}
    return {"base": cfg}


def conditions(cfg: ExperimentConfig) -> list[Condition]:
    conds = [Condition(d.name, d) for d in cfg.defenses]
    for snr in cfg.noise_snr_db:
        name = f"noise_{snr:g}dB"
        conds.append(
            Condition(
                name, DefenseConfig(name=name, steps=[DefenseStep(type="noise", params={"snr_db": snr})])
            )
        )
    names = [c.name for c in conds]
    if len(set(names)) != len(names):
        raise ConfigError("defense / noise condition names collide")
    return conds


def model_labels(cfg: ExperimentConfig) -> list[str]:
    names = [m.name for m in cfg.models]
    return [n if names.count(n) == 1 else f"{n}#{i}" for i, n in enumerate(names)]


def build_jobs(cfg: ExperimentConfig) -> list[Job]:
    jobs: list[Job] = []
    conds = conditions(cfg)
    for variant, vcfg in expand_variants(cfg).items():
        labels = model_labels(vcfg)
        for protocol in vcfg.effective_protocols():
            for mi, label in enumerate(labels):
                for seed in vcfg.seeds:
                    if "static" in vcfg.attacker_modes or not conds:
                        jobs.append(
                            Job(variant, vcfg, protocol, mi, label, seed, "static", CLEAN, (CLEAN, *conds))
                        )
                    if "adaptive" in vcfg.attacker_modes:
                        for c in conds:
                            jobs.append(Job(variant, vcfg, protocol, mi, label, seed, "adaptive", c, (c,)))
    if not jobs:
        raise ConfigError("experiment expands to zero jobs")
    return jobs
