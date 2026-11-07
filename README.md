<div align="center">

# 🎙 OpenKeystroke

**An open research platform for authorized acoustic side-channel experiments and privacy-defense evaluation.**

[![CI](https://github.com/AhmedJaffar/OpenKeystroke/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmedJaffar/OpenKeystroke/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-126%20passing-brightgreen)

</div>

OpenKeystroke measures **how much class information survives in short acoustic transients**, and **how well signal
transformations (defenses) reduce it**, with the statistical discipline research needs: group-disjoint splits, leakage checks,
repeated seeds, confidence intervals, and attackers that adapt to the defense.

> **Scope and ethics.** It classifies *predefined, non-sensitive acoustic categories* in controlled or explicitly consented
> laboratory data. It does **not** reconstruct text, passwords or messages, capture audio covertly, intercept communications, or
> ingest recordings that lack a documented consent record. Recordings stay local; nothing is uploaded.

## Highlights

| | |
|---|---|
| **One lifecycle** | `CREATED → VALIDATED → RUNNING → COMPLETED / FAILED / CANCELLED`, persisted in SQLite with an immutable config snapshot, dataset fingerprint, code revision and environment metadata |
| **Honest evaluation** | session / device / environment-disjoint protocols, leak detection, train-only fitting, validation-only model selection, cluster-bootstrap CIs, paired tests across seeds |
| **Defenses that must earn the label** | a defense counts as effective only with a significant drop (≥3 seeds) against a static **and** an adaptive attacker |
| **8 models** | nearest centroid, logistic regression, random forest, SVM, 1D CNN, 2D CNN, residual CNN, audio transformer |
| **6 defenses** | low-pass, band-pass, noise injection, transient attenuation, spectral smoothing, time-frequency masking (+ chains), with distortion, latency and Pareto analysis |
| **Reproducible** | `experiment reproduce` re-runs from the stored config and diffs every metric (max diff 0 at 1e-9 on the same machine) |
| **Three front ends, one schema** | CLI, Python API and a 7-workspace Streamlit dashboard share `ExperimentConfig` |

## Install

```bash
git clone https://github.com/AhmedJaffar/OpenKeystroke
cd OpenKeystroke
pip install -e ".[dev]"
openkeystroke doctor --self-test
```

## Quick start

```bash
openkeystroke dataset generate --out data/syn            # deterministic SYNTHETIC data
openkeystroke dataset validate data/syn
openkeystroke experiment run --preset model_comparison    # dataset → features → train → evaluate → artifacts
openkeystroke experiment inspect <id>
openkeystroke benchmark run --preset defense_comparison   # static vs adaptive attackers, distortion, Pareto set
openkeystroke experiment reproduce <id> --tol 1e-6
openkeystroke dashboard                                   # interactive workspace over the same registry
```

```python
from openkeystroke import Workspace, preset_config, run_experiment, get_summary

ws = Workspace("./.openkeystroke")
exp_id, status = run_experiment(preset_config("cross_device"), ws)
print(status, get_summary(exp_id, ws)["domain_shift"])
```

## Example results

All numbers below are **synthetic data**: they describe controlled simulated transients and are **not evidence of real-world
performance**. Held-out test partitions, 3 seeds, 6 classes (chance = 0.167).

**Model comparison, session-disjoint**

| Model | Accuracy (mean ± std) |
|---|---|
| 1D CNN | 0.847 ± 0.006 |
| Logistic regression | 0.715 ± 0.006 |
| Random forest | 0.694 ± 0.032 |
| SVM | 0.684 ± 0.022 |
| 2D CNN | 0.622 ± 0.006 |
| Nearest centroid | 0.552 ± 0.058 |

**Domain shift (1D CNN):** 0.94 in-distribution → 0.85 unseen sessions → 0.86 unseen devices.

**Defenses vs a 1D CNN.** An adaptive attacker recovers most of what a static one loses:

| Defense | Static drop | Adaptive drop |
|---|---|---|
| `lowpass_1k` | 0.559 | 0.441 |
| `smooth_25` | 0.455 | 0.323 |
| `noise_0dB` | 0.420 | **0.076** |
| `noise_20dB` | 0.010 | 0.003 (not effective) |

## The dashboard

Overview · Signal Explorer · Dataset Explorer · Experiment Studio · Model Evaluation · Defense Laboratory · Experiment Registry.
Every chart reads stored experiment artifacts; empty workspaces show empty states, never placeholder metrics.
Experiments launched from the UI run in bounded worker processes and can be cancelled.

## Validity rules enforced in code

- Splits are reproducible; group-disjoint protocols are leak-checked (indices, groups, identical waveforms) before every run.
- Standardisation, PCA and normalisation statistics are fit on the training partition only; variant selection uses validation.
- Synthetic datasets are labelled everywhere; real data requires a consent manifest.
- Pickle is never loaded from unverified files; checkpoints load with `torch.load(weights_only=True)`.
- Training results are independent of memory layout and of whether the feature cache is warm (regression-tested).

## Architecture

```
src/openkeystroke/
  data/         schema, synthetic generator, consent ingestion, splits, leakage, validation
  dsp/          STFT/iSTFT, mel/MFCC, spectral descriptors, transient + SNR estimation
  features/     extractors + fingerprint-keyed cache
  models/       classical + PyTorch models, trainer (checkpoints, resume, NaN recovery)
  evaluation/   metrics, calibration, bootstrap, seed statistics, summaries
  defense/      transformations, distortion metrics, Pareto analysis
  registry/     SQLite metadata + atomic artifact store
  experiments/  plan, runner, bounded local executor
  dashboard/    Streamlit app + backend
  cli.py  api.py
```

## Development

```bash
ruff check src tests && ruff format --check src tests
python -m mypy src
python -m pytest
```

CI runs lint, type checks and tests on Python 3.10–3.13.

## Known limitations

Synthetic data only so far; GPU/AMP paths are untested; bit-exact reproducibility assumes the same machine and thread count;
the 2D CNN is under-tuned. Validation on real, consented laboratory recordings is the main open item.

## About

Created and maintained by **[Ahmed Jaffar](https://github.com/AhmedJaffar)**. Issues and contributions are welcome,
especially from researchers with consented laboratory recordings and defense ideas.

## License

MIT. See [LICENSE](LICENSE).
