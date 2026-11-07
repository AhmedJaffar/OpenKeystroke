# Contributing to OpenKeystroke

OpenKeystroke is a research platform for authorized acoustic side-channel experiments and privacy defense evaluation. Contributions are welcome when they keep the research honest and reproducible.

## Ground rules

- Work only with controlled or explicitly consented laboratory data. Never add recordings that lack a documented consent record.
- The platform classifies predefined, non-sensitive acoustic categories. It does not reconstruct text, passwords, or messages, and contributions must not add that capability.
- Recordings stay local. Do not upload data, and do not commit datasets to the repo.

## Set up

```bash
git clone https://github.com/AhmedJaffar/OpenKeystroke
cd OpenKeystroke
pip install -e ".[dev]"
openkeystroke doctor --self-test
```

## Before you open a pull request

- Run the test suite and make sure it passes.
- Keep evaluation honest: group-disjoint splits, train-only fitting, and validation-only model selection are part of the design, not optional extras.
- A defense counts as effective only with a significant drop across at least 3 seeds against both a static and an adaptive attacker. Claims in docs and code should match that bar.
- If your change affects experiment results, note whether `experiment reproduce` still diffs clean.

## Pull requests

- Keep each pull request focused on one change.
- Describe what changed, why, and how you verified it.
- Add or update tests when behavior changes.
