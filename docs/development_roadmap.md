# FPGAI Development Roadmap

This roadmap converts FPGAI from iteration-driven research code into a professional open-source research compiler.

## Principles

1. YAML configuration and the `fpgai` CLI are the public interface.
2. `scripts/` is transitional, not the main API.
3. Paper/experiment artifact tooling belongs in `fpgai/paper_experiments/` and can be exposed through `fpgai experiment artifact ...`.
4. Development/debug helpers belong in `tools/dev/` or tests.
5. Iteration updates and one-off migration files should not remain in the public repository.
6. Raw experiment output should not be committed; reproducible summaries and paper experiments should be committed.
7. No paper claim should depend on placeholder estimator values or modeled experiment artifact without a limitation label.

## P1 — Script manifest and cleanup map

Create `scripts/MANIFEST.md` and classify every script as public-transition, paper-experiment artifact, dev-diagnostic, legacy-experiment, or delete-candidate.

Pass condition:

```text
Every file in scripts/ is classified.
No files are deleted yet.
```

## P2 — Remove obvious repository pollution

Remove tracked cache/system files:

```text
.DS_Store
.pytest_cache/
fpgai/**/__pycache__/
tests/**/__pycache__/
```

Update `.gitignore` so these do not return.

Pass condition:

```text
git status does not show cache/system files as tracked.
```

## P3 — YAML schema stabilization

Make public YAML examples match the real current config schema. Add tests that parse all YAML files and inspect minimal examples.

Pass condition:

```text
All configs/examples/*.yml parse.
All configs/experiments/*.yml parse.
fpgai inspect passes for the minimal inference example.
```

## P4 — Delete sprint update scripts

After reference-checking, remove historical sprint update scripts and one-off migration helpers.

Pass condition:

```text
No update_sprint*.py files remain in scripts/.
Tests still pass.
```

## P5 — Move experiment artifact collectors into package modules

Move paper experiments logic from scripts into:

```text
fpgai/paper_experiments/claims.py
fpgai/paper_experiments/training.py
fpgai/paper_experiments/vivado_impl.py
fpgai/paper_experiments/hardware_knobs.py
fpgai/paper_experiments/estimator_accuracy.py
fpgai/paper_experiments/communication.py
```

Expose through:

```text
fpgai experiment run --config configs/experiments/arxiv_paper.yml --out paper_experiments/arxiv
```

Pass condition:

```text
Experiment artifact logic is importable and tested from package modules.
```

## P6 — Create public sweep/experiment CLI

Replace direct calls to `scripts/run_fpgai_experiments.py` with:

```text
fpgai sweep --config configs/sweeps/<name>.yml --out experiments/<name>
```

Pass condition:

```text
README quickstart uses fpgai sweep, not Python script calls.
```

## P7 — Fix estimator exports properly

Every compile should emit a real estimator output:

```text
build/estimator/resource_prediction.json
```

This file must contain explicit validity metadata and must not use placeholder 0/1 values as predictions.

Pass condition:

```text
Estimator experiment artifact collector ignores placeholders and compares only valid exported predictions.
```

## P8 — Experiment output policy

Keep only reproducibility summaries and documentation in git. Raw generated experiment outputs should be ignored.

Pass condition:

```text
experiments/ contains README.md/.gitkeep only, or curated tiny examples.
Large outputs are not tracked.
```

## P9 — Documentation and contributor workflow

Add or update:

```text
docs/getting_started.md
docs/configuration.md
docs/experiments.md
docs/experiment artifact.md
docs/developer_guide.md
docs/testing.md
CONTRIBUTING.md or contributing.md
```

Pass condition:

```text
A new user can install, inspect a config, and run a small compile from docs alone.
```

## P10 — CI quality gate

Add CI for:

```text
package import
YAML parse
minimal config inspect
unit tests
```

Do not run Vivado in CI.
