# Config-First FPGAI Workflow

FPGAI should be usable primarily through YAML configuration files and the `fpgai` command-line interface.

## Intended user workflow

```bash
fpgai inspect --config configs/examples/inference_compile.yml \
  --json-output build/examples/inference_inspection.json
```

Future stable workflow target:

```bash
fpgai compile --config configs/examples/inference_compile.yml
fpgai sweep --config configs/sweeps/example.yml
fpgai paper collect --config configs/experiments/arxiv_paper.yml
```

## Repository policy

YAML files should describe repeatable compilation and experiment settings. Python scripts should not be the primary user interface for common compilation tasks.

Allowed script categories:

| Category | Purpose |
|---|---|
| public | Stable user-facing wrappers while CLI commands mature |
| paper | Artifact/experiment collection for manuscripts |
| dev | Maintainer diagnostics and migration tools |
| legacy_workflow | Historical development utilities; not recommended for new users |

## Current transition

The repository still contains many historical development scripts. They should not be deleted immediately. Instead, each script should be classified, documented, and either promoted to a stable CLI command or moved under a legacy/development namespace.
