# Repository structure and contribution ownership

FPGAI separates maintained source files from generated outputs. A contributor
should be able to understand a file's purpose from its name and directory.

## Maintained content

The following content belongs in the source repository:

- `fpgai/`: compiler, backend, runtime, validation, and reporting code.
- `tests/`: public unit, integration, contract, and regression tests.
- `configs/`: maintained compiler configurations and sweep definitions.
- `examples/`: small, documented user examples.
- `docs/`: current user and contributor documentation.
- `models/`: small maintained model fixtures and suite definitions.
- `scripts/`: documentation only; executable workflows belong in `fpgai/`.

## Generated content

The following directories are local outputs and must not be committed:

- `build/`
- `experiments/`
- `paper_results/`
- `paper_outputs/`
- `paper_tables/`
- `repo_audit/`
- `dev_audits/`
- `reports/`
- `examples/build/`

Vivado projects, Vitis HLS projects, bitstreams, XSA files, HWH files, logs,
runtime captures, and generated paper results are also excluded.

Paper experiment definitions and reproducibility code may be committed. Produced
results should be archived separately and referenced by version.

## Naming rules

Names must state one clear responsibility. Prefer names such as:

- `runtime_package_validator.py`
- `experiment_config_materializer.py`
- `vivado_implementation_report.py`
- `operator_registry.py`

Avoid names such as `utils.py`, `helpers.py`, `misc.py`, `new.py`, and
`run_all.py` because they do not explain ownership.

## Readability rules

- Keep control flow explicit.
- Prefer small functions with one purpose.
- Use descriptive variable and type names.
- Avoid metaprogramming and hidden import side effects.
- Explain compiler or hardware reasons, not obvious Python syntax.
- Do not create abstractions that make a direct implementation harder to read.

Production modules above 800 lines require review. Modules above 1,200 lines are
blocking repository-audit findings and must be split by responsibility before
new behavior is added. Test files use higher temporary limits because many
existing regression cases are still being organized.

## Repository checks

Run the non-destructive audit:

```bash
python -m fpgai.devtools.repository_audit
```

Preview generated files and caches that can be removed:

```bash
python -m fpgai.devtools.clean_generated_artifacts
```

Apply the cleanup only after reviewing the list:

```bash
python -m fpgai.devtools.clean_generated_artifacts --apply
```
