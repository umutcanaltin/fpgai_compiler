# Repository structure and package ownership

FPGAI keeps one canonical owner for each compiler responsibility. Generated tool
outputs and local research artifacts are not part of the maintained source tree.

## Python package ownership

| Package | Responsibility |
|---|---|
| `fpgai/engine/` | Compile orchestration, stage coordination, planning, and compile results. |
| `fpgai/capabilities/` | Operator/backend/architecture capability metadata used by planning and validation. |
| `fpgai/config/` | Python configuration loading, access helpers, and configuration contracts. |
| `fpgai/ir/` | FPGAI intermediate representation and IR passes. |
| `fpgai/frontend/` | Model import and frontend lowering. |
| `fpgai/operators/` | Logical operator contracts and operator loading. |
| `fpgai/implementations/` | Backend implementation contracts, selection, HLS/VHDL integration, and mixed-backend composition. |
| `fpgai/backends/` | Backend-specific code generation and tool integration. |
| `fpgai/analysis/` | Derived metrics, characterization, estimation, compatibility, and generated-artifact analysis. |
| `fpgai/validation/` | Numeric and behavioral validation with explicit pass/fail status. |
| `fpgai/benchmark/` | Reusable benchmark references, comparisons, setup, plots, and per-run benchmark artifacts. |
| `fpgai/experiments/` | Sweep and repeated-run orchestration. It does not own benchmark math or report serialization. |
| `fpgai/reporting/` | Serialization and presentation of compiler, validation, benchmark, HLS, Vivado, and runtime results. |
| `fpgai/runtime/` | Runtime packages, host/runtime interfaces, binary I/O, and board runtime helpers. |
| `fpgai/contracts/` | Package/dependency/version contracts. |
| `fpgai/registries/` | Stable registry infrastructure. |
| `fpgai/discovery/` | External package discovery and path safety. |
| `fpgai/numerics/` | Precision policies and numerical emulation helpers. |
| `fpgai/schemas/` | Versioned machine-readable schemas. |
| `fpgai/devtools/` | Non-runtime repository audits and development utilities. |

There is intentionally no parallel `fpgai/reports/`, `fpgai/benchmarking/`, or
`fpgai/compiler/` package. Reporting, benchmarking, capabilities, and compile
orchestration each have one owner.

## Repository-level directories

- `configs/` contains maintained YAML configurations and sweep definitions.
  This is distinct from `fpgai/config/`, which is the Python configuration API.
- `examples/` contains small maintained examples and package examples.
- `tests/` contains public unit, integration, contract, and regression tests.
- `models/` contains small maintained model fixtures.
- `docs/` contains current user and contributor documentation.
- `scripts/` contains thin executable entry points for workflows that are also
  implemented by importable FPGAI modules.

## Generated content

Generated outputs remain local and are excluded by `.gitignore`, including:

- `build/`
- `benchmark_results/`
- `benchmark_runs/`
- `experiment_results/`
- `experiment_runs/`
- `dev_audits/`
- `repo_audit/`
- Vitis HLS projects and logs
- Vivado projects, logs, bitstreams, XSA/HWH files, and simulation outputs
- runtime captures and generated benchmark artifacts

Generated results can be archived separately for reproducibility, but they are
not maintained source files.

## Naming rules

Names should describe capability or ownership, not development chronology.
Avoid internal iteration labels, temporary migration names, and publication-
specific names in maintained code. Prefer stable terms such as `benchmark`,
`validation`, `implementation`, `artifact_status`, and `reporting`.

## Readability rules

- Keep control flow explicit.
- Prefer small functions with one clear responsibility.
- Reuse canonical owners instead of creating compatibility copies.
- Remove dead modules when their functionality has no callers.
- Keep vendor-tool execution separate from parsing/analysis where practical.
- Do not duplicate a report or contract when an existing canonical artifact can
  be extended.

## Repository checks

Run the non-destructive repository audit:

```bash
python -m fpgai.devtools.repository_audit
```

Preview generated files and caches that can be removed:

```bash
python -m fpgai.devtools.clean_generated_artifacts
```

Apply cleanup only after reviewing the preview:

```bash
python -m fpgai.devtools.clean_generated_artifacts --apply
```
