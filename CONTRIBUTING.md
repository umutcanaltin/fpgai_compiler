# Contributing to FPGAI Compiler

Thank you for improving FPGAI. The most useful contributions are small, independently reproducible changes to compiler semantics, operator support, source emission, documentation, and evidence qualification.

**Important:** This repository is publicly viewable but currently has **no operative source-code license**. Before contributing, the maintainer must confirm the proposed contribution can be accepted and redistributed under an eventual authorized license. Repository access alone grants no permission to publish, redistribute, or reuse its contents.

## Before starting

Read the [project README](README.md), [compiler architecture](docs/ARCHITECTURE.md), [supported features](docs/SUPPORTED_FEATURES.md), [licensing and permissions](docs/LICENSE_DECISION.md), and [release constraints](docs/OPEN_SOURCE_RELEASE_CHECKLIST.md). Check for an existing issue or related implementation first. For operator extensions, follow [the operator guide](docs/COMPILER_OPERATOR_CONTRIBUTION.md).

A contribution should be **scoped to one component and one verifiable behavior**. Avoid unrelated formatting-only refactors in scientific source files, unnecessary compatibility wrappers, stale duplicate implementations or changes to preserved experimental records.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[onnx,test]"
python -m compileall -q src/fpgai_research
python -m pytest -q
```

Start from the checked-out source and ensure no older editable installation shadows the current `src/fpgai_research` package. Run `python -m pytest -q` for full regression when dependencies permit; state exactly which checks ran and which did not.

**Normal installation and tests must not call vendor tools**. Do not run Vitis/Vivado, native original experiments or a self-hosted GitHub Actions runner on behalf of another contributor. Vendor campaigns require explicit consent, licensed programs, hardware availability, and new create-only output directories.

## Source layout and ownership

| Component | Location | Contribution contract |
|---|---|---|
| ONNX import | `src/fpgai_research/frontend/` | Preserve operator attributes, static shapes, unsupported-case errors |
| Functional and architectural IR | `src/fpgai_research/model_ir/` | Preserve typed tensors, provenance and deterministic identities |
| Hardware legality and scoped requests | `src/fpgai_research/architecture/` | Fail closed on unsupported precision, tiling, resources or mappings |
| Core compile orchestration | `src/fpgai_research/compiler/` | Use canonical IR, avoid alternate compilers for each experiment |
| HLS emission | `src/fpgai_research/codegen/hls/` | Generated code must reflect resolved architecture semantics |
| Numerical validation | `src/fpgai_research/validation/` | Independent reference and strict tolerances |
| Vendor tools | `src/fpgai_research/toolchains/` | No implicit invocation; preserve original receipts |
| Research experiments | `src/fpgai_research/experiments/` and `experiments/compiler_ir/` | Explicit cohorts, inputs, denominators and limitations |
| Experimental RL | `rl_environment/` | Currently frozen; discuss substantial changes before editing |

Do not move previously source-pinned scientific interfaces or alter `rl_environment/` for an unrelated compiler fix.

## Expected pull-request workflow

1. **Describe the scientific or engineering question.** State the exact supported domain, limitations and intended behavior.
2. **Inspect existing code paths** before implementing. Reuse the typed Model IR and existing toolchain boundary.
3. **Add appropriate regression coverage:** valid input, invalid input, deterministic output, changed architecture decisions, and source-only capability without native dependencies.
4. **Keep experiments reproducible:** match the exact functional graph, target, numerical policy and source provenance; preserve all failures and excluded cases.
5. **Run the relevant tests** and document the command, Python version, optional dependencies, and observed outcome. If native HLS was not run, say so.
6. **Use focused commits and submit a PR** with a concise summary and any documentation changes.

New numerical results, timing claims or FPGA utilization numbers require their **actual original tool output** and provenance. An HLS estimated period is not routed WNS or hardware frequency. Do not extrapolate selected mutant detections to formal all-input correctness.

## Review checklist

- [ ] No private models, unpublished weights/datasets, credentials or proprietary/generated vendor artifacts
- [ ] Existing source semantics and public APIs preserved or deliberately documented as changed
- [ ] Unsupported variants fail with actionable errors; no silent hardware substitutions
- [ ] Relevant tests and documentation updated
- [ ] No accidental Vitis/Vivado or GitHub Actions vendor execution
- [ ] No modification to frozen original study folders or source-attributed receipts
- [ ] Rights of any third-party source or dataset explicitly identified

## Documentation contributions

Fixing unclear terminology, broken links, stale examples, missing operator restrictions and reproducibility instructions is valuable. Update user-facing guides in `docs/` and `examples/`. The long historical research notes in `paper/compiler_ir/` and `docs/maintainers/` are not the public API; avoid expanding them for a routine documentation correction.

## Vulnerabilities and support

Security-sensitive reports should follow [SECURITY.md](SECURITY.md) rather than opening a public issue with a payload or secret. Feature requests and normal bugs can use the GitHub issue templates once contributions have been coordinated with maintainers. There is no promised response SLA.

For attribution, use [CITATION.md](CITATION.md); no published DOI or open-source license is asserted.
