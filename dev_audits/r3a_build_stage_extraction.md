# R3A Build-stage extraction

This refactor extracts build-stage resolution helpers from `fpgai/engine/compiler.py` into `fpgai/engine/build_stages.py`.

## Intent

- Reduce `compiler.py` without changing compiler behavior.
- Keep existing compatibility imports from `fpgai.engine.compiler` for tests and callers.
- Establish a focused owner for `build.stages` validation and summary logic.

## Extracted symbols

- `BUILD_STAGE_KEYS`
- `cfg_has_path`
- `as_bool`
- `resolve_build_stages`
- `validate_build_stage_dependencies`
- `build_stage_summary`

## Compatibility

`compiler.py` imports the canonical functions with the old private names:

- `_BUILD_STAGE_KEYS`
- `_cfg_has_path`
- `_resolve_build_stages`
- `_build_stage_summary`

This keeps current tests and internal references working while reducing the compiler module.

## Validation command

```bash
python -m pytest -q \
  tests/test_build_stages.py \
  tests/test_paper_validation_trace.py \
  tests/test_paper_evidence_chain.py \
  tests/test_runtime_package.py \
  tests/test_generated_hls_explanation_and_numeric.py
```
