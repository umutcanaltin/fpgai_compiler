# FPGAI testing guide

FPGAI uses tests as compiler contracts. Sprint-specific tests may exist during
development, but public coverage should be organized around subsystem behavior:
configuration, memory/data movement resolution, code generation, numeric
validation, runtime packaging, and paper verification.

## Recommended quick smoke

```bash
python -m pytest -q \
  tests/test_training_memory_storage_contract.py \
  tests/test_memory_storage_effect.py \
  tests/test_runtime_package.py \
  tests/test_memory_semantics_classifier.py \
  tests/test_build_stages.py
```

## Test markers

- `unit`: fast tests for pure resolver/config/layer logic.
- `codegen`: generated-source and HLS interface contract tests.
- `validation`: numeric checks against Python/reference artifacts.
- `integration`: full compiler/user workflows that do not require FPGA tools.
- `hls`: tests requiring Vitis HLS or HLS execution artifacts.
- `vivado`: tests requiring Vivado.
- `fpga`: tests requiring a physical board.
- `slow`: long-running tests.

## Cleanup rule

Do not delete old tests until replacement coverage exists. First classify each
test as keep, merge, move, delete-after-replacement, obsolete, or invalid.
