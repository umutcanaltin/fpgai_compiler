# FPGAI F5A Selectable Fused Update Patch

Changed files:
- `fpgai/backends/hls/emit/top_train_cpp.py`
- `tests/test_optimizer_support_contract.py`
- `configs/examples/training_adam_kv260_fused_update_direct.yml` (new)
- `KNOBS_TUTORIAL.md`

Implementation:
- preserves `full_buffer` and `tiled_accumulate` as explicit options;
- enables `fused_update` for Dense + Adam + direct single-record training;
- defaults parameter-gradient storage to `recompute` only when fused update is selected;
- rejects contradictory materialized storage choices;
- removes complete and tiled weight-gradient buffers;
- consumes each recomputed weight gradient directly in the Adam update;
- recomputes gradients for export when requested;
- retains persistent Adam state arrays.

Validation performed:
- `tests/test_optimizer_support_contract.py`: 47 passed
- related training/HLS suites: 59 passed, 36 skipped
- full suite collection blocked in the patch environment because `onnx` and `onnxruntime` are not installed.

## F5A.1 compiler-gate correction

The initial package updated the HLS training emitter but missed the earlier validation in `fpgai/engine/compiler.py`. The corrected package now:

- enables `training.gradients.computation=fused_update` at compiler planning time;
- resolves omitted parameter-gradient storage to `recompute` only for fused update;
- accepts explicit `training.storage.parameter_gradient=recompute` for fused update;
- rejects BRAM/URAM requests for fused update because no gradient buffer is materialized;
- keeps DDR gradient storage unsupported until its real lowering exists;
- preserves BRAM as the default for full-buffer and tiled-accumulate modes.

Compiler-level regression: `48 passed` in `tests/test_optimizer_support_contract.py`.
