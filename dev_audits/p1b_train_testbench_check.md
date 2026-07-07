# P1B DDR/tiled testbench check

Finding:
- Inference `fpgai/backends/hls/testbench.py` already has a DDR/m_axi runtime-weight CSim branch, but it did not include `ddr_tiled` mode.
- Generated DDR-tiled inference tops therefore received a `weights_mem` port while the fallback testbench emitted a two-argument call.
- Training `fpgai/backends/hls/testbench_train.py` was checked. It already treats `ddr_tiled` and `ddr_tiled_mutable` as m_axi runtime-weight modes via `m_axi_weight_runtime` and passes `weights_mem.data()` to mode 0/1/2/3/4/5/6 calls.

Patch:
- Extends the inference DDR branch to include `ddr_tiled`, `runtime_ddr`, `m_axi`, and `external_ddr`.
- Adds regression tests for both inference and training DDR-tiled testbench generation.

Validation performed on patch copy:
- `python -m py_compile fpgai/backends/hls/testbench.py fpgai/backends/hls/testbench_train.py tests/test_hls_testbench_runtime_weights.py`
- `python -m pytest -q tests/test_hls_testbench_runtime_weights.py`
- Result: 2 passed.
