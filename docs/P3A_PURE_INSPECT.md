# P3A Pure Inspect Command

`fpgai inspect --config ...` must be a pure metadata/config inspection command.
It must not run compile, benchmark, Vitis HLS, Vivado, precision sweeps, or training comparison.

The bug fixed in this sprint was CLI dispatch order: the legacy top-level `--config`
check ran before explicit subcommand dispatch. Therefore `fpgai inspect --config ...`
was treated like `fpgai --config ...`, which calls `run_from_config(..., action="auto")`.
If `benchmark.enabled` is true, this launches the benchmark/HLS path.

The fix dispatches explicit subcommands first and uses legacy top-level `--config`
only when no subcommand is provided.
