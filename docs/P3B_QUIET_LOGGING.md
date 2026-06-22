# P3B — Quiet Compile Logging

This sprint makes FPGAI usable as a public CLI tool by preventing vendor-tool output from flooding the terminal by default.

## Behavior

```bash
fpgai compile --config configs/examples/inference_compile.yml
```

prints the FPGAI compile summary and log-file paths only. Internal compiler, Vitis HLS, and Vivado output is captured under:

```text
build/cli_logs/*_compile_stdout.log
build/cli_logs/*_compile_stderr.log
```

Verbose mode streams full output:

```bash
fpgai compile --config configs/examples/inference_compile.yml --verbose
```

Minimal mode prints only the shortest compile summary:

```bash
fpgai compile --config configs/examples/inference_compile.yml --quiet
```

`--verbose` and `--quiet` are mutually exclusive.

## Rationale

Open-source users should see concise FPGAI status messages by default. Full Vitis/Vivado logs remain available in files for debugging and reproducibility.
