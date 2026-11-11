# Cross-network FPGAI compiler experiment protocol (11 November 2026)

**State: experiment automation implemented; ZERO new Vitis/Vivado results are claimed in this document.** The author's last locally tested main revision before this patch is `953eff6a5aeb614416d947631adc2d8778327641` (2,409 pytest pass / 3 skip; 190 research audit pass). Source code and tests added alongside this protocol need **new workstation verification**.

## Research question

How do FPGAI's operator-local architecture choices, numeric formats and requested clock targets change the **numerically valid HLS cycle estimates**, **HLS area estimates** and **strictly timing- and DRC-qualified routed resource results** across different *admitted* static neural network topologies?

**Scope:** Four existing deterministic, untrained ONNX models (not public accuracy benchmarks):

| Network | Family | Structure | Input → output |
|---|---|---|---|
| `tiny_mlp` | Fully connected | Gemm → ReLU → Gemm → Softmax | [1,4] → [1,3] |
| `cnn_pool_dense` | Pooling CNN | Conv → ReLU → MaxPool → Conv → Flatten → MatMul → ReLU → Gemm | [1,1,6,6] → [1,2] |
| `cnn_strided_dense` | Strided CNN | Stride-2 Conv → ReLU → Flatten → Gemm → ReLU → MatMul | [2,1,7,8] → [2,3] |
| `cnn_two_conv_softmax` | Two-convolution CNN | Conv → Conv → ReLU → Flatten → Gemm → Softmax | [1,2,5,7] → [1,3] |

These are distinct operator mixes and tensor geometries, **not equal-sized neural-network workloads**. Compare architectural deltas *within* the same ONNX graph rather than claiming that one network is inherently faster than another.

## Prespecified matrix and fair controls

- Architectures: **scalar** (control) versus existing **operator-local independent** Conv/Gemm/MatMul mapping (2).
- Fixed-point CSim/HLS types: `fx16_6` (16 total / 6 integer bits) and `fx12_5` (12 total / 5 integer bits) (2). The implementation explicitly propagates each format into operator overrides, avoiding accidental fallback to default 16/6.
- Requested clock targets: 12 ns and 20 ns (2). Same KV260 FPGA part `xck26-sfvc784-2LV-c` across cases; identical testbench seed and source numerical-tolerance policy.
- Four networks × two mappings × two precisions × two clocks = **32 prepared design points**. Default 2 deterministic synthetic CSim input samples per case; never label them classification accuracy.
- Optional confirmatory target sweep: `--clocks 12 16 20 24` creates **64** design points. Use **one prespecified population** for any publication table, and report all blocked/failed points in the denominator.
- No "winning" architecture may be selected from one tool stage and reported as though the same workload and numeric format had independently been qualified at another stage.

## Exact commands — author workstation

From clean, installed `fpgai_compiler` checkout on `main`:

```bash
source .venv/bin/activate
python -m pytest -q tests/test_network_benchmark.py
python -m scripts.network_benchmark plan
# SOURCE GENERATION ONLY: creates fresh ONNX, HLS source, CSim testbench, Vitis Tcl, Vivado Tcl.
python -m scripts.network_benchmark prepare --output build/network_matrix_20261111_01
# DRY-RUN: no vendor process launched.
python -m scripts.network_benchmark run --root build/network_matrix_20261111_01 --stage vitis --first 0 --limit 1
# Explicit opt-in to licensed Vitis, only on configured workstation:
python -m scripts.network_benchmark run --root build/network_matrix_20261111_01 --stage vitis --first 0 --limit 1 --execute
# Do not start Vivado until exact-source E3 plus numerical CSim is qualified:
python -m scripts.network_benchmark run --root build/network_matrix_20261111_01 --stage vivado --first 0 --limit 1 --execute
# READ ONLY: collect source-attested results into a new immutable directory.
python -m scripts.network_benchmark collect --root build/network_matrix_20261111_01 --output build/network_matrix_report_20261111_01
```

Choose **fresh create-only `build/` paths**, and never rerun/overwrite a case with existing native receipts or project output. If Vitis/Vivado is missing or failed, retain the failed case and its logs; collect results with metrics as null/TBD. Execute further cases via bounded slices (`--first`, `--limit`) only after initial smoke results are checked. This protocol does **not** invoke GitHub Actions.

## Evidence and statistical interpretation

| Stage | Valid observations | Must remain unknown |
|---|---|---|
| E1 — prepared, hash-verified compiler source | ONNX shape/operator composition, IR source SHA, prepared HLS/testbench/native Tcl identity | Synthesis cycles, DSP/BRAM/LUT, real timing/Fmax, physical power and runtime |
| E2 — source-attested Vitis CSim passing, full generated testbench coverage | Fixed-point synthetic numerical agreement under stated tolerance | Hardware resources/cycles and classification accuracy |
| E3 — fresh source/tool-attested Vitis CSynth **plus E2 pass** | Best/worst HLS-estimated cycles, initiation interval and estimated LUT/FF/DSP/BRAM18/URAM; requested-clock `cycles × requested_ns` nominal **latency proxy** | Routability, achieved Fmax, actual board inference latency |
| E4 — matching source/RTL-attested Vivado OOC | Observed routed WNS, utilization, DRC; exact `strict_e4_qualified` requires 0 total violations, explicit clean status, positive route completeness, timing-met, valid CSim/E3 | Board-inference latency, energy, power, achieved Fmax unless actually measured |

The same model, mapping, precision and requested clock define matched scalar vs mapped pairs. A pair's cycle/LUT delta is **qualified only when both arms satisfy strict E4** and both have acceptable E3 cycles. Other paired differences are null, not secretly a success. Report E4 nonclean/negative-WNS observations descriptively, with failed status and their measured values, and never use them for favorable speedup claims.

Clock sweeps provide the highest **qualified tested target frequency**, not the true continuous maximum/achieved Fmax. For a fully matched board-level latency or power claim, perform separately scoped hardware instrumentation with exact source/bitstream identity and accepted board power/IO timing criteria. These experiments are deliberately small synthetic workload probes, not performance comparisons against published deep-learning networks.

## Artifacts and ownership

`scripts/network_benchmark.py` owns preparation, **explicit opt-in** staged vendor invocation through existing `toolchains.*.runner` interfaces, and read-only metrics aggregation through existing `results.summarize_case` and shared `suite_native_batch` E3/E4 qualifiers. It does not implement a second compiler, synthesis parser or plotting oracle. `tests/test_network_benchmark.py` owns source integrity, fake-data qualification adversaries and create-only rules. `Makefile` exposes planning/testing targets only. `build/` carries all new case models, HLS packages, native receipts, CSV/JSON/Markdown tables; none are tracked or promoted to paper evidence without actual author run and audit.

## Paper release locks

No new experimental result has been executed by committing this protocol. The exact-source Allo comparison, original C1/C2 reproducibility, timing-clean/DCR-clean hardware evidence, board power/latency and institutional redistribution approval remain separate open paper/release gates. Even a 32-case matrix with source-only outputs cannot close those gates.

## 12 November 2026 — Vitis HLS startup preflight and local recovery

The author verified source preparation and all 27 original benchmark tests but `vitis_hls` was not on their active shell `PATH`. This fails **before HLS launches** and is not a compiler or HLS-result observation. A Python virtual environment does not configure AMD binaries. Check actual installation and current shell:

```bash
command -v vitis_hls || true
command -v vivado || true
# Locate the REAL installed settings script / executable (example search roots; optional paths may not exist):
find /tools /opt /usr/local "$HOME" -type f \( -name vitis_hls -o -name settings64.sh \) 2>/dev/null | grep -Ei 'Vitis|Vivado' | head -50
```

If an AMD 2023.2 installation is found, `source "<actual-installation>/settings64.sh"` in that same shell, then `command -v vitis_hls` and `vitis_hls -version`; the tool itself determines its actual version. The benchmark now supports zero-execution doctor mode:

```bash
python -m scripts.network_benchmark doctor --stage vitis
python -m scripts.network_benchmark doctor --stage vivado
# Or explicitly select a verified executable if PATH was not configured:
python -m scripts.network_benchmark doctor --stage vitis --executable /REAL/ABSOLUTE/PATH/TO/vitis_hls
```

**Do not rerun `prepare`.** The source-prepared `build/network_matrix_20261111_01` remains usable. When doctor returns `"available": true`, try one case:

```bash
python -m scripts.network_benchmark run --root build/network_matrix_20261111_01 --stage vitis --first 0 --limit 1 --execute
# Or append --executable /REAL/ABSOLUTE/PATH/TO/vitis_hls
```

If native execution produces an invocation receipt, do not run the same case twice: inspect its logs and use a new create-only experiment directory for any replacement. This *new* code and tests must first pass the local focused pytest and research audit. No native E2/E3/E4 results were produced by author at the time of this protocol update.
