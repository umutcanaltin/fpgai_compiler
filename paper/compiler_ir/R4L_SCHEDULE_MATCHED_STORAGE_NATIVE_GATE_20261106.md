# R4k original E1 verified and R4l guarded native E2/E3 gate — 6 November 2026

**Evidence status:** R4k source-only, schedule-matched E1 was executed by the author and **40 focused regression tests passed** on the workstation. The new R4k original create-only study succeeded with status `MATCHED_ROW_SOURCE_SCHEDULE_REAL_HOST_E1_ACCEPTED`, stored at `build/r4k_source_schedule_matched_storage_20261106_01`; frozen source revision **776e3a7a4b599fd114718f6a03ab73e06be17f5c**. It performed **two real GCC compilations/executions** within the E1 acceptance test and **zero new Vitis/Vivado calls**. The user-reported plan pins the 32/6 graph, 12.0 ns clock, 1.5 ns explicit uncertainty, tolerance 0.02, and input model architecture.

These two E1 arms vary H-memory layout/capacity but share the **same row-local source event order** and independently verified functional and architecture identities. The row source has **240 fewer C++ scratch array elements**, with all intermediate/output values and original non-address event traces matching at E1. It is a controlled SOURCE schedule comparison, not a claim that Vitis HLS will infer the same pipeline, clock schedule, RAM ports or timing.

## R4l scientific question

For the newly generated R4k 32/6 matched source pair, do both original Vitis HLS CSim fixed-point checks pass E2, and, if so, what do their source-attested HLS E3 resource estimates, worst cycles and estimated clocks show when their **source loop/reader schedule is held constant**?

R4l does NOT compare a new arm with the frozen original mixed-schedule full-H HLS run for causal resource attribution. The only controlled comparison is between the **TWO new R4k packages**, `full_storage` and `row_storage`. The separate historical R4d data remain the previous mixed-schedule observation.

## New implementation and files

- `src/fpgai_research/experiments/cst_r4l_schedule_matched_storage_native.py` — SHA-verifies all original R4k source/E1 outputs and the frozen R4a/R4d native HLS lineage; dry-run, explicit opt-in real Vitis execution, and separate read-only E2/E3 qualification.
- `tests/test_cst_r4l_schedule_matched_storage_native.py` — synthetic source packages and mocked vendor results for no-native planning, SHA tamper, missing confirmation, prior partial attempt, vendor-before-qualify ordering, E2 fail and E3 null behavior, and fully qualified resource deltas. Mock values are illustrative **NOT R4l HLS measurements**.
- `paper/compiler_ir/R4L_SCHEDULE_MATCHED_STORAGE_NATIVE_GATE_20261106.md` — this preregistered evidence protocol.
- Existing `paper/compiler_ir/FIRST_PAPER_SCIENTIFIC_COMPLETION_20261106.md` and draft PR #2 are updated to mark R4k E1 **observed** but R4l native **not yet performed**.

## Input integrity and safety gates

1. Check complete original frozen R4a/R4d original SHA inventories, including both E2/E3 accepted original 32/6 arms, original row-H C++ SHA and Tcl SHA. Never rerun their original Vitis cases.
2. Check the new, create-only R4k E1 result status, source revision, source/architecture/functional hashes, GCC count, every original listed artifact SHA and both host outputs' identical source logical trace hash (after removing address witnesses). Require two untouched original packages with identical testbench, precision 32/6, model, part/clock/tolerance and Tcl.
3. **Dry-run by default.** Native tool invocation requires BOTH `--execute-native` and `--confirm-isolated-native`. Before any vendor run, reject **either arm's** existing invocation receipt, log directory or project directory. If an attempt was interrupted, do NOT overwrite or restart it; preserve the partial evidence.
4. If explicitly executed, run both new, separate `vitis_source` packages through the canonical `run_vitis()` runner, then separately inspect process receipt and source-attested CSim/HLS artifact qualification. Process exit 0 is NOT numeric qualification.
5. Read-only `--audit-native` reports **E2-qualified counts** and **E3-qualified counts**. An arm's E3 requires a native invocation file AND accepted E2. Both-arm resource differences are numeric only if BOTH arms independently E3-qualified; otherwise they remain `null`. No physical E4 result is inferred.
6. Preserve source-only SHA inventory unchanged. Running Vitis may add vendor outputs ONLY inside the NEW R4k package subdirectories. Do not overwrite original R4k source/testbench/Tcl/host outputs, other R4a/R4d evidence, or a partially attempted arm. No GitHub Actions.

## Safe workstation procedure

First run the **source-only, vendor-free** checks:

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m pytest -q \
  tests/test_cst_r4l_schedule_matched_storage_native.py \
  tests/test_cst_r4k_matched_row_schedule_storage.py \
  tests/test_cst_r4j_memory_instance_source_sites.py

python -m fpgai_research.experiments.cst_r4l_schedule_matched_storage_native \
  --study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01
~~~

If and only if the above passes and the R4l dry run reports exactly two planned new vendor calls, the new matched Vitis experiment may be run **ONCE**:

~~~bash
python -m fpgai_research.experiments.cst_r4l_schedule_matched_storage_native \
  --study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --vitis-executable /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls \
  --execute-native --confirm-isolated-native
~~~

Afterwards, inspect the source-preserving native audit:

~~~bash
python -m fpgai_research.experiments.cst_r4l_schedule_matched_storage_native \
  --study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --audit-native
~~~

**Status for publication:** R4k E1 accepted and source-verified; R4l new native tests and runs are **not yet executed on the author's workstation**. Until the E2 and E3 gates complete, the R4k storage-only hardware resource deltas are **UNKNOWN**, not zero or measured. E3 is HLS-estimated only, not placed/routed FPGA throughput or frequency. The first paper remains **MAJOR REVISION / NOT SUBMISSION READY**.

## Verified R4l original Vitis evidence — 6 November 2026

The author pulled the R4l branch at `a3cc5cafd8a53d8edab334bae888058a80db7ef0`, ran **35/35** focused R4l/R4k/R4j tests, and verified dry-run with exactly **two planned Vitis calls and zero calls made**. The read-only preflight confirmed original source revision `776e3a7a4b599fd114718f6a03ab73e06be17f5c`, R4a revision `fe99d627a5a16ce75e5d4df4542bb0f0f93e6ab0`, and R4d revision `abe6aa37e5c799f27312b7f3a17309c6122ed7cb`.

The author subsequently explicitly executed **both NEW** R4k Vitis HLS 2023.2 source packages (`full_storage`, `row_storage`) ONCE. Both native processes exited zero, both E2 CSim gates passed **0/128 failures** (max absolute error `3.7956994241e-06`, RMSE `8.1167162467e-07`), and both were E3 HLS qualified. The independent read-only `--audit-native` confirmed the paired qualification and **zero native calls during audit**:

| HLS-only metric (32/6) | `full_storage` | `row_storage` | row minus full |
|---|---:|---:|---:|
| Estimated BRAM18K | 13 | 13 | **0** |
| Estimated DSP | 21 | 19 | **−2** |
| Estimated LUT | 5,095 | 4,755 | **−340** |
| Estimated FF | 2,120 | 1,647 | **−473** |
| Estimated URAM | 0 | 0 | 0 |
| HLS worst cycles | 22,954 | 22,602 | **−352** |
| HLS estimated clock period (ns) | 9.857 | 9.500 | **−0.357** |

The C++ source SHA-256 for the new full-storage arm is `03b0cc275ba27294d99adb14d597717d2a9035853b572b4dff6d7e6dcc4df98f`; its original CSynth XML SHA is `854d1ac848de0dd7e8edb4302108b9a7dceb7a982aab7902a8101b8ef1a6c703`. The row-storage arm's C++ SHA-256 is `aa0a7a2f684f3c0e6e64f6b4e3084c8f0dc93ae152318bd5339d9646b4894e9e`, and its CSynth XML SHA is `fe929aaa9db08990e5f7556d750d7d1eb7ef081c94b855424cccea38eb2644eb` (also the previously qualified R4e row-H RTL source case). The Tcl and CSim testbench digests are identical across the R4l source arms.

**New observed conclusion:** When original source loop/reader issue order is held constant, choosing row-retained versus full scratch address allocation reduces HLS-estimated DSP, LUT and FF and improves estimated worst cycles/clock period for this one 32/6 matched fanout model. It does **not** reduce HLS-estimated BRAM count. This does **not** show equal HLS hardware schedules, a physical RAMB reduction, routed timing, real hardware frequency, power or generality across graph topologies. E4 remains `null`, not measured.

**IMPORTANT: both R4l vendor calls are already complete. Do not execute R4l `--execute-native` again.** Only run read-only audits and source-attested new analyses. Preserve all original Vitis outputs, reports and invocation receipts. The recommended next study is a separate read-only three-point decomposition of the already completed R4e/R4l matched source cases; its HLS differences remain descriptive source-controlled contrasts, not a fully randomized factorial causal estimate.
