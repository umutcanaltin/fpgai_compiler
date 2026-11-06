# R4z: 24 ns four-arm Vitis CSim/CSynth — prospectively isolated native experiment

**Status:** New R4z source staging, guarded native runner, read-only auditor and tests are committed. These NEW tests have not yet run on the author's workstation, and no new R4z vendor experiment has executed.

## Original R4y evidence now verified

At source commit `dde56d405b951431b426e121a32596a0a789c973`, the author passed **60/60** focused regression tests in **48.71 s**, then successfully ran `make research-r4y-plan`, `make research-r4y-prepare` and `make research-r4y-audit`. This is a fully prepared four-arm **source-only** cohort at 24 ns/1.5 ns, not new HLS numerical/synthesis or mapped hardware evidence. Four distinct original source digests, one common numerical testbench SHA `50aff21ba41562dd81e2655022fe43211cfee6183fa4c54aee24838f25ce341d` and one common Tcl SHA `f0fc4740ecf7b8cccdaa92491ac99da15269f5c1409060d9a82939e441fe771d` are SHA-audited. Original frozen R4k revision `776e3a7a4b599fd114718f6a03ab73e06be17f5c` is unmodified. All E2/E3/E4 and board latency results remain null.

## Exact controls and result gates

R4z copies only `model.cpp, tb.cpp, run_hls.tcl, manifest.json` from EACH original R4y source-only case into a **new create-only directory**. Inputs are byte-attested to their original source root; 17 hashes (16 case inputs plus study summary) are locked and rechecked before every action. Read-only R4y provenance audit is rerun; no original file or native lock is modified.

Four cases at identical 24 ns, 1.5 ns HLS uncertainty, xck26 part, ap_fixed<32,6>:

| Source storage | Pipeline source intervention | Case |
|---|---|---|
| full | canonical baseline | full_storage__baseline |
| full | producer MAC-loop PIPELINE II=1 **requested** | full_storage__producer_mac_ii1_requested |
| row | canonical baseline | row_storage__baseline |
| row | producer MAC-loop PIPELINE II=1 **requested** | row_storage__producer_mac_ii1_requested |

No case is described as DSP-pipelined until actual RTL inspection confirms it; **top-level HLS II is not necessarily achieved producer-loop II**.

Default execution path is a dry-run. Actual Vitis execution requires BOTH `--execute-native` and `--confirm-isolated-native`, a bounded timeout and a real tool executable. A one-time attempt lock is written BEFORE the first native call; it remains even after any interrupted/partial vendor run. Never delete/override locks to obtain a different result.

Read-only `audit_native()` uses the canonical repository evidence parser and `qualify_case` to distinguish original-source E2 CSim pass/failure from E3 HLS synthesis estimates. Cases with failed or unattested E2/E3 contribute to the denominator but have null quantitative performance. It records numerical error, HLS worst cycles, estimated clock, approximate top-level II, HLS LUT/FF/DSP/BRAM/URAM, and **HLS requested-period × worst-cycles** as an explicitly **NOT-measured** schedule product. For correctly matched E3-qualified arms only, it reports storage contrasts at each pipeline setting, within-storage pipeline contrasts, and the complete 2x2 interaction (difference-in-differences). Failed or absent cases force those comparisons to null. Physical routed E4, verified MAC-loop II and measured FPGA/board/DMA latency are ALWAYS null.

## Safe workstation verification first — zero native calls

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
(
set -euo pipefail
git switch research/first-paper-source-memory-bridge-20261106
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106

python -m pytest -q \
  tests/test_cst_paper_r4z_24ns_four_arm_native.py \
  tests/test_cst_paper_r4y_24ns_pipeline_source_prepare.py \
  tests/test_cst_paper_r4x_feasible_only_core_latency.py \
  tests/test_cst_paper_r4w_feasible_clock_sweep.py

# R4y is ALREADY prepared; do not regenerate or overwrite.
make research-r4y-audit

test ! -e build/r4z_24ns_four_arm_native_20261107_01
make research-r4z-native-prepare
make research-r4z-native-dry-run
make research-r4z-native-audit
)
~~~

These targets have NO Vitis, Vivado, board or GCC execution. If any check fails, STOP and preserve both original and new study roots.

## Eventual real Vitis execution, separately authorized, not run

Only AFTER successful new synthetic tests and SHA source prep, the author may intentionally execute four fresh HLS cases, once, with:

~~~bash
python -m fpgai_research.experiments.cst_paper_r4z_24ns_four_arm_native \
  --study build/r4z_24ns_four_arm_native_20261107_01 \
  --execute-native --confirm-isolated-native \
  --vitis-executable /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls \
  --timeout-seconds 7200
make research-r4z-native-audit
~~~

No new native result has yet been observed. Once E2/E3 are independently accepted, a SEPARATE four-arm, same-clock Vivado routed study with source/ROM SHA, complete setup and hold, DRC severity, all-path constraints, resources and false-acceptance checks remains required. A board implementation is a further independent integration experiment with instrumented end-to-end inference latency. The 24 ns clock is an exploratory conservative engineering candidate, NOT guaranteed board safe. Paper priority remains C1 source-event contracts and C2 policy-independent decisions, with independent prior-art and CPU-cost evaluations.
