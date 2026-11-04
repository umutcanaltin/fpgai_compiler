# FPGAI compiler-IR — one-command scientific experiment series

This folder runs the **existing, actual compiler-paper experiment implementations**, in a source-pinned sequential workflow with one terminal command. It does not write or delete any RL/Gym code, does not start background jobs, and does not claim native HLS or external baseline results without evidence.

## One terminal command — local PC

Run from a **new clean compiler-only clone** (do not pull/change a checkout currently running a source-pinned RL L0 experiment):

```bash
git clone https://github.com/umutcanaltin/fpgai_research.git fpgai_research_compiler_paper
cd fpgai_research_compiler_paper
bash experiments/compiler_ir/run_all.sh --bootstrap --profile paper
```

The optional `--bootstrap` creates a dedicated `.venv-compiler-ir/` and installs only compiler research test and ONNX reference dependencies from the checked-out repository. It does **not** upgrade the original RL checkout's environment, install Torch, or invoke Vitis. If you have already installed dependencies, omit `--bootstrap`. The complete source-based series automatically runs all E0/E1 available campaigns, original native **source preparation and zero-call audits**, actual ONNX Runtime/GCC conformance, and the paper CPU table.

**Output** is a create-only directory named `build/compiler_ir/series/YYYYMMDDTHHMMSSZ_<gitSHA>` with `series_manifest.json`, `series_summary.json`, stage-specific raw `cases.csv` / `summary.json` / generated HLS packages and `logs/*.log`. The manifest records the exact Git SHA, config hash, step commands, exit codes, evidence tiers and per-artifact SHA-256 hashes. No original case folder is overwritten.

The default status is successful when all available **CPU** stages pass. External data requirements (actual trained MNIST, original vendor tool, prior-art executable comparisons and unsupported Conv/branching capability) remain visible as **separate research gates**, not invented successes.

## Preview, quick test, and complete CPU series

```bash
# Preview all exact subprocess commands. No files or vendor calls:
bash experiments/compiler_ir/run_all.sh --profile paper --plan

# Fast end-to-end smoke of the *same* orchestration:
bash experiments/compiler_ir/run_all.sh --profile quick

# Full CPU + GCC + independent ONNX + original native-source preparation:
bash experiments/compiler_ir/run_all.sh --profile paper
```

All commands execute in the foreground and save per-stage logs. A failed prerequisite blocks dependent steps; independent experiment families still run, so one failing campaign does not destroy the remaining evidence. A failure produces nonzero exit code. If interrupted, inspect the manifest and use `--resume <run-folder>` to run **unstarted** steps at the same Git revision and config. Already completed, failed, or partly written create-only stages are never overwritten or silently rerun. To redo a failed study, start a **fresh** output root.

Example resume (substitute the actual path printed by your run):

```bash
bash experiments/compiler_ir/run_all.sh --resume build/compiler_ir/series/YOUR_RUN_FOLDER
```

Do not `git pull`, change the configuration, or edit compiler source while an original run is executing.

## Actual trained MNIST assets (optional and never fabricated)

The research repository's original trained MNIST artifacts are not part of the public clone. Once available, provide their **verified original** MLP and/or CNN folder to the same command:

```bash
bash experiments/compiler_ir/run_all.sh --profile paper \
  --trained-mlp /absolute/path/to/original/mnist_mlp \
  --trained-cnn /absolute/path/to/original/mnist_cnn
```

The original MLP study executes the strict ONNX frontend, complete Flatten → Gemm(bias) → Relu → Gemm(bias) lowering, independent ONNX Runtime numerical reference and generated GCC host C++ for selected source-pinned examples. It separately prepares original Vitis HLS source/testbench/Tcl without calling the tool. Real CNN admission is a **coverage audit** only: the current restricted row-lifetime HLS path does not compile arbitrary Conv/branched CNNs, and its rejection is not a success claim. If private files are missing/incompatible, the corresponding steps are visibly **BLOCKED/FAILED**. No new weights or training are synthesized. Real batch-one MNIST checks semantics, **not** B-row source scratch reduction.

## Run original licensed AMD Vitis HLS on *your* separate machine

Only after sourcing your **original AMD Vitis HLS environment** and ensuring that the new clone is not the source checkout running the frozen RL native study:

```bash
# Example: source the appropriate installed AMD/Vitis/*/settings64.sh
# Exact path and version depend on your workstation.
command -v vitis_hls
vitis_hls -version

bash experiments/compiler_ir/run_all.sh --profile paper \
  --with-native --confirm-isolated-native \
  --vitis-executable vitis_hls
```

This explicitly executes **two original licensed Vitis study families** after CPU checks and create-only preparation: 12 matched whole/column-major/one-row sources and 6 matched B=1/B=2/B=M source variants under the configured part/clock. Every invocation is logged by the existing original native runner, and all cases are independently audited. The subsequent scorecards must preserve failed or absent numerical/synthesis results as unqualified; a process exit code does not establish CSim correctness, HLS timing closure, post-route Fmax, FPGA speedup, or BRAM reduction.

Hardware parameters are defined in [series.json](series.json). Set the real target part, Vitis-supported clock period and numeric tolerance **before** launching a run. The source revision and config SHA are frozen into that run; do not edit the config after experiment preparation. The Vitis suite is serial and may be resource intensive: never point it at the running L0 RL checkout.

## Certified optimizer decision interface (first version)

The full series now runs `decision_interface`: a policy-independent, typed action and source-compiled legal-action mask study, with fixed seeded random/heuristic one-action proposals, exact rejection reasons and selected original GCC numerical checks. This is **not trained RL**. The compiler recomputes source validity per action; it currently uses atomic row-stage legality and **does not yet infer per-element completion from local Phi**. Results appear under `decision_interface/summary.json` and case-level CSV. The novel mapping-conditioned IR proof and the later RL sample-efficiency study remain distinct publication gates.

## Included experiment matrix (default paper profile)

| Evidence | Stage | What actually runs |
| --- | --- | --- |
| E0/E1 | compiler_tests | All CST tests and architecture interfaces, temporal schedule and composition tests; no RL experiments |
| E1 | onnx_crosscheck | Real ONNX Runtime vs actual generated GCC C++ on representative synthetic nonlinear models |
| E0/E1 | array, row, chain, cpu_paper_table | 96 layout cases, 18 two-operator row pairs, 12 three/four-operator cases; verified case-level numerical checks and create-only paper table |
| E0/E1 | schedule_mutations | Complete planned 12-graph control campaign, valid schedules and five designed invalid intervention classes |
| E0 | independent_oracle | All 3,600 finite permutations compared with a separately implemented reference |
| E0 | scaling | 15 graphs, 2–512 rows, 126 raw checker timing samples with three repetitions |
| E1 | nonlinear | 16 source cases across six fixed static Flatten/Gemm/Relu/Gemm models, six actual host numerical checks and three traces |
| E1 | tensor_source, row_block_source | 12 + 6 original-Vitis **prepared sources** with independent audit/empty native report; no vendor call |
| E0/E1 | trained_mlp_* | Only if original trained assets supplied: ONNX coverage, actual GCC source/ONNX Runtime numerical tests, future Vitis source packaging |
| E0 | trained_cnn_preflight | Only if original trained CNN supplied: exact operator/support coverage audit (not a fake generated Conv path) |
| E2/E3 | *native execute/report* | **Only with explicit vendor opt-in** and isolated-checkout confirmation; original licensed Vitis CSim and CSynth measured where available |

**Still unsolved research gates:** A scientifically defensible novel IR must be compared directly against pinned Allo, ST-Flow and StreamTensor artifacts. The restricted CST row-chain path does not yet provide arbitrary Conv2D or branching lifetime compilation. Hardware timing, resource and power results require original vendor tools and suitable physical implementation receipts. These must not be inferred from passing CPU tests.

## Automation safety

- **Clean tracked Git checkout required**; any tracked modifications cause a preflight failure.
- **Never overwrites original cases**; creates a new per-revision, timestamped run directory.
- **No silent vendor execution**; both native flags required and `vitis_hls` must be found.
- **No RL code touched or imported**, no Docker/services/background processes, no automatic `git pull`.
- **No changing the paper based on missing results**; summary clearly labels CPU source, host GCC, original native (if executed), and external unknowns.
- **Exact-case reproducibility** uses input/source/config SHA, immutable campaign outputs, and per-stage log/file checksums. CPU timings are descriptive and may vary between machines; host float is not ap_fixed.
- `--require-all` returns nonzero if optional local requirements remain blocked, or original native/trained models were not requested. It cannot assert unknown scientific novelty or external comparator parity.

If an original experiment fails, share `series_manifest.json`, `series_summary.json` and the corresponding `logs/<step>.log`; those preserve exactly which case/phase failed and the original source revision.


## Final local workstation preflight (read-only)

Run this *after* bootstrap and *before* starting the full study:

```bash
bash experiments/compiler_ir/run_all.sh --profile paper --doctor
```

It prints an exact commit SHA, clean-checkout state, installed numpy/pytest/ONNX/ONNX Runtime, C++ compiler, free disk, and explicitly marks whether the CPU suite can start. It does **not** execute research experiments or invoke Vitis. For native readiness detection only, add `--with-native`; that probes `vitis_hls -version` but cannot verify the tool's license, target part, numerical correctness or synthesis feasibility. If you supply `--trained-mlp /actual/folder`, the doctor checks the original files and source hashes. An unsupplied original model is reported as absent, not synthesized.

**Run sequence on your PC:** first `--doctor`, then `--profile quick`, then `--profile paper`. To test original HLS, use the `--with-native --confirm-isolated-native` flags **only from an isolated compiler checkout after the matching source Vitis environment is set**, preferably first in `quick` profile, then in `paper` profile. Every invocation must be treated as an experiment, not a guarantee of a usable scientific measurement. The `series_summary.json` now explicitly separates original-HLS **pairwise qualification counts** from whether the command returned successfully. `--require-all` checks local trained-model and HLS outcomes but cannot establish priority over published compilers.

**Important:** The native Vitis experiment families use current configuration values from `series.json` (default KV260 part, 32 ns period, 8.64 ns uncertainty, 0.05 absolute fixed-point error tolerance). These were chosen as preliminary study parameters, **not experimentally optimized thresholds or achieved clock figures**. Confirm their relevance to your installed Vitis version and scientific hypothesis before creating a new source-pinned run. A strict binary/version check does not establish actual Vitis licensing or device support.

## New E0/E1 two-reader source campaigns (2026-11-02)

Both `quick` and `paper` profiles now include `fanout_exact_e0` and `fanout_source_e1`. The former performs a finite, independently replayed **abstract** fan-out tensor-base allocation study with explicit budget-unknown and infeasible controls. The latter generates a **restricted actual MatMul/two-ReLU/Add/ReLU C++ diamond** and executes three paired shapes (six cases total) under GCC with independent NumPy host-float checks and per-element source read/write traces. The pair compares disjoint 4MN source scratch elements to verified-after-last-reader 3MN source scratch elements, **not physical BRAM, RTL, Fmax or Vitis latency**. The studies and their own tests do not call vendor tools or unfreeze original RL.

First source-pinned complete run on [code revision d2b5a45a7](https://github.com/umutcanaltin/fpgai_research/actions/runs/41429203318): **23/23 quick and 23/23 paper stages passed**. For additional replicated evidence, run in the isolated compiler-only checkout, never over an active source-pinned RL workstation experiment.

## Source-pinned L6a mapped graph edge experiment

The one-command `graph_phi_e1` stage (quick **and** paper profile) exercises bounded **non-diamond** static topological 2D MatMul(constant RHS), ReLU and same-shape Add graphs. It preserves separately resolved MatMul Phi, emits actual GCC-executed source, checks first-class tensor edge value READ/COMMIT/last-use/source-address witnesses and host-float NumPy reference, and retains hashes/numeric error in `graph_phi_e1/cases.csv`. Three graph families × two mapping policies × two source scratch layouts = **12 cases**. No original Vitis is invoked, and source scratch savings are not measured BRAM.

Verified original code SHA `c1298ea35bf526b743f39c910615aa72f29afadf`; [30/30 quick and 30/30 paper original run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41450443712), [raw retained outputs](https://github.com/umutcanaltin/fpgai_research/actions/runs/41450443712/artifacts/13144403380). A native original Vitis request is **still** explicitly opt-in and should use an isolated compiler-paper clone, not the active source-pinned RL research checkout. Full contract and evidence boundary: [graph-wide CST source certificates](../../paper/compiler_ir/GRAPH_PHI_BOUNDARY_CERTIFICATES.md).

## L6a/L7 source-verified broadcast/Concat and independent access-map oracle

Two newly source-pinned one-command stages are included in quick and paper profiles: `graph_access_e1` and `graph_semantic_oracle_e0`. The former runs four rank-two broadcast/Concat graph configurations, two MatMul Phi mapping policies and two source scratch plans (**16 actual GCC numerical + 16 original source traces**); the latter has four positive controls and four synthetic correlated-semantic source-index mutants, independently checked with NumPy grids (4/4 mutant rejects; production *intentionally shared-wrong* lifetime-only E0 accepts 4/4). Only finite source logical correctness is qualified.

Original source `dbf6aade8228cbbe703037fb5108715cd0dd64f9`; [32/32 quick and 32/32 paper series](https://github.com/umutcanaltin/fpgai_research/actions/runs/41452625112), [raw evidence](https://github.com/umutcanaltin/fpgai_research/actions/runs/41452625112/artifacts/13146491650). No native Vitis tools were run. Source/readiness and last-use guarantees are NOT physical timing or RAM-port guarantees. [Detailed scientific scope](../../paper/compiler_ir/NONIDENTITY_GRAPH_ACCESS_AND_ORACLE.md).

## NCHW Conv E0 and original reader-LOAD E1 campaign

The new `conv_phi_boundary_e0_e1_numeric` stage in both one-command profiles exercises four explicit stride/pad/kernel cases with two independent Conv Phi mapping policies (**8 cases**). E0: per-kernel logical NCHW read/owner/last-use and an independent NumPy sliding-window oracle. E1: actual original canonical Conv HLS C++ compiled/run with GCC, real Conv reader per-kernel source load addresses + owner trace, independent numerical NumPy comparison. Not E1 verification of all scalar READ/COMMIT timing or emitted shared scratch memory; **zero original Vitis calls**. Source `6af10f9e2ec71810181b51560d227d8dba133303`, [original quick/paper 33/33](https://github.com/umutcanaltin/fpgai_research/actions/runs/41456356235), [raw case artifacts](https://github.com/umutcanaltin/fpgai_research/actions/runs/41456356235/artifacts/13146878908). Technical [Conv proof obligations](../../paper/compiler_ir/CONV2D_NCHW_BOUNDARY_PROOF.md).

## L6a generalized NCHW DAG scientific experiment `nchw_graph_e0_partial_e1`

4 graph/shape configurations, 2 independently selected Conv mappings and 2 **logical** scratch-layout certificates; 16 paper cases, 8 real GCC source executions (the canonical source is identical across the paired E0 layout hypotheses). Includes graph-wide Conv/Relu/Add/channel Concat/MaxPool, first-class value-edge witnesses, independent NumPy index-grid oracle, host-float numerical outputs and **actual Conv intermediate LOAD index/Phi trace**. Other source READ/COMMIT events and logical shared scratch addresses are NOT emitted/attested. Original Vitis calls 0, no physical resource/clock claim.

Source SHA `fcb1f621bb6c0eaadae8f2206e823f59bd2bf21c`, [34/34 quick + 34/34 paper original run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41473339211), [raw artifacts](https://github.com/umutcanaltin/fpgai_research/actions/runs/41473339211/artifacts/13156047647). [Precise gates and scientific limitations](../../paper/compiler_ir/GENERAL_NCHW_CST_EDGE_CERTIFICATES.md).

## 35th/36th source stages: all-intermediate GCC E1 and real shared scratch source

`nchw_all_intermediates_e1` checks **8 original GCC C++ sources**, **50 operator tensor observations** and **1,764 intermediate elements**, with independent NumPy comparison for EVERY stage. `nchw_real_source_scratch_e1` generates and executes two ACTUALLY different C++ source storage layouts per 4 graph settings × 2 Phi policies: **8 paired disjoint-vs-shared C++ scratch arrays**, **16 source variants and 32 original GCC runs** for tensor and Conv-load checks. **All 16 numerical and all 16 Conv LOAD/Phi source checks passed**; every pair shows local C++ array capacity savings (minimum 8 elements). The source optimizer is conservative first-fit, not exact or physical FPGA RAM qualification.

Original immutable code `319ab4e94a57019d0249dbe2254a925e748e4ace`; [complete 36/36 quick and 36/36 paper run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917065), [raw archived case CSVs and original logs](https://github.com/umutcanaltin/fpgai_research/actions/runs/41476917065/artifacts/13157444356). Native Vitis CSim/CSynth calls **0**. This does not prove actual BRAM savings, RAM banking, timing, II or whole-source event order. [Detailed scientific proof gates](../../paper/compiler_ir/ACTUAL_NCHW_INTERMEDIATE_AND_SCRATCH_EVIDENCE.md).

## Full NCHW source READ/WRITE and matched original Vitis preparation

Mandatory `nchw_real_source_rw_e1` runs 3 mixed NCHW graph families ×2 independent Conv mapping policies ×2 genuinely different generated C++ storage plans = **12 original GCC sources**. The actual source subscript observer checks produced-tensor READ/WRITE event order and scratch lifetime with independent dynamic NumPy counts, including extra C++ ReLU/MaxPool reads; 12/12 passed with **3,424 reads and 1,928 writes** at source `77fe5dfcd96ad81ff8d71f36fe57cb88a47411e4` ([original run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41480551778)).

Mandatory **source-only** `nchw_native_pair_packages_e1` prepared **12 actual original Vitis HLS cases, six paired disjoint/shared variants**, original code `e5502be096bd4b8ab0cf15b82e2ad4c96d0eac84`, [38/38 quick and paper original run](https://github.com/umutcanaltin/fpgai_research/actions/runs/41481941961), [raw case sources](https://github.com/umutcanaltin/fpgai_research/actions/runs/41481941961/artifacts/13159804282). Same model, Phi, input, CSim testbench, part, clock, uncertainty and tolerance per pair, source hash differs. Original vendor invocations: **zero**; hardware qualified: zero. For source-only inspection, native executor opt-in and full workstation commands see [exact handoff](../../paper/compiler_ir/NCHW_SOURCE_ACCESS_AND_NATIVE_HLS_HANDOFF.md). No native tool runs automatically in the standard series; never infer BRAM savings or timing from C++ source array sizes.


## 2026-11-03 three-contribution CST paper experiment series

The canonical single command now includes C1
(mapping-conditioned, source-level tensor-value lifetime
contracts), C2 (bounded certificate and independent oracle
controls) and C3 (**RL-compatible policy-independent**
architecture decisions). New source-only stages:

- `rl_compatible_ir_r3_r4`: 2/3/4 operator chain graph
  observation invariance, typed/action-mask checks and
  identical source/certificate results for identical
  policy-neutral externally supplied actions; no agent
  training, HLS reward or claimed sample-efficiency.
- `two_phi_counterfactual_e1`: secondary read-only causal
  analysis of the existing 24-case original GCC
  `two_phi_readers_e1` stage. Reports ineffective
  interventions and separately measured real source
  last-use/declared scratch differences, not simply
  different Phi hashes. The evidence index binds the
  summary to the original same-run GCC CSV SHA-256.

Start with:
```bash
bash experiments/compiler_ir/run_all.sh --profile paper --plan --skip-onnx
bash experiments/compiler_ir/run_all.sh --profile paper \
  --skip-onnx --output build/cst_c123_paper_20261103
```
Omit `--skip-onnx` if the independent ONNX packages are
installed and desired. Both studies are create-only and
source-pinned. Running this does **not** continue historical
`build/nchw_bram_vivado_rom_v6` Vivado; original physical
runs require separate explicit opt-in and re-audit.
A passed CPU series never establishes external peer-system
feature parity or a trained RL performance benefit.
