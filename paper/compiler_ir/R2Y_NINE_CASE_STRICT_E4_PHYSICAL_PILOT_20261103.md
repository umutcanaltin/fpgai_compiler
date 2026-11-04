# R2y — Prospective nine-case, seven-pair strict Vivado OOC source-pinned pilot

**2026-11-03 original status: workstation verification PASSED; physical pilot PARTIALLY EXECUTED, 1/9 original Vivado cases attempted.** Clean source-preparation revision `303209deaf660334a798337c187cff8dbaae6e27`: **30/30 focused tests and 1,595/1,595 full tests passed**; all nine source/RTL/ROM-pinned Vivado cases were prepared and independently audited, with 0 original Vivado invocations before native execution. The first opt-in native run, `case_000`, **exited 0 and produced a source-attributed routed observation with disclosed DRC warnings**, classified `ROUTED_WITH_DISCLOSED_DRC_NOT_E4` and **strict_clean_DRC_E4=false**. The original Vivado result/receipt is now frozen; **8/9 physical cases not yet attempted, 0 strict E4-qualified cases and 0 E4-qualified pairs**. No actual DRC rule IDs, routed WNS, resource values or warning counts were pasted yet; do not infer them from process success. The remaining cases must not be described as physically measured. This pilot must never overwrite the original 32-case R2u Vitis study or any R2v/R2w/R2x analysis v1 output.

## Why an actual new physical pilot is necessary

Original R2x static RTL hierarchy was verified on the workstation at code revision **`e5ea6a14d279be786492f338bdf1e51ba684ba73`**: **23/23 focused** and **1,588/1,588 full pytest**, **32/32 source-attested Verilog cases**, **32/32 top `fpgai_model` static type graphs analyzable**, and **64/64 source-attested matched RTL type-graph comparisons analyzable**. Original Vitis source revision remained `0c6d85f189062df8d5157237f3c8ac74ac8da4b8`, with no additional native calls.

The original R2x type graph found **10 recognized memory-like module types and 10 recognized memory-like type instances** for table lookup versus **6 and 6** for switch lookup in the two-reader graph, and **16/16 versus 8/8** in the four-reader graph. This is a bounded *static source type/instance graph*, NOT elaboration, physical memory mapping, RAM ports or cell counts. All 14 E3-qualified address-lookup comparisons in the earlier R2v study tied in **HLS-estimated BRAM** regardless of this source HDL organization difference. Across the complete 64 R2v contrasts, **2 estimated-BRAM decreases, 47 ties, 8 increases, 7 E3 unknown**; original source scratch reduction rarely survived HLS synthesis.

The earlier `R2j/R2l/R2m` physical six-pair study, performed under **different 32 ns HLS source conditions**, returned **six descriptive routed BRAM ties but zero strict clean-DRC E4-qualified pairs**. We must not pool that source protocol with the newer R2u **8 ns** HLS factorial or infer an original physical effect from the HLS RAM directive. The remaining scientific decision is whether a tightly selected, original **source-matched Vivado OOC** cohort can produce independently qualified physical area/timing results or whether the paper remains a bounded negative source-to-synthesis translation case study.

## Prespecified NEW routed cohort

R2y reuses the existing canonical `toolchains.vivado.project.make_vivado_tcl`, `toolchains.vivado.runner.run_vivado`, `toolchains.vivado.rtl_inputs.inventory_vitis_rtl`, original ROM attestation and `toolchains.vivado.report.parse_vivado_reports` without rewriting those toolchain semantics.

Source input: preserved original `build/r2u_hls_factorial_32_original_v1/`. Before creating output, require the complete original 32 Vitis case/source/factor checks, **32/32 E2, 30/32 E3**, original `0c6d85f...` source revision and the exact **8 ns** HLS requested clock. Preselect exactly **nine** E3-qualified original HLS cases:

| Original case | Graph | Storage | Address | Binding | Padding | Role |
|---|---|---|---|---|---|---|
| case_000 | 2 readers | Disjoint | Table | Auto | Exact | Graph 0 common control |
| case_001 | 2 readers | Disjoint | Table | Auto | Pad 64 | Padding intervention |
| case_002 | 2 readers | Disjoint | Table | Requested BRAM | Exact | Binding intervention |
| case_008 | 2 readers | Element-exact | Table | Auto | Exact | Source-allocation intervention |
| case_016 | 4 readers | Disjoint | Table | Auto | Exact | Graph 1 common control |
| case_017 | 4 readers | Disjoint | Table | Auto | Pad 64 | Padding intervention |
| case_018 | 4 readers | Disjoint | Table | Requested BRAM | Exact | Binding intervention |
| case_020 | 4 readers | Disjoint | Switch | Auto | Exact | Address-generation intervention |
| case_024 | 4 readers | Element-exact | Table | Auto | Exact | Source-allocation intervention |

**Seven fixed matched contrasts** (intervention minus control): `000→008` storage, `000→002` binding, `000→001` padding; `016→024` storage, `016→020` address, `016→018` binding, `016→017` padding.

**No original case_004 or case_012** is eligible for a source-attributed E3-based routed comparison because both failed the effective HLS clock budget in the 8 ns original protocol. Therefore the table→switch factor is routed only on the four-reader graph; there is no attempt to hide the unavailable two-reader address comparison.

The selected `016→024` source-storage comparison is particularly discriminating because the corresponding original source-attested HLS estimated BRAM_18K **decreased by one**. The graph 0 storage comparison did not show an HLS BRAM estimate reduction, while bindings/padding can increase estimated HLS BRAM. This fixes scientifically useful positive and negative controls *before routing*, not after seeing new physical results.

## Preparation and native evidence gates

Implementation `src/fpgai_research/experiments/cst_r2y_controlled_physical_pilot.py`; tests `tests/test_cst_r2y_controlled_physical_pilot.py`. Source preparation:

- Read-only re-audit all frozen original R2u package/Tcl/source/report SHA evidence and 32/32 E2 / 30/32 E3 status, verify each selected original Vitis `rtl_output_sha256` and `rom_output_sha256`, reject symlinks, missing ROM assets and source changes.
- Verify each pair holds **graph, Phi architecture, numerical input/testbench/HLS Tcl and all other three source factors constant**. Preserve the target part and exact 8 ns requested clock; require the original HLS 1.5 ns clock uncertainty in the original source manifest. **The canonical Vivado OOC Tcl sets an 8 ns clock period but does NOT carry over Vitis's 1.5 ns uncertainty automatically**. Thus within-R2y routed comparisons share Vivado constraints, but physical WNS must not be equated directly with HLS's effective 6.5 ns scheduling budget. Do not pool this cohort with the old 32 ns R2j campaign.
- Copy each selected original source `model.cpp`, testbench and case manifest, original named Verilog and source ROM assets, and stage read-only source-derived ROM working copies in a **NEW** `build/r2y_vivado_pilot_9_original_v1/` directory. Generate its own fixed Vivado out-of-context script; **no board pins and no bitstream**.
- The preparer never calls a vendor tool. The native execution path requires **both** `--execute-native` and `--confirm-isolated-native`, preflights no prior original Vivado receipt/log/project output, and refuses to overwrite an already attempted case.
- The read-only auditor checks copied source, Verilog, ROM and Vivado script SHA; genuine original vendor invocation/route report identity and freshness; original $readmem input conservation or independently verified vendor-read post-run absence; complete routing, source-attributed WNS and all physical resource counts.
- Only a **fully routed**, nonnegative timing case with **verified ZERO DRC violations and ZERO critical warnings** qualifies as strict **`STRICT_CLEAN_DRC_VIVADO_OOC_E4`**. Fully routed timing observations with disclosed DRC warnings remain a **separate descriptive tier**, never clean E4. If a case is absent, fails routing, retains invalid ROM provenance or fails DRC/timing, no pairwise E4 resource numbers are invented. A successfully completed Vivado process does **not** automatically establish E4.
- The seven-pair report explicitly separates E4-qualified physical BRAM/LUT/FF/DSP differences from warning-only descriptive physical resource deltas and all UNKNOWN outcomes. It does **not** report achieved deployed maximum Fmax, board performance, watts or end-to-end functional equivalence.

Original source-only and vendor input artifacts are immutable. This pilot writes only a new physical package/output and, on explicit native execution, native results in **that new study**. No GitHub Actions or frozen RL changes.

## First original routed case: source-pinned receipt, warning-only outcome

The original R2y physical folder `build/r2y_vivado_pilot_9_original_v1/` contains an **already executed** `case_000/vivado/` with its Vivado receipt and reports. The process exited successfully, and the R2y verifier accepted a fully routed timing observation **only at its DRC-warning-disclosed descriptive level**. **No clean-DRC E4 result was achieved** by this case, and no pair can qualify from one observed arm. It is essential to inspect the original SHA-attested DRC rule dictionary, WNS and resource counts before describing any mechanisms. The original outputs are **immutable**; do not invoke `--start-case 0` again, touch the Tcl, or remove the reports to force qualification.

The original study's first audited result has status `ROUTED_WITH_DISCLOSED_DRC_NOT_E4`; this does not say which violations occurred. An independent diagnostic read-only snapshot is the next task. Even if Vivado reports all remaining 8 cases routed with similar warnings, no strict E4 resource comparison exists without BOTH members of a pair passing the clean DRC gate. Warning-only observations may be discussed separately with their full DRC rules and original source/report hashes.

## Linux workstation commands

First pull and validate the committed code. **Do not run original Vitis HLS again.**

```bash
cd ~/Desktop/github_projects/fpgai_research

git status --short
git pull --ff-only origin main
git rev-parse --short HEAD

python -m compileall -q src/fpgai_research experiments/compiler_ir tests

python -m pytest -q \
  tests/test_cst_r2y_controlled_physical_pilot.py \
  tests/test_cst_r2x_rtl_hierarchy_analysis.py \
  tests/test_cst_r2w_rtl_memory_forensics.py \
  tests/test_cst_r2v_hls_factorial_review.py

python -m pytest -q
```

Only after tests pass, prepare the new original R2y study in a **fresh, unused** path:

```bash
python -m fpgai_research.experiments.cst_r2y_controlled_physical_pilot \
  --prepare \
  --study build/r2u_hls_factorial_32_original_v1 \
  --output build/r2y_vivado_pilot_9_original_v1

python -m fpgai_research.experiments.cst_r2y_controlled_physical_pilot \
  --audit \
  --study build/r2y_vivado_pilot_9_original_v1

python -m fpgai_research.experiments.cst_r2y_controlled_physical_pilot \
  --execute \
  --study build/r2y_vivado_pilot_9_original_v1 \
  --start-case 0 --max-cases 1
```

The last command is **DRY RUN**. No vendor process should start. Source preparation/attestation should report nine ready cases and zero physical observations.

If source/tests/attestation all qualify and the actual `vivado` CLI is available, run one isolated native case first:

```bash
python -m fpgai_research.experiments.cst_r2y_controlled_physical_pilot \
  --execute \
  --study build/r2y_vivado_pilot_9_original_v1 \
  --execute-native --confirm-isolated-native \
  --start-case 0 --max-cases 1
```

Inspect original vendor completion, source/ROM integrity, DRC and route/time status before committing to the remaining cases. If the first case exposes routing or original ROM problems, **preserve failed originals and do not overwrite/retry them**; changes require a distinct prospective source revision and study output path. Cases are indexed by the order of the nine selected IDs, not their original numeric suffix (for example, new index 4 maps to original `case_016`).

If appropriate, the remaining **nonoverlapping** cases use `--start-case 1 --max-cases 8` with both native flags. Do not attempt to rerun case index 0. After all 9 original cases are attempted (whether pass/fail/unknown), perform a separate create-only export:

```bash
python -m fpgai_research.experiments.cst_r2y_controlled_physical_pilot \
  --export \
  --study build/r2y_vivado_pilot_9_original_v1 \
  --output build/r2y_vivado_pilot_9_original_report_v1
```

Read `build/r2y_vivado_pilot_9_original_report_v1/summary.json` including **all nine per-case statuses** and **all seven paired E4/UNKNOWN classifications**. Preserve every raw Vivado original generated source, invocation receipt and report SHA. No result is valid for the paper until original workstation outputs exist and are independently re-audited.

## Manuscript decision after the physical study

A positive strict E4 pair may support an exactly scoped **two-graph, nine-arm OOC source-to-routed implementation observation** under the fixed 8 ns requested clock, not a universal compiler area claim. A set of routed BRAM ties, failures or DRC warnings is equally publishable **negative translation evidence**, but cannot be misreported as strict physical equality if warnings prevent E4 qualification. Scientific novelty still depends on nontrivial IR/temporal-contract capabilities and independently grounded related-work comparisons; neither a conventional interval-coloring optimum nor a source-lookup implementation difference alone establishes firstness.

References: [original R2u/R2t protocol](R2T_R2U_ALLOCATION_AND_HLS_MECHANISM_PROTOCOL_20261103.md), [R2v verified HLS factors](R2V_READONLY_NATIVE_FACTORS_PROTOCOL_20261103.md), [R2w original RTL SHA](R2W_SOURCE_ATTESTED_RTL_MEMORY_FORENSICS_20261103.md), [R2x original type graph](R2X_STATIC_HLS_VERILOG_HIERARCHY_PROTOCOL_20261103.md), [formal source-event theorem scope](FORMAL_SOURCE_EVENT_REFINEMENT_AND_TENSOR_BASELINE_20261103.md).
