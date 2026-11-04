# R2x — Read-only static Verilog hierarchy analysis of original R2u Vitis outputs

**Status (2026-11-03): ORIGINAL R2x WORKSTATION VERIFIED AND FROZEN.** Source-analysis commit `e5ea6a14d279be786492f338bdf1e51ba684ba73`: **23/23 focused**, **1,588/1,588 full regression**. Original create-only `build/r2x_original_r2u_static_hierarchy_v1/` completed, with **32/32 original generated RTL SHA-attested source cases, 32/32 static-top-known cases, and 64/64 matched static-top-known pairs**. There were **zero new vendor calls**. This is static source-level type reachability, not physical resource mapping. Original R2x v1 is frozen. Next prospective native pilot is [R2y nine-case/strict-E4 study](R2Y_NINE_CASE_STRICT_E4_PHYSICAL_PILOT_20261103.md). R2x is a source-level **read-only** HLS-generated Verilog analysis, not a vendor run or physical implementation. The original R2u source revision `0c6d85f189062df8d5157237f3c8ac74ac8da4b8` and original R2v and R2w analysis directories remain frozen. **No Vitis/Vivado commands, new hardware results, or RL training are performed.**

## Verified original evidence motivating R2x

R2w code revision `bb98983a438efe9e65c0204041895353730f964e` passed **22/22 focused**, **1,580/1,580 complete tests**, and its create-only original read-only study `build/r2w_original_r2u_rtl_memory_forensics_v1/` completed: all **32 original generated RTL cases** and **64 original matched RTL pairs** were SHA-attested to their original Vitis invocation receipts. Of the 64 factor comparisons, **57 E3-qualified**, **7 unqualified**. No new vendor calls.

The original Verilog file inventory depends strongly on address generation:

| Original graph | Address table | Switch-case address |
|---|---|---|
| Two readers | **23** RTL files, **10** memory-named files, **7** scratch-named files | **24** RTL files, **6** memory-named files, **7** scratch-named files |
| Four readers | **30** RTL files, **16** memory-named files, **9** scratch-named files | **32** RTL files, **8** memory-named files, **9** scratch-named files |

The originally verified R2v HLS BRAM_18K estimate results were **2 decreases, 47 ties, 8 increases and 7 E3-unknowns** across **64** comparisons. The **14 eligible address-lookup comparisons all tied in HLS BRAM estimates**, despite differences in RTL file inventories. Therefore file count and lexical memory-named filenames clearly cannot be equated to memory cell counts.

## Original R2x verified module-type/recognized-instance results

The original 32 per-case statuses were all `STATIC_TYPE_GRAPH_ONLY`, with no unsupported parser cases in this fixed generated HDL cohort. The recognized **memory-named module types / recognized memory-named type instances** by address form were:

| Source graph | Table lookup | Switch lookup |
|---|---|---|
| Two readers | **10 / 10** | **6 / 6** |
| Four readers | **16 / 16** | **8 / 8** |

The same counts repeated across all original storage, requested binding and padding factor levels for each graph and address mode. Thus the static module-type graph supports R2w's reported **source RTL structural difference**, not a general mapped BRAM change. The recognized per-type module instances are not elaborated with actual parameter/generate multiplicities and **cannot** be promoted to a count of actual instantiated RAM primitives. The HLS-estimated BRAM in all **14 eligible** address comparisons still tied, and original physical evidence is unchanged.

## New structural question and conservative method

Can we identify *module definitions that actually appear as static module-type dependencies*, rather than just RTL filenames? R2x reuses the unchanged original R2w HLS source/Tcl/testbench/CSynth and full generated RTL/ROM SHA verification. It then scans only the original SHA-attested Verilog files using a **bounded, deliberately incomplete Verilog source parser**:

1. Identify simple `module ... endmodule` definitions and their defining RTL filenames. A duplicate module type, invalid/unrecognized declaration or unsupported module header produces an explicit **`UNKNOWN_UNSUPPORTED_OR_AMBIGUOUS_VERILOG`** result, **not** a zero-memory claim.
2. Recognize a subset of named module-type instances with and without `#(...)` parameter overrides, including named external primitive-token candidates. Ignore text in comments and strings; report only structural candidates, never infer a mapped FPGA cell.
3. Starting from named RTL module `fpgai_model`, compute provisional **type-level** reachability. A missing top module produces **`UNKNOWN_TOP_MODULE_NOT_DEFINED`**, with null reachability counts and no assertion that inferred memory is absent.
4. Count provisionally reachable scratch-named and RAM/ROM/MEM/FIFO-named **module TYPES**, recognized static named instances, candidate external RAMB/URAM/XPM instance tokens, and literal `reg [bits] name[integer:integer]` unpacked array declarations. Module type reachability is **not elaborated or synthesized instance multiplicity**; generate conditions, unrecognized external modules, include files and macros are not fully interpreted. A declaration may be unused or optimized away.
5. Pair all **64** original same-graph interventions, separating any incomplete static-top interpretation from exact original E3 qualification. Unqualified E3 pairs keep `null` HLS BRAM deltas even when both RTL sources are attested. Source-reported static module deltas are never routed BRAM or achieved Fmax.

The study writes an independent new create-only analysis folder `build/r2x_original_r2u_static_hierarchy_v1/` with `summary.json`, `cases.csv`, and `contrasts.csv`. The original Vitis source directory, original R2v and original R2w output folders are **read-only**.

**This scanner is not a general Verilog parser.** It cannot prove actual instance counts, memory port legality, physical mapped RAM blocks, static timing, DRC closure, or exact equivalence between C++ and RTL. An unrecognized module reference, named `RAMB18E2` instance, `reg` declaration, or filename should not be converted into a hardware resource utilization claim. All real HLS figures remain E3 estimates; E4 still requires qualified native Vivado.

## Workstation commands: no new vendor jobs

```bash
cd ~/Desktop/github_projects/fpgai_research

git status --short
git pull --ff-only origin main
git rev-parse --short HEAD

python -m compileall -q src/fpgai_research experiments/compiler_ir tests

python -m pytest -q \
  tests/test_cst_r2x_rtl_hierarchy_analysis.py \
  tests/test_cst_r2w_rtl_memory_forensics.py \
  tests/test_cst_r2v_hls_factorial_review.py

python -m pytest -q
```

**Historical original R2x commands already executed. The existing v1 outputs are now frozen and MUST NOT be re-run/overwritten.** To reproduce a separate source cohort, use a distinct new source revision and completely new output path. Do not rerun any original R2u, R2v, R2w, R2t, R2s, or R2r v1 paths:

```bash
python -m fpgai_research.experiments.cst_r2x_rtl_hierarchy_analysis \
  --study build/r2u_hls_factorial_32_original_v1 \
  --output build/r2x_original_r2u_static_hierarchy_v1
```

Inspect `summary.json` fields `static_top_reachable_case_count`, `static_top_reachable_pair_count`, `cases[].parse_status`, `cases[].provisionally_reachable_memory_module_types`, `cases[].static_instance_edges` and all 64 `contrasts[]`. No result may be cited until original local workstation tests and source-pinned R2x analysis actually complete. If the parser fails on valid Vitis HDL constructs, **preserve the exact failure/UNKNOWN** and update the supported syntax using tests; never reclassify an unsupported case as a successful zero-memory result.

## Scientific decision gate after R2x

If type-level memory topology correlates with source-only lookup changes while HLS BRAM estimates tie, document this as a **specific static-source representation difference, not an implemented hardware memory reduction**. If source module hierarchy is ambiguous under macros/generate/opaque IP, the study must report that weakness; a conservative tool-independent parser such as Surelog/Verible, or Vivado elaboration, may be necessary for a stronger causal statement, with separately attested input hashes and tool versions.

A truly complete publication still needs independent related-work mechanism comparisons, a non-trivial scientific IR gap or explicitly scoped negative systems finding, a real generated first-MAC in-place comparator, and qualified native E4 if any routed BRAM/physical performance claim is made. Original six historical routed BRAM ties with 0 strict clean-DRC E4 remain unchanged. [R2w verified source audit](R2W_SOURCE_ATTESTED_RTL_MEMORY_FORENSICS_20261103.md) and [R2v verified HLS factorial](R2V_READONLY_NATIVE_FACTORS_PROTOCOL_20261103.md) contain the prior results.
