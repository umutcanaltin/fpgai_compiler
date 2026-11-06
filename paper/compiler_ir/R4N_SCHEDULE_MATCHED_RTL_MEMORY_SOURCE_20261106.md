# R4n — read-only SHA-attested RTL memory for new schedule-matched R4l HLS

**6 November 2026.** New R4n source and tests are IMPLEMENTED but **NOT YET WORKSTATION-TESTED**. The author's completed R4l native Vitis HLS output is already qualified E2/E3 for BOTH new 32/6 storage arms; **DO NOT execute either native package again.** R4n runs zero Vitis/Vivado/GCC processes and must not change any original experiment file.

## Verified scientific context

The author's R4m analysis passed **28/28 focused tests** and completed a create-only study under `build/r4m_three_point_HLS_ledger_20261106_01`. All original R4e and new R4l 32/6 observations pass E2/E3, original and new row source + CSynth report SHA hashes agree, and three-point source contrast deltas add exactly.

| Comparison (source change) | HLS BRAM18K | DSP | LUT | FF | Worst cycles | Est clock ns |
|---|---:|---:|---:|---:|---:|---:|
| Original full/global → new full/row-source schedule | 0 | −2 | +105 | +322 | +762 | −0.445 |
| New full/row schedule → new row-retained/row schedule | 0 | −2 | −340 | −473 | −352 | −0.357 |
| Original full/global → original row/row schedule | 0 | −4 | −235 | −151 | +410 | −0.802 |

All three original HLS cases estimate 13 BRAM18K. One native-qualified 32/6 graph was studied; original R4d 16/6 and 32/8 native arms remain UNKNOWN. Source-issue order is fixed in the B→C comparison but no assertion of equal HLS hardware-cycle scheduling is possible.

## New scientific question

Do the **NEW original R4l** `full_storage` and `row_storage` Vitis-generated RTL sources show different memory module type declarations, source RAM dimension defaults, or literal Verilog instance-site parameter overrides that may help explain HLS-estimated logic changes and the BRAM18K tie?

### Implementation and original file connectivity

- `src/fpgai_research/experiments/cst_r4n_schedule_matched_rtl_memory.py` — entrypoint. Calls *only* `cst_r4l_schedule_matched_storage_native.audit_native`, then reuses `cst_r4g_rtl_memory_bridge.inspect_arm` to check **exact original Vitis invocation receipt and all generated RTL/ROM SHA manifests**, `cst_r4h_hls_memory_source_crosswalk.memory_source_signatures` to inspect bounded memory module source, `cst_r4i_rtl_memory_parameter_defaults.parameter_defaults` to recognize original module parameter defaults, and `cst_r4j_memory_instance_source_sites.scan_source_sites` / `compare_scratch_source_sites` to scan named literal source instance overrides and reconcile scratch-site pairs. The R2x parser is reused underneath; there is no parallel new full-Verilog parser.
- `tests/test_cst_r4n_schedule_matched_rtl_memory.py` — synthetic immutable original RTL inventories for exact three-scratch-site comparisons, default-versus-override counterexamples, E2/E3 unknown, unsupported symbolic values, SHA tampering, symlinks, create-only outputs and complete population.
- `paper/compiler_ir/R4N_SCHEDULE_MATCHED_RTL_MEMORY_SOURCE_20261106.md` — this protocol and file dependency index.
- `paper/compiler_ir/FIRST_PAPER_SCIENTIFIC_COMPLETION_20261106.md`, `paper/compiler_ir/R4M_THREE_POINT_HLS_SOURCE_CONTRAST_20261106.md`, and draft PR #2 updated to record the author's verified original R4m run and new R4n gate.

### Qualification and interpretation

R4n refuses any source SHA mismatch, missing/symlinked original RTL, invalid file names, incomplete new native arm population, or unqualified numerical E2/E3 source. A matched R4l arm whose RTL is not available or whose unsupported lexical syntax prevents complete parsing is reported UNKNOWN and is **not** given a fabricated capacity.

Even where R4n recognizes 32-bit / 8-bit address / literal depth values, it labels them **source-module defaults** or **source-site literal parameters**, not elaborated instance dimensions or actual RAMB counts. Multiple Verilog source-site declarations are not multiplied into physical resource use. Unsupported generate conditions, defparam, symbolic overrides, port topology, platform packing, and vendor memory binding are **not elaborated**. R4n reports an independent R4l HLS BRAM18K delta only if both original E3 arms have qualified integer HLS estimates. It never reports physical E4.

Expected data must be DISCOVERED from signed original RTL, never supplied from the older mixed-schedule R4e case. In particular, R4n must **not** assume original R4e scratch 214/134 depths or three instance sites are reproduced in the new source-controlled full-storage RTL until independently observed. New R4l row-source is source/CSynth SHA-identical to historical R4e row, but the newly generated Verilog is separately audited anyway.

## Safe workstation verification — NO VENDOR CALLS

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106

python -m pytest -q \
  tests/test_cst_r4n_schedule_matched_rtl_memory.py \
  tests/test_cst_r4m_three_point_source_contrast.py \
  tests/test_cst_r4l_schedule_matched_storage_native.py

python -m fpgai_research.experiments.cst_r4n_schedule_matched_rtl_memory \
  --r4k-study build/r4k_source_schedule_matched_storage_20261106_01 \
  --original-r4d build/r4d_numeric_format_source_20261106_01 \
  --original-r4a build/r4a_row_retained_paper_20261106_01 \
  --output build/r4n_schedule_matched_RTL_memory_20261106_01
~~~

The output folder is create-only. If it exists, omit `--output` to inspect a read-only report rather than overwrite. R4n's own original workstation result does not exist yet; no original RTL dimension outcomes from R4n may be cited before that run.

## Physical gate after R4n

A separate E4 study should use a new output root and independent SHA-attested original `full_storage` and `row_storage` R4l RTL **plus all original ROM data** to prepare matched Vivado out-of-context implementation. It must not modify the native Vitis projects. Collect actual synthesized/placed/route cell/resource counts, real timing slack and DRC, timing constraints, memory primitive mapping, bitstream/board assumptions, and explicit board qualification limits before claiming a physical FPGA improvement. The R4n source-only and R4m HLS-level results can motivate that experiment but cannot substitute for it.

**First paper remains MAJOR REVISION / NOT SUBMISSION READY.**

## Observed original R4n workstation audit (6 November 2026)

The author pulled research revision `9b5f834c0fc6629061e4435919e061113bd36bca`, ran **27/27 focused R4n/R4m/R4l tests**, and completed the create-only read-only R4n study `build/r4n_schedule_matched_RTL_memory_20261106_01`. It read and SHA-attested both original R4l Vitis HLS RTL and ROM inventories without a single new vendor call or original file modification. R4n's paired status is **`SOURCE_ATTESTED_STATIC_RTL_SCRATCH_SITES_PAIRED_NOT_ELABORATED`**.

| Actual Vitis-generated source evidence | R4l `full_storage` | R4l `row_storage` |
|---|---:|---:|
| RTL `.v/.sv` files | 24 | 24 |
| ROM `.dat/.mem` payloads | 3 | 3 |
| Reachable source memory module types | 5 | 5 |
| Static memory-related source-site declarations | 8 | 8 |
| Scratch source-site declarations | 3 | 3 |
| Scratch `DataWidth` (source literal) | 32 | 32 |
| Scratch `AddressWidth` (source literal) | 8 | 8 |
| Scratch `AddressRange` (source literal, each site) | 214 | 134 |
| Scratch source-declaration data-bit product *per site* | 6,848 | 4,288 |
| Accumulator / counter source depth | 256 | 256 |
| Weight-ROM source defaults / payload multisets | Identical | Identical |
| HLS estimated BRAM18K | 13 | 13 |

All three source scratch site names `scratch_U`, `scratch_1_U`, `scratch_2_U` carry supported **literal named parameter overrides**, each `214→134` entries, so the R4n site-level difference is **−80 entries**. Counting three matching declaration sites gives a numerical sum of **−240 entry-slots**; this is source-level accounting, **NOT** 240 freed physically mapped SRAM words. The native R4l full/row outputs have identical 3-ROM payload SHA multisets, and no change in reachable source module-type counts, memory module-type counts or memory source-site counts.

Both native arms remain original E2/E3 qualified, and R4n explicitly returns `mapped_BRAM18K_delta = null`, `placed_routed_clock_delta = null`, `physical_E4 = null`. Thus this study strongly links the R4k storage reduction to actual HLS-generated **Verilog source declarations and literal instantiation parameters** while providing no evidence of mapped RAMB count, physical area, timing or Fmax. Static candidate primitive count zero means only this bounded source parser found none; it does **not** assert zero real RAMB cells after Vivado.

The original R4n create-only report is immutable and should not be regenerated in place. Next gate: independent matched Vivado OOC physical preparation, using separate staged copies of SHA-attested RTL+ROM assets and unchanged source clocks/part, then strict routed timing/DRC qualification. No original Vitis native rerun.
