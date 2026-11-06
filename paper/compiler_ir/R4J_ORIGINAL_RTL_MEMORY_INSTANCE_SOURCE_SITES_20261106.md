# R4i observed result and R4j read-only memory instance-site gate — 6 November 2026

## Observed workstation R4i evidence

The author's workstation reported **24/24** focused R4i/R4h/R4g tests passing. Its original-study read-only R4i report completed with ZERO new Vitis/Vivado calls. The report generated separately under build/r4i_rtl_memory_parameter_defaults_20261106_01.

The only natively E2/E3-qualified new R4d/R4e format remains **32/6** (both H storage arms). The new 16/6 and 32/8 native arms remain UNKNOWN/unexecuted. The original failed R4b 16/6 native attempt is a separate frozen study.

The original 32/6 HLS-generated Verilog **module default** parameters (not elaborated instances) are:

| Parameter/source evidence | full_H | row_H | Row minus full |
|---|---:|---:|---:|
| Scratch DataWidth | 32 | 32 | 0 |
| Scratch AddressWidth | 8 | 8 | 0 |
| Scratch AddressRange | 214 | 134 | **−80 entries** |
| Scratch default declared product, bits | 6,848 | 4,288 | **−2,560** |
| Accumulator default depth | 256 | 256 | 0 |
| Weight ROM default profiles | 30×64, 29×128, 30×128 | same | unchanged |
| HLS BRAM18K estimate | **13** | **13** | **0** |

The scratch default depth reduction is **37.38%**. All three weight ROM initialization payload SHA digests match across original arms; original source, generated RTL and native receipts remained unchanged.

**Critical:** R4i source-module defaults have not been checked against **actual instantiation arguments**. The numeric difference −80 entries is NOT the earlier compiler E1 source-storage saving of 240 cells. HLS resource estimates are NOT routed/physical BRAM allocations.

## R4j — newly implemented, not yet workstation-tested

New source: src/fpgai_research/experiments/cst_r4j_memory_instance_source_sites.py

New tests: tests/test_cst_r4j_memory_instance_source_sites.py

R4j reuses the original R4g SHA-verified Vitis RTL and ROM source audit, and separately the R4i original parameter-default audit. Rather than trusting filenames, it checks the bytes/SHA of every scanned original Verilog file and reuses R2x's bounded module/type recognition and static source-edge inventory. It scans for memory-related **source instance declarations** in reachable Verilog module types.

Each identified source site records parent module type, memory module type, instance name, file and line. For the supported subset of instantiations it distinguishes:

- No explicit source-site parameter override.
- Exact decimal named parameter overrides, including DataWidth, AddressWidth and AddressRange.
- Unsupported expressions, positional arguments, duplicate names, unbalanced or ambiguous syntax, and defparams: UNKNOWN, not evidence of default values.
- Textual source-default versus source-site-literal parameter tuples, if all three required parameters are source-attested and the source syntax supports the comparison.

Even a fully recognized literal source instance declaration is NOT a proof that the instance survives generate conditions, has exactly one physical instance, receives its default source parameters after full elaboration, maps to RAMB, or yields a specific number of resource blocks. R4j labels these limitations explicitly. Its original source-date denominator remains all three new numerical formats; missing native evidence stays UNKNOWN rather than becoming a negative measurement.

The R4j scanner is a **conservative subset recognizer**. Any unrecognized source case must be reported as UNKNOWN; no assumption of completeness or silently omitted site should be interpreted as a hardware observation. Compare R4j's source-site candidates to the original R2x static module edges rather than inventing a separate full Verilog parser.

## Next safe workstation commands (zero vendor calls)

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106

python -m pytest -q \
  tests/test_cst_r4j_memory_instance_source_sites.py \
  tests/test_cst_r4i_rtl_memory_parameter_defaults.py \
  tests/test_cst_r4h_hls_memory_source_crosswalk.py

python -m fpgai_research.experiments.cst_r4j_memory_instance_source_sites \
  --study build/r4d_numeric_format_source_20261106_01 \
  --original-study build/r4a_row_retained_paper_20261106_01 \
  --output build/r4j_memory_instance_source_sites_20261106_01
~~~

The last command writes a **new create-only** SHA-inventoried report outside both original evidence roots. Do NOT reexecute any 32/6 original Vitis HLS case; do NOT modify original source, generated RTL, native receipts or ROM assets.

## Next causal research experiment

If R4j can characterize original scratch source instantiations, next inspect schedule-controlled storage-only changes, not further uncontrolled iterations over source memory size. A stronger comparative experiment holds same graph, weights, precision, target, requested period, uncertainty, numerical tolerance and **source issue order** constant while varying only selected materialization decisions. HLS and placed-and-routed reports then determine whether total BRAM, DSP, LUT, FF, cycles or physical timing respond. Report both successful and negative/UNKNOWN cases; do not claim source arrays map one-to-one into BRAM blocks.

The first paper remains **MAJOR REVISION / NOT SUBMISSION READY**. R4j code/tests have been committed but not yet executed on the author's workstation.

## Original R4j workstation results and source-site aggregation correction (6 November)

The author's original R4j/R4i/R4h workstation run reported **30/30 tests passing**, created a separate R4j report at build/r4j_memory_instance_source_sites_20261106_01, and performed **zero new Vitis/Vivado calls**. The 32/6 pair has complete SHA-attested source-site observations for both arms.

| Original Verilog source site | full_H AddressRange | row_H AddressRange | DataWidth | AddressWidth |
|---|---:|---:|---:|---:|
| scratch_U | 214 | 134 | 32 | 8 |
| scratch_1_U | 214 | 134 | 32 | 8 |
| scratch_2_U | 214 | 134 | 32 | 8 |
| acc_U | 256 | 256 | 32 | 8 |
| count_U | 256 | 256 | 32 | 8 |

All source parameter overrides for the three scratch sites are named decimal literals and match the respective default source module parameters. The three weight ROMs retain matching 30×64, 29×128 and 30×128 source-site parameters. All memory observations are textual declaration sites, **not** proven instantiated/physical hardware.

### Aggregation flaw diagnosed and fixed in source (pending workstation retest)

Original R4j requires exactly one scratch site in each arm for a qualified paired source summary. As the original generated RTL contains **three** scratch sites each, R4j incorrectly returned UNKNOWN for pairs[1] despite complete arm-level source-site evidence. This does NOT invalidate the original R4g/R4h/R4i source-hash evidence or vendor results. The original report is preserved unchanged.

The corrected R4j implementation pairs source sites by exact parent module, memory module, and instance name; it rejects missing/duplicate sites and unsupported source tuples; and it reports a scalar −80 entry per-source-site depth only when every matched site agrees in width, address width and default/override values. It retains per-site evidence and no global scalar for heterogeneous source tuples. **This code and four added tests are committed but not yet workstation-tested**. The optional new report must be create-only.

Three source declaration sites each change by −80 entries, yielding a **numerical source-site sum of −240 entry-slots**. This matches the earlier R4a source-level −240 scalar-cell figure numerically, but no per-element identity mapping or elaborated memory allocation is established. Do not call this a physical or causal 240-word saving. HLS BRAM18K remains 13 vs 13, with the original resource/cycle tradeoff unchanged.

### New CPU-only verification (preserve original report)

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
python -m pytest -q \
  tests/test_cst_r4j_memory_instance_source_sites.py \
  tests/test_cst_r4i_rtl_memory_parameter_defaults.py \
  tests/test_cst_r4h_hls_memory_source_crosswalk.py

python -m fpgai_research.experiments.cst_r4j_memory_instance_source_sites \
  --study build/r4d_numeric_format_source_20261106_01 \
  --original-study build/r4a_row_retained_paper_20261106_01
~~~

Optional new report destination: build/r4j_memory_instance_source_sites_corrected_20261106_01, using --output, only if it does not already exist. Never overwrite the original R4j report or rerun the original 32/6 vendor experiments.
