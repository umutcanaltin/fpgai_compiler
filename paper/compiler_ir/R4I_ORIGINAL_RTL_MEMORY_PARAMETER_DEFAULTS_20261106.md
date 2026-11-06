# R4h original source results and R4i dimension-default audit — 6 November 2026

## New observed evidence from workstation

The author's original R4h/HLS-RAM/R2x tests **19/19 passed**, and the create-only R4h report completed without any new Vitis/Vivado call. The original source-attested 32/6 pair has the following HLS-generated **module-default** parameter values:

| Memory module parameter | full_H | row_H |
|---|---:|---:|
| scratch_RAM_AUTO_1R1W DataWidth | 32 | 32 |
| scratch_RAM_AUTO_1R1W AddressWidth | 8 | 8 |
| scratch_RAM_AUTO_1R1W AddressRange | **214** | **134** |
| acc_RAM_AUTO_1R1W DataWidth | 32 | 32 |
| acc_RAM_AUTO_1R1W AddressWidth | 8 | 8 |
| acc_RAM_AUTO_1R1W AddressRange | 256 | 256 |
| w0 ROM default Width × Range | 30 × 64 | 30 × 64 |
| w1 ROM default Width × Range | 29 × 128 | 29 × 128 |
| w2 ROM default Width × Range | 30 × 128 | 30 × 128 |
| R4e HLS BRAM18K estimate | **13** | **13** |

At unchanged 32-bit default data width, scratch RAM declared source default depth was reduced by **80 entries**, or **37.38%** of its 214-entry control default. The *declaration-level product* changes from 214×32 = 6848 bits to 134×32 = 4288 bits: **−2560 bits**.

The accumulator source default is unchanged at 256×32 = 8192 bits. The three weight ROM defaults and SHA-256 hashes of their initialization payloads are unchanged; source module names change only because loop names differ. The scratch module SHA differs and the accumulator module SHA matches between arms.

**CRITICAL LIMIT:** these parameters are literal defaults in original Vitis-generated module SOURCE, NOT independently verified instantiated parameter values, actual allocated RAM depths, physical BRAM packing, or placed FPGA RAMB18 instances. The 8-bit source address-width default remains unchanged. There is no established causal explanation yet for the **13 vs 13** HLS BRAM estimate.

Likewise the separate R4a source-level H-storage saving cited earlier (240 scalar cells) and the 80-entry HLS scratch-module-default reduction are **different measurements**; they must not be directly substituted or equated. Read-only R4h did not establish the elaboration/optimization relation between those quantities.

## R4i workstation-tested; new R4j original-source site protocol

**Update:** The author's workstation reported 24 focused R4i/R4h/R4g tests passing, then completed the create-only original R4i source-default report at build/r4i_rtl_memory_parameter_defaults_20261106_01. The original 32/6 output source-qualifies the 214 versus 134 scratch default depth, same 32-bit default width and 8-bit default address width, unchanged 256-word accumulator defaults, three matching ROM default profiles, and **zero HLS estimated BRAM18K reduction (13 vs 13)**. It launched zero vendor calls and modified no original evidence. A new [R4j original source memory-instance site audit](R4J_ORIGINAL_RTL_MEMORY_INSTANCE_SOURCE_SITES_20261106.md) is now implemented, but remains UNTESTED on the author's workstation.

## Original R4i protocol (now verified on workstation)

New code: src/fpgai_research/experiments/cst_r4i_rtl_memory_parameter_defaults.py

New tests: tests/test_cst_r4i_rtl_memory_parameter_defaults.py

R4i consumes the existing R4h→R4g→R4e original SHA-attestation chain and extracts **only finite, literal DataWidth, AddressWidth and AddressRange module parameter defaults** where the exact symbolic array-template declaration is present in the signed source excerpts. It compares the two original 32/6 sources, preserves all 3 prospective formats in the denominator, and reports:

- scratch module default widths/ranges and **row-minus-full** default depth/data-bit-product;
- whether both default source address widths and accumulator parameter profiles match;
- multisets of three weight ROM source default profiles, and unchanged initialization ROM content digests independently attested through R4h;
- qualified HLS estimated BRAM delta, kept separate from the source dimension comparison;
- explicitly null mapped RAM capacity, unexamined instance overrides, E4 and actual FPGA clock.

If parameter syntax is symbolic, ambiguous, incomplete, truncated, missing, or an array template is unsupported, R4i reports **UNKNOWN**, not a guessed width/size. A source declaration can have legal literal defaults even if the module instance later overrides those parameters; R4i does not evaluate instance overrides. Original vendor cases and generated RTL files are never rewritten.

### Safe CPU-only workstation verification

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106

python -m pytest -q \
  tests/test_cst_r4i_rtl_memory_parameter_defaults.py \
  tests/test_cst_r4h_hls_memory_source_crosswalk.py \
  tests/test_cst_r4g_rtl_memory_bridge.py

python -m fpgai_research.experiments.cst_r4i_rtl_memory_parameter_defaults \
  --study build/r4d_numeric_format_source_20261106_01 \
  --original-study build/r4a_row_retained_paper_20261106_01 \
  --output build/r4i_rtl_memory_parameter_defaults_20261106_01
~~~

The final command creates a new separate original-read-only report. It cannot overwrite original R4a/R4d evidence or an existing report folder. If the output path already exists, omit --output for a fresh read-only print instead of overwriting it.

## Next causal experimental gates

1. Verify the original **module instance parameter values** and memory port/connectivity in a new source-SHA-pinned hierarchy-aware read-only analysis, accounting for parameter overrides. Static source defaults alone are insufficient to conclude that hardware contains an 214-word or 134-word scratch RAM.
2. Contrast the original 214/134 HLS defaults with the compiler-level 240-cell source storage saving through an independent storage-layout witness rather than assuming the two deltas are numerically interchangeable.
3. Design a new, separately source-pinned matched experiment holding producer/reader execution ordering constant while modifying **only** selected memory materialization; verify E1 and E2 numerical semantics and HLS schedule/RTL before comparing any storage benefits.
4. Only claim real FPGA BRAM18K/clock gains after qualified, source-matched Vivado implementation with clean DRC/route and original native provenance; no existing successful Vitis case should be rerun or modified.

Paper status remains **MAJOR REVISION**. E3 savings are mixed (logic/DSP lower, BRAM tie, more cycles); the new R4i experiment is source-only and has **not** yet passed workstation tests.
