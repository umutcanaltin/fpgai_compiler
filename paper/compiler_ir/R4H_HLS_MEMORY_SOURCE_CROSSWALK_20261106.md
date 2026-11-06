# R4g workstation results and R4h static memory source crosswalk

6 November 2026. **R4g observed on the author's workstation; R4h code/tests committed but not yet workstation-executed.** All original studies and native cases remain unchanged.

## R4g observed evidence, not a new vendor execution

The author's original R4g/R4f/R2x regression reported **20/20 tests passing**. Its read-only source-attested Vitis HLS RTL examination at 32 total / 6 integer bits confirmed:

| R4g metric | Full H | Row H | Row minus full |
|---|---:|---:|---:|
| E3 HLS estimated BRAM18K | 13 | 13 | 0 |
| Original RTL files | 25 | 24 | -1 |
| Static reachable module types | 28 | 26 | -2 |
| Memory-related module types | 5 | 5 | 0 |
| Memory-related candidate instances | 8 | 8 | 0 |
| Scratch module types | 1 | 1 | 0 |
| Weight ROM initialization files | 3 | 3 | 0 |
| Source primitive instance candidates | 0 | 0 | 0 |

The original full-H and row-H HLS-generated acc_RAM_AUTO_1R1W source modules had identical SHA-256 digests, while the two scratch_RAM_AUTO_1R1W source modules had different SHA-256 digests. All three original weight ROM initialization payload hashes matched between the arms, even where generated filenames differed.

The R4g output resides in the create-only build/r4g_static_rtl_memory_bridge_20261106_01 directory. Original R4a source revision: fe99d627a5a16ce75e5d4df4542bb0f0f93e6ab0. Original R4d source revision: abe6aa37e5c799f27312b7f3a17309c6122ed7cb.

These source-level observations **do not demonstrate** physical RAMB allocations, memory capacity, inferred BRAM port count, routing success, or the reason for the 13-to-13 estimated BRAM tie.

## R4h: new conservative read-only RTL source crosswalk

Implemented code: src/fpgai_research/experiments/cst_r4h_hls_memory_source_crosswalk.py
New tests: tests/test_cst_r4h_hls_memory_source_crosswalk.py

R4h consumes the EXISTING R4g original SHA-attested read-only audit, then examines only the same-named generated Verilog files corresponding to its statically reachable named memory-module types. For each file it checks the SHA-256 again, returns bounded original source snippets and recognizes **only fully literal** scalar reg array widths and element ranges. Symbolic parameters and expressions remain UNKNOWN, NOT inferred or evaluated; even recognized literal source declarations are NOT physical RAM allocations.

The paired review distinguishes identically named original HDL modules whose bytes are unchanged from ones whose SHA differs, records unpaired type names, and compares the multiset of original, signed ROM content digests across arms. Both new 16/6 and 32/8 formats remain UNKNOWN because their original R4d vendor calls have not been made. R4h executes **zero Vitis/Vivado**, does not modify original files, and optionally writes a separate create-only, SHA-inventoried report.

## Workstation commands: CPU-only, source-preserving

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
git pull --ff-only origin research/first-paper-source-memory-bridge-20261106

python -m pytest -q \
  tests/test_cst_r4h_hls_memory_source_crosswalk.py \
  tests/test_cst_r4g_rtl_memory_bridge.py \
  tests/test_cst_r2x_rtl_hierarchy_analysis.py

python -m fpgai_research.experiments.cst_r4h_hls_memory_source_crosswalk \
  --study build/r4d_numeric_format_source_20261106_01 \
  --original-study build/r4a_row_retained_paper_20261106_01 \
  --output build/r4h_hls_memory_source_crosswalk_20261106_01
~~~

Do not rerun the original two successful 32/6 Vitis arms or overwrite any R4a/R4d native evidence. If the R4h report discovers symbolic parameters (for example DataDepth or AddressWidth), these require a **new separate, source-pinned dimension/binding analysis** before causal explanations.

## Next publication decisions

1. Identify exact original scratch and accumulator memory widths, depths, ports and mapping behavior, with uncertainty shown for symbolic source expressions.
2. Separate a materialization-only compiler intervention from producer/consumer loop schedule changes, so savings in source memory cannot be confused with different arithmetic scheduling.
3. Run new controlled HLS/physical experiments only after E2 numerical acceptance, evidence attestation and advance experimental specification.
4. Maintain the prior observed BRAM tie, E3 logic reductions and longer worst-case cycles as mixed outcomes, not generalized hardware superiority.

**Paper: MAJOR REVISION / NOT SUBMISSION READY.**
