# R2w — Read-only forensics of original Vitis-generated RTL and memory names

**Status on November 3, 2026: R2w ORIGINAL WORKSTATION VERIFIED AND FROZEN.** Clean analysis revision `bb98983a438efe9e65c0204041895353730f964e`: **22/22 focused and 1,580/1,580 full regression tests passed**, followed by read-only source-attested analysis of the preserved R2u original source revision `0c6d85f189062df8d5157237f3c8ac74ac8da4b8`. The create-only `build/r2w_original_r2u_rtl_memory_forensics_v1/` succeeded: **32/32 original RTL case inventories and 64/64 conditional RTL pairs match original Vitis invocation SHA receipts**, with **30 E3-qualified and 7 E3-UNKNOWN HLS factor comparisons** retained; **0 Vitis/Vivado calls**. These are lexical RTL inventories, not physical RAM instances. The original R2w output is frozen—do not rerun or overwrite its v1 path. The original R2u/R2v results are **COMPLETE, SOURCE-PINNED and MUST NOT BE REEXECUTED OR MUTATED**. This R2w postprocessing is a new strictly create-only and vendor-free analysis of the original Vitis outputs. The active paper is **MAJOR REVISION**, not publication-ready.

## Completed and frozen original scientific evidence

Original R2u vendor source revision: `0c6d85f189062df8d5157237f3c8ac74ac8da4b8`. Original R2v analysis source revision: `23e998a0553e3eec6f05c5d3cd09a9b4a81af83e`. The user's local R2v focused **26/26**, full pytest **1,572/1,572**, and read-only `build/r2v_r2u_native_factorial_review_v1/` succeeded. **32 original Vitis invocations, 32/32 SHA-attested CSynth XMLs, 32/32 CSim E2, 30/32 effective-clock-qualified HLS CSynth E3, and 2/32 rejected for the effective clock (case_004 and case_012)**.

Of exactly **64 planned conditional same-graph factor contrasts**, **57 E3-qualified / 7 UNKNOWN**. At HLS-estimated **BRAM_18K** granularity:

| Factor, intervention minus control | Reduced | Tied | Increased | E3 UNKNOWN |
|---|---:|---:|---:|---:|
| Scratch (disjoint → committed-element exact) | 2 | 13 | 0 | 1 |
| Address (static array table → source switch) | 0 | 14 | 0 | 2 |
| Binding (auto → requested `ram_2p` BRAM) | 0 | 10 | 4 | 2 |
| Declared scratch capacity (exact → padded to 64) | 0 | 10 | 4 | 2 |
| **Total** | **2** | **47** | **8** | **7** |

The R2v audit recorded original `case_004` and `case_012` as **FAILED_HLS_EFFECTIVE_CLOCK_BUDGET**, not 0 BRAM or ties, even though their raw XML files are SHA-attested. These are Vitis HLS estimates only. **No new Vivado place/route or physical BRAM/Fmax measurements exist.** The earlier six warning-disclosed OOC routed comparisons remain distinct: **six BRAM ties, 0 strict clean-DRC E4**.

## Original source-attested RTL inventory results

The user printed every original R2w case inventory. Within each graph, address lookup form—not scratch layout, requested binding or declared padding—separated the observed HDL **file counts** and **memory-named filenames**:

| Original graph | Address mode | Verilog files per case | Scratch-named files per case | Memory-named files per case |
|---|---|---:|---:|---:|
| Two readers | Table (`case_000..003`, `008..011`) | **23** | **7** | **10** |
| Two readers | Switch (`case_004..007`, `012..015`) | **24** | **7** | **6** |
| Four readers | Table (`case_016..019`, `024..027`) | **30** | **9** | **16** |
| Four readers | Switch (`case_020..023`, `028..031`) | **32** | **9** | **8** |

All cases had `rtl_observation_status=LEXICAL_HDLSOURCE_ONLY`. This shows that changing the table lookup to switch-case **changes original HLS emitted HDL file organization** within both graphs, and can reduce the number of *filenames containing lexical memory tokens* despite more total generated HDL files. It does **NOT** show a reduction in actual memory instances, available RAM ports, memory capacity, HLS estimated BRAM (the 14 qualified address comparisons were all estimated BRAM ties), or Vivado routed BRAM. Filenames may refer to unused/dead definitions or helper modules. These original results now motivate a **separate module-definition/instantiation/hierarchy study**, not a physical implementation claim.

## Question addressed by R2w

Why can reducing source scratch capacity produce no reduction in HLS-estimated BRAM, while padding or requesting a particular storage binding sometimes increases it? This is **a mechanism investigation**, not a claim of new FPGA memory performance. A source HLS pragma does not itself identify which storage primitive is inferred or implemented; the original CSynth XML resource count cannot be reverse-labeled as a particular RAMB cell.

R2w inventories the **already existing, original HLS-generated Verilog/SystemVerilog and literal local ROM files** for all 32 original cases. For each:

- Re-run the existing **read-only** R2u native case/source/Tcl/numerical/clock qualification, with its recorded source revision.
- Require exactly the completed original 32 E2, 30 E3 and 57 qualified pair counts. The 2 timing-rejected cases stay in the population.
- Compare the **complete generated RTL and ROM filename→SHA-256 maps** to the original `vitis_invocation.json` fields `rtl_output_sha256` and `rom_output_sha256`. A mismatch is **rejected**, not silently trusted. The CSim/CSynth original reports retain their separate existing SHA evidence.
- Inspect the raw original Vitis CSynth text report `fpgai_model_csynth.rpt` **only if SHA-attested in the original invocation**. Capture a bounded number of scratch-related text lines and their original report digest.
- Report names of original RTL files containing the lexical identifier `scratch`, names that suggest memory/ROM/RAM/FIFO, and occurrences of RAMB18E*, RAMB36E*, URAM288 and XPM memory tokens in source text. Those are **LEXICAL** observations only; a token may occur in a comment, declaration, unused module or uninstantiated component.
- For 64 original source-matched conditional contrasts, compare lexical file counts and snippets only if **both original RTL inventories are SHA-attested**. Preserve all missing/malformed results as `UNKNOWN`, never as zero. The 7 E3-rejected contrasts keep null estimated BRAM deltas even if original RTL files are present.

**Not measured:** actual netlist RAMB primitive allocation; in-hardware port legality; padding retention after implementation; real LUTRAM, URAM or BRAM inference; physical utilization, routing, timing, power. The study can offer candidate original RTL modules to inspect, not prove the cause of the old physical BRAM ties. It may reveal **no scratch-named RTL at all** because of HLS name lowering or registers; this must be reported as `NO LEXICAL MATCH`, not proof that no memory exists.

## Code and evidence scope

- New read-only module: `src/fpgai_research/experiments/cst_r2w_rtl_memory_forensics.py`.
- New tests: `tests/test_cst_r2w_rtl_memory_forensics.py`. The fixtures are **synthetic** original invocation/Verilog/ROM/report files and deliberately exercise a post-run RTL bit change, a ROM bit change, a text-report bit change, a symlink, and missing native/HDL files. These tests do **not** simulate any native vendor resource measurement.
- Reuse unchanged: `cst_r2u_hls_mechanism_execute.audit_native`, `cst_r2v_hls_factorial_review.review`, and `toolchains.vivado.rtl_inputs.inventory_vitis_rtl`. No changes to the original source generator or vendor runner.

R2w writes a completely **new, create-only** `build/r2w_original_r2u_rtl_memory_forensics_v1/` containing `summary.json`, `cases.csv`, and `contrasts.csv`. It never writes into the original R2u/R2v folders. The code deliberately does not invoke Vitis, Vivado or the RL environment.

## Commands on the current workstation

```bash
cd ~/Desktop/github_projects/fpgai_research
git status --short
git pull --ff-only origin main
git rev-parse --short HEAD

python -m compileall -q src/fpgai_research experiments/compiler_ir tests
python -m pytest -q \
  tests/test_cst_r2w_rtl_memory_forensics.py \
  tests/test_cst_r2v_hls_factorial_review.py
python -m pytest -q
```

**Historical original R2w commands, ALREADY COMPLETED: NEVER rerun or overwrite the existing v1 paths.** The commands below document reproducibility only:

```bash
python -m fpgai_research.experiments.cst_r2w_rtl_memory_forensics \
  --study build/r2u_hls_factorial_32_original_v1 \
  --output build/r2w_original_r2u_rtl_memory_forensics_v1
```

Inspect the resulting `summary.json`, especially `source_attested_RTL_case_count`, `conditional_pairs_with_both_RTL_sha_attested`, `case_evidence[].scratch_name_rtl_files`, `case_evidence[].memory_name_rtl_files`, and `case_evidence[].report_scratch_snippets`. Report every missing/unknown case rather than forcing all 32 to qualify.

Do **not** rerun original `--execute-native`, `--prepare`, `R2t`, `R2s`, or the read-only R2v original create-only `v1` output. If R2w fails because the generated RTL contains unsupported file formats, symlinks, unrecognized ROM dependencies, or unmatched SHA receipts, keep the original artifacts unchanged and diagnose that specific failure; do not weaken evidence gates merely to pass.

## Remaining publication gates

The paper still needs a demonstrated *qualified E4 routed* before/after comparison or must remain an explicit **negative translation result** under disclosed DRC limitations; the original HLS RTL may need deeper **structural module connectivity or native Vivado netlist inspection**, not only lexical token counts. The full source-event theorem does not ensure actual HLS port/cycle semantics. Independent related-work artifact/capability verification against Allo, StreamTensor, ST-Flow, CODO and closely related work remains necessary. The hypothetical first-MAC in-place source-storage convention is still **not implemented as an emitted GCC/HLS design**. No user-facing manuscript should claim the full compiler product is finished or that trained RL improved quality.

See [verified R2v HLS factorial analysis](R2V_READONLY_NATIVE_FACTORS_PROTOCOL_20261103.md), [R2t/R2u original protocol](R2T_R2U_ALLOCATION_AND_HLS_MECHANISM_PROTOCOL_20261103.md), [formal source-event scope](FORMAL_SOURCE_EVENT_REFINEMENT_AND_TENSOR_BASELINE_20261103.md), and [active paper](MANUSCRIPT_DRAFT.md).
