# Existing first-paper evidence: inventory before further experiments

**Review date:** 6 November 2026. **Read-only repository audit.** This inventory was assembled from the repository's original source-revision-pinned research protocol and workstation result documents; **the original local `build/` bytes are not accessible via GitHub**. Completion below means **documented completed workstation execution**, not that this reviewer independently opened or rehashed the local raw output folder. A fresh local filesystem search and receipt audit are required before deriving a new report.

**Important location correction:** the original 2026 research protocols were run from the **earlier** `~/Desktop/github_projects/fpgai_research` checkout. The active `~/Desktop/github_projects/fpgai_compiler` checkout is a separate working tree, and a directory missing from its `build/` is *not evidence that the earlier experiment was never run or that its artifacts were deleted*. Do not re-run, overwrite or delete original paths.

## Historical evidence inventory (first paper)

| Study | Original recorded result | Expected preserved artifact | Qualification |
|---|---|---|---|
| R2q source lifetime baseline | 27 Phi comparisons: 18 committed-value scratch reductions, 9 ties; **0** reductions under hypothetical first-MAC in-place accounting | `build/r2q_*` (refer to protocol for exact path) | Source-only; accounting convention matters |
| R2s Phi scaling | 147 source configurations, 126 matched non-scalar interventions; **102** scratch reductions, **24** ties, **0** increases under committed-value accounting. Max **12 source slots**, max **5.882%** of original declared mutable arrays on tested cases; on largest sampled shape maximum **0.389%** | `build/r2s_phi_scaling_147_source_v1/` | Verified E0 source, not hardware |
| R2r emitted C++ | Six original GCC baselines; **216/216** selected mutants detected = **192** memory-operation / **24** arithmetic; original 222 GCC attempts | `build/r2r_emitter_shadow_216_mutants_original_v1/` | Host C++ instrumentation; not proof of all-source safety |
| R2t tensor optimum | **36/36** fixed-order contiguous-tensor minima proved; **12** improve on greedy whole-tensor first-fit | `build/r2t_tensor_exact_36_original_v1/` | E0 source optimization |
| R2u native memory factorial | **32/32** Vitis invocations and CSim passes; **30/32** qualified HLS estimates; `case_004` and `case_012` fail effective-clock criterion | `build/r2u_hls_factorial_32_original_v1/` | Source revision `0c6d85f189062df8d5157237f3c8ac74ac8da4b8`; part `xck26-sfvc784-2LV-c`, clock request 8 ns, uncertainty 1.5 ns |
| R2v original read-only HLS review | **64/64** factor contrasts; **57** E3-qualified, **7** unknown. BRAM18K response **2 decreases / 47 ties / 8 increases / 7 unknown** | `build/r2v_r2u_native_factorial_review_v1/` | Qualified Vitis estimates, no routed resource claims |
| R2w original read-only generated RTL | **32/32** RTL SHA-attested cases, **64/64** pairs | `build/r2w_original_r2u_rtl_memory_forensics_v1/` | Lexical source RTL, not RAM cells |
| R2x original read-only module hierarchy | **32/32** RTL cases and **64/64** matched type graphs statically interpretable | `build/r2x_original_r2u_static_hierarchy_v1/` | Static module-type reachability, not post-synthesis elaboration |
| R2i/R2j/R2l/R2m original routing | Six selected descriptive routed BRAM **ties** with disclosed DRC warnings; **zero** strict E4-qualified matched pairs, 30 of 36 pairs physically unobserved | `build/r2m_original_six_physical_paper_v2/` plus original `r2i/r2j/r2l` studies | Separate 32-ns protocol, not the 8-ns R2u factorial |
| R2y strict physical pilot | One of nine original Vivado cases routed, with disclosed DRC; **zero** strict clean-DRC E4 pairs | Original R2y v1 output (see its protocol) | Eight unattempted; no physical saving claimed |
| R2z/R3a mixed CNN/Dense | Six source packages and **12/12** generated C++ GCC intermediate-stage numerical checks. Six Vitis E2 and four E3; **2/3** qualified matched pairs | Original R2z and R3a output folders, see R3c protocol | Qualified original independent mapping had **624→911** and **2365→5436** cycles and unchanged HLS BRAM |
| R3d/R3e/R3f C1/C2 | Six actions = **3/2/1**; five oracle checks and 25 selected mutants; 48 actions = **21 certified / 24 rejected / 3 unknown** | `build/r3d_c1c2_finite_contract_gate_v1/`; `build/r3e_c1_bounded_counterexamples_v1/`; `build/r3f_c2_variable_graph_coverage_v1/` | Source-only bounded evidence |
| R3g/R3h semantics and catalog | **133** source trace variants, zero two-interpreter disagreements; **48** compiler transitions to build masked catalog, **zero additional** after selecting cached certified candidate | `build/r3g_c1_independent_event_oracle_v1/`; `build/r3h_c2_compiled_once_selection_v1/` | Source-only; same bounded admitted specification |
| R3i Allo external parity | Protocol inspected, **NOT RUN** | No qualified original external result established | External research gap |
| R3j/R3k held-out semantics and limits | Code exists; **no standalone original source-pinned workstation receipts verified** in the last recorded checkpoint | Planned `build/r3j_...` / `r3k_...` | Never count planned 216 traces or 192 actions as completed experiments |
| New 6 Nov first-paper contract ablation | **Observed from user terminal:** 24 deliberately source-value-safe but Phi-order-nonconformant controls accepted by reduced checker versus **0** by full checker; source revision `51aab66304b9efecc887608d44267604fecbd57f` | In current compiler checkout: `build/first_paper_contract_ablation_20261106_01/` | Internal source-predicate ablation; invokes R3j `run()` but **is not a separately stamped R3j original experiment** |
| New 6 Nov HLS/RTL bridge | **Implemented, not executed against original raw case folder** | Not yet produced | Needs original R2u folder; must not fabricate from documentation-only counts |

## Scientific conclusions that are already supported

1. On the admitted static MatMul graph subset, tensor-element lifetime accounting finds logical scratch reuse; sensitivity to accumulator accounting is large.
2. The selected emitted C++ tests detect source-memory and arithmetic mutations even when the code's printed memory address witness is not trustworthy. The detector attribution and output-preserving breakdown should be verified from the actual original per-family receipts.
3. HLS BRAM reduction is neither frequent nor guaranteed: storage layout changes reduce the BRAM18K estimate in **2** controlled eligible comparisons. Overall 64 factorial contrasts include seven timing-unqualified cases.
4. Source HDL filename / static memory-named module structure changes do not establish a change in HLS BRAM or physical RAM. In the original **14 qualified address-mode matched comparisons**, HLS BRAM estimates tied despite observable RTL source structure changes.
5. A new 6 November source-only internal ablation demonstrates added checking power for the Phi issue-order predicate on 24 selected adverse controls. This population was constructed using the existing bounded model and is not an independent external benchmark.

## Why the new search returned no HLS folder

The user executed a `find` restricted to `~/Desktop/github_projects/fpgai_compiler/build` after moving the nested copy. Historical research docs use the earlier `~/Desktop/github_projects/fpgai_research/build`. A **missing folder in one checkout is not a failed or unexecuted experiment**. No file search has yet been run on the author's complete workstation root in this turn.

## Safe workstation inventory, NO source modification and NO vendor calls

The following checks paths only. They never invoke Vitis, Vivado, source compilation, package installation, git clean, or deletion.

```bash
for root in \
  "$HOME/Desktop/github_projects/fpgai_research" \
  "$HOME/Desktop/github_projects/fpgai_compiler" \
  "$HOME/Desktop/github_projects/fpgai_first_paper"; do
  if [ -d "$root/build" ]; then
    echo "--- $root/build ---"
    find "$root/build" -maxdepth 3 -type d \( \
      -name 'r2u_hls_factorial_32_original_v1' -o \
      -name 'r2v_r2u_native_factorial_review_v1' -o \
      -name 'r2w_original_r2u_rtl_memory_forensics_v1' -o \
      -name 'r2x_original_r2u_static_hierarchy_v1' -o \
      -name 'r2r_emitter_shadow_216_mutants_original_v1' -o \
      -name 'r3g_c1_independent_event_oracle_v1' -o \
      -name 'r3h_c2_compiled_once_selection_v1' \
    \) -print
  else
    echo "--- no build directory: $root ---"
  fi
done
```

To discover other preserved study directories without choosing filenames in advance:

```bash
for root in "$HOME/Desktop/github_projects/fpgai_research" "$HOME/Desktop/github_projects/fpgai_compiler"; do
  if [ -d "$root/build" ]; then
    echo "--- $(basename "$root") top-level research outputs ---"
    find "$root/build" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
      grep -E '^(r2|r3|cst|compiler_ir)' | sort
  fi
done
```

Only if an original R2u folder actually exists and its hash audit passes should the *new* HLS/RTL bridge be executed into a fresh create-only output directory. Never rerun the original 32 Vitis calls.

## Priority after the inventory

- First: locate the **real original raw files** and perform read-only hash verification, not new native runs.
- Second: analyze the existing HLS/RTL output with the new bridge only if no existing equivalent analysis already answers the question. R2w and R2x already establish RTL structural differences with unchanged HLS BRAM under address lookup changes, so avoid presenting that as a newly discovered contribution.
- Third: focus genuinely missing science: matched external source-equivalence/contract capabilities, richer compositional coverage, generated C++ event refinement, and a separately qualified routed experiment if hardware-savings claims are retained.
- Keep RL source, generated build outputs, and branch history unchanged. The **first paper remains MAJOR REVISION**, not publication-ready merely because tests pass.
