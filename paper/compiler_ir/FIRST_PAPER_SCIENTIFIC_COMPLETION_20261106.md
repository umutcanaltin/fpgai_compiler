# First FPGAI paper: scientific completion gates

**Decision, 6 November 2026.** Pause development on the RL hardware-optimization branch. Preserve the branch, its original checkpoints, and all experiments without modification. This paper addresses **architectural semantics, safe tensor-element storage reuse, generated-source verification, and measured source-to-HLS translation**. Trained RL optimization is a separate publication.

**Read the [existing evidence inventory](EXISTING_EVIDENCE_INVENTORY_20261106.md) before any additional study.** Most original memory, GCC, RTL and HLS experiments were already completed in the former `fpgai_research` checkout. The inventory distinguishes finished work from genuinely missing experiments and provides safe, read-only file discovery.

**Status: MAJOR REVISION / NOT SUBMISSION READY.** New code and tests are proposed experiments, not observed results. Never overwrite or silently rerun source-pinned native observations.

**Prospective better-hardware mechanism:** [Verified selective intermediate materialization protocol](VERIFIED_SELECTIVE_MATERIALIZATION_RESEARCH_20261106.md) defines a *new* materialize / forward / selectively spill compiler experiment. This is not an observed hardware improvement. It specifically addresses why earlier exact scratch allocation saved too little of the total generated mutable state to change HLS BRAM consistently.

**Native R4b update (6 November):** [Original case-003 E2 failure report and read-only diagnostic command](R4B_NATIVE_CASE003_FAILED_CSIM_20261106.md). Both full-H and row-H Vitis 2023.2 processes exited 0 but failed E2 CSim; E3-qualified resource comparison is **unavailable**. Preserve the two original vendor runs. The user's subsequent selected stdout extract reports identical aggregate CSim results (28/128 failed outputs per arm, maximum error 0.055395125 against tolerance 0.02); see the new [R4c read-only numerical triage protocol](R4C_CSIM_NUMERICAL_TRIAGE_20261106.md). No numerical correction or extra native case has been established. **The author's workstation has now verified 23 focused tests and completed R4d 3/3 actual GCC E1 pairs** at source revision `abe6aa3`; see [R4d protocol](R4D_MATCHED_NUMERIC_PRECISION_20261106.md). The new [R4e gate](R4E_NUMERIC_FORMAT_NATIVE_GATE_20261106.md) has now passed **6 focused workstation tests**, and the author's original matched **32/6** Vitis pair qualified E2 and E3 on both arms. Crucially, estimated BRAM18K **tied 13 vs 13**, while row H used 4 fewer DSP, 235 fewer LUT, 151 fewer FF and 410 more worst cycles; see the [R4e result record and untested R4f read-only report protocol](R4E_32_6_WORKSTATION_HLS_RESULTS_AND_R4F_REVIEW_20261106.md). These are HLS estimates, not placed-and-routed physical results; new 16/6 and 32/8 have not been natively executed. The subsequent workstation **R4f read-only report passed 6/6 focused tests** and confirmed exactly **1/3 E3-qualified format pairs** with zero new vendor calls. The [R4g source-attested static RTL memory bridge](R4G_READ_ONLY_RTL_MEMORY_BRIDGE_20261106.md) has now passed **20 workstation tests** and confirms the original 32/6 same-memory-module-count and HLS BRAM tie. The two scratch-RAM source SHA hashes differ, accumulator-RAM source SHA is unchanged, and all three weight ROM payload hashes match. The [R4h per-module source SHA crosswalk](R4H_HLS_MEMORY_SOURCE_CROSSWALK_20261106.md) has now **passed 19 workstation tests** and identified original 32/6 HLS-generated scratch-RAM **module defaults** of 32 bits × **214 entries** in full H versus 32 bits × **134 entries** in row H, while the accumulator defaults are identical at 32 × 256 and the three ROM width/depth defaults and payload digests agree. This is a **source default**, not proven elaborated memory allocation. The [R4i literal-default analysis](R4I_ORIGINAL_RTL_MEMORY_PARAMETER_DEFAULTS_20261106.md) has now passed **24 focused workstation tests**, reproducing the original-source default difference of **−80 entries / −2560 declared data bits**, with unchanged accumulator/ROM defaults and a 13 vs 13 HLS BRAM estimate. It does NOT equate this source default change to the earlier 240-cell compiler source saving or to elaborated/mapped memory. A new [R4j original RTL memory-instance source-site study](R4J_ORIGINAL_RTL_MEMORY_INSTANCE_SOURCE_SITES_20261106.md) is implemented but NOT YET workstation-tested, with explicit UNKNOWN for symbolic/positional overrides and no native calls.

## Scientific hypotheses and the required contrasts

| Question | Prespecified comparison | Acceptable conclusion |
|---|---|---|
| What additional guarantee does the mapping contract provide? | Identical graph, Phi, and event traces under (a) readiness, lifetime and Phi owner coordinates without exact READ issue order, and (b) the same predicates plus declared Phi READ issue order | The extra predicate rejects selected value-safe, architecture-order-invalid traces while retaining positive controls. This is an internal ablation, not an external compiler benchmark. |
| Is the abstract contract observed in generated C++? | SHA-pinned emitted operations, real load/store instrumentation, selected faults that preserve the printed source address witness | Detect selected actual-access faults; distinguish 192 memory-operation mutants from 24 arithmetic mutants; separately measure how many preserve the final numerical output. |
| When does source reuse affect HLS memory? | Identical graph/clock/precision/architecture, controlled allocation and storage materialization, paired E3 reports | Measure qualified HLS BRAM18K deltas and report unknowns. Correlate with source-attested static RTL, but do not equate source RTL module names with physical RAM instances. |
| Is there an actual hardware benefit? | Clean-DRC placed/routed matched arms with identical constraints | Report absolute BRAM, DSP, LUT, FF, WNS, numerical error and latency. Warned or timing-failed runs are not qualified. |
| Is the mechanism scientifically distinguishable? | Pinned, executed Allo and related system comparison with the same source obligations | Report exactly which invariant each implementation checks. Allo already supports source-equivalence checks. Errors and unsupported probes are not competitor failures. |
| Does the contract generalize? | At least one mixed Conv2D/Dense/fanout graph under the SAME contract checker | Show positive and independently specified negative cases. Otherwise narrow paper to static MatMul chains and fanouts. |

## Completed evidence, without new claims

- Original generated C++ experiment: 192 selected load/store mutations plus 24 arithmetic mutations, all detected in a selected six-graph source cohort. Source-value and final-output acceptance must be distinguished when interpreting these.
- Mapping-aware committed-value scratch accounting: 18 of 27 comparisons reduce logical slots, 9 tie. Hypothetical first-MAC in-place accounting yields 0 reductions. This is not evidence of physical BRAM savings.
- Exact contiguous whole-tensor source placement: 36 completed reference cases, 12 better than greedy first-fit under fixed event ordering.
- Frozen 32-arm memory HLS factorial: source revision 0c6d85f189062df8d5157237f3c8ac74ac8da4b8, part xck26-sfvc784-2LV-c, Vitis requested clock 8 ns, uncertainty 1.5 ns, effective HLS criterion 6.5 ns, absolute C simulation tolerance 0.01. All 32 E2 pass; 30 E3 pass. Sixty-four conditional comparisons: 2 BRAM18K estimate reductions, 47 ties, 8 increases and 7 unqualified.
- Existing source-attested Verilog inventory and static module analysis cover the same 32 cases. Their reachable, memory-named module candidates are not physical RAM cells.
- Bounded semantics: five exact storage oracle agreements, 25 rejected selected mutants, 133 event traces and zero disagreements between interpreters sharing one static model specification.
- Typed action interface: 48 source actions with 21 certified, 24 rejected, 3 uncertified; masking the entire catalog costs 48 source compiler transitions.
- Separate complete Conv/Dense HLS cohort: six E2 passes, four E3 passes. Qualified mappings increase cycles 624 to 911 and 2365 to 5436; both strided arms fail a 9 ns effective criterion (10 ns request less 1 ns uncertainty) with estimates of 9.181 ns.
- Earlier six routed fanout comparisons are descriptive BRAM ties, with zero strict clean-DRC-qualified pairs. The newer physical pilot attempted one of nine cases and did not satisfy strict E4.
- Pinned Allo study NOT RUN. At the last recorded workstation checkpoint, the 216-trace held-out source and 192-action source limit studies were IMPLEMENTED, NOT YET EXECUTED. Do not count their planned outcomes as measurements.

## New experiment A: matched source-contract ablation

Implementation: src/fpgai_research/experiments/cst_paper_contract_ablation.py

Uses the predeclared 24-profile, 216-trace source population already defined for the held-out fanout experiment. Holds the source trace fixed while adding/removing only the operator-local Phi READ issue-order predicate; the baseline retains Phi owner-coordinate checks. It does NOT simulate an independent competitor, final-output checker or physical FPGA.

Run in a separate clean first-paper checkout:

    python -m pytest -q tests/test_cst_paper_contract_ablation.py tests/test_cst_r3j_c1_heldout_semantics.py
    python -m fpgai_research.experiments.cst_paper_contract_ablation --output build/first_paper_contract_ablation_20261106_01

The output folder is create-only with a JSON summary, per-trace CSV and SHA-256 digest inventory. These new numbers are not workstation-verified yet.

## New experiment B: HLS BRAM versus static RTL structure

Implementation: src/fpgai_research/experiments/cst_paper_memory_bridge.py

Reuses the existing read-only R2x source/XML/Verilog SHA audits, joins all 64 same-graph conditional contrasts, preserves qualified and unqualified E3 outcomes separately, and produces a descriptive contingency comparing estimated BRAM18K changes with changes in reachable memory-named RTL source module candidates. A static type graph is not a post-synthesis netlist or placed BRAM count.

    python -m pytest -q tests/test_cst_paper_memory_bridge.py tests/test_cst_r2x_rtl_hierarchy_analysis.py
    python -m fpgai_research.experiments.cst_paper_memory_bridge \
      --study /absolute/path/to/build/r2u_hls_factorial_32_original_v1 \
      --output build/first_paper_memory_bridge_20261106_01

The original 32-case study must remain UNMODIFIED. The new cross-stage contingency has not yet been executed on the workstation and no Vitis/Vivado calls are made by this analysis.

**New implemented source intervention:** [R4a row-retained intermediate materialization study](R4A_ROW_RETAINED_MATERIALIZATION_EXPERIMENT_20261106.md) with guarded R4b native Vitis qualification. Its implementation changes the GLOBAL source event order, preserves per-operator Phi read order, and reuses a single H row across producer rows. **The R4a E1 workstation run is now observed:** 26 passing targeted tests, quick 3/3 and full 5/5 matched GCC pairs at source `fe99d627...`; see [workstation results](R4A_WORKSTATION_SOURCE_RESULTS_20261106.md). Source outputs still require read-only local SHA audit before final manuscript use. **One R4b native pair was attempted; both arms failed E2 CSim, and no new qualified E3 or physical E4 result exists.** Do not report hardware improvement from source-only savings.

## Validation before merging or publication use

Use a separate clean clone so the original RL environment/branch is untouched:

    git fetch origin
    git switch research/first-paper-source-memory-bridge-20261106
    python -m pip install --no-deps -e .
    python -m compileall -q src/fpgai_research/experiments tests
    python -m pytest -q tests/test_cst_paper_contract_ablation.py tests/test_cst_paper_memory_bridge.py
    python -m pytest -q
    git rev-parse HEAD
    git status --short

Record the exact Git SHA and preserve raw outputs in new create-only directories. Do not count passing synthetic unit fixtures as native replication. No GitHub Actions, Vitis, Vivado, Allo or hardware run is claimed from these code commits.

## Outstanding scientific gates

1. Run and independently audit new A and B. Preserve adverse results, complete denominators and unknown outcomes.
2. Execute and audit already implemented held-out semantics and source limit studies (in new original folders if absent), including compiler transition cost and failed controls.
3. Identify output-preserving emitted C++ mutations and establish a bounded emitted loop/event refinement condition, using direct actual-access checks rather than printed addresses only.
4. Extend the SAME contract checker to a mixed operator graph, or explicitly narrow generality.
5. Execute the pinned external compiler comparison and compare equivalent semantic obligations, not just API names.
6. R4e now supplies **one** qualified 32/6 matched HLS comparison, with a BRAM tie and mixed logic/cycle tradeoff. Complete read-only R4f audit, isolate numerical effects, explain inferred memory mapping, and design a new controlled vendor/clean-DRC physical E4 study before claiming memory or throughput superiority. Preserve all original native cases.
7. Assemble a rights-cleared replication artifact and revise the manuscript using only attested outcomes; obtain independent scientific review and author approval.

**First-paper submission remains blocked** until the verified distinctive mechanism and independent external comparison justify the claims. Completing an engineering sprint or writing a stronger abstract does not itself close this scientific gate.
