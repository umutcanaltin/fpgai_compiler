# First FPGAI paper: scientific completion gates

**Decision, 6 November 2026.** Pause development on the RL hardware-optimization branch. Preserve the branch, its original checkpoints, and all experiments without modification. This paper addresses **architectural semantics, safe tensor-element storage reuse, generated-source verification, and measured source-to-HLS translation**. Trained RL optimization is a separate publication.

**Read the [existing evidence inventory](EXISTING_EVIDENCE_INVENTORY_20261106.md) before any additional study.** Most original memory, GCC, RTL and HLS experiments were already completed in the former `fpgai_research` checkout. The inventory distinguishes finished work from genuinely missing experiments and provides safe, read-only file discovery.

**Status: MAJOR REVISION / NOT SUBMISSION READY.** New code and tests are proposed experiments, not observed results. Never overwrite or silently rerun source-pinned native observations.

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
6. Design a new single-factor, original-source, matched Vitis then clean-DRC Vivado memory-materialization study with actual reported BRAM/clock/logic and numerical qualification, without revisiting frozen old vendor cases.
7. Assemble a rights-cleared replication artifact and revise the manuscript using only attested outcomes; obtain independent scientific review and author approval.

**First-paper submission remains blocked** until the verified distinctive mechanism and independent external comparison justify the claims. Completing an engineering sprint or writing a stronger abstract does not itself close this scientific gate.
