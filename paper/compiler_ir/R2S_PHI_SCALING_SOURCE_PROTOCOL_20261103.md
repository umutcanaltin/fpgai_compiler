# R2s — Falsifiable Phi scaling and lifetime-convention bound study

## Completed original evidence — source revision f38a98cb (2026-11-03)

**WORKSTATION VERIFIED, FROZEN.** After the final schema/test correction, the original `f38a98cb96a96097174204773462a53b8e6c02e2` source checkout passed **19/19 focused** and **1,543/1,543 full pytest**, then the user executed the complete **create-only** `build/r2s_phi_scaling_147_source_v1/` and separate read-only `build/r2s_phi_scaling_147_source_audit_v1.json`. Its saved source revision, full **147 configurations / 126 controlled non-scalar Phi interventions**, and exact result population agreed with the independent same-source E0 recomputation audit. **No new GCC, Vitis, Vivado or RL calls**.

| Outcome over 126 scalar-controlled Phi changes | Committed-value E0 source | Hypothetical first-MAC in-place E0 |
|---|---:|---:|
| Reduced capacity | **102** | **0** |
| Unchanged capacity | **24** | **126** |
| Increased capacity | **0** | **0** |
| Largest observed source-slot reduction | **12** | **0** |

All **126/126** tested interventions satisfied the reviewer-proposed source inequality `S_Phi <= R*(max_reader_PE-1)`, with **0** observed counterexamples. **This is empirical consistency within a finite seven-shape, R=2..4, PE=1/2/4, serial-source schedule; it is NOT a proof of a general bound.** The maximum absolute committed-value reduction as a fraction of the scalar program's **complete declared mutable-array population** was **5.882352941176471%** across the 126 contrasts. For the largest studied tensor `(m,k,n,q)=(16,2,16,16)`, the largest percentage was only **0.38910505836575876%** (four readers, all PE4, 12 slots saved from scalar source scratch 1036 to 1024; baseline full mutable population 3084). The full source-only study sampled seven specified tensor shapes, not all shapes in the dimensional range.

The in-place convention is **hypothetical**, not emitted R2h C++; R2h declares complete accumulators and counters separately from source scratch. Therefore the 102 savings establish source-committed scratch behavior **under one accounting convention**, not physical memory savings or map-dependent benefit under an alternative accumulator implementation. Preserved original `v1` results must **never be rerun, deleted, or overwritten**. The audit checks original SHA/source-derived arrays/CSV status on the matching source checkout, **not independent GCC execution or hardware evidence**.

**Second local workstation smoke (2026-11-03):** At corrected revision `f5e75293e877f845388df8c0ec0a05d5dcd0fffd`, **19/20 focused** and **1,542/1,543 full pytest** passed. The sole remaining failure was a **test's stale case-level key name**: `source_scratch_nominal_16bit_bits` is valid on a *per-intervention comparison*, but actual per-*configuration* case evidence uses `original_r2h_scratch_nominal_16bit_bits`. All CSV export, create-only, read-only audit and input tampering checks passed on that workstation run. The test has been corrected to use the existing case schema; another assertion now explicitly compares the two differently named, equivalent nominal-bit fields. **The corrected version is not yet workstation verified; the 147-case original study is not run.**

**First local pre-experiment smoke (2026-11-03):** At revision `a76301d1a4dfbaa6d1d35a46a23ba77efbc35b86`, the user's workstation found **5 failing tests, 11 passing focused; 5 failing and 1,535 passing full regression**. One test incorrectly compared an integer against a parenthesized Boolean in the whole-tensor-capacity inequality. Four tests failed because `csv.DictWriter` was supplied four additional scalar-vs-intervention capacity fields outside its declared columns. **These are an assertion error and an export-schema defect, not observed Phi bound counterexamples or original R2s results**. Subsequent GitHub patches fix both, add a single-field CSV tampering control, require the source-evidence column set to match **before** creating a frozen original directory, and enforce the exact CSV header in the read-only auditor. **The patched code is not yet workstation-verified**, and the original full 147-case R2s study has **NOT run**. Run focused/full tests again before starting the create-only original study.

**Status, 2026-11-03:** prospectively committed source-only experiment/tests, **not workstation executed yet**. The original R2r emitted-source mutation campaign is **VERIFIED and frozen** at `fbdd34ea363647dbea700223f923d0dad8547732` (22 focused / 1,531 full pytest, 6 original unmutated GCC baselines, **216/216 generated-C++ mutants detected: 192 dynamic source shadow, 24 existing R2h checker**, zero survivors or invalid failures; 222 GCC attempts). The separate read-only audit agreed; no new vendor or RL calls. Original R2q is also frozen at `7890c6f363cc9dd61c8bfa29ae121107ee3d86de` (36 source configurations, 18 Phi reductions under committed-value but **0** under hypothetical first-MAC in-place). The primary paper is **MAJOR REVISION**.

## Motivation and falsifiable hypotheses

The scientific review suggested a bounded upper limit for the **source committed-value scratch-capacity reduction** induced by changing reader mapping from scalar to `PE=w`: `S_Phi <= R * (w-1)` source scalar slots. The review also suggested that the relative saving shrinks rapidly for larger dimensions and disappears under a hypothetical in-place reader-output memory model. These are **testable conjectures, not established theorems**. The bound may depend on the exact issue order, partial-width final tiles, fanout degree and all-reader mapping. A violation must be **retained and reported**, never silenced to preserve the hypothesis.

R2s reuses the **existing** R2f source issue/event verifier, R2h source-level minimum slot-layout compiler, and R2q committed-versus-in-place accounting; it does **not** invent an alternative source event order. The independent source observation is the exact source-slot liveness capacity under a **serial** event issue policy, not achieved FPGA BRAM or physical memory-bank behavior.

## Prespecified finite study

Original source program: `src/fpgai_research/experiments/cst_r2s_phi_scaling.py`; checks: `tests/test_cst_r2s_phi_scaling.py`. Exactly **seven** dimension tuples `(m,k,n,q)`:

| ID | Dimensions | Purpose |
|---|---|---|
| 0 | (2,2,3,4) | Existing small R2q baseline |
| 1 | (2,3,5,4) | Existing unequal source-reduction dimension |
| 2 | (3,3,4,4) | Existing moderate baseline |
| 3 | (4,2,4,8) | More output columns |
| 4 | (8,2,8,8) | Mid-sized workload |
| 5 | (16,2,16,8) | Largest matrix rows/reduction, narrower reader outputs |
| 6 | (16,2,16,16) | Maximum `m,n,q` admitted by this bounded fanout source compiler |

All have fanout readers `R ∈ {2,3,4}`. For each graph, evaluate exactly **seven** reader PE assignments:

- All scalar (all reader PEs 1), as fixed baseline.
- First reader PE2, last reader PE2, all readers PE2.
- First reader PE4, last reader PE4, all readers PE4.

This gives **7 shapes × 3 fanout degrees × 7 policies = 147** complete source E0 configurations; **126** nonscalar, same-graph controlled Phi interventions. `simd=1` throughout. No unspecified software or HLS variants are silently inserted.

For each configuration, the actual compiler produces and verifies the source event schedule and scratch slot assignment. R2q baseline code separately reports: disjoint no reuse; feasible tensor-granular first-fit; committed-value element-exact source slots; hypothetical in-place first-MAC lifetime peak; actual R2h separate full-size accumulator/counter declarations and total mutable-array elements. This is **not** an emitted fused in-place variant or original GCC experiment.

For each **nonscalar** intervention, record signed `scalar_slots - intervened_slots` for both lifetime interpretations; do not clip negative effects. Let `w=max(reader_PEs)`; calculate the **proposed** bound `B=R*(w−1)`, `observed<=B`, and the percentage effect relative to the original all-declared mutable element population. Do **not** call the bound proven because all tested cases satisfy it. A counterexample would refute it for this issue-policy definition. For the largest shape, report all 18 non-scalar comparisons without hiding zero effects. The experiment also reports **nominal 16-bit source scratch bits** but makes no claim about BRAM blocks, bank ports, FPGA timing or achieved occupancy.

The complete original R2s result is **verified and immutable**, as recorded at the top of this protocol. The code's test suite uses only the first shape (21 configurations and 18 contrasts) and includes a purposely forged counterexample to ensure bound violations are reported, plus corruption and source-SHA tests.

## Read-only source receipts and immutable original execution

The original experiment writes only to a **fresh, create-only** directory:

- `cases.json`: all configurations and the actual source-event SHA.
- `comparisons.json`: all 126 controlled Phi contrasts with both accounting conventions and the proposed bound outcome.
- `comparisons.csv`: table-readable full contrast population, including negative effects.
- `summary.json`: full denominators, reductions/ties/increases, violations and largest cases.

The separate read-only audit recomputes the original source E0 results with the **same clean original Git revision**, verifies all case/contrast/summary and CSV fields. It is **not independent GCC execution**, a formal mathematical proof or native FPGA verification. Run both at the **same source revision**, and preserve original output paths permanently.

## Commands on the existing Linux workstation

No GitHub Actions/tokens, no Vitis/Vivado and no RL training.

```bash
cd ~/Desktop/github_projects/fpgai_research
git status --short
git pull --ff-only origin main
git rev-parse --short HEAD

python -m compileall -q src/fpgai_research experiments/compiler_ir tests

python -m pytest -q \
  tests/test_cst_r2s_phi_scaling.py \
  tests/test_compiler_ir_series.py

python -m pytest -q
```

**Historical commands of the ALREADY COMPLETED, frozen v1 original study** (documentation only; DO NOT rerun or overwrite these paths):

```bash
python -m fpgai_research.experiments.cst_r2s_phi_scaling \
  --output build/r2s_phi_scaling_147_source_v1 \
  --shape-limit 7

python -m fpgai_research.experiments.cst_r2s_phi_scaling \
  --study build/r2s_phi_scaling_147_source_v1 \
  --audit-output build/r2s_phi_scaling_147_source_audit_v1.json

cat build/r2s_phi_scaling_147_source_v1/summary.json
```

The original results are **strictly create-only and now frozen**. Do not repeat any frozen original R2h/R2i/R2j/R2l/R2m/R2n/R2o/R2p/R2q/R2r campaign. The legacy `experiments/compiler_ir/series.py` quick profile includes a one-shape (**21 cases**) R2s stage; the paper profile schedules all seven (**147**). Execute only this standalone R2s command for now.

## Acceptance, limits, next scientific steps

If the source-bound conjecture fails, preserve the complete negative cases. If it holds in every tested case, **mathematical proof is still absent**, and any source-only saving must be reported as a fraction of total declared mutable elements rather than isolated scratch. In particular, even a validated `R*(w-1)` source-slot bound must not be translated to a routed BRAM or power claim.

If hypothetical first-MAC in-place differences remain zero, continue to classify **mapping-dependent storage savings as allocation-model-sensitive** rather than broadly physical. A true fused/in-place emitted GCC implementation or native HLS CSynth comparison is necessary before changing that interpretation.

Remaining gates: prove/refute the issue-order-bound analytically; stronger tensor-granular **feasible optimum** (integer programming rather than simply first-fit); HLS address generation × scratch padding × binding factorial; independent generated-source memory/refinement proof obligations; prior-art mechanism-level comparison (StreamTensor, Filament, MODeL, CODO) and a genuinely budgeted C3/RL decision-interface consumer evaluation. The present R2s study does not test any of those directly.

See [independent reviewer scientific triage](INDEPENDENT_REVIEW_TRIAGE_AND_R2Q_PROTOCOL_20261103.md), [R2q baseline interpretation](MANUSCRIPT_DRAFT.md), [R2r mutation protocol](R2R_EMITTED_SOURCE_SHADOW_MUTATION_PROTOCOL_20261103.md), and [current scientific status](CURRENT_SCIENTIFIC_STATUS.md).
