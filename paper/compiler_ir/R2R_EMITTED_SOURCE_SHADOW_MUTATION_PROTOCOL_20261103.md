# R2r — Actual emitted C++ shadow-memory and mutation-adequacy experiment

**Scientific status (2026-11-03): ORIGINAL SOURCE EXPERIMENT WORKSTATION VERIFIED** at clean original revision `fbdd34ea363647dbea700223f923d0dad8547732`. Original local focused **22/22** and full **1,531/1,531** pytest passed. The full 6-graph/216-mutant create-only study ran: **6/6 unmodified GCC baselines qualified, 216/216 source C++ mutants detected (192 dynamic shadow, 24 original R2h checker), zero survivors, invalid/preflight/unknown outcomes**; 222 original GCC attempts. Independent **read-only** source SHA/identity/outcome audit agreed, but it did **not** replay GCC. Original artifacts frozen; later commands document reproduction only and must not overwrite these `v1` outputs. Original R2q is completed at **`7890c6f363cc9dd61c8bfa29ae121107ee3d86de`**, 17 focused and 1,516 full tests passed, with its original 36-case E0 study and separate read-only audit frozen. R2q confirmed **18/27 Phi interventions saved committed-value scratch** but **0/27 saved hypothetical first-MAC in-place peak**; source-only and no new GCC. The primary paper remains **MAJOR REVISION**.

## Why R2r matters scientifically

The independent reviewer found that the R2f E0 event certificates and R2h GCC address prints share an underlying issue specification. This makes an accidentally incorrect **emitted memory operation** a stronger falsification target than another synthetic Python event mutation. R2r adds a *separately injected, actual-execution* shadow memory guard to generated R2h C++ and systematically mutates the C++ **memory operations**. The original trusted R2h verifier still checks source event order, emitted runtime slot address witness, all intermediate values and final NumPy correctness.

Our shadow guard does **not** trust the printed `FPGAI_SLOT` or `FPGAI_R2G` labels to define occupancy. It maintains an independent `shadow_owner[physical_slot]` and `shadow_remaining[logical_value]`, enforcing:

1. No double commit and no overwriting an occupied physical slot.
2. Every actual scratch **load** accesses the slot occupied by the intended committed value, while it still has at least one permitted read.
3. All required reads occur before that physical slot is released.
4. Every logical intermediate was committed exactly once and all obligations discharged at function exit.

Read multiplicity comes **directly from the recognized static fanout topology**, not from the C++ trace or proposed issue-order policy. For the producer H, every one of the R MatMul readers consumes each H scalar **q times**, and each reader/Add intermediate is consumed **once** by its downstream Add or Relu. All wrapped C++ loads/stores use the shadow memory's **actual slot parameter**, whereas existing `FPGAI_SLOT` prints remain the original unaltered claimed address. This adversarial separation makes it possible to catch a bad actual access even if the source prints the expected labels.

These are **host-only instrumented source safeguards**, not independently proved binary memory instrumentation, a generalized arbitrary-DAG machine model or an RTL/Vivado verifier. The shadow still shares the admitted topology, names and tensor shapes with the compiler; future independent/oracular byte-level memory tests remain needed for stronger proof.

## Exact prospective matrix

File `src/fpgai_research/experiments/cst_r2r_emitter_shadow_mutation.py`, test `tests/test_cst_r2r_emitter_shadow_mutation.py`.

Six bounded graphs use **2, 3 or 4 fanout MatMul readers**, two representative graph shapes for each degree, and scalar/all-pes2 reader policies, all with exact `per_element_optimal` source scratch. Each has an **original baseline instrumented GCC execution** that must pass complete R2h source/numeric/slot checks. Only after baseline success, **36 distinct C++ mutants per graph**, for exactly **216 total mutants** and **222 attempted GCC executions** across the complete original cohort, run.

Every graph has **9 predetermined mutation families × 4 payload variations**:

| Family | What C++ changes | Intended detecting guard |
|---|---|---|
| `producer_store_slot` | Actual H physical scratch STORE index shifts, printed slot trace remains unchanged | Shadow physical owner / original source address |
| `producer_read_slot` | Actual H READ index shifts, labels unchanged | Shadow wrong-owner/read obligation |
| `reader_store_slot` | Actual MatMul output STORE index shifts | Shadow ownership |
| `add_left_read_slot` | Actual Add left operand LOAD index shifts | Shadow actual read |
| `add_right_read_slot` | Actual Add right operand LOAD index shifts | Shadow actual read |
| `add_store_slot` | Actual Add output STORE index shifts | Shadow ownership |
| `final_read_slot` | Actual Relu operand LOAD index shifts | Shadow actual read |
| `skip_producer_store` | One H element's actual STORE is suppressed but its printed COMMIT still occurs | Shadow missing store/owner |
| `producer_numeric` | Producer arithmetic perturbation in real C++ | Original independent numerical E1 checker |

Each modified source is stored **verbatim** with its SHA-256 plus one case receipt. Keep *all* `DETECTED_DYNAMIC_SHADOW`, `DETECTED_R2H_CHECKER`, `SURVIVED`, `INVALID_COMPILE` and `UNKNOWN_RUNTIME_FAILURE` outcomes. Compile-invalid mutants are **not** credited as detected unsafe source, and surviving mutants are **not** hidden. An independent **read-only SHA and deterministic source-regeneration audit** checks baseline C++/GCC stdout, every mutant source and outcome denominator, but does not rerun GCC.

The full 216-case original experiment **has now been verified** with the exact outcomes above. This is mutation adequacy for the prespecified selected fault families only, and must never be described as a correctness proof for arbitrary source/RTL.

## Verification steps on existing laptop checkout

No GitHub Actions or tokens and no native HLS/vendor calls.

```bash
cd ~/Desktop/github_projects/fpgai_research

git status --short
git pull --ff-only origin main

python -m compileall -q src/fpgai_research experiments/compiler_ir tests

python -m pytest -q \
  tests/test_cst_r2r_emitter_shadow_mutation.py \
  tests/test_compiler_ir_series.py

python -m pytest -q
```

The R2r focused test itself contains an **actual one-graph, 36-mutant GCC experiment in pytest's own ephemeral directory**. It does not touch the frozen original R2h/R2q outputs or create any persistent paper experiment. It should classify zero survivors, zero compiler-invalid and zero unexplained errors for the test graph; if it does not, investigate and **retain the failure**, do not change the expectation merely to make tests green.

**Historical exact commands for the already-completed original study** (create-only; DO NOT rerun/overwrite the existing `v1` output). For a distinct new cohort use entirely new versioned paths and its own source revision:

```bash
python -m fpgai_research.experiments.cst_r2r_emitter_shadow_mutation \
  --output build/r2r_emitter_shadow_216_mutants_original_v1 \
  --graphs 6 \
  --mutants-per-graph 36

python -m fpgai_research.experiments.cst_r2r_emitter_shadow_mutation \
  --study build/r2r_emitter_shadow_216_mutants_original_v1 \
  --audit-output build/r2r_emitter_shadow_216_mutants_audit_v1.json

cat build/r2r_emitter_shadow_216_mutants_original_v1/summary.json
```

Both paths **strictly create-only**. If a path already exists, never delete or overwrite it: inspect the original partial failure data or choose a new source-pinned version. The auditor compares every mutant's C++ content with exact deterministic regeneration from its saved baseline C++ and manifest. Preserve any *survivors*, compile errors and failure receipts.

In the overall `experiments/compiler_ir/series.py` plan, quick profile runs only **1 graph × 4 mutations** and paper profile runs the full **6 × 36**, under `emitter_shadow_mutation_e1`. Do not run the whole old native series to test this new stage.

## Scientific acceptance and remaining gates

A complete, **compiled** and rejected matrix with no survivors supports **mutation adequacy only for these exact nine fault families and source graphs**. It does **not** prove that arbitrary emitted C++ exactly refines E0, that a maliciously forged source cannot collude with the shadow, that all memory violations are caught, or that hardware port/cycle semantics hold. Distinguish dynamic shadow catches from the earlier source/NumPy checker, and disclose all detection rates as **detected / attempted / compiling / valid** separately.

After this, the independent review still asks for: formal and empirically tested final-reader theorem; larger source scaling and classical tensor/ILP optimum; HLS binding/padding/addressing factorial explaining the source-to-BRAM mismatch; artifact-level comparisons with StreamTensor/Filament/MODeL/CODO; and a meaningful capacity-delta outcome or actual budget-controlled proposer experiment for C3. **The RL-compatible interface remains secondary engineering evidence, not an invented trained-RL improvement**.
