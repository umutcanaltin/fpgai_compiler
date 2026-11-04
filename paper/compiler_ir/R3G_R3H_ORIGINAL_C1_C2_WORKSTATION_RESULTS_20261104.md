# Frozen R3g/R3h C1/C2 workstation source checkpoint — 4 November 2026

**Original observed source revision:** `3309076814714e61906363e21e3d50b932519f9d`. This is an original terminal report supplied by the author. The author ran these commands in the existing local checkout; this repository update **does not rerun, mutate, or replace** any of the original experiment directories.

The author's complete Python regression returned **1,751/1,751 passed** in **162.15 s**, following **64/64 focused passed** in **4.30 s**. The `python -m compileall -q src/fpgai_research` command produced no reported errors.

## Original R3g — independently interpreted bounded C1 event semantics

- Saved original directory: `build/r3g_c1_independent_event_oracle_v1` (**create-only, frozen**).
- Prespecified static source cases: **5**.
- Independently compared C1 source trace variants: **133** (canonical traces, adjacent swaps and additional adverse mutations).
- Independent source event interpreter vs compiler status disagreements: **0 / 133**.
- Original source SHA-256 revision: `3309076814714e61906363e21e3d50b932519f9d`.
- Read-only audit output SHA-256: `847efcd501860bb5e58757766a8ed64440e14f558dac7c666f42096e96ab542d`.
- Vendor calls during source campaign/audit: **0**.

**Interpretation:** agreement holds for these 133 finite source traces, within the existing bounded 2–4 reader MatMul/Relu/Add-chain source semantics. Both interpreters **share the same admitted static model specification**. This cannot prove correctness for arbitrary graphs, all malformed traces, generated RTL or real FPGA source-cycle schedules. It is not a competitor benchmark.

## Original R3h — compiled-once, policy-neutral C2 action catalog

- Saved original directory: `build/r3h_c2_compiled_once_selection_v1` (**create-only, frozen**).
- Prespecified source graphs: **6**.
- Original total finite typed candidate actions: **48**.
- Actual original compiler source transitions to construct all action catalogs: **48**.
- **Additional** original compiler source transitions to select a precomputed E0-certified action: **0**.
- Original source SHA-256 revision: `3309076814714e61906363e21e3d50b932519f9d`.
- Read-only audit output SHA-256: `510746a5b78cc4b0b4c247b15d24510081f4685a2dd0cd2db064edec3f7dc72f`.
- Vendor/agent-training calls: **0**.

**Interpretation:** a catalog can return the already compiler-certified next state without a **second** source compile after a full mask was built. It is not evidence that full catalog precomputation is cheaper than directly compiling a single blind action or that RL policies improve. The mask costs **one compiler transition per candidate**.

## Allo R3i: only protocol inspected

The user executed `--protocol` and received `external_artifact_executed=false`, `result_status=NOT_RUN`, pinned external Allo revision `abc270c7a1919c780df2b7d7e99de45e81c175b9`. No Allo source or PAST verifier was executed. **Allo capability comparison remains UNKNOWN.** No claim of superior source verification is justified by R3i.

## Scientific evidence status

The combined original finite experiments now include R3d (6 source actions), R3e (5 fixed-order slot oracle cases, 25 targeted mutations), R3f (48 source actions), R3g (133 interpreted source traces), and R3h (48 cached source actions), **each with separate revision-pinned observations**. This is NOT an additive statistical population (some cases share constructs), and must not be reported as independent random samples.

C1 and C2 have stronger **internal bounded source evidence**. Remaining external novelty/comparison, scope beyond static fanout/chain MatMul, and independent manuscript scientific review are still unsatisfied. The CNN native cohort is separate and its negative HLS E3 observations are unchanged.

**Publication decision:** MAJOR REVISION / NOT YET SUBMISSION-READY. Further extensions must produce new revision-specific artifacts rather than modifying these saved source receipts.
