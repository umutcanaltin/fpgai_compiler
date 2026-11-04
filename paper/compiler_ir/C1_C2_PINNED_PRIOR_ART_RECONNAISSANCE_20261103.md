# C1/C2 independent prior-art artifact reconnaissance (3 November 2026)

**This is an initial reproducible source inspection, NOT an executed equivalent-configuration comparison or proof of exclusive novelty.** Claims in the compiler IR paper must remain **MAJOR REVISION** pending actual competing-artifact test cases.

## Public artifacts pinned on 3 November 2026

| Project | Public repository | Observed default-branch HEAD | Relevant overlap |
|---|---|---|---|
| Allo (PLDI 2024) | https://github.com/cornell-zhang/allo | `abc270c7a1919c780df2b7d7e99de45e81c175b9` | Composable kernel, memory, communication and type customizations; source/schedule verification |
| CODO (ISCA 2026) | https://github.com/sjtu-zhao-lab/codo-artifact | `e11913db4da01946d66ac5215c262c53da98aa30` | FPGA dataflow violation detection, data-movement optimizations, scheduling |
| ST-Flow (PACT 2026) | https://github.com/UCLA-VAST/ST-Flow | `2daafad8a0b116cf4fb84ecceb9b58e16bed725e` | Spatial/temporal/hybrid dataflow compilation |

These commits are **observed remote repository HEADs**, not locally installed, executed, or reviewed end-to-end. Versions could change later; always fetch the pinned commit for the formal study. Public paper sources: Allo [arXiv 2404.04815](https://arxiv.org/abs/2404.04815), CODO [arXiv 2604.12618](https://arxiv.org/abs/2604.12618), StreamTensor [arXiv 2509.13694](https://arxiv.org/abs/2509.13694), and Filament [PLDI 2023](https://doi.org/10.1145/3591234).

### Concrete initial Allo code observation

The inspected `allo/verify.py` at the public default branch defines `verify(schedule_a, schedule_b)`. It builds two schedules targeting `vhls`, serializes C output, and passes these files to PAST verification; non-equivalence yields a generated-code diff. This means Allo has **substantive source/schedule-equivalence verification** and must not be described as lacking verification. The inspected file alone does *not* answer whether it also checks our exact **multi-reader last-use scratch alias** and **per-operation source issue-order** relations.

**Next independent artifact experiment required:** implement source-admitted and rejected 2-reader MatMul → Add workloads in the pinned Allo version with exactly matched shape, input weights, intermediate placement, and target assumptions. Test (i) expected safe schedule; (ii) premature last-reader overwrite; (iii) changed reader issue order with preserved value readiness; (iv) rejected PEs/illegal memory design. Explicitly distinguish "unsupported test representation" from "representable but verifier accepted" and "representable and rejected". Record source files, tool versions, exit codes, actual verifier semantics, and the same numerical/target constraints.

**Do not** claim an invalid source program *should* compile in a competitor in order to manufacture a failure. If Allo cannot encode a proposed source-visible action under its supported interface, establish and document that concrete limitation from its actual artifact and reviewer-visible test, not a guess.

### C1/C2 paper novelty risk

Allo's source verification, CODO's dataflow-violation checker, Filament's timing types, and StreamTensor's typed streaming/fusion are meaningful prior art. Our proposed contribution must be differentiated as a **specific bounded model-level source event contract linked to the *same* policy-independent complete action transition interface**, if independently verified. Using a GNN-ready graph observation, an action mask, or Conv/MatMul support alone does **not** establish novelty.

No agent is trained for this paper. The separately scoped future agent paper could test whether such contract APIs actually reduce reward/learning cost. Original R3a/R3c HLS negative outcomes remain frozen and must be included, not hidden.


## Deeper Allo artifact inspection: same pinned revision, source not executed

Additional **read-only** files inspected at `abc270c7a1919c780df2b7d7e99de45e81c175b9`:

- `allo/memory.py`: defines `Layout` with sharding/replication placements and `Memory` configurations for implementation resources and port patterns. This is meaningful memory-type/source-layout support, not mere tensor shapes.
- `tests/test_schedule_memory.py`: contains real `reuse_at`, buffer/partition and Conv2D line-buffer memory schedule tests. This rules out claims that Allo lacks explicit source reuse or Conv-related memory management.
- `tests/test_schedule_compose.py`: includes use-def-chain, nested composition, partition composition and reuse-at composition controls. This rules out generic claims that Allo does not reason about inter-kernel composition.
- `tests/test_verify.py`: has positive schedule equivalence tests for GEMM, fusion, memory layouts, reordering and composition, **plus negative composition non-equivalence tests**. The `allo.verify(schedule_a,schedule_b)` method in `allo/verify.py` builds `vhls` source and delegates to PAST; it is a distinct equality/semantics verification route that must not be omitted from comparison.

This substantially narrows the potential difference. FPGAI's source-event oracle validates **a finite, typed sequence of individual tensor-element commit/read/last-use and Phi issue obligations**, while Allo's **inspected** test suite emphasizes source/schedule equivalence, static memory layout primitives and use-def composition. **This description compares the particular inspected code paths only.** It is **not evidence that Allo cannot express/check an equivalent last-reader condition through its other transformations or emitted code**.

The next equivalent-feature experiment must distinguish:

| Question | Our bounded implementation | Allo pinned-artifact evidence | Status |
|---|---|---|---|
| Two independently customized MatMul readers of shared producer | R2f/R2h E0 and actual GCC source checks | Allo `compose`, `get_equivalent_variables`, use-def-chain tests | **Not run side-by-side** |
| Source-valid and incorrect final values | Selected GCC/NumPy and mutation controls | `allo.verify` positive and negative equivalence controls via PAST | **Both have relevant mechanisms; matched test missing** |
| Intermediate scratch reuse / memory type | R2c fixed-order source slot packing and `verify_allocation` | `reuse_at`, `partition`, `buffer_at`, typed `Memory` | **Both have relevant mechanisms; definitions differ** |
| Explicit all-reader last-use safety at tensor-element granularity | R2f symbolic source-event checker, R2h actual load/store address witness | No matched equivalent checker evaluated | **UNKNOWN, not absent** |
| Operator-Phi issue-order conformance independent of source-value safety | R2f `verify_fanout_issue` and new independent R3g differential plan | No comparable exact test executed | **UNKNOWN, not absent** |
| Policy-neutral complete-design transition + E0/UNKNOWN/REJECTED mask | R2p facade, R3d/R3f source evidence | No equivalent decision-interface study executed | **UNKNOWN, not absent** |

**Important reviewer warning:** A positive result from our checker and an "unsupported" result from a competitor only support a restricted capability conclusion if we demonstrate the equivalent source design and all attempted legal encoding routes. Prior-art code that fails for an unrepresentable or wrongly translated experiment is NOT evidence of an unsound competitor.

The academic scope remains a **bounded source compiler IR experiment**; do not claim the first compiler to offer type-safe customization, scheduling verification, memory reuse or graph-based optimization.


## R3i — isolated, pinned external source capability probe (NEW CODE, UNEXECUTED)

The repository now contains `src/fpgai_research/experiments/cst_r3i_external_allo_capability.py` and pure opt-in tests. These **do not install, import or run Allo during normal compiler test execution**. The external experiment is intentionally segregated from frozen R3d–R3h source measurements and emits a new create-only result folder.

An isolated Allo environment must point to **exact** checkout `abc270c7a1919c780df2b7d7e99de45e81c175b9` and use that exact source package. The driver checks the original git SHA and clean tracked source. It then independently specifies a 2×2 float32 MatMul producer with two MatMul consumers and compares three source functions using **Allo's own** `allo.verify` / PAST path:

1. `safe` vs itself (positive baseline);
2. `safe` vs a source program reordering the two independent readers (output-equivalence control);
3. `safe` vs a source program writing zeros to the shared H tensor *after the first reader but before the second reader* (negative output mutation).

It keeps each actual verifier response as `EXECUTED_TRUE`, `EXECUTED_FALSE` or `ERROR_NOT_A_NEGATIVE_CAPABILITY_RESULT`. A dependency, codegen or PAST failure is an **ERROR/UNKNOWN**, never evidence that an absent capability has been demonstrated. The test runs Allo's relative `tmp` output inside a separate temporary directory, without editing this research repo.

Read-only protocol inspection (always safe, no external compiler):

```bash
python -m fpgai_research.experiments.cst_r3i_external_allo_capability --protocol
```

After preparing a **separate compatible Allo/PAST virtual environment** and a clean external checkout at the exact SHA, the actual optional probe is:

```bash
python -m fpgai_research.experiments.cst_r3i_external_allo_capability \
  --allo-checkout /absolute/path/to/isolated/allo \
  --output build/r3i_external_allo_capability_v1 \
  --confirm-isolated-external
```

Do **not** run the optional command in an environment where `import allo` resolves to a different checkout; the driver refuses that mismatch. Do **not** install Allo/PAST directly into the validated `fpgai_research` virtual environment solely for this probe.

**Important scientific limitation:** Allo whole-output equivalence is NOT the same property as FPGAI's element-level source event/last-reader contract. An Allo verification failure for the overwritten program is relevant but does not prove Allo has or lacks our typed lifetime certificate. The **full matched capability proof** (representation of an explicit typed last-read/physical alias proposal, identical error semantics and action-mask API) is still outstanding after this probe. The artifact has **NOT** been executed at writing, so all R3i results are UNKNOWN.
