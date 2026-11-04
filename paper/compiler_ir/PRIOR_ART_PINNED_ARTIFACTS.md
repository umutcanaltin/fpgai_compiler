# Pinned prior-art comparison inputs — compiler-paper L8 gate

**Recorded 2026-11-02. Status: verified public upstream SOURCE POINTERS; no external baseline run has yet been reproduced.** This is not an empirical SOTA claim, proof of novelty, or claim that our five-op diamond is supported by every referenced system.

| System | Verified reproducibility pointer | Direct conceptual overlap / baseline obligation | Current status |
|---|---|---|---|
| **ST-Flow** (PACT 2026) | [UCLA-VAST/ST-Flow](https://github.com/UCLA-VAST/ST-Flow), annotated release tag `PACT26` -> commit `b249f322fbfb6e746e7c8a3a09800a2298c79757`, [artifact DOI 10.5281/zenodo.21881993](https://doi.org/10.5281/zenodo.21881993) | Per-kernel spatial/temporal scheduling, reuse-aware resource sharing, generated Vitis C++ and streamed communication. Inspect exact inter-kernel tensor readiness, fanout and source checks; do not assume features are absent | PINNED, NOT EXECUTED |
| **Allo** (PLDI 2024) | [cornell-zhang/allo](https://github.com/cornell-zhang/allo), inspected `main` snapshot `abc270c7a1919c780df2b7d7e99de45e81c175b9` (**not** established as PLDI evaluation commit) | Compose schedule, compute, memory, datatype and communication. Compare conflicts, data ownership and equivalent source lowering; cannot call compositional scheduling novel on its own | SNAPSHOT PINNED, PAPER VERSION NOT RECONCILED |
| **HIDA / ScaleHLS** (ASPLOS 2024) | [UIUC-ChenLab/ScaleHLS-HIDA](https://github.com/UIUC-ChenLab/ScaleHLS-HIDA), inspected `main` snapshot `fd79f6c946593c290d6d285c2a2efcab72fbe9c6` (not necessarily published evaluation version) | Hierarchical dataflow composition, task-local scheduling and graph-level HLS optimization. Audit fanout/dataflow layout and cross-kernel dependence obligations | SNAPSHOT PINNED, PAPER VERSION NOT RECONCILED |
| **StreamTensor** (MICRO 2025) | [official documentation](https://hanchenye.com/streamtensor/) and [paper](https://arxiv.org/abs/2509.13694); no verified runnable evaluated source revision in this pass | Iterative tensor types, stream layout converters, multi-kernel fusion and buffer allocation; its FIFO/token scheduling may overlap our purported availability/reuse contribution | DOCUMENTATION IDENTIFIED, EXECUTABLE ARTIFACT NOT QUALIFIED |
| **Wavelet / Let It Flow** (PLDI 2026) | [plum-umd/wavelet](https://github.com/plum-umd/wavelet), inspected `main` snapshot `9ef3a89aa6b28254820a81e23367c24562eea903`; [archived artifact version 0.1.2](https://doi.org/10.5281/zenodo.19393250) | Formally verified dataflow compilation/capability type system. Our E0/E1 finite certificates are **not** a substitute for its Lean proof | SNAPSHOT PINNED, ARTIFACT NOT EXECUTED |

## Controlled comparison protocol

1. Freeze per-competitor commit, compiler/runtime dependencies, exact model shapes, weights, input sets, numeric formats, HLS target, clock constraints and requested mapping when the system actually supports an equivalent configuration. Record failures and unsupported pairs in the denominator.
2. Compare separately **capability/expressiveness** (whether exact dual MatMul fan-out + independent choices is representable), **source-level legality** (which mutations or incorrect physical source aliases can be rejected and with what evidence), and **hardware QoR** (only if paired original Vitis CSim/CSynth and matching correctness gates succeed).
3. Never treat a compiler's rejection of an unsupported mode as a safety failure; likewise, do not label a competitor's unsupported contract “wrong” or our source scratch reduction “measured BRAM.”
4. Distinguish synthetic E0 mutation ablations against deliberately weak representations from externally reproduced published systems. The former are NOT credible SOTA performance/novelty comparisons.
5. Report one table of exact overlapping capability claims, one table of artifact/source evidence, one paired source/numeric table, and hardware results ONLY if original native runs exist. Full graph/DAG semantics and branch banks/ports remain unresolved in our own compiler.

**Publication gate remains OPEN** until executable upstream comparison artifacts are actually run or a carefully scoped availability/limitations audit is recorded. An inspected Git ref and paper abstract alone establish neither novelty nor superiority.

## Automated Git-pin preflight (not a peer experiment)

Use `python -m fpgai_research.experiments.cst_prior_art_environment
--peer ST-Flow --checkout ../ST-Flow` to inspect the exact expected
revision, clean status and source remote. This invokes **zero
compilers**. Allo/HIDA/Wavelet source pins are snapshots, not
verified publication evaluation revisions. StreamTensor is
`NO_VERIFIED_RUNNABLE_REVISION`. A matching source checkout,
even with a genuine upstream `PACT26` tag, **does not** demonstrate
numeric parity, legal/illegal Phi-conditioned graph contracts,
source event checking, matched tool versions or performance.
Those original executions and peer-adapter programs are still
outstanding. The tool cannot assert firstness or provide a fair
external baseline result.
