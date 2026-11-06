# R4a — first-paper row-retained materialization experiment

**6 November 2026: IMPLEMENTED, NOT YET WORKSTATION-EXECUTED.** These are new prospective source studies. No original R2r/R2u/R2v/R2w/R2x receipts or RL branches have been changed.

## Scientific question

Does a certified row-local producer/consumer schedule permit smaller H scratch materialization while retaining every consumer's independently mapped Phi issue order and passing actual GCC intermediate-value checks?

The canonical R2g emits all m*n producer H elements before any readers begin. R4a uses that very same arithmetic source emitter and Phi ordinal decoder but changes the SOURCE loop order:

    for each row:
        produce n complete H elements
        issue all original per-row fanout reads for every reader
        each reader updates its original acc and count arrays
        release the n row-buffer slots after all required reads
    execute the original Add and ReLU tail

The changed source event sequence is checked through the existing R2f readiness, all-consumer last-use, Phi issue-order and exact source slot replay algorithms. A new source execution verifier compiles and runs the generated C++, checks actual source address-variable witnesses and every intermediate/final value against an independent NumPy reference.

This is ROW RETENTION, not zero-buffer forwarding, streaming among real concurrent hardware actors, or a new general FPGA memory allocator. The H buffer shrinks from m*n to n elements; source scratch savings equal (m-1)*n. Full reader accumulator and counter arrays remain present and are included in total mutable array accounting. Vitis and physical BRAM savings are NOT established.

## New code and experiment populations

- src/fpgai_research/codegen/hls/row_retained_fanout_cpp.py — certified source event schedule and canonical R2g C++ transformation
- src/fpgai_research/validation/row_retained_fanout_cpp.py — full generated-source GCC, address and all-intermediate numerical validation
- src/fpgai_research/experiments/cst_r4a_row_retention_study.py — original create-only GCC and native Vitis PACKAGE PREPARATION; does not run Vitis
- tests/test_row_retained_fanout_cpp.py and tests/test_cst_r4a_row_retention_study.py — source/counterexample/GCC/packaging checks

Predeclared profiles (readers, m, k, n, q, mapping):

| Readers | Shape (m,k,n,q) | Phi mapping | Full H cells | Row H cells | Candidate saved H cells |
|---:|---|---|---:|---:|---:|
| 2 | (2,3,3,4) | Scalar | 6 | 3 | 3 |
| 3 | (2,2,4,4) | Mixed | 8 | 4 | 4 |
| 4 | (3,2,3,4) | Mixed | 9 | 3 | 6 |
| 2 | (16,4,16,8) | Mixed | 256 | 16 | 240 |
| 4 | (16,4,16,4) | Mixed | 256 | 16 | 240 |

**Protocol correction before original R4a execution (6 November 2026).** The initial planned 4-reader case used dimensions `(32,4,32,8)` and failed the 16-per-dimension admission bound in both the existing R2f model fixture and source contract. Only the test suite was run; this case never reached native HLS or an original R4a study. The proposed fifth case is now `(16,4,16,4)` with 256 full-H / 16 row-H source cells. We intentionally did **not** widen the established verification domain to 32 for the sake of a favorable size. A future >16 study requires a separately justified domain extension, guards and numerical qualification. Test coverage now checks the entire prospective cohort against the bound.

Quick: first three cases; paper: all five cases. Each has full-H control and row-H intervention, with identical model, numeric input, weights and effective mappings. Both arms use the canonical source packaging layer. Prospective Vitis settings: part xck26-sfvc784-2LV-c, requested clock 12 ns, explicit uncertainty 1.5 ns, numeric CSim absolute tolerance 0.02. These are new prospective settings and are not pooled with old R2u (8 ns) or R2i (32 ns) experiments.

## Source execution steps on the EXISTING checkout only

Run from the existing active research branch in your original fpgai_compiler working directory. Never clone the repo again, remove build folders, or overwrite frozen results.

    cd ~/Desktop/github_projects/fpgai_compiler
    git branch --show-current
    git diff --exit-code
    git diff --cached --exit-code
    git pull --ff-only origin research/first-paper-source-memory-bridge-20261106
    python -m pip install --no-deps -e .

Focused GCC and scientific-source tests:

    python -m pytest -q tests/test_row_retained_fanout_cpp.py tests/test_cst_r4a_row_retention_study.py

Dry plan only, no vendor runs and no files:

    python -m fpgai_research.experiments.cst_r4a_row_retention_study --profile paper

Create-only actual host GCC source experiment, quick first:

    python -m fpgai_research.experiments.cst_r4a_row_retention_study --profile quick --output build/r4a_row_retained_quick_20261106_01

Only if focused tests and quick original source study pass:

    python -m fpgai_research.experiments.cst_r4a_row_retention_study --profile paper --output build/r4a_row_retained_paper_20261106_01

Every case writes both generated source files, both actual GCC stdout traces, independent numerical/source receipts, both canonical Vitis source/Tcl/testbench packages, SHA-256 artifact inventory and an explicit source-only report. No Vitis or Vivado jobs are launched. Failed studies retain FAILED.json and original partial outputs; use a NEW name after a source correction. Do not silently rerun an original create-only folder.

## Guarded real Vitis HLS stage (ONLY after E1 source passes)

The newly implemented src/fpgai_research/experiments/cst_r4b_row_retention_native.py is opt-in and reuses the existing native runner, report parser and strict timing/CSim qualifiers. It independently rechecks the full original R4a source, source-manifest, host receipts, Tcl, testbench and SHA artifact inventory first.

Read-only source audit and DRY RUN of the new, separate R4a quick study (ZERO vendor calls):

    python -m fpgai_research.experiments.cst_r4b_row_retention_native --study build/r4a_row_retained_quick_20261106_01 --audit-native
    python -m fpgai_research.experiments.cst_r4b_row_retention_native --study build/r4a_row_retained_quick_20261106_01 --start-case 0 --max-cases 1

After independently checking source test and audit receipts, a deliberately explicit single matched pair can be executed on the author's licensed Vitis workstation:

    python -m fpgai_research.experiments.cst_r4b_row_retention_native --study build/r4a_row_retained_quick_20261106_01 --start-case 0 --max-cases 1 --execute-native --confirm-isolated-native --vitis-executable /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls

This opt-in command makes two original Vitis calls, one per arm. It refuses any selected arm that has already been invoked or has any partial log/project folder. A process exit 0 does NOT mean that E2 and E3 qualified.

Read-only raw Vitis attribution and complete paired qualification, including absent/failed pairs as UNKNOWN:

    python -m fpgai_research.experiments.cst_r4b_row_retention_native --study build/r4a_row_retained_quick_20261106_01 --audit-native

The full paper profile uses its own separate R4a source root and must not reuse the quick original output. Do not proceed to native execution while new source tests are failing. Preserve failed and partial original vendor attempts.

## Scientific acceptance criteria and limitations

- Both GCC arms must compile and execute, pass all intermediate/final numerical checks, and preserve the originally specified per-reader Phi issue order.
- The row arm must pass the exact unit-slot source alias checker; premature reuse, missing readers, bad issue order or changed actual source accesses must be detected by appropriate independent numerical/source conditions.
- Report TOTAL declared mutable arrays, not scratch alone, and retain any negative outcomes.
- After independent audit of new GCC E1 receipts, plan a NEW source-pinned Vitis CSim/CSynth study with proper E2/E3 timing and numeric gates. Source-package preparation is NOT an HLS result.
- Qualified source-paired BRAM18K, FF, LUT, DSP, worst cycles and clock estimates must be measured. Timing-failed comparisons are UNKNOWN. Later physical BRAM claims require new matched, clean-DRC Vivado evidence.
- External prior-art comparison and true cross-operator Conv2D/Dense contract coverage remain incomplete. Row buffering/fusion and stream scheduling themselves are not novel.
- No trained RL performance, arbitrary dataflow correctness, real hardware cycle concurrency or generalized FIFO/port certificate can be inferred from this bounded experiment.

**Publication decision remains major revision.**
