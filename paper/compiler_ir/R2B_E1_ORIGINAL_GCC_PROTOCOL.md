> **POST-EXECUTION UPDATE — November 3, 2026:** The originally
> prospective 24-case E1 GCC study described below has
> **now been executed on the user's workstation** at original
> source SHA `5d449ca595dbdfdf7a837ce862eb5fa8f89ce9ce`:
> 24/24 GCC compilations, executions, independent NumPy
> numerical and typed event checks passed; original
> saved-source trace read-only audit verified 24/24.
> 12/18 layout-scoped contrasts changed actual final
> H-only READ rank/consumer, and 6/18 changed
> last-reader identity. Because each of 9 unique
> graph/Phi interventions had two storage-layout
> controls, the unique counts are **6/9 and 3/9**.
> Original full pytest **1253 passed**.
> This is a bounded SOURCE-ORDER result only,
> not source scratch capacity, real physical
> PE timing, peer novelty or RL learning evidence.
> [Verified original results](R2B_E1_ORIGINAL_RESULTS_20261103.md).
> The new second-level reanalysis code is
> implemented but **not workstation-tested yet**.
>
# Prospective R2b: original GCC-executed, Phi-conditioned CST source schedule

**NEW PROSPECTIVE DESIGN, not the historical R2 source study.**
2026-11-03. Research compiler only. Source tests and 24-case
original executions in this proposal **have not yet run on the
workstation** at the time of commit. No physical/Vitis results
or novelty conclusions are precommitted as findings.

## Original observation motivating the new design

- Historical original fixed serial-source reader scheduling:
  **24 GCC cases, 18 effective Phi reader interventions**,
  18 changed real source C++/event SHA, but **0 changed
  source last-reader ordinals and 0 scratch capacity changes**.
  This is an immutable negative result.
- The new R2b E0 one-rule PE-grouped interleaving
  was original source-pinned at
  **90bf86d666bfeb5fcad56a56734db43cd3dc37f2**;
  user reported **1245 full tests passed,
  12/12 logical certificates, 9/9 planned Phi
  intervention effects on source-issue last-read
  ordinals**, 0 GCC and 0 Vitis calls.
  These are **E0 logical schedule effects only**.

## New E1 scientific target

Under the SAME Functional five-op MatMul diamond,
same weights, input per graph, fixed source issue algorithm,
and storage-layout policy, independently edit only
a reader \`Phi.pes\` and test whether the **actual
generated and executed C++ scratch-array READ order**
changes the per-element final-reader source-event ordinal.

This is distinct from:

- changing a requested \`Phi\` hash/metadata;
- changing the exact scratch-layout policy between
  two otherwise fixed-mapping cases;
- claiming physical RAM capacity savings, RTL
  parallel issue or HLS achieved Fmax;
- claiming that other published compilers cannot
  represent the same condition.

A new, non-historical module
\`codegen/hls/r2b_interleaved_branch.py\` emits
individual genuine \`scratch[index]\` reads,
per-output real C++ reduction arithmetic, intermediate
COMMIT stores, Add and Relu. Each READ witness uses
the **same C++ \`r2b_index\` symbol** as the source
load. E0 exact typed source events must certify
each layout before emission; early use and
pre-last-reader aliasing remain concrete rejects.
This is a **serialized source program** even though
the Phi labels include PEs; no physical PE/SIMD
parallelism is inferred.

\`validation/r2b_interleaved_cpp.py\` runs a
*new original host GCC compilation + execution*
in an isolated temporary directory for each
case. It checks full typed READ/COMMIT/owner
sequence from the **actual C++ output**,
re-verifies bounded source address safety and
per-element actual last-reader ordinals, and
numerically compares model output against the
separate \`validation.reference.evaluate\`
NumPy functional graph evaluator. Both event
and numeric tampering are explicitly tested:
the output can be numerically wrong while
event indices look unchanged. Preserve original
GCC stdout SHA and the complete raw trace.

## Frozen prospective 24-case design

Three source graphs with shapes
\`(m,k,n,q)\` = \`(2,2,3,4)\`,
\`(2,3,5,4)\`, \`(3,3,4,4)\`.
Four \`Phi\` policies: scalar,
reader0-only \`pes=2\`, reader1-only
\`pes=2\`, both readers \`pes=2\`.
Two source scratch allocations, per case:
fully disjoint and globally last-reader
reuse (aliases \`g\` after all shared \`h\`
reads finish). **24 original case sources;
18 same-graph/layout Phi comparisons**
against scalar, 12 controlled storage
pair comparisons. Report all outcomes,
including no-ops, rejected, unknown
and negative effects.

Input and all three constant weight matrices
are deterministic, frozen for a graph;
no MNIST trained weights or changed graph
shape across an intervention. The reference
uses float32 functional semantics and the
new GCC source uses host float; the numeric
tolerances and complete original source
hash are preserved in each receipt.

Source scratch array counts **exclude**
\`acc_a\`, \`acc_b\`, \`add_a\`,
\`add_b\` temporaries. Therefore a smaller
scratch array is **NOT** an audited reduction
in total hardware memory or LUT/FF/BRAM.

## Required workstation procedure

\`\`\`bash
cd ~/Desktop/github_projects/fpgai_research
git pull --ff-only origin main
python -m compileall -q src/fpgai_research tests
python -m pytest -q tests/test_cst_r2b_source_campaign.py
python -m pytest -q

# NEW, create-only original 24-case E1 study. Do not reuse an old folder.
python -m fpgai_research.experiments.cst_r2b_source_campaign \
  --output build/cst_r2b_original_gcc_e1_v1

# Independently reparse the source/trace artifacts without GCC reruns:
python -m fpgai_research.experiments.cst_r2b_source_audit \
  --study build/cst_r2b_original_gcc_e1_v1

cat build/cst_r2b_original_gcc_e1_v1/summary.json
\`\`\`

The runner is create-only; when interrupted or
failing mid-study it preserves original source and a
\`FAILED.json\` receipt. It does not automatically
retry or overwrite an incomplete original case.
Do NOT delete the original folder to force a passing
result: diagnose code, modify a new source revision,
and run a **new** independently named campaign.
No historical serial R2, v6 Vivado or frozen RL
source is modified.

The full original case folder contains
\`source.cpp\`, \`original_case.json\` with
full input source provenance, raw
\`original_gcc_trace.txt\`, and
\`verified_e1.json\`. The 24-case study
contains original \`cases.csv\`,
\`summary.json\`; the read-only audit
recomputes source/trace SHA-256 and observed
per-value final-reader positions.

## Claim gates, once actual results are observed

**If** E1 actual source last-reader positions
change under held-fixed algorithm and same
layout/graph/input, we may claim a **bounded
source-order dependent temporal effect** of
the selected reader mapping under the
prospectively fixed schedule. A source
number is NOT a physical hardware clock
time, source scratch layout savings are
NOT mapping-induced savings unless shown,
and a host E1 success is NOT independent
external-compiler firstness.

If changes are zero or numerical/source
trace checks fail, report the negative
result; never redefine event hashes as
lifetime evidence. For broader scientific
distinctness, continue with genuinely
independent alternative compiler artifact
reproduction, larger general graphs,
mapping-lifetime legality discriminators,
native HLS CSim/CSynth under matched
numeric constraints and clean-DRC physical
qualification. R5 RL learning performance
requires separately trained/controlled
agents, not the existence of a CST
decision interface.


## Stronger causal discriminator: not merely a global event ordinal

The original E0 9/9 mapping intervention result counts the
global index of the last H READ among READ and COMMIT events.
That index may change because another output COMMIT appears
earlier/later even if H's final consumption ordering is
unchanged. Consequently this new E1 campaign **independently
reconstructs** from actual GCC stdout:

1. Global last source-event index for each H tensor element;
2. Last-H-READ rank counting **only H READ operations**;
3. Final reader consumer identity and its output index.

A purported source-executed mapping-conditioned lifetime
effect is admitted only when **(2) or (3)** changes with the
functional graph, input, storage layout and issue-policy
algorithm all fixed. Changes in (1) alone remain explicit
negative evidence. All source per-element witnesses are
SHA-bound and independently recomputed from raw traces
by the read-only source auditor.

The emit/validator code is implemented in GitHub but
has NOT been executed on the user's workstation yet.
The previous serial source E1 result **0/18** is neither
overwritten nor automatically superseded by new E0
logical schedules. Even genuine E1 H-only read rank
differences do NOT imply reduced source scratch capacity:
the R2b source emitter allocates separate reader
accumulators and its whole-tensor scratch offsets are
held fixed for a given layout.
