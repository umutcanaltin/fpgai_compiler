> **Later prospective R2b executed-GCC outcome (2026-11-03):**
> The originally negative 0/18 fixed serial-source
> experiment remains unchanged. The NEW, separately
> source-pinned, fixed Phi-aware alternate reader
> issue policy completed 24/24 original GCC C++
> cases and independent source/numeric auditing,
> observing changed actual final H-only READ
> rank/consumer in **6 of 9 unique graph/Phi
> interventions** (12/18 layout-scoped),
> and changed last-reader identity in **3/9**
> (6/18 layout-scoped). This demonstrates a
> **bounded mapping-conditioned source order
> effect** under the new policy, NOT a
> Phi-dependent scratch/BRAM capacity effect,
> HLS timing or external firstness. See
> [E1 original results](R2B_E1_ORIGINAL_RESULTS_20261103.md).
>
# CST Model IR — original R2 negative finding and prospective schedule-causality experiment

**Status: originally observed negative finding; NEW follow-on experiment is
NOT implemented, executed, or a publication result.** Research CST Model
IR only, November 3, 2026. See
[three candidate contributions](PAPER_CONTRIBUTIONS_AND_RL_INTERFACE.md).

## Original scientific finding (immutable)

On the workstation, the source-pinned
`build/cst_c123_paper_20261103` CPU/GCC paper series at
`1cefdd8566658cb6b69e12444c9b5ce9ba6e4a31`
completed 40/40 stages. Its controlled two-reader
`two_phi_readers_e1` ran **24 original GCC-source cases**
(three shapes × four reader mappings × two scratch layouts).
Secondary analysis
`two_phi_counterfactual_e1/summary.json`, original CSV SHA
`d22219c4e6afbde1ab784a7e9de29cf876b2873f60ed8125dea009f2f7780898`,
reported **18/18 effective Phi interventions**, 18/18 source C++
hash differences, 18/18 source-event SHA differences, but
**0/18 last-use hash differences and 0/18 source scratch-capacity
differences**. All 18 outcomes remain in the denominator.
The output explicitly says:
`lifetime_discrimination_established_by_this_study=false`;
`mapping_conditional_capacity_difference_established=false`;
`novelty_relative_to_prior_art_established=false`.

This is evidence that a particular bounded *Phi-only*
intervention did **not** change final source-level lifetime or
local C++ scratch capacity under the chosen sequential stage-order
lowering. It does **not** demonstrate that CST can never
produce such a change in a different legal execution schedule,
nor establish scientific priority.

## Why this test was negative

Inspect the actual implementation, not its Phi hash:

- `model_ir/branch_matmul_contract.py`:
  `branch_matmul_source_events` produces all of the producer
  stage before reader0, then reader1, then Add and ReLU.
  Each reader's `mapped_matmul_order` changes its internal
  operation/owner ordering, but the **inter-stage source
  ordering remains fixed**.
- `codegen/hls/branch_matmul.py`:
  `emit_branch_matmul_hls` emits the three stage loops
  sequentially, with no actual interleaving, DATAFLOW or
  reader-specific channel scheduling.
- `disjoint_branch_offsets`:
  the declared scratch addresses depend on source-order
  liveness and disjoint-versus-reuse policy, not an
  optimizer-calculated `Phi`-dependent final reader release.
  The last-read event ordinal can remain unchanged even
  if the owner coordinate sequence differs.

The negative original result thus exposes a limitation
of **the current scheduling/lowering policy**, not
a numerical compiler failure. It must not be repaired by
altering historical original CSV, replacing original source,
silently changing its denominator or redefining "last use"
to mean a changed owner hash.

## Prospective R2b: a test capable of falsifying the stronger thesis

**Research question:** under a *single explicitly specified*
topological scheduler and fixed Functional model, numeric policy,
memory budget and original source semantics, does a legal change
in one consumer's architectural `Phi` change actual earliest
read/last-read constraints, physical source slot reuse or
the set of compiler-accepted interleavings?

### Minimal prespecified graph

`X @ W0 -> H` feeds two independently mapped MatMul readers,
`H @ W1 -> A` and `H @ W2 -> B`; then Add -> ReLU.
Include odd row/width shapes, at least 3 distinct widths,
and matched separate single-reader and two-reader controls.

### Exact comparison arms

1. **S0 serial control:** reproduce original `reader0-then-reader1`
   stage-order execution. Expect owner/source-order differences
   but no guaranteed last-read or scratch-capacity effect;
   negative results are retained.
2. **S1 actual mapped interleaving:** implement a deterministic
   scheduler that derives individual ready/read operations
   from producer and both readers' `Phi` and their exact
   reduction-completion requirements. Use actual source
   snippets/accesses in emitted C++, NOT only an abstract
   event-list schedule. The entire schedule generator
   must be held fixed between interventions.
3. **S2 constrained storage:** apply the same allocator and
   memory budget to each S1 schedule, with an independent
   address-safety reference checker and exhaustive small
   oracle where tractable. Last-use and slot deltas must be
   measured on original, host-executed READ/COMMIT traces.
4. **Negative/mutation controls:** unchanged `Phi`, owner-hash
   relabeling without changed source access order, fixed
   serial schedule with changed `Phi`, premature release,
   missing second-reader read, incomplete reduction
   contribution and illegal address alias. Require exact
   tensor/element/consumer/address counterexamples.

### Required measurements per original source case

Retain all cases, including rejects, no-ops and unknowns:
full Functional SHA; original weights/data SHA; independent
reader `Phi` projections; source C++ and numerical oracle
SHA; SOURCE event sequence with actual element indices
and COMMIT/READ/RELEASE time in **host source order**
(not hardware cycles); last-read ordinal by (tensor,element)
plus consumer and output; source scratch peak allocated
elements; source self-consistency; independent NumPy
output; CPU verifier elapsed time; optionally original
Vitis CSim/CSynth under separately source-pinned matched
fixed-point conditions. Do not substitute synthetic
manifest traces for actual emitted source reads.

**Firstness proof remains separate:** an internal weak
baseline is useful for isolation but is *not*
external-peer state-of-the-art parity. ST-Flow, Allo,
HIDA, StreamTensor and other available artifacts must be
independently investigated for equivalent features.

### Formal accept/reject gates

- **R2b FAILS or INCONCLUSIVE** if only owner/source hashes
  differ, if all new host source read-last-use/capacity
  effects are zero, or if one numeric/trace check fails
  in a claimed accepted case.
- **R2b internal evidence** requires source-executed
  nonzero mapping-conditioned change under a prespecified
  held-fixed scheduling algorithm, a correct independent
  result oracle, and precise counterexample discrimination.
- **Novelty over prior work** stays UNKNOWN even after
  R2b succeeds unless source-pinned external published
  comparator capabilities are matched.
- **Native QoR** remains UNKNOWN until independent
  original HLS and physical matched pairs; the existing
  warning-only v6 route is merely exploratory.

No result for the unimplemented follow-on S1–S2 must
appear in tables, conclusions or software-ready status.


## Implemented first prospective R2b step: new bounded E0 issue-order contract

At a NEW source revision *after* all historical R2/Vivado
observations, research code adds:

- `model_ir/interleaved_branch_schedule.py`: an immutable
  producer-complete, alternated two-reader logical schedule
  based on one shared work-conserving **PE-group** issue rule.
  Under this explicitly fixed rule, PE count changes
  the actual planned order of individual reader-value
  access; the original serial emitter did not.
- `experiments/cst_r2b_interleaved_schedule.py`:
  prespecified three 5-op MatMul diamonds ×
  scalar/reader0-PE2/reader1-PE2/both-PE2,
  including unchanged controls, 12 original
  E0 logical case certificates and finite
  independent per-element last-read ordinals.
- `tests/test_cst_r2b_interleaved_schedule.py`:
  confirms valid full original/reader event
  multiplicity, controlled logical Phi-induced
  last-read changes, precise premature-read and
  unsafe-alias counterexamples, and unsupported
  tile/fold policies rejected rather than guessed.

**CRITICAL status:** This is *only an E0 speculative
source-order scheduling prototype*. It is NOT an
executed source schedule (E1), no GCC/C++ has been
generated by this new module, no independent
numerical oracle for it has run, and no Vivado
or Vitis measurements exist. It MUST NOT be
confused with the 18-case original E1 negative
result, and MUST NOT be cited as qualifying the
strong mapping-conditional real-source
lifetime thesis. The new code/tests have been
pushed but not yet verified on the user's workstation.

To start the first original R2b E0 pilot, in a
clean updated checkout:

```bash
python -m pytest -q tests/test_cst_r2b_interleaved_schedule.py
python -m fpgai_research.experiments.cst_r2b_interleaved_schedule \
  --output build/cst_r2b_logical_e0_v1
```

Do not include its `logical_h_last_use_changed_interventions`
in an E1 actual-source causal-effect table.
**Next implementation required before a strong
R2b paper claim:** emit the exact E0 schedule
as independently instrumented C++ source,
run original GCC, verify actual source
READ/COMMIT/RELEASE and address alias
events versus an independent NumPy oracle,
and validate per-element release changes
only on the actual executed host trace.
Only then test native HLS as a separately
pinned follow-on and external peer parity.
