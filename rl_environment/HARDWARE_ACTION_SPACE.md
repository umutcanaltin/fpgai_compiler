# FPGAI RL hardware action space — first implementation sprint

**Branch:** `feature/rl-hardware-knobs-environment-20261104`
**Status (2026-11-04): source implementation committed separately from main; local
source-only pytest and native Vitis tests still pending.**

The user explicitly reopened RL interface development on 4 November 2026,
after the 1 November compiler-paper freeze. This branch works on *source-only
hardware action representation and verification* first. It does **not**
restart historical HLS studies, modify existing shared-GNN weights, relax
native evidence gates, or assert that a new Conv policy has been trained.

## Design rule: real knobs are not imaginary hardware

The canonical compiler has exactly these eight `ArchitectureRequest`
properties. All eight are exposed in a typed, bounded *proposal* catalog,
and all changes are sent through the original architecture resolver,
Model IR refinement and HLS C++ emitter. The catalog is not claimed to
enumerate every legal integer value or satisfy hardware timing.

| Field | Action scope | Notes |
|---|---|---|
| `precision_bits` | Model-global | Supported widths from `TargetLimits.supported_widths`; existing HLS emitter uses one `ap_fixed` type across the graph |
| `integer_bits` | Model-global | Must be within selected precision; cross-operator casts do not exist |
| `parallelism` | Per compute operator | Real compiler request; target max parallel units is a hard source capacity check |
| `pes` | Per compute operator | Real compiler request; combination checked with parallelism × SIMD |
| `simd` | Per compute operator | Real compiler request; generated reduction-loop mapping |
| `serialization` | Per compute operator | Real fold/serialization request, no promise of timing improvement |
| `tile` | Per compute operator | Full-rank tuple: MatMul/Gemm M/N/K; Conv only output channel/height/width may vary, non-output axes stay full |
| `pipeline_ii` | Per compute operator | `None, 1, 2, 4, 8` bounded proposal menu; requested, **not achieved** synthesis II |
| `clock_ns` | Model-global continuous | Canonical 0.001 ns choices within 2–32 ns; native Vitis controls effective timing; **not** measured Fmax |

Currently admitted compute operators: **MatMul, Gemm, Conv**. Supported
scalar operators (for example Relu, Flatten, MaxPool, Softmax) remain
in the functional graph, carry baseline scalar mapping and are not
silently given nonexistent PE/tile actions. Thus a Conv→Pool→Dense graph
can have independent Conv and Dense architecture choices while sharing
a single current numerical format. The original functional model,
weights and tensor dependencies remain immutable across actions.

The inventory explicitly labels the following **not action-ready**
in this backend: BRAM/URAM allocation, memory banking/port counts,
AXI DMA width and bursts, clock domains and CDC, board/part switching,
physical placement/routing strategies and power constraints. Some
related features exist in other compiler layers; they do not have
validated `HardwareEdit -> compiler source -> source-attested Vitis`
actions here. We must implement and validate them before giving them
to a learning policy. Device part is currently a task constraint,
not a step-editable knob.

## Source-backed planning interface

The public interface is `rl_environment.hardware_space`:

- `HardwareState`: complete immutable network requests, fixed part,
  requested clock and compiler target limits, optionally including exactly
  five native resource budgets (LUT, FF, DSP, BRAM18K, URAM) and a numerical
  absolute-error bound. Unknown constraints remain `null` and have explicit
  presence masks; no fabricated device capacity or error threshold.
- `HardwareEdit(scope, field, value, operator_index)`: one typed
  model-global or operator-local change. Invalid edits leave state
  unchanged, with an explicit rejection reason.
- `transition`: *one actual source compile* using
  `resolve_scoped -> compile_architecture`; returns typed source
  SHA and complete state identity when admitted, or a rejection.
- `candidate_actions`: complete bounded, deterministic discrete
  menus for all available categories in the current source domain; the
  agent's continuous-clock head is separate, so no clock grid is implied.
  An overflowing request fails rather than silently truncating options.
- `action_features`: policy-facing typed, variable-length candidate
  descriptors without unknown native costs filled as zeros.
- `action_mask`: finite source-compilability decisions over exact
  proposed actions, not an estimated HLS physical feasibility mask.
- `HardwarePlanningEnvironment.offers`: the compiled mask, typed action
  features, and continuous-clock bounds in one source-only handoff.
- `graph_observation`: variable-size tensor/operator graph,
  iteration shapes, current hardware requests and constraints; no
  raw learned weights, HLS resource labels or prior reward.
- `HardwarePlanningEnvironment`: reset/propose interface with
  no Vitis, reward or synthetic physical feedback. Calling `step`
  raises until an actual independently audited native backend exists.

Every accepted action retains source and exact functional SHA.
A clock edit may leave `model.cpp` identical while changing the
source-bound requested period. We therefore hash complete
`(functional graph, architecture requests, part, period, target limits)`
as a distinct RL source state. A source-legal action is not an
HLS-qualified or E4/board-qualified design.

### Read-only command

After switching to the branch and installing it editable:

~~~bash
python -m pip install --no-deps -e .
python -m pytest -q tests/test_rl_full_hardware_space.py

fpgai-rl-hardware-space \
  --onnx build/demo/cnn_pool_dense.onnx \
  --part xc7z020clg400-1 \
  --clock-ns 10 \
  --max-parallel-units 8
~~~

For a later native RL task, provide exact budgets using
`--resource-caps LUT FF DSP BRAM18K URAM` and an independently chosen
`--max-absolute-error TOLERANCE`. These inputs are **hard constraints**, not
synthetic reward labels. Omitting them records explicit unknowns.

This only reads ONNX and emits JSON to stdout. It creates no cases,
does not invoke Vitis/Vivado, and does not claim reward/learning.
The existing `rl_environment.gym_env`/historical Task, catalog,
reward, study and audit files remain unmodified.

## Next research-development sprints

| Sprint | Scope | Evidence gate |
|---|---|---|
| HW-RL-01 | Full eight-field typed action inventory, bounded per-op menus, source-only masks and variable-size weight-private graph observation | Source regression with MatMul and multioperator Conv CNN; no backend calls; **implemented, local tests pending** |
| HW-RL-02 | Introduce a new original-native Vitis backend for *this exact complete action state* with exact part, clock, source digest, immutable create-only case and independent CSim/CSynth qualification | One small original MatMul and one Conv toy native case, independently audited E2/E3; no stub metrics |
| HW-RL-03 | Gymnasium `Env` with variable-size action features/masks and continuous clock head, invoking the HW-RL-02 real evaluator on each authorized step | `reset` read-only, `step` always a fresh case, consistent 5-resource/numerical/timing receipt, non-stale imports |
| HW-RL-04 | Shared GNN actor for MatMul↔Conv↔CNN transfer with variable graph sizes and category-specific knob masks; preserve one shared policy across episodes | CPU gradients, correct invalid masking, no heldout leakage, no pretrained checkpoint |
| HW-RL-05 | Prospective matched on-line learning comparison vs random and no-GNN controls; original Vitis calls fixed per task/seed | Separate training/validation/test, same budgets, per-network qualified HLS nanoseconds and uncertainty; negative results retained |
| HW-RL-06 | Optional system/board/physical action domains once compiler implements and verifies typed memory/AXI/CDC/board semantics | No fictional DMA or power knob; original physical board verification separately required |

For native qualification, the established semantics remain: feasible
whole-top estimated HLS latency = qualified worst-case cycles ×
requested period. CSim numerical error, five resource caps, and
effective HLS scheduling timing are hard gates. Native call count
is a strict budget. No 100×10 or 100×30 campaign is authorized by this
source-only sprint. All past episode/checkpoint/receipt directories
remain immutable.

## Scientific hypothesis and falsifiers

**RQ-HW-1:** Can one typed compiler-neutral policy observation and legal
action interface cover mixed CNN/dense network topologies and all
*currently source-implemented* architecture knobs without corrupting
model semantics?

This source sprint evaluates representational admission and rejection,
not learned speedups. Native studies must separately test:

1. Whether additional knob categories increase qualified hardware
   architecture diversity at a fixed **real Vitis call budget**.
2. Whether source masks reduce failures that would otherwise reach
   real Vitis; source legality must not be called native feasibility.
3. Whether a single shared graph policy achieves lower heldout
   **HLS-estimated nanoseconds** than matched random/no-GNN baselines,
   with credible variation across networks/seeds.
4. Whether architectural edits affect actual HLS output; if source
   variants synthesize identically, document the equivalence and
   remove ineffective action dimensions rather than claim progress.

Existing Allo/HIDA/StreamTensor/FINN/hls4ml prior art and the current
FPGAI C1/C2 scientific paper must be considered before claiming a
new DSE/IR invention. Do not claim trained transfer, device
programming or measured power on source-only evidence.
