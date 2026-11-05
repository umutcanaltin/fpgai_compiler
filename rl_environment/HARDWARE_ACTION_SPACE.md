# FPGAI RL hardware action space — first implementation sprint

**Branch:** `feature/rl-hardware-knobs-environment-20261104`
**Status (2026-11-04): HW-RL-01 local verification passed: 30 targeted
tests, and 1,982 passed / 21 skipped in the full suite at commit
`4242a55`. The installed console script then exposed a packaging bug:
`pyproject.toml` discovered only `src/`, although
`rl_environment/` lives at repository root. A packaging correction and
independent-process CLI regressions are now on this feature branch but
must be reinstalled and tested locally. No native Vitis experiments or RL
training are included in this fix.**

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
python -m pytest -q   tests/test_rl_full_hardware_space.py   tests/test_rl_installed_cli_packaging.py

# This invokes the installed script, not a pytest-only source import:
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

The package now uses explicit setuptools source mappings for both
`src/fpgai_research` and root `rl_environment`, including its
subpackages. A clean wheel and editable installation should therefore
have the same import paths; test from outside the repository with
`(cd /tmp && fpgai-rl-hardware-space --help)`. No ad-hoc `PYTHONPATH`
fix is required or recommended.

This only reads ONNX and emits JSON to stdout. It creates no cases,
does not invoke Vitis/Vivado, and does not claim reward/learning.
The existing `rl_environment.gym_env`/historical Task, catalog,
reward, study and audit files remain unmodified.

### HW-RL-04 dependency gate (5 November 2026)

The shared mixed-network GNN agent requires PyTorch (the `rl` project
extra), Gymnasium and ONNX. In particular, `pip install --no-deps -e .`
**does not install PyTorch** and the PyTorch-dependent test module is
skipped if the library is absent. This is not evidence of a functioning
agent. The repaired CLI now fails early with an actionable explanation
before reading ONNX, creating a native case or running Vitis.

From the activated workstation environment:

```bash
python -m pip install -e '.[rl,onnx]'
python -c "import torch, gymnasium, onnx; print(torch.__version__)"
python -m pytest -q tests/test_rl_hardware_optional_dependencies.py tests/test_rl_hardware_agent.py tests/test_rl_hardware_gym.py
```

Then rerun the **same source-only** `fpgai-rl-hardware-agent` preview
with the existing ONNX, samples, part, caps and fresh output root. Do
not add `--execute-native` until the installed policy, action masks
and source-only preview all pass. Missing PyTorch cannot be replaced
with a random/fake policy, a guessed HLS reward or an automatic vendor
fallback. GitHub-only source edits do not verify that workstation
PyTorch is installed or the agent's gradient step works.

## HW-RL-02 — source-attested original Vitis HLS evaluation

**Local CPU feedback and patch (2026-11-04):** The user verified the
new native CLI package install; local HW-RL-02 tests found three failures
from an exact Python tuple-versus-JSON array comparison in the stored
`complete_architecture[*].request.tile`. The full CPU suite reported
2,000 passed, 22 skipped and those same three failures. The action plan
now canonicalizes its complete source record to JSON-native data before
saving and auditing, without relaxing strict equality or any E2/E3 gate.
Regression tests cover both default and non-empty Conv tiles and nested
tile tampering. **New patch verification on the workstation is pending.**



**Status:** New source path and CPU-only negative tests are implemented
on this branch, but **the new code and actual native Conv/MatMul HLS have
not yet been run on the user's workstation**. No RL step/reward, learning
curve, or improved FPGA latency is claimed.

The new `rl_environment.hardware_native` path reuses the exact
`compile_state -> write_vitis_package -> run_vitis -> summarize_case ->
audited_hls_timing` workflow already implemented and independently
tested for the existing FPGAI research compiler. It does not use an
estimated/cached reward oracle or modify old frozen Gym studies.

- A task must provide the exact ONNX model, original 1..256 finite
  float32 C-contiguous input samples with shape
  `[N,*model.input_shape]`, explicitly selected FPGA part, positive
  numerical error limit, and five resource budgets in the order
  LUT/FF/DSP/BRAM_18K/URAM.
- The bounded RL design action state can be reconstructed from a typed
  list of source-legal `HardwareEdit` objects in `--edits-json`.
- **Dry run is the default**: no case directory, licensed tool call,
  inferred latency or reward. `--execute-native` must be explicitly
  selected with a fresh `--output` directory.
- Before any output allocation, reject stale installed compiler
  package, missing Vitis binary, noncanonical 2023.2 XML clock, invalid
  numerical sample cohort and unexpected existing case directory.
- Source+sample identity, clock uncertainty, real CSim testbench and
  emitted HLS model source hashes are preserved in immutable source
  and action receipts. The output case is never overwritten.
- Independently re-audit the real Vitis invocation identity, qualified
  E2/CSim numerical metrics, source-attested E3 XML, actual five resource
  estimates and effective HLS scheduling clock. Rejected or absent
  measurements stay `null`, not zeros, surrogate predictions or
  fake synthesis reports.
- Native HLS currently requires the policy-produced requested clock to
  be canonical to **0.01 ns** for tool-report attribution. The source
  planner still accepts **0.001 ns** continuous-valued actions. This
  is a verified-report format limitation, not a finite clock choice grid.
- The task samples are embedded in the generated CSim testbench. Do not
  publish private/proprietary sample arrays or the native case directory
  without appropriate review.
- A qualified result is **only** an HLS-estimated latency:
  `worst_case_cycles * requested_clock_ns`. It is not routed timing,
  real achievable Fmax, physical board inference or a learned reward.

### Local smoke sequence (does not reuse old native cases)

The example model must exist (or create it using
`python examples/make_neural_network_onnx.py cnn_pool_dense
build/demo/cnn_pool_dense.onnx` with a fresh destination).

Prepare a small **reproducible synthetic numerical input cohort**,
not measured hardware data:

~~~bash
python - <<'PY'
from pathlib import Path
import numpy as np
from fpgai_research.frontend.onnx import load_onnx
m = load_onnx("build/demo/cnn_pool_dense.onnx")
shape = m.tensor(m.inputs[0]).shape
rng = np.random.default_rng(123)
p = Path("build/rl_hw02_cnn_samples_seed123.npy")
if p.exists():
    raise FileExistsError(p)
np.save(p, rng.uniform(-0.05, 0.05, size=(2, *shape)).astype(np.float32))
print(p)
PY
~~~

Inspect the source-only task **without** any Vitis call:

~~~bash
fpgai-rl-hardware-native \
  --onnx build/demo/cnn_pool_dense.onnx \
  --samples build/rl_hw02_cnn_samples_seed123.npy \
  --part xc7z020clg400-1 \
  --clock-ns 10.00 \
  --resource-caps 20000 30000 100 80 0 \
  --max-absolute-error 0.1
~~~

**The above example resource values are illustrative only**, not
verified PYNQ-Z2 device capacities; replace them with independently
confirmed task budgets before any source-accepted native qualification.
The dry-run command creates no `--output` artifacts.

After reviewing the source plan, testing the new source-only tests and
verifying actual device limits, optionally execute one *new* native
Vitis run by appending `--output build/rl_hw02_cnn_native_01
--vitis /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls
--execute-native`. Preserve that output even on failure. Independently
replay with `--audit --output build/rl_hw02_cnn_native_01`, with all
original task arguments unchanged, and compare the reports. Native
evaluation will return exit code 3 when unqualified.

**Do not** launch a new 100-episode RL study from HW-RL-02.
HW-RL-03 must first bind the source/action interface to this exact
independently audited native case, then implement the reward-producing
Gymnasium loop.

## HW-RL-02 measured native baseline — 2026-11-04

The user successfully executed an original Vitis HLS 2023.2 CNN experiment
and independently audited the same immutable case, with **no second vendor
call**. This is the first qualified, real-source HLS latency baseline for
the new hardware-space backend. It is **not** a routed FPGA performance
measurement or a pretraining result.

| Native case | Requested clock | Worst cycles | HLS clock estimate | Effective scheduling | Status |
|---|---:|---:|---:|---:|---|
| `build/rl_hw02_cnn_native_01` | 10.00 ns | 690 | 8.462 ns | 7.300 ns | **Infeasible** (−1.162 ns scheduling margin) |
| `build/rl_hw02_cnn_native_14ns_01` | 14.00 ns | 646 | 10.213 ns | 10.220 ns | **Qualified** (+0.007 ns scheduling margin) |

The 14 ns case produced a source-attested HLS-estimated latency of
**646 × 14 ns = 9,044 ns**, CSim max absolute error **0.0003907345**,
LUT **10,238**, FF **3,734**, DSP **45**, BRAM18K **1**, URAM **0**.
All five resource estimates were within the explicitly chosen task
caps. The scheduling margin is **7 picoseconds**, so do not call this
robust physical timing. The 10 ns case's worst-cycle count is diagnostic
only and must not be used as a *qualified* HLS latency.

The native cases remain immutable and are **not** silently imported as
RL training experience. Future learning curves must identify any
baseline seeding or pretraining explicitly. The first Gym episode starts
with no qualified previous best; its first qualifying fresh case receives
the established first-qualified reward, not a reward calculated against
this external 14 ns case.

## HW-RL-03 — original-Vitis Gymnasium integration

**Status:** New `rl_environment.hardware_gym.GymHardwareNativeEnv`
source and CPU-only unit tests added to the isolated RL branch.
**Tests and first actual Gym-native step have not been run locally.**

- Gymnasium-compatible `reset` and `step` return fixed padded
  node/edge tensors for a variable-size typed network, explicit node,
  edge and action masks, field/category-specific action features and
  task budgets/accuracy context. No tensor weight values are exported.
- Each action is a tuple `(source_action_index, requested_clock_ns)`.
  Index zero changes the clock alone; other indices edit one existing
  compiler-supported architecture knob and may change the same
  requested clock simultaneously. The architecture choices originate
  from `candidate_actions` and the original compiler `action_mask`.
- The clock head is **continuous-valued**, not a discrete menu; a
  hardware case is normalized to **0.01 ns** because original Vitis
  HLS 2023.2 clock-report attribution is proven only at that
  resolution. No native Fmax estimate is invented.
- `reset`, graph observation and CLI preview invoke zero vendor
  tools and do not allocate a native case. Each native `step` needs
  explicit `execute_native=True`, makes one *new*
  `episode_xxxxxx/case_xxxx`, and independently re-audits original
  compiled source, original numerical CSim, original resource/timing
  evidence and exact task/sample identity. Cases and original vendor
  receipts are never overwritten. A step-level outcome lives at the
  episode level, separate from original vendor files.
- Source-invalid, masked, repeated or no-op proposals use **zero**
  licensed calls and return no reward. Infeasible but genuinely
  original-audited HLS cases receive `-0.05`. The first qualified
  case gets `-0.005`; subsequent best improvements receive
  `ln(previous_best/current_best)-0.005`, preserving existing
  latency-only `reward.py` semantics. Rejected/unknown timing
  never receives a fabricated qualified latency.
- Source-only and CI tests simulate a vendor response solely to test
  the Gym state machine; **test fixture receipts are not evidence of
  a real HLS run**. A real native Gym case on the user's workstation
  remains a separate acceptance gate.
- No learned shared-GNN agent, checkpoints, 100-episode study or
  physical FPGA metrics are produced by this sprint. The old
  `gym_env.py`, agents, training tasks and paper experiments remain
  untouched.

### Source-only Gym preview and one native step

If Gymnasium is missing, install `gymnasium>=1,<2` in the active
virtual environment. Then reinstall the editable repository to
register the new console entry point. This model and fixed sample file
are the originals from HW-RL-02.

~~~bash
python -m pytest -q tests/test_rl_hardware_gym.py
python -m pip install --no-deps -e .

# Read-only, NO native calls or experiment output allocation:
fpgai-rl-hardware-gym \
  --onnx build/demo/cnn_pool_dense.onnx \
  --samples build/rl_hw02_cnn_samples_seed123.npy \
  --part xc7z020clg400-1 \
  --resource-caps 53200 106400 220 280 0 \
  --max-absolute-error 0.1 \
  --initial-clock-ns 14.00 \
  --output-root build/rl_hw03_cnn_pilot_01
~~~

Use a **new** experiment root for each study. The following is an
explicit one-original-Vitis, no-training smoke test; action index zero
changes only the requested clock from 14.00 to 15.00 ns. This is **not**
an assumption that 15 ns will qualify:

~~~bash
fpgai-rl-hardware-gym \
  --onnx build/demo/cnn_pool_dense.onnx \
  --samples build/rl_hw02_cnn_samples_seed123.npy \
  --part xc7z020clg400-1 \
  --resource-caps 53200 106400 220 280 0 \
  --max-absolute-error 0.1 \
  --initial-clock-ns 14.00 \
  --output-root build/rl_hw03_cnn_pilot_01 \
  --action-index 0 \
  --action-clock-ns 15.00 \
  --vitis /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls \
  --execute-native
~~~

The command will refuse an existing episode/case and will not rerun
an old experiment in-place. Review the generated `original_vitis_audit`
and `step_0000.json` before any multi-episode training.

## HW-RL-04: shared mixed-network GNN actor–critic — source implementation

**Observed acceptance of HW-RL-03 (4 November, user workstation):**
75 focused tests passed, one skipped; complete CPU suite **2,032 passed,
17 skipped**. The read-only native Gym preview displayed **23 nodes,
22 edges, 160 source-accepted actions** from the eight-operator CNN.
One actual original-Vitis Gym step selected clock-only action 0 at
**15.00 ns**, produced independently audited E3 qualified latency
**646 × 15 = 9,690 ns**, original HLS scheduling margin **+0.105 ns**,
numerical error **0.0003907345** and first-qualified reward **−0.005**.
No policy weights were trained in that HW-RL-03 smoke; neither design
has physically routed/board-verified timing.

**New implementation on branch (5 November, workstation tests pending):**

- `hardware_gym` now publishes `action_target_node` (exact operator node
  index or −1 for global edits), plus up to **30 genuinely observed**
  native-trial rows and `history_mask`. The rows expose the original
  action, selected clock, qualified status, real reward, cycles, error,
  HLS clock estimate, resource ratios, source E3 status and best-known
  *qualified* latency; unavailable values retain presence semantics.
  No raw tensor values or unknown Vitis results are invented.
- `hardware_agent.SharedHardwareGraphPolicy` reuses the existing
  directed `gym_agent.MessagePass` primitive with the new mixed
  tensor/operator graph. It combines per-operator node embeddings,
  source-legal action features, task constraints and attention over
  real trial history. The actor selects a masked discrete compiler
  hardware edit; a conditional **Beta continuous clock head** selects
  2–32 ns, normalized to 0.01 ns for original Vitis report attribution.
  A single critic is shared across networks. No per-network policy.
- `HardwareGraphTrainer.update` uses the existing latency-only Gym
  reward, entropy 0.001, gamma 0.95, gradient clip 5, Adam 0.0005 and
  feasibility auxiliary (weight 0.2, positive class weight 8). It
  **requires the matching live Gym step, exact selected action, exact
  original reward and a second independent read-only native-case audit**
  before optimizer updates. It does not run Vitis or accept an
  unaudited result dict as a learning label.
- `fpgai-rl-hardware-agent` previews a deterministic untrained GNN
  action without creating a study or invoking Vitis by default.
  `--execute-native` allows **one** fresh original Vitis case,
  independent audit, and **one** gradient update only; no 100-episode
  campaign is implicitly authorized. Private checkpoints are tensor-
  only NPZ without optimizer state and support **evaluation restore
  only**, not an incorrectly advertised resumable training run.
- New CPU-only tests cover mixed graph sizes, invalid/padded action
  masks, legal target nodes, continuous Beta clock head, GNN gradient,
  trial-history masking, source-data privacy, tamper rejection and
  no-feedback/no-case update refusal. Mocked E3-like dictionaries are
  **fixtures only** and do not demonstrate native hardware performance.

After testing the source locally and ensuring PyTorch is installed:

~~~bash
git pull --ff-only origin feature/rl-hardware-knobs-environment-20261104
python -m pip install --no-deps -e .
python -m pytest -q \
  tests/test_rl_hardware_agent.py \
  tests/test_rl_hardware_gym.py \
  tests/test_rl_installed_cli_packaging.py
python -m pytest -q

# Preview only: no Vitis, no cases, no training.
fpgai-rl-hardware-agent \
  --onnx build/demo/cnn_pool_dense.onnx \
  --samples build/rl_hw02_cnn_samples_seed123.npy \
  --part xc7z020clg400-1 \
  --resource-caps 53200 106400 220 280 0 \
  --max-absolute-error 0.1 \
  --initial-clock-ns 14.00 \
  --seed 101 \
  --output-root build/rl_hw04_cnn_agent_01
~~~

**Only if these tests pass**, an explicit single-step pilot is possible
with the same argument list and `--execute-native --vitis
/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls`. This step may select an
HLS-infeasible action and earn −0.05, which is a legitimate measured
negative example; missing original E3 evidence halts with no reward.
Never reuse `rl_hw03_cnn_pilot_01` or `rl_hw02_cnn_native_14ns_01`.
A single policy update proves nothing about RL generalization. Next,
pre-register a fair multiple-task, multiple-seed original-Vitis
comparison against masked random and no-message-passing ablations, with
equal native-call budgets and independent test cohorts.

## Next research-development sprints

| Sprint | Scope | Evidence gate |
|---|---|---|
| HW-RL-01 | Full eight-field typed action inventory, bounded per-op menus, source-only masks and variable-size weight-private graph observation | Source regression with MatMul and multioperator Conv CNN; no backend calls; **30 targeted and 1,982 full tests passed before packaging fix; installed CLI needs recheck** |
| HW-RL-02 | Original-native Vitis backend for exact source state, samples, part, clock, immutable cases and independent CSim/E3 gate | **CPU tests passed; native CNN 10 ns infeasible and 14 ns qualified (9,044 ns HLS estimate); other model families pending** |
| HW-RL-03 | Gymnasium native-step environment with bounded padded variable-sized graph, source action masks and continuous clock output, original Vitis per authorized step | **Source/test implementation added; local CPU and first real Gym-native step pending** |
| HW-RL-04 | Shared mixed-network graph actor–critic with exact per-operator node targets, real-trial history attention and continuous-clock Beta head | **Source + CPU test implementation added; local tests and first actual one-step online gradient pending** |
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
