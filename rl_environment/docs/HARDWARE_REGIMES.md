# Hardware-RL environment experiment regimes

Six scientific settings share **one** compiler-grounded native Gym, real Vitis
HLS backend and independent case audit. Regimes control the source-legal
**task schedule** and the requested-clock **action space**; they are not six
reimplemented simulators and never substitute cached/surrogate rewards.

| Network dimension | Clock dimension | Experiment question |
| --- | --- | --- |
| `same` (one immutable ONNX model) | `fixed` (identical initial clock for all tasks) | Can the optimizer tune one model's architecture at one target? |
| `same` | `task_fixed` (different fixed target per task) | Does one learned policy adapt architectural choices to target timing across episodes of the *same* model? |
| `same` | `search` (learnable 2–32 ns requested clock) | Does clock + architecture co-design help on a known model? |
| `general` (2+ distinct ONNX structural graphs per schedule) | `fixed` | Can the policy transfer among graphs under one timing target? |
| `general` | `task_fixed` | Does one policy adapt to new graphs **and** different fixed target clocks? |
| `general` | `search` | Does shared architecture + clock search generalize across graphs? |

Select the regimes with the existing `fpgai-rl-hardware-campaign` CLI:

```bash
# Source-only previews: no native calls, output dirs or checkpoints.
fpgai-rl-hardware-campaign --tasks build/single_tasks.json \\
  --network-regime same --clock-regime fixed \\
  --method ppo --episodes 1 --calls-per-episode 2

fpgai-rl-hardware-campaign --tasks build/multi_tasks.json \\
  --network-regime general --clock-regime search \\
  --method ppo --episodes 2 --calls-per-episode 2

# Exact native authorization is always separate and opt-in:
# --execute-native --max-total-vitis-calls N --output-root NEW_PATH
```

### Input manifest and validation

Use the existing `fpgai-research/hardware-rl-tasks/v1` JSON manifest with
per-task `id`, `split`, ONNX path, sample path, FPGA part, five resource caps,
maximum absolute error, and `initial_clock_ns` (canonical .01 ns).
For `task_fixed`, duplicate the same model in two task entries with
different initial clocks, or include multiple graph types at distinct clocks.
For `fixed`, **every scheduled task must have the same initial clock**.
For `general`, the scheduled tasks must have at least two distinct ONNX
operator/topology/shape fingerprints; merely changing weights, filenames or
producer metadata is insufficient. A source-only ONNX graph fingerprint is
stored in the original campaign plan; ONNX source file hashes still protect
train/validation/test data leaks. The `auto` network regime is retained for
legacy compatibility only, and does **not** establish generalization.

### Scientific separation of fixed versus search clocks

`fixed` and `task_fixed`: the native Gym permits only the task's original
clock period, disables the clock-only discrete option, uses an equal-bound
clock action space and gives the policy no Beta clock latent. Its actor and
PPO score-function likelihood are **discrete-only**; the auxiliary feasibility
head conditions on a constant neutral clock fraction, so the clock head does
not receive gradients from the fixed-clock task.

`search`: the original architecture-conditioned Beta policy samples a
continuous clock latent over 2–32 ns, canonically rounded to .01 ns for Vitis;
PPO replays the sampled latent. The clock-only action remains valid. All
three modes retain source, numerical, resources, CSim, timing, immutable
native evidence and read-only original-case replay gates. HLS-estimated
latency is not FPGA board throughput or physical timing.

### Paper protocol

For each regime/method/seed, pre-register equal original Vitis budgets, one
FPGA part, numerical inputs, compiler revision, timeout and uncertainty rule.
Use disjoint task IDs/ONNX bytes for train, validation and test; for *structural*
heldout generalization, additionally keep tested structural fingerprints
absent from the training split. Report how many distinct operator graphs and
network identities were actually observed (episodes repeating the same task
do not count as distinct networks). Separately compare trained/frozen policies,
random, no-message-passing, fixed-clock and clock-search under equal budgets.

A source-only plan is not vendor acceptance. The original eight HLS smoke
calls at `f8ffe2d` and the post-refactor two HLS calls at `2f6caef`
remain valid for their **own** pinned source versions only. Any new regime
requires fresh create-only original Vitis runs and independent audits on its
new source revision; do not rewrite those case folders or claim hardware
superiority from tests/smokes.

### Opt-in fixed-clock versus clock-search ablation

Ordinary matched paired comparisons reject different clock action spaces.
To compare `fixed` against `search` (or `task_fixed`) deliberately,
use `--allow-clock-regime-ablation` on the existing
`fpgai-rl-hardware-compare` CLI. Both runs must still be independently
audited frozen heldout campaigns with **identical** ONNX/sample/task manifest,
split, seed, source revision, episode schedule and exact original Vitis budget.
The exported descriptive summary records both clock modes and the explicit
ablation authorization; it does not claim statistical significance. Different
clock action spaces are different treatments, not naturally interchangeable
method labels. This opt-in does not authorize HLS calls or source changes.
