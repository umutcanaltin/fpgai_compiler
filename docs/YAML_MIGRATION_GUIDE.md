# FPGAI YAML migration guide — W0-lite

This guide documents the first migration direction from legacy/sprint-era YAML keys to the canonical public format.

## Pipeline II

Deprecated:

```yaml
hls:
  pipeline_ii: 2
```

Canonical:

```yaml
optimization:
  pipeline:
    ii: 2
```

## Parallelization

Deprecated:

```yaml
hls:
  unroll_factor: 4
```

Canonical:

```yaml
optimization:
  parallel:
    unroll_factor: 4
```

## Input movement

Deprecated nested style:

```yaml
data_movement:
  inputs:
    import:
      interface: m_axi
      transport: ps_runtime
```

Canonical direct style:

```yaml
data_movement:
  inputs:
    interface: m_axi
    transport: ps_runtime
```

## Output movement

Deprecated nested style:

```yaml
data_movement:
  outputs:
    export:
      interface: axi_stream
      transport: dma
```

Canonical direct style:

```yaml
data_movement:
  outputs:
    interface: axi_stream
    transport: dma
```

## Weight mode

Deprecated:

```yaml
data_movement:
  ps_pl:
    weights:
      mode: embedded
```

Canonical:

```yaml
weights:
  mode: embedded
```

## Training batch size

Deprecated:

```yaml
training:
  batch_size: 4
```

Canonical:

```yaml
training:
  batch:
    size: 4
```

## Gradient accumulation

Deprecated:

```yaml
training:
  gradient_accumulation:
    steps: 2
    mode: average
```

Canonical:

```yaml
training:
  accumulation:
    steps: 2
    mode: average
```

## Optimizer-state storage

Deprecated:

```yaml
training:
  storage:
    optimizer_state: bram
```

Canonical:

```yaml
memory:
  optimizer_state_storage: bram
```


## Memory storage

Deprecated:

```yaml
memory:
  storage:
    weights: bram
    activations: bram
    gradients: bram
    optimizer_state: bram
```

Canonical:

```yaml
memory:
  weight_storage: bram
  activation_storage: bram
  gradient_storage: bram
  optimizer_state_storage: bram
```

## Training execution/storage

Deprecated:

```yaml
training:
  execution:
    batch_size: 4
    epochs: 1
  storage:
    gradients: bram
```

Canonical direction:

```yaml
training:
  batch:
    size: 4
    epochs: 1
memory:
  gradient_storage: bram
```

## Sweep and paper YAML files

Sweep files using roots such as `defaults`, `design_points`, `parameters`, and `materialize_configs` are classified as `sweep_template`, because they materialize compiler YAML instead of being direct compiler configs. Paper aggregation files using roots such as `paper`, `inputs`, `claim_levels`, `limitations`, and `vivado` are classified as `paper_artifact_spec`. They are still audit-visible, but they are not treated as unknown compiler YAML keys.

## Current behavior

W0-lite reports deprecated aliases but keeps them accepted to avoid breaking existing configs immediately.

Later sprints will:

```text
W0  — complete YAML cleanup and migration policy
W   — stricter schema and unknown-key validation
Q0  — production examples based on canonical YAML only
```

## Repository audit before migration

Use `docs/YAML_REPO_AUDIT.md` as the current migration queue. Files listed with deprecated aliases should be migrated first. Files listed with unknown or unclassified keys should be inspected before public examples are created, because an accepted YAML key must either materialize into generated artifacts/reports/runtime or reject clearly.

## W0-lite/Q0 safe migration notes

The repo examples in `configs/examples/` were migrated where safe from old storage and
batch aliases to canonical keys:

- `training.execution.batch_size` → `training.batch.size`
- `training.execution.epochs` → `training.batch.epochs`
- `training.storage.weights` → `memory.weight_storage`
- `training.storage.activations` → `memory.activation_storage`
- `training.storage.gradients` → `memory.gradient_storage`
- `training.storage.optimizer_state` → `memory.optimizer_state_storage`
- `memory.storage.weights` → `memory.weight_storage`
- `memory.storage.activations` → `memory.activation_storage`
- `memory.storage.gradients` → `memory.gradient_storage`
- `memory.storage.optimizer_state` → `memory.optimizer_state_storage`

Backward compatibility remains in the compiler for now. Later W0/W strict cleanup can
warn or reject legacy aliases by mode after configs and examples are fully migrated.

## Batch 2 example migration status

The Q0 example suite uses canonical keys for direct compiler examples. Legacy aliases remain accepted by the compiler for older configs, but new public examples should use:

- `weights.mode` instead of `data_movement.ps_pl.weights.mode`
- `data_movement.inputs.interface` / `transport` instead of nested `inputs.import.*`
- `data_movement.outputs.interface` / `transport` instead of nested `outputs.export.*`
- `memory.weight_storage`, `memory.activation_storage`, `memory.gradient_storage`, and `memory.optimizer_state_storage` instead of `memory.storage.*`
- `training.batch.*` and `training.accumulation.*` instead of older `training.execution.*` or `training.gradient_accumulation.*`
