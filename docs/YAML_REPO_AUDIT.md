# Repository YAML schema audit

- status: `audit_only`
- scope: `W0-lite repo-level audit of configs/ and examples/ YAML files.`
- files scanned: `56`
- files with deprecated aliases: `10`
- files with unknown/unclassified keys: `0`
- unreadable files: `0`

## Aggregate status counts

- `canonical`: `1405`
- `deprecated_alias`: `30`
- `legacy_or_internal`: `848`
- `paper_artifact_spec`: `34`
- `section_container`: `288`
- `sweep_template`: `1070`

## Files with deprecated aliases

- `configs/examples/default_compile.yml`
- `configs/examples/inference_compile.yml`
- `configs/examples/quick_compile.yml`
- `configs/examples/training_compile_smoke.yml`
- `configs/suite/cnn_avgpool.yml`
- `configs/suite/cnn_mnist.yml`
- `configs/suite/cnn_no_pool.yml`
- `configs/suite/mlp_leakyrelu.yml`
- `configs/suite/mlp_mnist.yml`
- `configs/suite/mlp_sigmoid.yml`

## Files with unknown or unclassified keys

None.

## Migration queue

- `data_movement.ps_pl.weights.mode -> weights.mode` — `10` files
- `data_movement.pl_ps.output -> data_movement.outputs` — `1` files
- `data_movement.pl_ps.output.compression -> data_movement.outputs.compression` — `1` files
- `data_movement.pl_ps.output.compression.enabled -> data_movement.outputs.compression.enabled` — `1` files
- `data_movement.pl_ps.output.mode -> data_movement.outputs.mode` — `1` files
- `data_movement.pl_ps.output.precision -> data_movement.outputs.precision` — `1` files
- `data_movement.pl_ps.output.precision.int_bits -> data_movement.outputs.precision.int_bits` — `1` files
- `data_movement.pl_ps.output.precision.total_bits -> data_movement.outputs.precision.total_bits` — `1` files
- `data_movement.pl_ps.output.precision.type -> data_movement.outputs.precision.type` — `1` files
- `data_movement.pl_ps.output.size_bytes -> data_movement.outputs.size_bytes` — `1` files
- `data_movement.ps_pl.input -> data_movement.inputs` — `1` files
- `data_movement.ps_pl.input.compression -> data_movement.inputs.compression` — `1` files
- `data_movement.ps_pl.input.compression.codec -> data_movement.inputs.compression.codec` — `1` files
- `data_movement.ps_pl.input.compression.enabled -> data_movement.inputs.compression.enabled` — `1` files
- `data_movement.ps_pl.input.mode -> data_movement.inputs.interface` — `1` files
- `data_movement.ps_pl.input.precision -> data_movement.inputs.precision` — `1` files
- `data_movement.ps_pl.input.precision.int_bits -> data_movement.inputs.precision.int_bits` — `1` files
- `data_movement.ps_pl.input.precision.total_bits -> data_movement.inputs.precision.total_bits` — `1` files
- `data_movement.ps_pl.input.precision.type -> data_movement.inputs.precision.type` — `1` files
- `data_movement.ps_pl.input.size_bytes -> data_movement.inputs.size_bytes` — `1` files
- `optimization.parallel_policy -> optimization.parallel.policy` — `1` files

## Per-file findings

### `configs/examples/default_compile.yml`
- statuses: canonical=36, deprecated_alias=2, legacy_or_internal=111, section_container=11
- deprecated aliases: `2`
  - `data_movement.ps_pl.weights.mode` → `weights.mode`
  - `optimization.parallel_policy` → `optimization.parallel.policy`

### `configs/examples/inference_compile.yml`
- statuses: canonical=29, deprecated_alias=1, legacy_or_internal=85, section_container=9
- deprecated aliases: `1`
  - `data_movement.ps_pl.weights.mode` → `weights.mode`

### `configs/examples/quick_compile.yml`
- statuses: canonical=47, deprecated_alias=20, legacy_or_internal=92, section_container=10
- deprecated aliases: `20`
  - `data_movement.pl_ps.output` → `data_movement.outputs`
  - `data_movement.pl_ps.output.compression` → `data_movement.outputs.compression`
  - `data_movement.pl_ps.output.compression.enabled` → `data_movement.outputs.compression.enabled`
  - `data_movement.pl_ps.output.mode` → `data_movement.outputs.mode`
  - `data_movement.pl_ps.output.precision` → `data_movement.outputs.precision`
  - `data_movement.pl_ps.output.precision.int_bits` → `data_movement.outputs.precision.int_bits`
  - `data_movement.pl_ps.output.precision.total_bits` → `data_movement.outputs.precision.total_bits`
  - `data_movement.pl_ps.output.precision.type` → `data_movement.outputs.precision.type`
  - `data_movement.pl_ps.output.size_bytes` → `data_movement.outputs.size_bytes`
  - `data_movement.ps_pl.input` → `data_movement.inputs`
  - `data_movement.ps_pl.input.compression` → `data_movement.inputs.compression`
  - `data_movement.ps_pl.input.compression.codec` → `data_movement.inputs.compression.codec`
  - ... `8` more

### `configs/examples/training_compile_smoke.yml`
- statuses: canonical=29, deprecated_alias=1, legacy_or_internal=103, section_container=9
- deprecated aliases: `1`
  - `data_movement.ps_pl.weights.mode` → `weights.mode`

### `configs/experiments/arxiv_paper.yml`
- statuses: canonical=1, paper_artifact_spec=22
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/suite/cnn_avgpool.yml`
- statuses: canonical=17, deprecated_alias=1, legacy_or_internal=54, section_container=7
- deprecated aliases: `1`
  - `data_movement.ps_pl.weights.mode` → `weights.mode`

### `configs/suite/cnn_mnist.yml`
- statuses: canonical=17, deprecated_alias=1, legacy_or_internal=54, section_container=7
- deprecated aliases: `1`
  - `data_movement.ps_pl.weights.mode` → `weights.mode`

### `configs/suite/cnn_no_pool.yml`
- statuses: canonical=17, deprecated_alias=1, legacy_or_internal=54, section_container=7
- deprecated aliases: `1`
  - `data_movement.ps_pl.weights.mode` → `weights.mode`

### `configs/suite/mlp_leakyrelu.yml`
- statuses: canonical=17, deprecated_alias=1, legacy_or_internal=54, section_container=7
- deprecated aliases: `1`
  - `data_movement.ps_pl.weights.mode` → `weights.mode`

### `configs/suite/mlp_mnist.yml`
- statuses: canonical=17, deprecated_alias=1, legacy_or_internal=54, section_container=7
- deprecated aliases: `1`
  - `data_movement.ps_pl.weights.mode` → `weights.mode`

### `configs/suite/mlp_sigmoid.yml`
- statuses: canonical=17, deprecated_alias=1, legacy_or_internal=54, section_container=7
- deprecated aliases: `1`
  - `data_movement.ps_pl.weights.mode` → `weights.mode`

### `configs/sweeps/communication_ablation.yml`
- statuses: sweep_template=12
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/ddr_memory_strategy.yml`
- statuses: sweep_template=17
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/full_factor_pipeline.yml`
- statuses: sweep_template=70
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/full_factor_pipeline_stratified36.yml`
- statuses: sweep_template=70
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/hardware_knob_validation.yml`
- statuses: sweep_template=127
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/inference_policy.yml`
- statuses: sweep_template=18
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/inference_precision.yml`
- statuses: sweep_template=20
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/memory_binding_strategy.yml`
- statuses: sweep_template=17
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/memory_strategy.yml`
- statuses: sweep_template=20
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/model_suite.yml`
- statuses: sweep_template=9
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/parallelism_feasible_envelope.yml`
- statuses: sweep_template=122
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/parallelism_policy.yml`
- statuses: sweep_template=35
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/pipeline_policy.yml`
- statuses: sweep_template=69
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/pipeline_policy_strength.yml`
- statuses: sweep_template=70
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/precision_selection.yml`
- statuses: sweep_template=21
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/training_accelerator.yml`
- statuses: sweep_template=58
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/training_accumulated_batch.yml`
- statuses: sweep_template=66
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/training_batch_multistep.yml`
- statuses: sweep_template=62
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/training_convergence.yml`
- statuses: sweep_template=11
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/training_multi_epoch_convergence.yml`
- statuses: sweep_template=74
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/training_native_accumulated_batch.yml`
- statuses: sweep_template=66
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `configs/sweeps/vivado_bridge.yml`
- statuses: paper_artifact_spec=12
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/boards/kr260_training.yml`
- statuses: canonical=69, legacy_or_internal=7, section_container=11
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/boards/kv260_inference.yml`
- statuses: canonical=33, legacy_or_internal=7, section_container=9
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/boards/pynq_z2_inference.yml`
- statuses: canonical=33, legacy_or_internal=7, section_container=9
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/build/cpp_only.yml`
- statuses: canonical=33, legacy_or_internal=7, section_container=9
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/build/hls_project.yml`
- statuses: canonical=33, legacy_or_internal=7, section_container=9
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/build/hls_synthesis.yml`
- statuses: canonical=51, legacy_or_internal=7, section_container=11
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/build/vivado_bitstream.yml`
- statuses: canonical=52, legacy_or_internal=7, section_container=11
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/build/vivado_project.yml`
- statuses: canonical=38, legacy_or_internal=7, section_container=9
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/inference/cnn_m_axi_input.yml`
- statuses: canonical=48, legacy_or_internal=7, section_container=10
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/inference/cnn_stream_input.yml`
- statuses: canonical=48, legacy_or_internal=7, section_container=10
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/inference/mnist_mlp_embedded.yml`
- statuses: canonical=48, legacy_or_internal=7, section_container=10
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/inference/mnist_mlp_import_weights.yml`
- statuses: canonical=57, legacy_or_internal=7, section_container=10
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/paper/memory_strategy_sweep.yml`
- statuses: sweep_template=12
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/paper/pipeline_parallel_sweep.yml`
- statuses: sweep_template=12
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/paper/precision_sweep.yml`
- statuses: sweep_template=12
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/reference/full_options_reference.yml`
- statuses: canonical=111, section_container=12
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/training/batch_accumulation.yml`
- statuses: canonical=70, legacy_or_internal=7, section_container=11
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/training/mnist_mlp_cross_entropy.yml`
- statuses: canonical=69, legacy_or_internal=7, section_container=11
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/training/mnist_mlp_training_adam.yml`
- statuses: canonical=78, legacy_or_internal=7, section_container=11
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/training/mnist_mlp_training_momentum.yml`
- statuses: canonical=75, legacy_or_internal=7, section_container=11
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/training/mnist_mlp_training_sgd.yml`
- statuses: canonical=69, legacy_or_internal=7, section_container=11
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/training/tiled_training_axi_stream.yml`
- statuses: canonical=73, legacy_or_internal=7, section_container=11
- no deprecated or unknown/unclassified keys detected by W0-lite.

### `examples/training/tiled_training_m_axi.yml`
- statuses: canonical=73, legacy_or_internal=7, section_container=11
- no deprecated or unknown/unclassified keys detected by W0-lite.

## Most common YAML paths

- `version` — `31` files
- `data_movement` — `30` files
- `model` — `30` files
- `model.format` — `30` files
- `model.path` — `30` files
- `pipeline` — `30` files
- `pipeline.mode` — `30` files
- `pipeline.outputs` — `30` files
- `pipeline.outputs.top_kernel_name` — `30` files
- `project` — `30` files
- `project.clean` — `30` files
- `project.name` — `30` files
- `project.out_dir` — `30` files
- `targets` — `30` files
- `targets.platform` — `30` files
- `targets.platform.board` — `30` files
- `targets.platform.clocks` — `30` files
- `targets.platform.part` — `30` files
- `operators` — `29` files
- `operators.defaults` — `29` files
- `operators.defaults.activation_insert` — `29` files
- `operators.defaults.activation_insert.alpha` — `29` files
- `operators.defaults.activation_insert.except_last` — `29` files
- `operators.defaults.activation_insert.kind` — `29` files
- `operators.supported` — `29` files
