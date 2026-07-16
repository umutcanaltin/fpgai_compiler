# FPGAI Feature Matrix

This document defines what FPGAI features are allowed to claim publicly.

A feature is `supported` only when all of these are true:

1. A YAML knob exists.
2. The compiler reads the knob.
3. Generated HLS/C++ artifacts materially change.
4. The compile manifest or report records the applied setting.
5. A regression test proves the generated artifact changed.

## Status labels

| Status | Meaning |
|---|---|
| `supported` | Implemented, materialized in generated artifacts, recorded, and tested. |
| `supported_estimate_only` | Implemented as a pre-HLS analytical estimate, but not a measured HLS/Vivado result by itself. |
| `partial` | Some implementation exists, but at least one required proof point is missing. |
| `experimental` | Research/development path exists, but it is not a stable user-facing feature. |
| `not_yet_supported` | Public docs/configs should not claim support. |

## Core compiler features

| Feature | Required public meaning | Inference scope | Training scope | Current status | Required work before public support |
|---|---|---|---|---|---|
| YAML-first inspect/compile | User can inspect and compile from YAML through `fpgai` CLI. | supported | supported | supported | Keep tests and docs current. |
| Precision selection | YAML precision changes generated HLS/C++ data types and reports the applied precision. | supported | supported | supported | Keep materialization and generated-type regression tests current. |
| Memory/storage selection | YAML weight delivery and storage choices change generated HLS interfaces and parameter storage pragmas. | supported | supported | supported | Keep generated-interface and BIND_STORAGE regression tests current. |
| Tiling selection | YAML tiling choices change generated HLS dense/conv tiled helpers, tile template arguments, training backward/update call-site materialization, and architecture metadata. | supported | supported | supported | Training dense/conv forward and backward/update tiling are materialized in generated HLS call sites and metadata. |
| Pipeline selection | YAML pipeline style/II changes planner metadata and generated HLS layer call pipeline-II arguments. | supported | supported | supported | Keep planner-to-codegen pipeline regression tests current for inference and training. |
| Parallelization selection | YAML PE/SIMD/unroll/partition knobs change planner metadata, generated HLS call arguments, and array partition mode pragmas. | supported | supported | supported | Keep planner-to-codegen parallel regression tests current for inference and training. |
| Resource estimation | Compile/inspect emits pre-HLS LUT/FF/DSP/BRAM estimates. | supported_estimate_only | supported_estimate_only | supported_estimate_only | Keep clearly labeled as estimate unless compared to HLS/Vivado. |
| Timing estimation | Compile/inspect emits pre-HLS cycle/latency/timing estimates. | supported_estimate_only | supported_estimate_only | supported_estimate_only | Keep clearly labeled as estimate unless compared to HLS/Vivado. |
| Design-space exploration | DSE evaluates configured candidates and emits estimate-based recommendations with compile-ready/materialized-knob metadata. | supported_estimate_only | supported_estimate_only | supported_estimate_only | No exhaustive search is claimed; recommendations are pre-HLS estimates from configured candidates only. |
| HLS generation | Compiler emits HLS C++/headers/testbench/project artifacts. | supported | supported | supported | Keep generated artifact tests. |
| HLS run/artifact collection | When enabled and tools exist, compiler runs/collects HLS logs/reports. | supported | supported | supported | Keep optional-tool boundary documented. |
| Vivado bridge generation | Board-aware Vivado bridge scripts are generated for supported boards. | supported | supported | supported | Keep Vivado implementation clearly separate from compile. |
| Runtime package | Compile emits runtime package metadata and copies existing runtime-facing files. | supported | supported | supported | Keep hardware presence flags truthful. |
| Inference correctness benchmark | Supported inference flows compare outputs against reference/ONNX Runtime. | supported | not_applicable | supported | Keep benchmark limitations documented; training validation is separate. |
| Training code generation | Training configs generate HLS artifacts for forward, loss, gradient, update, optimizer, and training-specific data movement paths. | not_applicable | supported | supported | This is a generated-artifact/codegen claim, not a physical-board convergence claim. |
| Training single-step reference reports | Compile reports can record CPU/reference single-step training summaries such as loss_before, loss_after, and training estimate metadata. | not_applicable | supported | supported | This covers report generation and reference-summary traceability; numerical tolerance scope must remain in tests. |
| Deterministic multi-epoch training schedule | One canonical epoch/batch/order contract drives the HLS CSim testbench, float reference, fixed-point reference, execution counts, curves, and checkpoints. | not_applicable | supported | supported | This is a schedule/artifact claim. Real-board training and convergence remain separate. |
| Training multi-step convergence | Multi-step training/convergence sweeps and loss-curve artifacts exist for research validation. | not_applicable | experimental | experimental | Do not claim stable convergence/generalization until reproducible task metrics and real FPGA runs are validated and summarized as paper artifacts. |
| Training on-board runtime | Physical FPGA training runs and board-measured loss/timing artifacts. | not_applicable | experimental | experimental | Not stable until board runtime artifacts record bitstream, board, dataset, loss/timing, and reference comparison. |
| Communication optimization | Compiler models input/weight/output/aux tensor-edge data movement, per-edge precision/compression, transfer estimates, and generated HLS communication annotations. | supported | supported | supported | Compression codecs are modeled unless implemented_in_hls=true; measured board DMA speedup remains outside this claim. |

## Implementation order

1. Precision selection
2. Memory/storage selection
3. Tiling selection
4. Pipeline selection
5. Parallelization selection
6. DSE knob truth
7. Training code generation
8. Training single-step reference reports
9. Training multi-step convergence
10. Training on-board runtime
11. Communication optimization

---

# P2F Detailed YAML Implementation Matrix

This section is the repo-owned implementation-status matrix for YAML-controlled FPGAI behavior. It refines the feature-level table above into concrete YAML paths, owner files, artifact effects, report effects, validation coverage, and next actions.

The matrix follows the FPGAI product rule:

```text
user YAML decision
→ compiler/planner decision
→ generated HLS/Vivado/runtime artifact effect
→ report/manifest traceability
→ validation result
```

A YAML option must not be presented as fully supported unless it has a real artifact effect or a clear rejection/reporting behavior. Planner-only or report-only behavior must stay explicitly labeled.

## P2F status labels

| Status | Meaning |
|---|---|
| `implemented` | YAML is read, compiler behavior changes, generated artifacts/reports reflect the decision, and tests or real runs validate the path. |
| `partially_implemented` | Some implementation exists, but at least one required artifact/report/test point is incomplete. |
| `planning_only` | Compiler/planner records the decision, but generated artifacts do not yet materially change. |
| `report_only` | Reports mention the setting, but the compiler does not yet use it to change artifacts. |
| `unsupported` | The setting should be rejected or clearly marked unsupported. |
| `deprecated` | Accepted only for compatibility and should point to a canonical replacement. |
| `duplicate_or_legacy` | Multiple paths exist for similar behavior; future cleanup should canonicalize them. |
| `needs_validation` | Implementation appears present, but a test or real artifact/run must still prove it. |
| `needs_inspection` | Owner paths exist, but exact support level has not yet been audited. |

## P2F non-duplication rule

Future work must change, refactor, or complete the existing owner implementation when one exists. Do not add duplicate compiler paths, detached code generators, detached report systems, or one-off scripts.

Before each implementation sprint:

1. Inspect the current owner files.
2. Explain the mechanism and artifact effect.
3. Explain the reporting effect.
4. Explain the validation/tests.
5. Ask for approval.
6. Patch the existing implementation only after approval.

## 1. Project, output, and pipeline mode

| YAML path | Accepted values / example | Owner files | Compiler/planner effect | Artifact effect | Report effect | Validation coverage | Status | Next action |
|---|---|---|---|---|---|---|---|---|
| `project.name` | string | `fpgai/config/loader.py`, `fpgai/engine/compiler.py` | Names output/project metadata. | Output directory and manifest metadata use the name. | Manifest/result summary include project/output paths. | General compile tests. | `implemented` | Keep stable. |
| `project.out_dir` | path | `fpgai/config/loader.py`, `fpgai/engine/compiler.py` | Selects output root. | Generated C++/HLS/reports/runtime files are written under this directory. | Result summary reports output path. | General compile tests. | `implemented` | Keep stable. |
| `pipeline.mode` | `inference`, `training_on_device` | `fpgai/engine/compiler.py`, `fpgai/engine/training.py`, `fpgai/engine/result.py` | Selects inference or training compile path. | Selects inference or training generated artifacts/testbench/reporting paths. | Manifest/result summary reports pipeline mode. | Training/inference compile tests. | `implemented` | Keep training/runtime boundaries explicit. |

## 2. Build stages

| YAML path | Accepted values / example | Owner files | Compiler/planner effect | Artifact effect | Report effect | Validation coverage | Status | Next action |
|---|---|---|---|---|---|---|---|---|
| `build.stages.cpp` | boolean | `fpgai/engine/build_stages.py`, `fpgai/engine/compiler.py` | Enables generated HLS C++ source. | Emits generated C++/headers. | Pipeline stage summary records `generate_cpp`. | `tests/test_build_stages.py` | `implemented` | Keep as canonical. |
| `build.stages.host_cpp` | boolean | `fpgai/engine/build_stages.py`, `fpgai/engine/compiler.py`, `fpgai/runtime/hostcpp.py` | Enables host C++ generation. | Emits host C++ artifacts. | Result summary reports host C++ dir. | `tests/test_build_stages.py` | `implemented` | Keep as canonical. |
| `build.stages.testbench` | boolean | `fpgai/engine/build_stages.py`, `fpgai/backends/hls/testbench.py`, `fpgai/backends/hls/testbench_train.py` | Enables testbench artifact generation. | Emits inference/training testbench C++. | HLS artifacts and generated file metadata expose testbench when present. | HLS/testbench tests. | `implemented` | Expand later in model-behavior sprints. |
| `build.stages.hls_project` | boolean | `fpgai/engine/build_stages.py`, `fpgai/engine/compiler.py` | Enables HLS project generation. | Emits HLS project/TCL/source layout. | Pipeline stage summary records `generate_hls_project`. | `tests/test_build_stages.py` | `implemented` | Keep stable. |
| `build.stages.hls_synthesis` | boolean | `fpgai/engine/build_stages.py`, `fpgai/engine/compiler.py` | Runs HLS when toolchain is available. | Creates HLS logs/reports/csynth artifacts. | Result summary and manifest report HLS status. | Optional-tool tests and live runs. | `implemented` | Keep optional-tool boundary clear. |
| `build.stages.vivado_project` | boolean | `fpgai/engine/build_stages.py`, `fpgai/engine/compiler.py`, `fpgai/backends/vivado/run_bridge.py` | Requests Vivado bridge/project flow. | Generates/runs Vivado bridge project artifacts when enabled. | Vivado bridge summary and pipeline stages. | Vivado bridge tests and live P2E2 validation. | `implemented` | Keep YAML-driven flow. |
| `build.stages.vivado_implementation` | boolean | `fpgai/engine/build_stages.py`, `fpgai/backends/vivado/run_bridge.py` | Requests Vivado implementation. | Produces implementation reports/artifacts when tools pass. | `vivado_implementation_report.json`, result summary. | Live compact/DDR P2E2 validation. | `implemented` | Extend later with deeper timing/power analysis. |
| `build.stages.bitstream` | boolean | `fpgai/engine/build_stages.py`, `fpgai/backends/vivado/run_bridge.py` | Requests bitstream/XSA generation. | Produces `.bit`, `.hwh`, `.xsa` when tool flow passes. | `bitstream_report.json`, runtime package flags. | Live compact/DDR P2E2 validation. | `implemented` | Keep artifact detection honest. |
| `build.stages.runtime_package` | boolean | `fpgai/engine/build_stages.py`, `fpgai/runtime/package.py` | Requests runtime package creation. | Creates runtime package manifest and copies available deployable artifacts. | Result summary reports package/deployability flags. | `tests/test_runtime_package.py`; live P2E2 package validation. | `implemented` | Real board run remains separate. |

## 3. Toolchain, Vivado, bitstream, and board fit

| YAML path | Accepted values / example | Owner files | Compiler/planner effect | Artifact effect | Report effect | Validation coverage | Status | Next action |
|---|---|---|---|---|---|---|---|---|
| `toolchain.vitis_hls.executable` | path/name | `fpgai/config/loader.py`, `fpgai/engine/compiler.py` | Selects HLS executable. | HLS run uses selected executable when enabled. | HLS logs/status. | Toolchain manifest tests. | `implemented` | Keep optional-tool handling. |
| `toolchain.vivado.executable` | path/name | `fpgai/backends/vivado/run_bridge.py`, `fpgai/engine/compiler.py` | Selects Vivado executable for bridge. | Vivado bridge runs selected tool. | Vivado bridge reports include tool invocation metadata. | Vivado bridge tests. | `implemented` | Keep path validation/reporting clear. |
| `toolchain.vivado.settings64` | path | `fpgai/backends/vivado/run_bridge.py` | Sources Xilinx environment before Vivado. | Enables real Vivado runs in configured shell. | Vivado bridge report records settings path. | Live P2E/P2E2 runs. | `implemented` | Keep explicit in reports. |
| `targets.platform.board` | `kv260`, `kr260`, `pynq_z2`, etc. | `fpgai/backends/vivado/boards.py`, `fpgai/engine/compiler.py` | Selects board/part/PS type and capacity profile. | Vivado bridge project targets board-specific part/BD settings. | Board fit and Vivado bridge summary. | `tests/test_vivado_bridge_boards.py` | `implemented` | Extend only through board registry. |
| `targets.platform.fit_policy` | `report_only`, `warn`, `enforce`, `block_over_limit` | `fpgai/config/loader.py`, `fpgai/engine/compiler.py`, `fpgai/engine/result.py` | Resolves board-fit policy. | Can block Vivado/bitstream when over limit and policy enforces. | Fit policy gate reports policy/source/request. | Fit-policy tests. | `implemented` | Keep priority over legacy paths. |
| `hardware.fit_policy` | same as above | Same as above | Legacy/alternate policy source. | Same as above. | Fit policy gate reports source. | Fit-policy tests. | `duplicate_or_legacy` | Keep accepted until canonical cleanup. |
| `build.fit_policy` | same as above | Same as above | Build-stage level fit gate source. | Can block Vivado/bitstream when over limit and policy enforces. | Fit policy gate reports `policy_source: build.fit_policy`. | Live compact/DDR P2E2 validation. | `implemented` | Keep as supported if documented as canonical for build gating. |

## 4. Memory storage

| YAML path | Accepted values / example | Owner files | Compiler/planner effect | Artifact effect | Report effect | Validation coverage | Status | Next action |
|---|---|---|---|---|---|---|---|---|
| `memory.weight_storage` | `embedded`, `bram`, `uram`, `ddr`, `ddr_preload`, `ddr_tiled` | `fpgai/engine/memory.py`, `fpgai/engine/planner.py`, `fpgai/reports/memory_semantics.py`, HLS emitters | Resolves weight storage strategy. | Changes constants/interfaces/storage pragmas depending on mode. | Memory semantics and hardware knob reports. | Focused memory tests: `114 passed`. | `partially_implemented` | DDR-tiled generated-HLS/testbench/runtime/classifier coverage exists; next action is selected compile-level HLS/Vivado validation. |
| `memory.storage.weights` | legacy/canonical alias candidate | `fpgai/engine/planner.py` | Fallback/alternate weight storage source. | Same target behavior as weight storage when wired. | Reports may expose resolved storage. | Needs exact alias tests. | `duplicate_or_legacy` | Canonicalize against `memory.weight_storage`. |
| `memory.activation_storage` | `bram`, `uram`, `ddr`, etc. | `fpgai/engine/memory.py`, `fpgai/engine/training.py`, HLS emitters | Resolves activation storage preference. | May affect buffers/pragmas depending on path. | Training/memory reports. | Memory/training tests. | `partially_implemented` | Prove generated artifact effect per value. |
| `memory.gradient_storage` | storage region | `fpgai/engine/training.py` | Resolves gradient storage for training. | Should affect training buffers/export paths. | Training plan/report. | Training memory tests. | `partially_implemented` | Verify generated training HLS effect. |
| `memory.optimizer_state_storage` | storage region | `fpgai/engine/training.py` | Resolves optimizer-state storage. | Should affect optimizer state buffers/export paths. | Training plan/report. | Training memory tests. | `partially_implemented` | Verify optimizer state artifact effect. |

### Required memory terminology

Use precise terms in reports and docs:

| Term | Required meaning | Current support status |
|---|---|---|
| `embedded_weights` | Weights compiled as constants; no runtime weight payload. | `implemented` for small fixed models. |
| `bram_weights` | Local on-chip BRAM-backed weight buffers. | `partially_implemented`; verify source/pragmas per design. |
| `uram_weights` | Local on-chip URAM-backed weight buffers. | `partially_implemented`; verify source/pragmas per design. |
| `ddr_preload_weights` | Full external weight payload loaded into local W/B arrays before compute. | `implemented/needs_validation`; not scalable DDR-resident execution. |
| `ddr_tiled_weights` | Only tiles are loaded from DDR during compute; no full local model replica. | `partially_implemented`: structural generated-HLS support exists for Dense/Conv inference, runtime package payload handling, testbench handling, and memory semantics classification; selected compile-level HLS/Vivado validation is still required. |
| `mutable_training_weights` | Training weights can be updated/exported/imported according to training mode. | `partially_implemented`; validate per optimizer/storage mode. |

## 5. Data movement

| YAML path | Accepted values / example | Owner files | Compiler/planner effect | Artifact effect | Report effect | Validation coverage | Status | Next action |
|---|---|---|---|---|---|---|---|---|
| `data_movement.input.load` | interface/transport/tile config | `fpgai/engine/communication.py`, `fpgai/reporting/data_movement.py`, `fpgai/engine/compiler.py` | Resolves input tensor edge. | Should affect HLS interface/runtime transfer metadata. | Data movement reports. | Communication tests. | `partially_implemented` | Confirm canonical vs legacy path behavior. |
| `data_movement.inputs.import` | legacy/canonical candidate | Same as above | Alternate input edge spelling. | Same intended effect. | Data movement reports. | Needs alias tests. | `duplicate_or_legacy` | Canonicalize. |
| `data_movement.weights.load` | interface/transport/policy | `fpgai/engine/communication.py`, `fpgai/engine/planner.py`, `fpgai/engine/training.py` | Resolves weight import/load behavior. | Changes weight payload/interface requirements. | Data movement and hardware knob reports. | Import-weight tests. | `partially_implemented` | Verify for embedded/import/DDR/training modes. |
| `data_movement.weights.import` | legacy/canonical candidate | Same as above | Alternate weight import spelling. | Same intended effect. | Data movement reports. | Needs alias tests. | `duplicate_or_legacy` | Canonicalize. |
| `data_movement.weights.export` | interface/transport/policy | `fpgai/engine/communication.py`, `fpgai/engine/training.py` | Resolves training weight export behavior. | Should affect training output/interface/runtime package. | Training/data movement reports. | Training movement tests. | `partially_implemented` | Prove generated HLS export path. |
| `data_movement.output.store` | interface/transport | `fpgai/engine/communication.py`, `fpgai/reporting/data_movement.py` | Resolves output tensor edge. | Affects output interface/runtime transfer metadata. | Data movement reports. | Communication tests. | `partially_implemented` | Verify artifact effect per interface. |
| `data_movement.outputs.export` | legacy/canonical candidate | Same as above | Alternate output edge spelling. | Same intended effect. | Data movement reports. | Needs alias tests. | `duplicate_or_legacy` | Canonicalize. |
| `data_movement.gradients.export` | `interface=m_axi`, `transport=ps_runtime`, `policy=full` currently | `fpgai/engine/compiler.py`, `fpgai/engine/training.py` | Resolves gradient export contract. | Should expose gradient output/export path when supported. | Movement contract validation. | `tests/test_training_movement_gradient_export.py` | `partially_implemented` | Expand or reject unsupported combinations clearly. |
| `data_movement.optimizer_state.export` | interface/transport/policy | `fpgai/engine/communication.py`, `fpgai/engine/training.py` | Resolves optimizer state export. | Should affect training runtime payload/export artifact. | Data movement/training reports. | Needs focused tests. | `partially_implemented` | Add artifact-effect validation when implemented. |
| `data_movement.ps_pl.*` / `data_movement.pl_ps.*` | legacy PS/PL directions | `fpgai/engine/communication.py`, `fpgai/engine/training.py` | Legacy directional movement config. | May affect same tensor edges. | Reports should expose replacement/canonical behavior. | Legacy config tests. | `duplicate_or_legacy` | Maintain temporarily; document canonical replacement. |

## 6. Precision and numerics

| YAML path | Accepted values / example | Owner files | Compiler/planner effect | Artifact effect | Report effect | Validation coverage | Status | Next action |
|---|---|---|---|---|---|---|---|---|
| `numerics.precision_mode` | `float32`, `fixed`, mixed modes | `fpgai/engine/planner.py`, `fpgai/numerics/precision_policy.py`, HLS emitters | Selects global precision mode. | Changes generated type/layout when supported. | Precision layout/effect reports. | Precision config/codegen tests. | `implemented` | Keep materialization tests current. |
| `numerics.defaults.activation` | ap/fixed spec | `fpgai/engine/planner.py`, `fpgai/numerics/precision_policy.py` | Resolves activation type width. | Changes generated activation typedefs/metadata. | Precision reports. | Mixed precision tests. | `implemented` | Confirm HLS artifact differences in matrix tests. |
| `numerics.defaults.weight` | ap/fixed spec | Same as above | Resolves weight type width. | Changes generated weight typedefs/params. | Precision reports. | Mixed precision tests. | `implemented` | Keep source-diff validation. |
| `numerics.defaults.bias` | ap/fixed spec | Same as above | Resolves bias type width. | Changes generated bias typedefs/params. | Precision reports. | Mixed precision tests. | `implemented` | Keep source-diff validation. |
| `numerics.defaults.accum` | ap/fixed spec | Same as above | Resolves accumulator type width. | Changes generated accumulator typedefs. | Precision reports. | Mixed precision tests. | `implemented` | Keep source-diff validation. |
| `numerics.training` | training-specific precision overrides | `fpgai/engine/training.py`, `fpgai/numerics/precision_policy.py` | Resolves training numerics. | Should affect training generated code/testbench. | Training plan/report. | Training precision tests. | `partially_implemented` | Verify generated training source effects per field. |
| Layerwise precision rules | op/layer-specific specs | `fpgai/engine/layerwise_precision.py`, `fpgai/numerics/precision_policy.py` | Applies per-op precision metadata. | Should change per-layer generated code. | Precision reports include rules/tags. | Layerwise precision tests. | `implemented/needs_validation` | Confirm per-layer artifact effect coverage. |
| `analysis.precision_sweep.*` | sweep config | `fpgai/analysis/precision_sweep.py`, `fpgai/reporting/precision_effect.py` | Runs configured precision experiments. | Emits sweep outputs, not a single compile artifact setting. | Precision sweep reports. | Sweep tests. | `implemented` | Keep separate from selected hardware precision. |

## 7. Parallelism

| YAML path | Accepted values / example | Owner files | Compiler/planner effect | Artifact effect | Report effect | Validation coverage | Status | Next action |
|---|---|---|---|---|---|---|---|---|
| `optimization.parallel_policy` | policy preset | `fpgai/engine/planner.py` | Selects default PE/SIMD/unroll/partition policy. | Affects generated calls when not manually overridden. | Hardware knob contract and planner notes. | Policy tests. | `implemented` | Keep manual override priority. |
| `optimization.parallel.pe` | positive int | `fpgai/engine/planner.py`, HLS emitters | Sets processing-element/unroll plan. | Should affect generated layer call/template arguments. | Parallel/pipeline effect reports. | Parallel tests. | `implemented/needs_validation` | Keep artifact-diff tests. |
| `optimization.parallel.simd` | positive int | Same as above | Sets SIMD/input parallelism plan. | Should affect generated layer call/template arguments. | Parallel/pipeline reports. | Parallel tests. | `implemented/needs_validation` | Keep artifact-diff tests. |
| `optimization.parallel.unroll_factor` | positive int | Same as above | Sets loop unroll plan. | Should affect unroll arguments/pragmas. | Hardware knob contract. | Materialization tests. | `implemented/needs_validation` | Prove source/pragmas per operator. |
| `optimization.parallel.partition_factor` | positive int | Same as above | Sets array partition factor. | Should affect array partition pragmas/metadata. | Hardware knob contract. | Materialization tests. | `implemented/needs_validation` | Prove generated pragmas. |
| `optimization.parallel.array_partition_mode` | `cyclic`, `block`, etc. | `fpgai/engine/planner.py`, HLS emitters | Selects partition mode. | Should affect generated HLS partition pragma mode. | Hardware knob contract. | Needs focused test if missing. | `partially_implemented` | Confirm accepted values and artifact effect. |

## 8. Pipeline

| YAML path | Accepted values / example | Owner files | Compiler/planner effect | Artifact effect | Report effect | Validation coverage | Status | Next action |
|---|---|---|---|---|---|---|---|---|
| `optimization.pipeline.style` | `conservative`, `balanced`, `aggressive` | `fpgai/engine/planner.py`, HLS emitters | Resolves pipeline style and default II. | Should affect generated pipeline-II arguments/pragmas. | Parallel/pipeline effect and hardware knob reports. | Pipeline tests. | `implemented/needs_validation` | Keep source-diff tests per style. |
| `optimization.pipeline.ii` | positive int | `fpgai/engine/planner.py`, HLS emitters | Overrides initiation interval. | Should affect generated pipeline pragma/call argument. | Reports requested/effective II. | Pipeline tests. | `implemented/needs_validation` | Prove HLS source uses requested II. |
| `optimization.pipeline_ii` | legacy alias | `fpgai/engine/planner.py` | Alternate II source. | Same intended effect. | Should report canonical replacement. | Alias tests needed. | `duplicate_or_legacy` | Migrate to `optimization.pipeline.ii`. |
| `optimization.parallel.pipeline_style` | legacy/alternate path | `fpgai/engine/planner.py` | Fallback pipeline style. | Same intended effect. | Should be marked legacy if not canonical. | Existing tests likely. | `duplicate_or_legacy` | Migrate to `optimization.pipeline.style`. |

## 9. Tiling

| YAML path | Accepted values / example | Owner files | Compiler/planner effect | Artifact effect | Report effect | Validation coverage | Status | Next action |
|---|---|---|---|---|---|---|---|---|
| `optimization.tiling.dense` | dense tile spec | `fpgai/engine/planner.py`, `fpgai/backends/hls/emit/dense_tiling_codegen.py` | Resolves dense tile sizes. | Emits dense tiling helpers/arguments/metadata where supported. | Tiling/effect reports. | `tests/test_dense_tiling_codegen.py`, tiling materialization tests. | `implemented/needs_validation` | Keep source and HLS artifact checks. |
| `optimization.tiling.conv` | conv tile spec | `fpgai/engine/planner.py`, `fpgai/backends/hls/emit/conv_tiling_codegen.py` | Resolves convolution tile sizes. | Emits conv tiling helpers/arguments/metadata where supported. | Tiling/effect reports. | `tests/test_conv_tiling_codegen.py`, tiling materialization tests. | `implemented/needs_validation` | Keep source and HLS artifact checks. |
| `optimization.tiling.layers` | per-layer tile specs | `fpgai/engine/planner.py` | Applies per-layer tiling overrides. | Should affect specific generated layer calls/helpers. | Planner/tiling reports. | Needs exact coverage. | `partially_implemented` | Verify per-layer artifact effect. |
| Data movement tiled edge options | edge-level tile config | `fpgai/engine/communication.py` | Resolves transfer tiling metadata. | Should affect runtime/data movement artifacts when implemented. | Data movement reports. | Needs focused tests. | `partially_implemented` | Separate transfer tiling from compute tiling. |

## 10. Inference testbench and model behavior

| YAML path / feature | Accepted values / example | Owner files | Compiler/planner effect | Artifact effect | Report effect | Validation coverage | Status | Next action |
|---|---|---|---|---|---|---|---|---|
| Inference testbench generation | `build.stages.testbench: true` | `fpgai/backends/hls/testbench.py`, `fpgai/benchmark/reference.py` | Enables inference testbench. | Emits testbench C++ and reference payloads when configured. | HLS artifacts and numeric reports where enabled. | `tests/test_generated_hls_explanation_and_numeric.py`, integration tests. | `partially_implemented` | P2L should unify model-behavior report. |
| Reference output comparison | benchmark/validation config | `fpgai/benchmark/compare.py`, `fpgai/benchmark/reference.py` | Compares outputs against reference. | Produces comparison metrics. | Benchmark/validation reports. | Integration correctness tests. | `partially_implemented` | Standardize JSON/MD report fields. |
| Intermediate comparison | optional/reference intermediate | `fpgai/benchmark/reference_intermediate.py`, `fpgai/benchmark/intermediate_compare.py` | Supports layer/intermediate analysis. | Produces intermediate comparison data when enabled. | Intermediate validation reports. | Needs exact coverage. | `partially_implemented` | P2L should integrate into compile reports. |

## 11. Training behavior

| YAML path / feature | Accepted values / example | Owner files | Compiler/planner effect | Artifact effect | Report effect | Validation coverage | Status | Next action |
|---|---|---|---|---|---|---|---|---|
| `training.optimizer.type` | `sgd`, `momentum`, `adam`, etc. | `fpgai/engine/training.py`, training HLS emitters | Selects optimizer behavior. | Should affect generated update logic/testbench. | Training plan/report. | Training optimizer tests. | `partially_implemented` | Verify source effect per optimizer. |
| `training.loss.type` | loss names | `fpgai/engine/training.py`, `fpgai/backends/hls/testbench_train.py` | Selects loss behavior. | Should affect training testbench/generated loss path. | Training plan/report. | Training loss tests. | `partially_implemented` | Confirm generated HLS effect and numeric validation. |
| `training.gradient_accumulation.mode` | `none`, `native`, `testbench`, `native_accumulated`, `testbench_accumulated` | `fpgai/engine/compiler.py`, `fpgai/engine/training.py`, training emitters | Resolves accumulation implementation status. | Changes native/generated or testbench-only accumulation path depending on mode. | Training plan and support status. | Accumulation tests. | `implemented/partially_implemented` | Keep `testbench_only` explicitly reported where relevant. |
| Training reference comparison | compile/benchmark config | `fpgai/benchmark/training_reference.py`, `fpgai/benchmark/training_compare.py` | Compares training step behavior. | Produces loss/gradient/update comparison metrics. | Training validation reports. | Training tests. | `partially_implemented` | P2M should standardize behavior report. |

## 12. Runtime package and deployment

| YAML path / feature | Accepted values / example | Owner files | Compiler/planner effect | Artifact effect | Report effect | Validation coverage | Status | Next action |
|---|---|---|---|---|---|---|---|---|
| Runtime package creation | `build.stages.runtime_package: true` | `fpgai/runtime/package.py`, `fpgai/engine/compiler.py` | Requests runtime package. | Copies manifests, HLS/runtime support, and available `.bit/.hwh/.xsa`. | Runtime package summary and deployability flags. | `tests/test_runtime_package.py`; live P2E2 validation. | `implemented` | Keep deployability distinct from runtime execution. |
| Board runtime execution result | future runtime config | `fpgai/runtime/*`, board runtime owners | Should run accelerator on physical board. | Produces measured output/latency/runtime artifact. | Master results `with_runtime_result`. | Not yet live validated. | `needs_validation` | P2N: do real board execution; do not fake. |

## 13. Vivado implementation, timing, and power behavior

| YAML path / feature | Accepted values / example | Owner files | Compiler/planner effect | Artifact effect | Report effect | Validation coverage | Status | Next action |
|---|---|---|---|---|---|---|---|---|
| Vivado implementation report parsing | implementation reports | `fpgai/reporting/vivado_bridge_artifacts.py`, `fpgai/reporting/vivado_impl_artifacts.py`, `fpgai/reporting/hardware_feasibility.py` | Parses available Vivado artifacts. | Records utilization/timing/power report paths and parsed values where supported. | Vivado implementation report JSON/MD. | Vivado report tests. | `partially_implemented` | P2O should expand timing/power/critical warning fields. |
| Timing metrics | WNS/TNS/clock estimates | Same as above | Interprets implementation timing when present. | No generated source effect; report parsing effect. | Timing fields in hardware feasibility/Vivado reports. | Needs live report coverage. | `partially_implemented` | Add robust parser tests using real report snippets. |
| Power/energy metrics | power report + runtime latency | Same as above plus runtime owners | Parses power when available; energy requires latency. | Report-only until runtime latency exists. | Power/energy fields when available. | Needs validation. | `partially_implemented` | Do not claim energy without measured/validated latency source. |
| AXI/DMA/BD summary | Vivado BD artifacts | `fpgai/backends/vivado/vivado_bridge.py`, reporting modules | Generates/records board design wiring. | BD/TCL/HWH artifacts show interfaces. | Vivado bridge reports. | Vivado bridge tests. | `implemented/needs_validation` | P2O should add clear interface summary. |

## 14. Paper/reporting integration

| YAML path / feature | Accepted values / example | Owner files | Compiler/planner effect | Artifact effect | Report effect | Validation coverage | Status | Next action |
|---|---|---|---|---|---|---|---|---|
| Paper master results | build dirs → result rows | `fpgai/reporting/paper_results.py` | Aggregates generated artifacts. | Reads reports/manifests, does not create hardware. | `master_results.json/csv/md`, schema docs. | Paper results schema tests; live P2E2 refresh. | `implemented` | Keep runtime rows honest. |
| HLS/Vivado artifact status | report files/artifacts | `fpgai/reporting/*`, `fpgai/analysis/*` | Classifies artifact status. | Reads real generated artifacts. | Tables and summaries. | Reporting tests. | `implemented/needs_validation` | Prefer artifact-backed claims only. |
| Static validation | generated-source/report checks | `fpgai/validation/*`, `fpgai/devtools/*`, tests | Validates artifact structure. | No hardware run. | Static validation rows. | Existing tests/audits. | `implemented/needs_validation` | Keep separate from runtime validation. |

## 15. Legacy and duplicate YAML paths to clean later

These paths are accepted or referenced today, but require canonicalization before public documentation claims a clean v1 schema.

| Legacy/duplicate path | Canonical direction | Owner files | Status | Cleanup rule |
|---|---|---|---|---|
| `data_movement.ps_pl.*` | Tensor-specific `data_movement.input/weights/...` paths | `fpgai/engine/communication.py`, `fpgai/engine/training.py` | `duplicate_or_legacy` | Keep accepted temporarily; report replacement. |
| `data_movement.pl_ps.*` | Tensor-specific output/export paths | Same as above | `duplicate_or_legacy` | Keep accepted temporarily; report replacement. |
| `data_movement.inputs.import` vs `data_movement.input.load` | Choose one canonical public spelling | `fpgai/engine/communication.py` | `duplicate_or_legacy` | Matrix and schema cleanup must decide final canonical spelling. |
| `data_movement.outputs.export` vs `data_movement.output.store` | Choose one canonical public spelling | `fpgai/engine/communication.py` | `duplicate_or_legacy` | Matrix and schema cleanup must decide final canonical spelling. |
| `memory.storage.weights` vs `memory.weight_storage` | Prefer one canonical spelling | `fpgai/engine/planner.py`, `fpgai/engine/memory.py` | `duplicate_or_legacy` | Keep fallback until migration guide and tests are updated. |
| `optimization.pipeline_ii` | `optimization.pipeline.ii` | `fpgai/engine/planner.py` | `duplicate_or_legacy` | Mark deprecated after examples migrate. |
| `optimization.parallel.pipeline_style` | `optimization.pipeline.style` | `fpgai/engine/planner.py` | `duplicate_or_legacy` | Keep fallback; document canonical path. |
| Informal `truth` terminology in older reports/modules | Professional artifact/validation terminology | Existing reporting modules and tests | `duplicate_or_legacy` | Rename gradually without breaking artifact compatibility. |

## P2F completion criteria

P2F is complete when:

1. This matrix exists in the repo-owned feature matrix document.
2. Future sprints refer to the existing owner files listed here before implementation.
3. Unsupported or partial behavior is not described as fully supported in user-facing docs.
4. Cleanup/refactor sprints use this matrix to decide whether to modify, deprecate, or remove paths.

## Immediate next sprint after P2F

The recommended next implementation sprint is memory completion because the repo already documents the largest remaining architecture boundary: current DDR behavior is mostly `ddr_preload_weights`, while real scalable external DDR execution requires validated `ddr_tiled_weights`.

Start with existing owners only:

```text
fpgai/engine/memory.py
fpgai/engine/planner.py
fpgai/engine/communication.py
fpgai/backends/hls/emit/*
fpgai/reports/memory_semantics.py
docs/MEMORY_SEMANTICS_PLAN.md
tests/test_memory_semantics_classifier.py
tests/test_memory_storage_effect.py
tests/test_training_memory_storage_contract.py
```

## P3D-F3A memory residency contract

FPGAI now emits `reports/memory_residency_contract.json` and `.md` from the existing data-movement reporting owner. The contract separates physical residency, transfer mechanism, local staging, mutability, and lifetime. It distinguishes full DDR preload into BRAM/URAM from genuine DDR-tiled residency and reports the current tensor-level representation gap for gradients and optimizer state. Realized BRAM/URAM/DDR behavior still requires generated-source, HLS, Vivado, and board validation.
