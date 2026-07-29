# FPGAI Extensibility Audit

Repository: `/home/umutcanaltin/Desktop/github_projects/fpgai_compiler`

FPGAI is the open research, validation, and benchmarking compiler platform. Morfics owns commercial productization, managed builds, deployment, hosted inference/training, operations, security, billing, and production support.

## Summary

- Extension families: 18
- Externally extensible today: 0
- Requiring core edits today: 18

## Findings

| Capability | Mechanism | Core edit | Priority | Recommended contract |
|---|---|---:|---|---|
| `onnx_import` | `central_frontend_pipeline` | `yes` | `critical` | `OnnxImporterRegistry` |
| `ir_operator` | `hard_coded_metadata_and_generic_op` | `yes` | `critical` | `OperatorRegistry` |
| `shape_and_type_inference` | `frontend_helpers` | `yes` | `high` | `ShapeInferenceRegistry` |
| `canonicalization` | `central_functions` | `yes` | `high` | `CanonicalizationRegistry` |
| `hls_implementation` | `central_codegen_dispatch` | `yes` | `critical` | `ImplementationRegistry` |
| `vhdl_and_rtl_implementation` | `backend_not_yet_first_class` | `yes` | `critical` | `RtlImplementationRegistry` |
| `training_reference` | `central_operator_dispatch` | `yes` | `critical` | `TrainingSemanticsRegistry` |
| `optimizer` | `string_conditionals` | `yes` | `high` | `OptimizerRegistry` |
| `loss` | `string_conditionals` | `yes` | `high` | `LossRegistry` |
| `board` | `central_board_database` | `yes` | `high` | `BoardRegistry` |
| `backend_and_toolchain` | `direct_backend_imports` | `yes` | `high` | `BackendRegistry` |
| `memory_policy` | `central_policy_logic` | `yes` | `high` | `MemoryPolicyRegistry` |
| `transport` | `central_contract_logic` | `yes` | `high` | `TransportRegistry` |
| `dataset` | `direct_dataset_helpers` | `yes` | `medium` | `DatasetRegistry` |
| `validation` | `direct_report_calls` | `yes` | `medium` | `ValidationRegistry` |
| `reporter` | `direct_report_calls` | `yes` | `medium` | `ReporterRegistry` |
| `runtime_package` | `stable_public_entry_with_internal_modules` | `yes` | `medium` | `RuntimeRegistry` |
| `model_package` | `path_based_model_loading` | `yes` | `high` | `ModelRegistry` |
