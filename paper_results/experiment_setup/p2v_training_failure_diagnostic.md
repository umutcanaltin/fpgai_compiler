# P2V training failure diagnostic

## Compile status
order	id	status	log
00	I0_baseline_fx16_embedded	0	paper_results/experiment_setup/compile_logs/00_I0_baseline_fx16_embedded.log
01	I1_precision_fx8_embedded	0	paper_results/experiment_setup/compile_logs/01_I1_precision_fx8_embedded.log
02	I2_precision_fx24_embedded	0	paper_results/experiment_setup/compile_logs/02_I2_precision_fx24_embedded.log
03	I3_parallel_pe2	0	paper_results/experiment_setup/compile_logs/03_I3_parallel_pe2.log
04	I4_parallel_pe4	0	paper_results/experiment_setup/compile_logs/04_I4_parallel_pe4.log
05	I5_pipeline_latency_first	0	paper_results/experiment_setup/compile_logs/05_I5_pipeline_latency_first.log
06	I6_pipeline_resource_first	0	paper_results/experiment_setup/compile_logs/06_I6_pipeline_resource_first.log
07	I7_weight_import_m_axi	0	paper_results/experiment_setup/compile_logs/07_I7_weight_import_m_axi.log
08	I8_deployable_bitstream_candidate	0	paper_results/experiment_setup/compile_logs/08_I8_deployable_bitstream_candidate.log
09	I9_board_runtime_candidate	0	paper_results/experiment_setup/compile_logs/09_I9_board_runtime_candidate.log
10	T0_sgd_tiled_m_axi	2	paper_results/experiment_setup/compile_logs/10_T0_sgd_tiled_m_axi.log
11	T1_momentum_tiled_m_axi	2	paper_results/experiment_setup/compile_logs/11_T1_momentum_tiled_m_axi.log
12	T2_adam_tiled_m_axi	1	paper_results/experiment_setup/compile_logs/12_T2_adam_tiled_m_axi.log
13	T3_cross_entropy_tiled_m_axi	2	paper_results/experiment_setup/compile_logs/13_T3_cross_entropy_tiled_m_axi.log
14	T4_tile32_m_axi	2	paper_results/experiment_setup/compile_logs/14_T4_tile32_m_axi.log
15	T5_tile128_m_axi	2	paper_results/experiment_setup/compile_logs/15_T5_tile128_m_axi.log
16	T6_accum_batch2_m_axi	2	paper_results/experiment_setup/compile_logs/16_T6_accum_batch2_m_axi.log
17	T7_deployable_training_bitstream	2	paper_results/experiment_setup/compile_logs/17_T7_deployable_training_bitstream.log
18	T8_real_fpga_training_curve_candidate	2	paper_results/experiment_setup/compile_logs/18_T8_real_fpga_training_curve_candidate.log

## T0_sgd_tiled_m_axi

### compile log: paper_results/experiment_setup/compile_logs/10_T0_sgd_tiled_m_axi.log
14:HLS returncode       : 1
17:HLS csynth report    : None
52:Prediction artifacts:
61:  - Vivado allowed by fit: True
63:HLS artifacts       : available
65:  - hls_ok: False
66:  - hls_returncode: 1
69:  - artifact_metadata: hls_artifact_metadata.json
78:  - vivado_allowed_by_board_fit: True
85:Vivado bridge       : available
89:  - vivado_bridge_generated: True
90:  - vivado_synth_requested: True
91:  - vivado_impl_requested: True
96:Runtime package     : created
97:  - package: runtime_package/package_manifest.json
117:  - run_hls: failed
118:  - training_artifacts: done
119:  - vivado_project: done
121:  - runtime_package: done
134:[OK] Wrote artifacts to: /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T0_sgd_tiled_m_axi
139:      "name": "manifest_or_artifact_report_present",
149:      "name": "hls_ok",
154:      "name": "vivado_implemented",
158:  "claim_level": "level_0_compiler_artifact",
159:  "configured_stage": "vivado_implementation",
160:  "expected_claim_level": "level_2_vivado_implementation",
161:  "failed_checks": [
164:      "name": "hls_ok",
169:  "status": "failed"

### HLS stdout/stderr
build/paper_experiments/training/T0_sgd_tiled_m_axi/hls/logs/vitis_hls_stdout.log:42:INFO: [HLS 200-1510] Running: csim_design -clean -argv /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T0_sgd_tiled_m_axi/input.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T0_sgd_tiled_m_axi/target.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T0_sgd_tiled_m_axi/training_reference/weights_before_ref.bin . 
build/paper_experiments/training/T0_sgd_tiled_m_axi/hls/logs/vitis_hls_stdout.log:43:INFO: [SIM 211-2] *************** CSIM start ***************
build/paper_experiments/training/T0_sgd_tiled_m_axi/hls/logs/vitis_hls_stdout.log:44:INFO: [SIM 211-4] CSIM will launch GCC as the compiler.
build/paper_experiments/training/T0_sgd_tiled_m_axi/hls/logs/vitis_hls_stdout.log:50:../../../../src/tb.cpp:159:25: error: missing terminating " character
build/paper_experiments/training/T0_sgd_tiled_m_axi/hls/logs/vitis_hls_stdout.log:56:../../../../src/tb.cpp:160:1: error: missing terminating " character
build/paper_experiments/training/T0_sgd_tiled_m_axi/hls/logs/vitis_hls_stdout.log:62:../../../../src/tb.cpp:166:25: error: missing terminating " character
build/paper_experiments/training/T0_sgd_tiled_m_axi/hls/logs/vitis_hls_stdout.log:68:../../../../src/tb.cpp:167:1: error: missing terminating " character
build/paper_experiments/training/T0_sgd_tiled_m_axi/hls/logs/vitis_hls_stdout.log:72:../../../../src/tb.cpp:161:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T0_sgd_tiled_m_axi/hls/logs/vitis_hls_stdout.log:79:../../../../src/tb.cpp:168:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T0_sgd_tiled_m_axi/hls/logs/vitis_hls_stdout.log:86:make: *** [csim.mk:79: obj/tb.o] Error 1
build/paper_experiments/training/T0_sgd_tiled_m_axi/hls/logs/vitis_hls_stdout.log:87:ERROR: [SIM 211-100] 'csim_design' failed: compilation error(s).
build/paper_experiments/training/T0_sgd_tiled_m_axi/hls/logs/vitis_hls_stdout.log:88:INFO: [SIM 211-3] *************** CSIM finish ***************
build/paper_experiments/training/T0_sgd_tiled_m_axi/hls/logs/vitis_hls_stdout.log:89:INFO: [HLS 200-111] Finished Command csim_design CPU user time: 1.13 seconds. CPU system time: 0.28 seconds. Elapsed time: 1.4 seconds; current allocated memory: 0.000 MB.

### Reports/status files
build/paper_experiments/training/T0_sgd_tiled_m_axi/manifest.json
build/paper_experiments/training/T0_sgd_tiled_m_axi/reports/generated_cpp_validation.json
build/paper_experiments/training/T0_sgd_tiled_m_axi/reports/hardware_knob_contract.json
build/paper_experiments/training/T0_sgd_tiled_m_axi/reports/layer_backend_status.json
build/paper_experiments/training/T0_sgd_tiled_m_axi/reports/movement_contract_validation.json
build/paper_experiments/training/T0_sgd_tiled_m_axi/reports/numeric_validation.json
build/paper_experiments/training/T0_sgd_tiled_m_axi/reports/runtime_package_validation.json
build/paper_experiments/training/T0_sgd_tiled_m_axi/reports/vivado_bd_validation.json
build/paper_experiments/training/T0_sgd_tiled_m_axi/reports/vivado_validation_report.json
build/paper_experiments/training/T0_sgd_tiled_m_axi/runtime_package/manifest.json
build/paper_experiments/training/T0_sgd_tiled_m_axi/runtime_package/package_manifest.json
build/paper_experiments/training/T0_sgd_tiled_m_axi/runtime_package/runtime_package_validation.json
build/paper_experiments/training/T0_sgd_tiled_m_axi/vivado_bridge/vivado_bridge_manifest.json

## T1_momentum_tiled_m_axi

### compile log: paper_results/experiment_setup/compile_logs/11_T1_momentum_tiled_m_axi.log
14:HLS returncode       : 1
17:HLS csynth report    : None
52:Prediction artifacts:
61:  - Vivado allowed by fit: True
63:HLS artifacts       : available
65:  - hls_ok: False
66:  - hls_returncode: 1
69:  - artifact_metadata: hls_artifact_metadata.json
78:  - vivado_allowed_by_board_fit: True
85:Vivado bridge       : available
89:  - vivado_bridge_generated: True
90:  - vivado_synth_requested: True
91:  - vivado_impl_requested: True
96:Runtime package     : created
97:  - package: runtime_package/package_manifest.json
117:  - run_hls: failed
118:  - training_artifacts: done
119:  - vivado_project: done
121:  - runtime_package: done
134:[OK] Wrote artifacts to: /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T1_momentum_tiled_m_axi
139:      "name": "manifest_or_artifact_report_present",
149:      "name": "hls_ok",
154:      "name": "vivado_implemented",
158:  "claim_level": "level_0_compiler_artifact",
159:  "configured_stage": "vivado_implementation",
160:  "expected_claim_level": "level_2_vivado_implementation",
161:  "failed_checks": [
164:      "name": "hls_ok",
169:  "status": "failed"

### HLS stdout/stderr
build/paper_experiments/training/T1_momentum_tiled_m_axi/hls/logs/vitis_hls_stdout.log:42:INFO: [HLS 200-1510] Running: csim_design -clean -argv /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T1_momentum_tiled_m_axi/input.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T1_momentum_tiled_m_axi/target.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T1_momentum_tiled_m_axi/training_reference/weights_before_ref.bin . 
build/paper_experiments/training/T1_momentum_tiled_m_axi/hls/logs/vitis_hls_stdout.log:43:INFO: [SIM 211-2] *************** CSIM start ***************
build/paper_experiments/training/T1_momentum_tiled_m_axi/hls/logs/vitis_hls_stdout.log:44:INFO: [SIM 211-4] CSIM will launch GCC as the compiler.
build/paper_experiments/training/T1_momentum_tiled_m_axi/hls/logs/vitis_hls_stdout.log:50:../../../../src/tb.cpp:159:25: error: missing terminating " character
build/paper_experiments/training/T1_momentum_tiled_m_axi/hls/logs/vitis_hls_stdout.log:56:../../../../src/tb.cpp:160:1: error: missing terminating " character
build/paper_experiments/training/T1_momentum_tiled_m_axi/hls/logs/vitis_hls_stdout.log:62:../../../../src/tb.cpp:166:25: error: missing terminating " character
build/paper_experiments/training/T1_momentum_tiled_m_axi/hls/logs/vitis_hls_stdout.log:68:../../../../src/tb.cpp:167:1: error: missing terminating " character
build/paper_experiments/training/T1_momentum_tiled_m_axi/hls/logs/vitis_hls_stdout.log:72:../../../../src/tb.cpp:161:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T1_momentum_tiled_m_axi/hls/logs/vitis_hls_stdout.log:79:../../../../src/tb.cpp:168:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T1_momentum_tiled_m_axi/hls/logs/vitis_hls_stdout.log:86:make: *** [csim.mk:79: obj/tb.o] Error 1
build/paper_experiments/training/T1_momentum_tiled_m_axi/hls/logs/vitis_hls_stdout.log:87:ERROR: [SIM 211-100] 'csim_design' failed: compilation error(s).
build/paper_experiments/training/T1_momentum_tiled_m_axi/hls/logs/vitis_hls_stdout.log:88:INFO: [SIM 211-3] *************** CSIM finish ***************
build/paper_experiments/training/T1_momentum_tiled_m_axi/hls/logs/vitis_hls_stdout.log:89:INFO: [HLS 200-111] Finished Command csim_design CPU user time: 1.05 seconds. CPU system time: 0.26 seconds. Elapsed time: 1.33 seconds; current allocated memory: 0.000 MB.

### Reports/status files
build/paper_experiments/training/T1_momentum_tiled_m_axi/manifest.json
build/paper_experiments/training/T1_momentum_tiled_m_axi/reports/generated_cpp_validation.json
build/paper_experiments/training/T1_momentum_tiled_m_axi/reports/hardware_knob_contract.json
build/paper_experiments/training/T1_momentum_tiled_m_axi/reports/layer_backend_status.json
build/paper_experiments/training/T1_momentum_tiled_m_axi/reports/movement_contract_validation.json
build/paper_experiments/training/T1_momentum_tiled_m_axi/reports/numeric_validation.json
build/paper_experiments/training/T1_momentum_tiled_m_axi/reports/runtime_package_validation.json
build/paper_experiments/training/T1_momentum_tiled_m_axi/reports/vivado_bd_validation.json
build/paper_experiments/training/T1_momentum_tiled_m_axi/reports/vivado_validation_report.json
build/paper_experiments/training/T1_momentum_tiled_m_axi/runtime_package/manifest.json
build/paper_experiments/training/T1_momentum_tiled_m_axi/runtime_package/package_manifest.json
build/paper_experiments/training/T1_momentum_tiled_m_axi/runtime_package/runtime_package_validation.json
build/paper_experiments/training/T1_momentum_tiled_m_axi/vivado_bridge/vivado_bridge_manifest.json

## T2_adam_tiled_m_axi

### compile log: paper_results/experiment_setup/compile_logs/12_T2_adam_tiled_m_axi.log
1:[ERROR] Unsupported runtime sequence command 'export_optimizer_state'. Supported commands are: accumulate_gradients, apply_accumulated_gradients, export_gradients, export_weights, import_weights, reset_accumulators, run_inference, run_training.

### HLS stdout/stderr

### Reports/status files

## T3_cross_entropy_tiled_m_axi

### compile log: paper_results/experiment_setup/compile_logs/13_T3_cross_entropy_tiled_m_axi.log
14:HLS returncode       : 1
17:HLS csynth report    : None
52:Prediction artifacts:
61:  - Vivado allowed by fit: True
63:HLS artifacts       : available
65:  - hls_ok: False
66:  - hls_returncode: 1
69:  - artifact_metadata: hls_artifact_metadata.json
78:  - vivado_allowed_by_board_fit: True
85:Vivado bridge       : available
89:  - vivado_bridge_generated: True
90:  - vivado_synth_requested: True
91:  - vivado_impl_requested: True
96:Runtime package     : created
97:  - package: runtime_package/package_manifest.json
117:  - run_hls: failed
118:  - training_artifacts: done
119:  - vivado_project: done
121:  - runtime_package: done
134:[OK] Wrote artifacts to: /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T3_cross_entropy_tiled_m_axi
139:      "name": "manifest_or_artifact_report_present",
149:      "name": "hls_ok",
154:      "name": "vivado_implemented",
158:  "claim_level": "level_0_compiler_artifact",
159:  "configured_stage": "vivado_implementation",
160:  "expected_claim_level": "level_2_vivado_implementation",
161:  "failed_checks": [
164:      "name": "hls_ok",
169:  "status": "failed"

### HLS stdout/stderr
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/hls/logs/vitis_hls_stdout.log:42:INFO: [HLS 200-1510] Running: csim_design -clean -argv /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/input.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/target.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/training_reference/weights_before_ref.bin . 
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/hls/logs/vitis_hls_stdout.log:43:INFO: [SIM 211-2] *************** CSIM start ***************
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/hls/logs/vitis_hls_stdout.log:44:INFO: [SIM 211-4] CSIM will launch GCC as the compiler.
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/hls/logs/vitis_hls_stdout.log:50:../../../../src/tb.cpp:159:25: error: missing terminating " character
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/hls/logs/vitis_hls_stdout.log:56:../../../../src/tb.cpp:160:1: error: missing terminating " character
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/hls/logs/vitis_hls_stdout.log:62:../../../../src/tb.cpp:166:25: error: missing terminating " character
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/hls/logs/vitis_hls_stdout.log:68:../../../../src/tb.cpp:167:1: error: missing terminating " character
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/hls/logs/vitis_hls_stdout.log:72:../../../../src/tb.cpp:161:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/hls/logs/vitis_hls_stdout.log:79:../../../../src/tb.cpp:168:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/hls/logs/vitis_hls_stdout.log:86:make: *** [csim.mk:79: obj/tb.o] Error 1
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/hls/logs/vitis_hls_stdout.log:87:ERROR: [SIM 211-100] 'csim_design' failed: compilation error(s).
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/hls/logs/vitis_hls_stdout.log:88:INFO: [SIM 211-3] *************** CSIM finish ***************
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/hls/logs/vitis_hls_stdout.log:89:INFO: [HLS 200-111] Finished Command csim_design CPU user time: 1.04 seconds. CPU system time: 0.28 seconds. Elapsed time: 1.35 seconds; current allocated memory: 0.000 MB.

### Reports/status files
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/manifest.json
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/reports/generated_cpp_validation.json
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/reports/hardware_knob_contract.json
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/reports/layer_backend_status.json
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/reports/movement_contract_validation.json
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/reports/numeric_validation.json
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/reports/runtime_package_validation.json
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/reports/vivado_bd_validation.json
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/reports/vivado_validation_report.json
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/runtime_package/manifest.json
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/runtime_package/package_manifest.json
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/runtime_package/runtime_package_validation.json
build/paper_experiments/training/T3_cross_entropy_tiled_m_axi/vivado_bridge/vivado_bridge_manifest.json

## T4_tile32_m_axi

### compile log: paper_results/experiment_setup/compile_logs/14_T4_tile32_m_axi.log
14:HLS returncode       : 1
17:HLS csynth report    : None
52:Prediction artifacts:
61:  - Vivado allowed by fit: True
63:HLS artifacts       : available
65:  - hls_ok: False
66:  - hls_returncode: 1
69:  - artifact_metadata: hls_artifact_metadata.json
78:  - vivado_allowed_by_board_fit: True
85:Vivado bridge       : available
89:  - vivado_bridge_generated: True
90:  - vivado_synth_requested: True
91:  - vivado_impl_requested: True
96:Runtime package     : created
97:  - package: runtime_package/package_manifest.json
117:  - run_hls: failed
118:  - training_artifacts: done
119:  - vivado_project: done
121:  - runtime_package: done
134:[OK] Wrote artifacts to: /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T4_tile32_m_axi
139:      "name": "manifest_or_artifact_report_present",
149:      "name": "hls_ok",
154:      "name": "vivado_implemented",
158:  "claim_level": "level_0_compiler_artifact",
159:  "configured_stage": "vivado_implementation",
160:  "expected_claim_level": "level_2_vivado_implementation",
161:  "failed_checks": [
164:      "name": "hls_ok",
169:  "status": "failed"

### HLS stdout/stderr
build/paper_experiments/training/T4_tile32_m_axi/hls/logs/vitis_hls_stdout.log:42:INFO: [HLS 200-1510] Running: csim_design -clean -argv /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T4_tile32_m_axi/input.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T4_tile32_m_axi/target.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T4_tile32_m_axi/training_reference/weights_before_ref.bin . 
build/paper_experiments/training/T4_tile32_m_axi/hls/logs/vitis_hls_stdout.log:43:INFO: [SIM 211-2] *************** CSIM start ***************
build/paper_experiments/training/T4_tile32_m_axi/hls/logs/vitis_hls_stdout.log:44:INFO: [SIM 211-4] CSIM will launch GCC as the compiler.
build/paper_experiments/training/T4_tile32_m_axi/hls/logs/vitis_hls_stdout.log:50:../../../../src/tb.cpp:159:25: error: missing terminating " character
build/paper_experiments/training/T4_tile32_m_axi/hls/logs/vitis_hls_stdout.log:56:../../../../src/tb.cpp:160:1: error: missing terminating " character
build/paper_experiments/training/T4_tile32_m_axi/hls/logs/vitis_hls_stdout.log:62:../../../../src/tb.cpp:166:25: error: missing terminating " character
build/paper_experiments/training/T4_tile32_m_axi/hls/logs/vitis_hls_stdout.log:68:../../../../src/tb.cpp:167:1: error: missing terminating " character
build/paper_experiments/training/T4_tile32_m_axi/hls/logs/vitis_hls_stdout.log:72:../../../../src/tb.cpp:161:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T4_tile32_m_axi/hls/logs/vitis_hls_stdout.log:79:../../../../src/tb.cpp:168:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T4_tile32_m_axi/hls/logs/vitis_hls_stdout.log:86:make: *** [csim.mk:79: obj/tb.o] Error 1
build/paper_experiments/training/T4_tile32_m_axi/hls/logs/vitis_hls_stdout.log:87:ERROR: [SIM 211-100] 'csim_design' failed: compilation error(s).
build/paper_experiments/training/T4_tile32_m_axi/hls/logs/vitis_hls_stdout.log:88:INFO: [SIM 211-3] *************** CSIM finish ***************
build/paper_experiments/training/T4_tile32_m_axi/hls/logs/vitis_hls_stdout.log:89:INFO: [HLS 200-111] Finished Command csim_design CPU user time: 1.05 seconds. CPU system time: 0.25 seconds. Elapsed time: 1.33 seconds; current allocated memory: 0.000 MB.

### Reports/status files
build/paper_experiments/training/T4_tile32_m_axi/manifest.json
build/paper_experiments/training/T4_tile32_m_axi/reports/generated_cpp_validation.json
build/paper_experiments/training/T4_tile32_m_axi/reports/hardware_knob_contract.json
build/paper_experiments/training/T4_tile32_m_axi/reports/layer_backend_status.json
build/paper_experiments/training/T4_tile32_m_axi/reports/movement_contract_validation.json
build/paper_experiments/training/T4_tile32_m_axi/reports/numeric_validation.json
build/paper_experiments/training/T4_tile32_m_axi/reports/runtime_package_validation.json
build/paper_experiments/training/T4_tile32_m_axi/reports/vivado_bd_validation.json
build/paper_experiments/training/T4_tile32_m_axi/reports/vivado_validation_report.json
build/paper_experiments/training/T4_tile32_m_axi/runtime_package/manifest.json
build/paper_experiments/training/T4_tile32_m_axi/runtime_package/package_manifest.json
build/paper_experiments/training/T4_tile32_m_axi/runtime_package/runtime_package_validation.json
build/paper_experiments/training/T4_tile32_m_axi/vivado_bridge/vivado_bridge_manifest.json

## T5_tile128_m_axi

### compile log: paper_results/experiment_setup/compile_logs/15_T5_tile128_m_axi.log
14:HLS returncode       : 1
17:HLS csynth report    : None
52:Prediction artifacts:
61:  - Vivado allowed by fit: True
63:HLS artifacts       : available
65:  - hls_ok: False
66:  - hls_returncode: 1
69:  - artifact_metadata: hls_artifact_metadata.json
78:  - vivado_allowed_by_board_fit: True
85:Vivado bridge       : available
89:  - vivado_bridge_generated: True
90:  - vivado_synth_requested: True
91:  - vivado_impl_requested: True
96:Runtime package     : created
97:  - package: runtime_package/package_manifest.json
117:  - run_hls: failed
118:  - training_artifacts: done
119:  - vivado_project: done
121:  - runtime_package: done
134:[OK] Wrote artifacts to: /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T5_tile128_m_axi
139:      "name": "manifest_or_artifact_report_present",
149:      "name": "hls_ok",
154:      "name": "vivado_implemented",
158:  "claim_level": "level_0_compiler_artifact",
159:  "configured_stage": "vivado_implementation",
160:  "expected_claim_level": "level_2_vivado_implementation",
161:  "failed_checks": [
164:      "name": "hls_ok",
169:  "status": "failed"

### HLS stdout/stderr
build/paper_experiments/training/T5_tile128_m_axi/hls/logs/vitis_hls_stdout.log:42:INFO: [HLS 200-1510] Running: csim_design -clean -argv /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T5_tile128_m_axi/input.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T5_tile128_m_axi/target.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T5_tile128_m_axi/training_reference/weights_before_ref.bin . 
build/paper_experiments/training/T5_tile128_m_axi/hls/logs/vitis_hls_stdout.log:43:INFO: [SIM 211-2] *************** CSIM start ***************
build/paper_experiments/training/T5_tile128_m_axi/hls/logs/vitis_hls_stdout.log:44:INFO: [SIM 211-4] CSIM will launch GCC as the compiler.
build/paper_experiments/training/T5_tile128_m_axi/hls/logs/vitis_hls_stdout.log:50:../../../../src/tb.cpp:159:25: error: missing terminating " character
build/paper_experiments/training/T5_tile128_m_axi/hls/logs/vitis_hls_stdout.log:56:../../../../src/tb.cpp:160:1: error: missing terminating " character
build/paper_experiments/training/T5_tile128_m_axi/hls/logs/vitis_hls_stdout.log:62:../../../../src/tb.cpp:166:25: error: missing terminating " character
build/paper_experiments/training/T5_tile128_m_axi/hls/logs/vitis_hls_stdout.log:68:../../../../src/tb.cpp:167:1: error: missing terminating " character
build/paper_experiments/training/T5_tile128_m_axi/hls/logs/vitis_hls_stdout.log:72:../../../../src/tb.cpp:161:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T5_tile128_m_axi/hls/logs/vitis_hls_stdout.log:79:../../../../src/tb.cpp:168:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T5_tile128_m_axi/hls/logs/vitis_hls_stdout.log:86:make: *** [csim.mk:79: obj/tb.o] Error 1
build/paper_experiments/training/T5_tile128_m_axi/hls/logs/vitis_hls_stdout.log:87:ERROR: [SIM 211-100] 'csim_design' failed: compilation error(s).
build/paper_experiments/training/T5_tile128_m_axi/hls/logs/vitis_hls_stdout.log:88:INFO: [SIM 211-3] *************** CSIM finish ***************
build/paper_experiments/training/T5_tile128_m_axi/hls/logs/vitis_hls_stdout.log:89:INFO: [HLS 200-111] Finished Command csim_design CPU user time: 1.05 seconds. CPU system time: 0.27 seconds. Elapsed time: 1.34 seconds; current allocated memory: 0.000 MB.

### Reports/status files
build/paper_experiments/training/T5_tile128_m_axi/manifest.json
build/paper_experiments/training/T5_tile128_m_axi/reports/generated_cpp_validation.json
build/paper_experiments/training/T5_tile128_m_axi/reports/hardware_knob_contract.json
build/paper_experiments/training/T5_tile128_m_axi/reports/layer_backend_status.json
build/paper_experiments/training/T5_tile128_m_axi/reports/movement_contract_validation.json
build/paper_experiments/training/T5_tile128_m_axi/reports/numeric_validation.json
build/paper_experiments/training/T5_tile128_m_axi/reports/runtime_package_validation.json
build/paper_experiments/training/T5_tile128_m_axi/reports/vivado_bd_validation.json
build/paper_experiments/training/T5_tile128_m_axi/reports/vivado_validation_report.json
build/paper_experiments/training/T5_tile128_m_axi/runtime_package/manifest.json
build/paper_experiments/training/T5_tile128_m_axi/runtime_package/package_manifest.json
build/paper_experiments/training/T5_tile128_m_axi/runtime_package/runtime_package_validation.json
build/paper_experiments/training/T5_tile128_m_axi/vivado_bridge/vivado_bridge_manifest.json

## T6_accum_batch2_m_axi

### compile log: paper_results/experiment_setup/compile_logs/16_T6_accum_batch2_m_axi.log
14:HLS returncode       : 1
17:HLS csynth report    : None
52:Prediction artifacts:
61:  - Vivado allowed by fit: True
63:HLS artifacts       : available
65:  - hls_ok: False
66:  - hls_returncode: 1
69:  - artifact_metadata: hls_artifact_metadata.json
78:  - vivado_allowed_by_board_fit: True
85:Vivado bridge       : available
89:  - vivado_bridge_generated: True
90:  - vivado_synth_requested: True
91:  - vivado_impl_requested: True
96:Runtime package     : created
97:  - package: runtime_package/package_manifest.json
117:  - run_hls: failed
118:  - training_artifacts: done
119:  - vivado_project: done
121:  - runtime_package: done
134:[OK] Wrote artifacts to: /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T6_accum_batch2_m_axi
139:      "name": "manifest_or_artifact_report_present",
149:      "name": "hls_ok",
154:      "name": "vivado_implemented",
158:  "claim_level": "level_0_compiler_artifact",
159:  "configured_stage": "vivado_implementation",
160:  "expected_claim_level": "level_2_vivado_implementation",
161:  "failed_checks": [
164:      "name": "hls_ok",
169:  "status": "failed"

### HLS stdout/stderr
build/paper_experiments/training/T6_accum_batch2_m_axi/hls/logs/vitis_hls_stdout.log:42:INFO: [HLS 200-1510] Running: csim_design -clean -argv /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T6_accum_batch2_m_axi/input.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T6_accum_batch2_m_axi/target.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T6_accum_batch2_m_axi/training_reference/weights_before_ref.bin . 
build/paper_experiments/training/T6_accum_batch2_m_axi/hls/logs/vitis_hls_stdout.log:43:INFO: [SIM 211-2] *************** CSIM start ***************
build/paper_experiments/training/T6_accum_batch2_m_axi/hls/logs/vitis_hls_stdout.log:44:INFO: [SIM 211-4] CSIM will launch GCC as the compiler.
build/paper_experiments/training/T6_accum_batch2_m_axi/hls/logs/vitis_hls_stdout.log:50:../../../../src/tb.cpp:159:25: error: missing terminating " character
build/paper_experiments/training/T6_accum_batch2_m_axi/hls/logs/vitis_hls_stdout.log:56:../../../../src/tb.cpp:160:1: error: missing terminating " character
build/paper_experiments/training/T6_accum_batch2_m_axi/hls/logs/vitis_hls_stdout.log:62:../../../../src/tb.cpp:166:25: error: missing terminating " character
build/paper_experiments/training/T6_accum_batch2_m_axi/hls/logs/vitis_hls_stdout.log:68:../../../../src/tb.cpp:167:1: error: missing terminating " character
build/paper_experiments/training/T6_accum_batch2_m_axi/hls/logs/vitis_hls_stdout.log:72:../../../../src/tb.cpp:161:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T6_accum_batch2_m_axi/hls/logs/vitis_hls_stdout.log:79:../../../../src/tb.cpp:168:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T6_accum_batch2_m_axi/hls/logs/vitis_hls_stdout.log:86:make: *** [csim.mk:79: obj/tb.o] Error 1
build/paper_experiments/training/T6_accum_batch2_m_axi/hls/logs/vitis_hls_stdout.log:87:ERROR: [SIM 211-100] 'csim_design' failed: compilation error(s).
build/paper_experiments/training/T6_accum_batch2_m_axi/hls/logs/vitis_hls_stdout.log:88:INFO: [SIM 211-3] *************** CSIM finish ***************
build/paper_experiments/training/T6_accum_batch2_m_axi/hls/logs/vitis_hls_stdout.log:89:INFO: [HLS 200-111] Finished Command csim_design CPU user time: 1.05 seconds. CPU system time: 0.23 seconds. Elapsed time: 1.27 seconds; current allocated memory: 0.000 MB.

### Reports/status files
build/paper_experiments/training/T6_accum_batch2_m_axi/manifest.json
build/paper_experiments/training/T6_accum_batch2_m_axi/reports/generated_cpp_validation.json
build/paper_experiments/training/T6_accum_batch2_m_axi/reports/hardware_knob_contract.json
build/paper_experiments/training/T6_accum_batch2_m_axi/reports/layer_backend_status.json
build/paper_experiments/training/T6_accum_batch2_m_axi/reports/movement_contract_validation.json
build/paper_experiments/training/T6_accum_batch2_m_axi/reports/numeric_validation.json
build/paper_experiments/training/T6_accum_batch2_m_axi/reports/runtime_package_validation.json
build/paper_experiments/training/T6_accum_batch2_m_axi/reports/vivado_bd_validation.json
build/paper_experiments/training/T6_accum_batch2_m_axi/reports/vivado_validation_report.json
build/paper_experiments/training/T6_accum_batch2_m_axi/runtime_package/manifest.json
build/paper_experiments/training/T6_accum_batch2_m_axi/runtime_package/package_manifest.json
build/paper_experiments/training/T6_accum_batch2_m_axi/runtime_package/runtime_package_validation.json
build/paper_experiments/training/T6_accum_batch2_m_axi/vivado_bridge/vivado_bridge_manifest.json

## T7_deployable_training_bitstream

### compile log: paper_results/experiment_setup/compile_logs/17_T7_deployable_training_bitstream.log
14:HLS returncode       : 1
17:HLS csynth report    : None
52:Prediction artifacts:
61:  - Vivado allowed by fit: True
63:HLS artifacts       : available
65:  - hls_ok: False
66:  - hls_returncode: 1
69:  - artifact_metadata: hls_artifact_metadata.json
78:  - vivado_allowed_by_board_fit: True
85:Vivado bridge       : available
89:  - vivado_bridge_generated: True
90:  - vivado_synth_requested: True
91:  - vivado_impl_requested: True
96:Runtime package     : created
97:  - package: runtime_package/package_manifest.json
117:  - run_hls: failed
118:  - training_artifacts: done
119:  - vivado_project: done
121:  - runtime_package: done
134:[OK] Wrote artifacts to: /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T7_deployable_training_bitstream
139:      "name": "manifest_or_artifact_report_present",
149:      "name": "hls_ok",
154:      "name": "vivado_implemented",
164:      "name": "runtime_package_created",
169:      "name": "runtime_package_validated",
173:  "claim_level": "level_0_compiler_artifact",
174:  "configured_stage": "bitstream_package",
175:  "expected_claim_level": "level_3_bitstream_package",
176:  "failed_checks": [
179:      "name": "hls_ok",
184:  "status": "failed"

### HLS stdout/stderr
build/paper_experiments/training/T7_deployable_training_bitstream/hls/logs/vitis_hls_stdout.log:42:INFO: [HLS 200-1510] Running: csim_design -clean -argv /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T7_deployable_training_bitstream/input.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T7_deployable_training_bitstream/target.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T7_deployable_training_bitstream/training_reference/weights_before_ref.bin . 
build/paper_experiments/training/T7_deployable_training_bitstream/hls/logs/vitis_hls_stdout.log:43:INFO: [SIM 211-2] *************** CSIM start ***************
build/paper_experiments/training/T7_deployable_training_bitstream/hls/logs/vitis_hls_stdout.log:44:INFO: [SIM 211-4] CSIM will launch GCC as the compiler.
build/paper_experiments/training/T7_deployable_training_bitstream/hls/logs/vitis_hls_stdout.log:50:../../../../src/tb.cpp:159:25: error: missing terminating " character
build/paper_experiments/training/T7_deployable_training_bitstream/hls/logs/vitis_hls_stdout.log:56:../../../../src/tb.cpp:160:1: error: missing terminating " character
build/paper_experiments/training/T7_deployable_training_bitstream/hls/logs/vitis_hls_stdout.log:62:../../../../src/tb.cpp:166:25: error: missing terminating " character
build/paper_experiments/training/T7_deployable_training_bitstream/hls/logs/vitis_hls_stdout.log:68:../../../../src/tb.cpp:167:1: error: missing terminating " character
build/paper_experiments/training/T7_deployable_training_bitstream/hls/logs/vitis_hls_stdout.log:72:../../../../src/tb.cpp:161:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T7_deployable_training_bitstream/hls/logs/vitis_hls_stdout.log:79:../../../../src/tb.cpp:168:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T7_deployable_training_bitstream/hls/logs/vitis_hls_stdout.log:86:make: *** [csim.mk:79: obj/tb.o] Error 1
build/paper_experiments/training/T7_deployable_training_bitstream/hls/logs/vitis_hls_stdout.log:87:ERROR: [SIM 211-100] 'csim_design' failed: compilation error(s).
build/paper_experiments/training/T7_deployable_training_bitstream/hls/logs/vitis_hls_stdout.log:88:INFO: [SIM 211-3] *************** CSIM finish ***************
build/paper_experiments/training/T7_deployable_training_bitstream/hls/logs/vitis_hls_stdout.log:89:INFO: [HLS 200-111] Finished Command csim_design CPU user time: 1.04 seconds. CPU system time: 0.26 seconds. Elapsed time: 1.29 seconds; current allocated memory: 0.000 MB.

### Reports/status files
build/paper_experiments/training/T7_deployable_training_bitstream/manifest.json
build/paper_experiments/training/T7_deployable_training_bitstream/reports/generated_cpp_validation.json
build/paper_experiments/training/T7_deployable_training_bitstream/reports/hardware_knob_contract.json
build/paper_experiments/training/T7_deployable_training_bitstream/reports/layer_backend_status.json
build/paper_experiments/training/T7_deployable_training_bitstream/reports/movement_contract_validation.json
build/paper_experiments/training/T7_deployable_training_bitstream/reports/numeric_validation.json
build/paper_experiments/training/T7_deployable_training_bitstream/reports/runtime_package_validation.json
build/paper_experiments/training/T7_deployable_training_bitstream/reports/vivado_bd_validation.json
build/paper_experiments/training/T7_deployable_training_bitstream/reports/vivado_validation_report.json
build/paper_experiments/training/T7_deployable_training_bitstream/runtime_package/manifest.json
build/paper_experiments/training/T7_deployable_training_bitstream/runtime_package/package_manifest.json
build/paper_experiments/training/T7_deployable_training_bitstream/runtime_package/runtime_package_validation.json
build/paper_experiments/training/T7_deployable_training_bitstream/vivado_bridge/vivado_bridge_manifest.json

## T8_real_fpga_training_curve_candidate

### compile log: paper_results/experiment_setup/compile_logs/18_T8_real_fpga_training_curve_candidate.log
14:HLS returncode       : 1
17:HLS csynth report    : None
52:Prediction artifacts:
61:  - Vivado allowed by fit: True
63:HLS artifacts       : available
65:  - hls_ok: False
66:  - hls_returncode: 1
69:  - artifact_metadata: hls_artifact_metadata.json
78:  - vivado_allowed_by_board_fit: True
85:Vivado bridge       : available
89:  - vivado_bridge_generated: True
90:  - vivado_synth_requested: True
91:  - vivado_impl_requested: True
96:Runtime package     : created
97:  - package: runtime_package/package_manifest.json
117:  - run_hls: failed
118:  - training_artifacts: done
119:  - vivado_project: done
121:  - runtime_package: done
134:[OK] Wrote artifacts to: /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T8_real_fpga_training_curve_candidate
139:      "name": "manifest_or_artifact_report_present",
149:      "name": "hls_ok",
154:      "name": "vivado_implemented",
164:      "name": "runtime_package_created",
169:      "name": "runtime_package_validated",
173:  "claim_level": "level_0_compiler_artifact",
174:  "configured_stage": "bitstream_package",
176:  "failed_checks": [
179:      "name": "hls_ok",
184:  "status": "failed"

### HLS stdout/stderr
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/hls/logs/vitis_hls_stdout.log:42:INFO: [HLS 200-1510] Running: csim_design -clean -argv /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T8_real_fpga_training_curve_candidate/input.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T8_real_fpga_training_curve_candidate/target.bin /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/build/paper_experiments/training/T8_real_fpga_training_curve_candidate/training_reference/weights_before_ref.bin . 
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/hls/logs/vitis_hls_stdout.log:43:INFO: [SIM 211-2] *************** CSIM start ***************
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/hls/logs/vitis_hls_stdout.log:44:INFO: [SIM 211-4] CSIM will launch GCC as the compiler.
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/hls/logs/vitis_hls_stdout.log:50:../../../../src/tb.cpp:159:25: error: missing terminating " character
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/hls/logs/vitis_hls_stdout.log:56:../../../../src/tb.cpp:160:1: error: missing terminating " character
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/hls/logs/vitis_hls_stdout.log:62:../../../../src/tb.cpp:166:25: error: missing terminating " character
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/hls/logs/vitis_hls_stdout.log:68:../../../../src/tb.cpp:167:1: error: missing terminating " character
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/hls/logs/vitis_hls_stdout.log:72:../../../../src/tb.cpp:161:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/hls/logs/vitis_hls_stdout.log:79:../../../../src/tb.cpp:168:21: error: expected ‘)’ before ‘;’ token
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/hls/logs/vitis_hls_stdout.log:86:make: *** [csim.mk:79: obj/tb.o] Error 1
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/hls/logs/vitis_hls_stdout.log:87:ERROR: [SIM 211-100] 'csim_design' failed: compilation error(s).
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/hls/logs/vitis_hls_stdout.log:88:INFO: [SIM 211-3] *************** CSIM finish ***************
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/hls/logs/vitis_hls_stdout.log:89:INFO: [HLS 200-111] Finished Command csim_design CPU user time: 1.06 seconds. CPU system time: 0.25 seconds. Elapsed time: 1.34 seconds; current allocated memory: 0.000 MB.

### Reports/status files
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/manifest.json
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/reports/generated_cpp_validation.json
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/reports/hardware_knob_contract.json
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/reports/layer_backend_status.json
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/reports/movement_contract_validation.json
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/reports/numeric_validation.json
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/reports/runtime_package_validation.json
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/reports/vivado_bd_validation.json
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/reports/vivado_validation_report.json
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/runtime_package/manifest.json
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/runtime_package/package_manifest.json
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/runtime_package/runtime_package_validation.json
build/paper_experiments/training/T8_real_fpga_training_curve_candidate/vivado_bridge/vivado_bridge_manifest.json

