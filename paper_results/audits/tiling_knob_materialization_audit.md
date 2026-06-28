# Tiling knob materialization audit

## Design directories
- `kv260_tiling_small`: `paper_experiments/full_pipeline_gate/sprint26_paper_matrix/runs/kv260_tiling_small`
- `kv260_tiling_medium`: `paper_experiments/full_pipeline_gate/sprint26_paper_matrix/runs/kv260_tiling_medium`
- `kv260_tiling_large`: `paper_experiments/full_pipeline_gate/sprint26_paper_matrix/runs/kv260_tiling_large`

## HLS source comparison
- identical HLS/source files: 102
- differing HLS/source files: 1

### Differing files
- `hls/fpgai_hls_proj/sol1/impl/ip/run_ippack.tcl`: unique normalized hashes = 3

### Interesting knob/materialization lines
## `kv260_tiling_small`
### `analysis/architecture_capabilities.json`
```text
architecture_capabilities.json:109:       "feature": "tiling",
```
```text
architecture_capabilities.json:123:       "detail": "Dense/Conv code generation emits tiled call sites for the requested tile sizes.",
```
```text
architecture_capabilities.json:148:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:149:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:155:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:156:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:252:       "feature": "tiling",
```
```text
architecture_capabilities.json:260:       "detail": "No tiling was requested.",
```
```text
architecture_capabilities.json:285:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:286:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:292:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:293:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:395:       "feature": "tiling",
```
```text
architecture_capabilities.json:409:       "detail": "Dense/Conv code generation emits tiled call sites for the requested tile sizes.",
```
```text
architecture_capabilities.json:434:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:435:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:441:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:442:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:538:       "feature": "tiling",
```
```text
architecture_capabilities.json:546:       "detail": "No tiling was requested.",
```
```text
architecture_capabilities.json:571:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:572:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:578:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:579:         "activation_region": "BRAM",
```
### `calibration/calibrated_model.json`
```text
calibrated_model.json:3:     "bram": 1.0,
```
```text
calibrated_model.json:14:     "bram",
```
### `calibration/compile_plan_for_calibration.json`
```text
compile_plan_for_calibration.json:11:   "global_resource_budget": {
```
```text
compile_plan_for_calibration.json:27:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:30:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:66:         "tiling": {
```
```text
compile_plan_for_calibration.json:100:       "tile": {
```
```text
compile_plan_for_calibration.json:121:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:124:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:159:         "tiling": {
```
```text
compile_plan_for_calibration.json:190:       "tile": {},
```
```text
compile_plan_for_calibration.json:207:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:210:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:246:         "tiling": {
```
```text
compile_plan_for_calibration.json:280:       "tile": {
```
```text
compile_plan_for_calibration.json:301:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:304:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:339:         "tiling": {
```
```text
compile_plan_for_calibration.json:370:       "tile": {},
```
```text
compile_plan_for_calibration.json:389:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:392:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:428:         "tiling": {
```
```text
compile_plan_for_calibration.json:462:       "tile": {
```
```text
compile_plan_for_calibration.json:483:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:486:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:521:         "tiling": {
```
```text
compile_plan_for_calibration.json:552:       "tile": {},
```
```text
compile_plan_for_calibration.json:569:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:572:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:608:         "tiling": {
```
```text
compile_plan_for_calibration.json:642:       "tile": {
```
```text
compile_plan_for_calibration.json:663:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:666:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:701:         "tiling": {
```
```text
compile_plan_for_calibration.json:732:       "tile": {},
```
```text
compile_plan_for_calibration.json:748:       "BRAM",
```
```text
compile_plan_for_calibration.json:749:       "URAM",
```
```text
compile_plan_for_calibration.json:753:     "array_partition_mode": "cyclic",
```
```text
compile_plan_for_calibration.json:773:     "policy_resource_awareness": {
```
```text
compile_plan_for_calibration.json:803:       "BRAM",
```
```text
compile_plan_for_calibration.json:804:       "URAM",
```
```text
compile_plan_for_calibration.json:807:     "weight_storage": "bram"
```
### `calibration/estimate_vs_hls.json`
```text
estimate_vs_hls.json:4:       "bram": 1.0,
```
```text
estimate_vs_hls.json:15:       "bram",
```
```text
estimate_vs_hls.json:26:       "bram": 0.0,
```
```text
estimate_vs_hls.json:33:       "bram": null,
```
```text
estimate_vs_hls.json:41:       "bram": 0.0,
```
### `design_space/results.json`
```text
results.json:5:     "resources": "operator_structural_v2",
```
```text
results.json:177:     "resource_estimation_mode": "analytical",
```
```text
results.json:199:         "optimization.parallel.array_partition_mode",
```
```text
results.json:202:         "optimization.tiling.dense",
```
```text
results.json:203:         "optimization.tiling.conv",
```
```text
results.json:212:         "resource_prediction",
```
```text
results.json:216:         "resource_score",
```
```text
results.json:221:       "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:385:     "resource_estimation_mode": "analytical",
```
```text
results.json:407:         "optimization.parallel.array_partition_mode",
```
```text
results.json:410:         "optimization.tiling.dense",
```
```text
results.json:411:         "optimization.tiling.conv",
```
```text
results.json:420:         "resource_prediction",
```
```text
results.json:424:         "resource_score",
```
```text
results.json:429:       "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:593:     "resource_estimation_mode": "analytical",
```
```text
results.json:615:         "optimization.parallel.array_partition_mode",
```
```text
results.json:618:         "optimization.tiling.dense",
```
```text
results.json:619:         "optimization.tiling.conv",
```
```text
results.json:628:         "resource_prediction",
```
```text
results.json:632:         "resource_score",
```
```text
results.json:637:       "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:802:       "resource_estimation_mode": "analytical",
```
```text
results.json:824:           "optimization.parallel.array_partition_mode",
```
```text
results.json:827:           "optimization.tiling.dense",
```
```text
results.json:828:           "optimization.tiling.conv",
```
```text
results.json:837:           "resource_prediction",
```
```text
results.json:841:           "resource_score",
```
```text
results.json:846:         "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:1010:       "resource_estimation_mode": "analytical",
```
```text
results.json:1032:           "optimization.parallel.array_partition_mode",
```
```text
results.json:1035:           "optimization.tiling.dense",
```
```text
results.json:1036:           "optimization.tiling.conv",
```
```text
results.json:1045:           "resource_prediction",
```
```text
results.json:1049:           "resource_score",
```
```text
results.json:1054:         "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:1218:       "resource_estimation_mode": "analytical",
```
```text
results.json:1240:           "optimization.parallel.array_partition_mode",
```
```text
results.json:1243:           "optimization.tiling.dense",
```
```text
results.json:1244:           "optimization.tiling.conv",
```
### `estimate_vs_hls/layer_validation/results.json`
```text
results.json:4:   "resource_model": "operator_structural_v4_inference_hls_sharing_training_problem_shared",
```
### `estimate_vs_hls/modules/results.json`
```text
results.json:44:   "top_resources": {
```
```text
results.json:50:   "unassigned_top_resources": {
```
```text
results.json:508:   "aggregation_note": "Primary operator totals exclude generated pipeline and loop helper reports. Helper resources are hierarchical subsets and must not be added to their parent function resources."
```
### `estimate_vs_hls/results.json`
```text
results.json:168:     "resource_estimation_mode": "analytical",
```
```text
results.json:190:         "optimization.parallel.array_partition_mode",
```
```text
results.json:193:         "optimization.tiling.dense",
```
```text
results.json:194:         "optimization.tiling.conv",
```
```text
results.json:203:         "resource_prediction",
```
```text
results.json:207:         "resource_score",
```
```text
results.json:212:       "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:320:     "resources": {
```
```text
results.json:331:     "resources": {
```
```text
results.json:344:             "resources": {
```
### `hls/codegen_meta.json`
```text
codegen_meta.json:35:   "tiling_resource_estimate": {
```
```text
codegen_meta.json:38:     "format": "fpgai.tiling_resource_model.v1",
```
```text
codegen_meta.json:39:     "path": "reports/tiling_resource_estimate.json",
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/a.g.ld.0.bc.clang.reflow.diag.yml`
```text
a.g.ld.0.bc.clang.reflow.diag.yml:852:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:872:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:892:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:910:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:934:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:954:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:974:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:992:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1010:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1030:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1041:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1052:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1063:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1074:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1085:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1096:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1107:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1118:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1129:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1140:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1151:   - Name:            Resource/Bind_Storage
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/activations.pp.0.cpp`
```text
activations.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
activations.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
activations.pp.0.cpp:6048: #pragma HLS ARRAY_PARTITION variable=temporary cyclic factor=4
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/autopilot.rtl.models.tcl`
```text
autopilot.rtl.models.tcl:55:       {MODELNAME deeplearn_layer_in_RAM_1P_BRAM_1R1W RTLNAME deeplearn_layer_in_RAM_1P_BRAM_1R1W BINDTYPE storage TYPE ram_1p IMPL bram LATENCY 2 ALLOW_PRAGMA 1}
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/batchnorm.pp.0.cpp`
```text
batchnorm.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
batchnorm.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
batchnorm.pp.0.cpp:15799: #pragma HLS array_partition variable=swap_table complete
```
```text
batchnorm.pp.0.cpp:15800: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15801: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15802: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15803: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K3 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15804: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K4 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15834: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15835: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15836: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15837: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K3 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15838: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K4 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15839: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15840: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15841: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15842: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K3 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15843: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K4 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15892: #pragma HLS array_partition variable=swap_table complete
```
```text
batchnorm.pp.0.cpp:15893: #pragma HLS BIND_STORAGE variable=second_order_float::sin_cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15894: #pragma HLS BIND_STORAGE variable=second_order_float::sin_cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15895: #pragma HLS BIND_STORAGE variable=second_order_float::sin_cos_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15922: #pragma HLS BIND_STORAGE variable=second_order_float::cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15923: #pragma HLS BIND_STORAGE variable=second_order_float::cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15924: #pragma HLS BIND_STORAGE variable=second_order_float::cos_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15925: #pragma HLS BIND_STORAGE variable=second_order_float::sin_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15926: #pragma HLS BIND_STORAGE variable=second_order_float::sin_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15927: #pragma HLS BIND_STORAGE variable=second_order_float::sin_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15963: #pragma HLS BIND_STORAGE variable=first_order_fixed_16::sin_cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15964: #pragma HLS BIND_STORAGE variable=first_order_fixed_16::sin_cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15993: #pragma HLS array_partition variable=swap_table complete
```
```text
batchnorm.pp.0.cpp:15994: #pragma HLS array_partition variable=neg_sin_table complete
```
```text
batchnorm.pp.0.cpp:15995: #pragma HLS array_partition variable=neg_cos_table complete
```
```text
batchnorm.pp.0.cpp:16065: #pragma HLS array_partition variable=swap_table complete
```
```text
batchnorm.pp.0.cpp:16066: #pragma HLS array_partition variable=neg_sin_table complete
```
```text
batchnorm.pp.0.cpp:16067: #pragma HLS array_partition variable=neg_cos_table complete
```
```text
batchnorm.pp.0.cpp:16136: #pragma HLS array_partition variable=neg_sin_table complete
```
```text
batchnorm.pp.0.cpp:16137: #pragma HLS array_partition variable=neg_cos_table complete
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml`
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:50:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:55:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:60:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:65:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:70:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:75:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:80:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:85:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:90:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:95:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:100:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:105:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:110:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:115:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:120:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:125:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:130:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:135:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:140:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:145:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:150:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:155:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:160:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:165:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:170:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:175:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/conv.pp.0.cpp`
```text
conv.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
conv.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
conv.pp.0.cpp:5792: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:5793: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:5794: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.pp.0.cpp:5817: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
conv.pp.0.cpp:6016: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6017: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6018: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6221: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6222: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6223: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6395: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6396: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/deeplearn.compgen.tcl`
```text
deeplearn.compgen.tcl:4: 	::AP::rtl_comp_handler deeplearn_layer_in_RAM_1P_BRAM_1R1W BINDTYPE {storage} TYPE {ram_1p} IMPL {bram} LATENCY 2 ALLOW_PRAGMA 1
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/deeplearn.pp.0.cpp`
```text
deeplearn.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
deeplearn.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
deeplearn.pp.0.cpp:184: #pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:185: #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:186: #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:190: #pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:202: #pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:203: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:204: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2
```
```text
deeplearn.pp.0.cpp:285: #pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:286: #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:287: #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=2
```
```text
deeplearn.pp.0.cpp:291: #pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:303: #pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:304: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:305: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2
```
```text
deeplearn.pp.0.cpp:9299: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9300: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9301: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9310: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
deeplearn.pp.0.cpp:9429: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9430: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9431: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9531: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9532: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9586: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9587: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9588: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9596: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
deeplearn.pp.0.cpp:9723: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9724: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9783: #pragma HLS ARRAY_PARTITION variable=B cyclic factor=PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9784: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9862: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9863: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9864: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9887: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
deeplearn.pp.0.cpp:10086: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:10087: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:10088: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml`
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:869:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:874:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:879:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:884:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:889:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:894:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:899:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:904:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:909:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:914:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:919:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:924:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:929:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:934:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:939:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:944:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:949:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:954:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:959:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:964:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:969:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:974:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:979:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:984:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:989:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:994:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2205:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2228:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2233:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2238:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2243:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/dense.pp.0.cpp`
```text
dense.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
dense.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
dense.pp.0.cpp:5785: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5786: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5787: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5796: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
dense.pp.0.cpp:5915: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5916: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5917: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6017: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6018: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6072: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6073: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6074: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6082: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
dense.pp.0.cpp:6209: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=PARTITION dim=1
```
```text
dense.pp.0.cpp:6210: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=PARTITION dim=1
```
```text
dense.pp.0.cpp:6269: #pragma HLS ARRAY_PARTITION variable=B cyclic factor=PARTITION dim=1
```
```text
dense.pp.0.cpp:6270: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=PARTITION dim=1
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/fpgai_params.pp.0.cpp`
```text
fpgai_params.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
fpgai_params.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/pool.pp.0.cpp`
```text
pool.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
pool.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
### `hls/fpgai_hls_proj/sol1/impl/ip/run_ippack.tcl`
```text
run_ippack.tcl:103: deeplearn_layer_in_RAM_1P_BRAM_1R1W
```
```text
run_ippack.tcl:483:         bram {
```
```text
run_ippack.tcl:484:             dict set properties bus_type_vlnv "xilinx.com:interface:bram:1.0"
```
```text
run_ippack.tcl:1559:         bram {
```
```text
run_ippack.tcl:1583:                 set current_bus_interface [add_bus_interface $core ${interface_name}_PORT$suffix bram master]
```
```text
run_ippack.tcl:1630:                 set current_bus_interface [add_bus_interface $core ${interface_name}_PORT$suffix bram master]
```
### `hls/fpgai_hls_proj/sol1/sol1_data.json`
```text
sol1_data.json:107:       "impl\/vhdl\/deeplearn_layer_in_RAM_1P_BRAM_1R1W.vhd",
```
```text
sol1_data.json:145:       "impl\/verilog\/deeplearn_layer_in_RAM_1P_BRAM_1R1W.v",
```
```text
sol1_data.json:772:           "URAM": "0",
```
```text
sol1_data.json:811:           "URAM": "0",
```
```text
sol1_data.json:850:           "URAM": "0",
```
```text
sol1_data.json:889:           "URAM": "0",
```
```text
sol1_data.json:928:           "URAM": "0",
```
```text
sol1_data.json:967:           "URAM": "0",
```
```text
sol1_data.json:999:           "URAM": "0",
```
```text
sol1_data.json:1031:           "URAM": "0",
```
```text
sol1_data.json:1063:           "URAM": "0",
```
```text
sol1_data.json:1102:           "URAM": "0",
```
```text
sol1_data.json:1141:           "URAM": "0",
```
```text
sol1_data.json:1173:           "URAM": "0",
```
```text
sol1_data.json:1205:           "URAM": "0",
```
### `hls/include/layers/activations.h`
```text
activations.h:300: #pragma HLS ARRAY_PARTITION variable=temporary cyclic factor=FPGAI_ACT_UNROLL
```
### `hls/include/layers/conv.h`
```text
conv.h:48: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.h:49: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.h:50: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.h:73: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
conv.h:272: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.h:273: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.h:274: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.h:477: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.h:478: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.h:479: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.h:651: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.h:652: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
### `hls/include/layers/dense.h`
```text
dense.h:41: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.h:42: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:43: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.h:52: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
dense.h:171: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.h:172: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:173: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.h:273: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:274: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:328: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:329: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.h:330: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.h:338: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
dense.h:465: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=PARTITION dim=1
```
```text
dense.h:466: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=PARTITION dim=1
```
```text
dense.h:525: #pragma HLS ARRAY_PARTITION variable=B cyclic factor=PARTITION dim=1
```
```text
dense.h:526: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=PARTITION dim=1
```
### `hls/reports/tiling_analysis.json`
```text
tiling_analysis.json:29:       "tile": {
```
```text
tiling_analysis.json:40:       "detail": "Tiling analysis is implemented for Dense and Conv layers.",
```
```text
tiling_analysis.json:44:       "tile": {
```
```text
tiling_analysis.json:73:       "tile": {
```
```text
tiling_analysis.json:84:       "detail": "Tiling analysis is implemented for Dense and Conv layers.",
```
```text
tiling_analysis.json:88:       "tile": {
```
### `hls/reports/tiling_performance_estimate.json`
```text
tiling_performance_estimate.json:5:     "note": "This is an analytical estimate for comparing tile choices. Final achieved II and latency should be taken from HLS reports.",
```
### `hls/reports/tiling_resource_estimate.json`
```text
tiling_resource_estimate.json:8:     "note": "This estimates local tile buffer storage only. It does not replace full HLS resource reports."
```
```text
tiling_resource_estimate.json:10:   "format": "fpgai.tiling_resource_model.v1",
```
### `hls/src/deeplearn.cpp`
```text
deeplearn.cpp:24: // FPGAI real dense tiling helper.
```
```text
deeplearn.cpp:50: #pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
```
```text
deeplearn.cpp:51: #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1
```
```text
deeplearn.cpp:52: #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.cpp:56: #pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
```
```text
deeplearn.cpp:68: #pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
```
```text
deeplearn.cpp:69: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
```
```text
deeplearn.cpp:70: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2
```
```text
deeplearn.cpp:151: #pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
```
```text
deeplearn.cpp:152: #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1
```
```text
deeplearn.cpp:153: #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=2
```
```text
deeplearn.cpp:157: #pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
```
```text
deeplearn.cpp:169: #pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
```
```text
deeplearn.cpp:170: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
```
```text
deeplearn.cpp:171: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2
```
```text
deeplearn.cpp:230: //   0: layer_0 (Dense) ii=2 pe=4 simd=4 part_in=? part_out=? part_w=? tile={'sizes': {'in': 8, 'out': 4}} memory={'weight_mode': 'embedded', 'activation_mode': 'buffer', 'weight_region': 'BRAM', 'activation_region': 'BRAM', 'gradient_region': None} sig=19d5c1026e6172b78199490fb19ed2f4609276bcb58ffb3247f4cd73ccb14b1b
```
```text
deeplearn.cpp:231: //   1: layer_1 (Relu) ii=2 pe=4 simd=1 part_in=? part_out=? part_w=? tile={'sizes': {}} memory={'weight_mode': 'embedded', 'activation_mode': 'stream', 'weight_region': 'BRAM', 'activation_region': 'BRAM', 'gradient_region': None} sig=09bc099ed45f804c1a3df2bbd5f907af96944d4c5fede1242504710155c9d925
```
```text
deeplearn.cpp:232: //   2: layer_2 (Dense) ii=2 pe=4 simd=4 part_in=? part_out=? part_w=? tile={'sizes': {'in': 4, 'out': 2}} memory={'weight_mode': 'embedded', 'activation_mode': 'buffer', 'weight_region': 'BRAM', 'activation_region': 'BRAM', 'gradient_region': None} sig=888ba83c641c72204d227163f339c47de6aa22bcdc4c26f18d74e1c1583a37be
```
```text
deeplearn.cpp:233: //   3: layer_3 (Softmax) ii=2 pe=4 simd=1 part_in=? part_out=? part_w=? tile={'sizes': {}} memory={'weight_mode': 'embedded', 'activation_mode': 'stream', 'weight_region': 'BRAM', 'activation_region': 'BRAM', 'gradient_region': None} sig=f21e31cd34d79fb173904136638eea3644adf2d088771862838d6e00b9bacbbb
```
```text
deeplearn.cpp:315: #pragma HLS BIND_STORAGE variable=layer_in type=ram_1p impl=bram
```
```text
deeplearn.cpp:337:     // tile: {'in': 8, 'out': 4}
```
```text
deeplearn.cpp:339:     // output placement: BRAM / size=16 bytes
```
```text
deeplearn.cpp:342: #pragma HLS BIND_STORAGE variable=layer_0_out type=ram_1p impl=bram
```
```text
deeplearn.cpp:344:     // Weight cache placement: BRAM / size=weight_cache bytes
```
```text
deeplearn.cpp:345:     // FPGAI storage binding requested for embedded parameter W0/B0; top-level BIND_STORAGE is disabled for initialized W/B arrays because Vitis HLS rejects initialized global/static parameter arrays bound this way. Use runtime-loaded URAM buffers for real URAM parameter storage.
```
```text
deeplearn.cpp:358:     // tile: {}
```
```text
deeplearn.cpp:360:     // output placement: BRAM / size=16 bytes
```
```text
deeplearn.cpp:363: #pragma HLS BIND_STORAGE variable=layer_1_out type=ram_1p impl=bram
```
```text
deeplearn.cpp:376:     // tile: {'in': 4, 'out': 2}
```
```text
deeplearn.cpp:378:     // output placement: BRAM / size=8 bytes
```
```text
deeplearn.cpp:381: #pragma HLS BIND_STORAGE variable=layer_2_out type=ram_1p impl=bram
```
```text
deeplearn.cpp:383:     // Weight cache placement: BRAM / size=weight_cache bytes
```
```text
deeplearn.cpp:384:     // FPGAI storage binding requested for embedded parameter W1/B1; top-level BIND_STORAGE is disabled for initialized W/B arrays because Vitis HLS rejects initialized global/static parameter arrays bound this way. Use runtime-loaded URAM buffers for real URAM parameter storage.
```
```text
deeplearn.cpp:397:     // tile: {}
```
```text
deeplearn.cpp:402: #pragma HLS BIND_STORAGE variable=layer_3_out type=ram_1p impl=bram
```
### `hls/src/fpgai_params.cpp`
```text
fpgai_params.cpp:18: // FPGAI storage binding: parameter arrays requested for BRAM.
```
```text
fpgai_params.cpp:19: // FPGAI note: file-scope BIND_STORAGE pragmas are disabled because Vitis HLS csynth rejects them on global const arrays.
```
```text
fpgai_params.cpp:20: // FPGAI storage binding: bram requested for W0; file-scope BIND_STORAGE disabled.
```
```text
fpgai_params.cpp:21: // FPGAI storage binding: bram requested for B0; file-scope BIND_STORAGE disabled.
```
```text
fpgai_params.cpp:22: // FPGAI storage binding: bram requested for W1; file-scope BIND_STORAGE disabled.
```
```text
fpgai_params.cpp:23: // FPGAI storage binding: bram requested for B1; file-scope BIND_STORAGE disabled.
```
### `hls_artifact_metadata.json`
```text
hls_artifact_metadata.json:46:         "tiling": {
```
```text
hls_artifact_metadata.json:59:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:60:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:104:         "tiling": {
```
```text
hls_artifact_metadata.json:114:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:115:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:160:         "tiling": {
```
```text
hls_artifact_metadata.json:173:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:174:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:218:         "tiling": {
```
```text
hls_artifact_metadata.json:228:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:229:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:1728:       "path": "hls/reports/tiling_resource_estimate.json",
```
```text
hls_artifact_metadata.json:1878:       "path": "reports/resource_prediction.json",
```
### `ir/comm_plan.json`
```text
comm_plan.json:52:         "region": "BRAM",
```
```text
comm_plan.json:82:         "region": "BRAM",
```
```text
comm_plan.json:112:         "region": "BRAM",
```
```text
comm_plan.json:142:         "region": "BRAM",
```
```text
comm_plan.json:172:         "region": "BRAM",
```
```text
comm_plan.json:202:         "region": "BRAM",
```
```text
comm_plan.json:232:         "region": "BRAM",
```
### `ir/compile_plan.json`
```text
compile_plan.json:18:       "tile": {
```
```text
compile_plan.json:87:         "tiling": {
```
```text
compile_plan.json:100:           "weight_region": "BRAM",
```
```text
compile_plan.json:101:           "activation_region": "BRAM",
```
```text
compile_plan.json:113:       "tile": {},
```
```text
compile_plan.json:177:         "tiling": {
```
```text
compile_plan.json:187:           "weight_region": "BRAM",
```
```text
compile_plan.json:188:           "activation_region": "BRAM",
```
```text
compile_plan.json:200:       "tile": {
```
```text
compile_plan.json:269:         "tiling": {
```
```text
compile_plan.json:282:           "weight_region": "BRAM",
```
```text
compile_plan.json:283:           "activation_region": "BRAM",
```
```text
compile_plan.json:295:       "tile": {},
```
```text
compile_plan.json:359:         "tiling": {
```
```text
compile_plan.json:369:           "weight_region": "BRAM",
```
```text
compile_plan.json:370:           "activation_region": "BRAM",
```
```text
compile_plan.json:377:   "global_resource_budget": {
```
```text
compile_plan.json:386:     "weight_storage": "bram",
```
```text
compile_plan.json:397:     "policy_resource_awareness": {
```
```text
compile_plan.json:430:       "BRAM",
```
```text
compile_plan.json:431:       "URAM",
```
```text
compile_plan.json:435:       "BRAM",
```
```text
compile_plan.json:436:       "URAM",
```
```text
compile_plan.json:443:     "array_partition_mode": "cyclic",
```
### `ir/memory_plan.json`
```text
memory_plan.json:21:       "region": "BRAM",
```
```text
memory_plan.json:39:       "region": "BRAM",
```
```text
memory_plan.json:57:       "region": "BRAM",
```
```text
memory_plan.json:66:         "tile": {
```
```text
memory_plan.json:82:       "region": "BRAM",
```
```text
memory_plan.json:91:         "tile": {},
```
```text
memory_plan.json:103:       "region": "BRAM",
```
```text
memory_plan.json:121:       "region": "BRAM",
```
```text
memory_plan.json:139:       "region": "BRAM",
```
```text
memory_plan.json:148:         "tile": {
```
```text
memory_plan.json:179:     "BRAM": 224
```
```text
memory_plan.json:186:       "BRAM",
```
```text
memory_plan.json:187:       "URAM",
```
```text
memory_plan.json:191:       "BRAM",
```
```text
memory_plan.json:192:       "URAM",
```
### `manifest.json`
```text
manifest.json:137:         "feature": "tiling",
```
```text
manifest.json:151:         "detail": "Dense/Conv code generation emits tiled call sites for the requested tile sizes.",
```
```text
manifest.json:176:           "weight_region": "BRAM",
```
```text
manifest.json:177:           "activation_region": "BRAM",
```
```text
manifest.json:183:           "weight_region": "BRAM",
```
```text
manifest.json:184:           "activation_region": "BRAM",
```
```text
manifest.json:280:         "feature": "tiling",
```
```text
manifest.json:288:         "detail": "No tiling was requested.",
```
```text
manifest.json:313:           "weight_region": "BRAM",
```
```text
manifest.json:314:           "activation_region": "BRAM",
```
```text
manifest.json:320:           "weight_region": "BRAM",
```
```text
manifest.json:321:           "activation_region": "BRAM",
```
```text
manifest.json:423:         "feature": "tiling",
```
```text
manifest.json:437:         "detail": "Dense/Conv code generation emits tiled call sites for the requested tile sizes.",
```
```text
manifest.json:462:           "weight_region": "BRAM",
```
```text
manifest.json:463:           "activation_region": "BRAM",
```
```text
manifest.json:469:           "weight_region": "BRAM",
```
```text
manifest.json:470:           "activation_region": "BRAM",
```
```text
manifest.json:566:         "feature": "tiling",
```
```text
manifest.json:574:         "detail": "No tiling was requested.",
```
```text
manifest.json:599:           "weight_region": "BRAM",
```
```text
manifest.json:600:           "activation_region": "BRAM",
```
```text
manifest.json:606:           "weight_region": "BRAM",
```
```text
manifest.json:607:           "activation_region": "BRAM",
```
```text
manifest.json:618:     "resource_prediction_json": "/home/umutcanaltin/Desktop/github_projects/fpgai_compiler/paper_experiments/full_pipeline_gate/sprint26_paper_matrix/runs/kv260_tiling_small/reports/resource_prediction.json",
```
```text
manifest.json:690:     "BRAM": 224
```
```text
manifest.json:827:       "resources": "operator_structural_v2",
```
```text
manifest.json:857:         "optimization.parallel.array_partition_mode",
```
```text
manifest.json:860:         "optimization.tiling.dense",
```
```text
manifest.json:861:         "optimization.tiling.conv",
```
```text
manifest.json:870:         "resource_prediction",
```
```text
manifest.json:874:         "resource_score",
```
```text
manifest.json:1039:       "resource_estimation_mode": "analytical",
```
```text
manifest.json:1061:           "optimization.parallel.array_partition_mode",
```
```text
manifest.json:1064:           "optimization.tiling.dense",
```
```text
manifest.json:1065:           "optimization.tiling.conv",
```
```text
manifest.json:1074:           "resource_prediction",
```
```text
manifest.json:1078:           "resource_score",
```
```text
manifest.json:1083:         "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
manifest.json:1247:       "resource_estimation_mode": "analytical",
```
### `reports/board_fit.json`
```text
board_fit.json:6:     "limiting_resource": "target_clock_mhz",
```
```text
board_fit.json:9:     "resources": {
```
```text
board_fit.json:52:       "uram": {
```
```text
board_fit.json:64:   "normalized_resources": {
```
### `reports/hardware_knob_contract.json`
```text
hardware_knob_contract.json:62:         "generated HLS template args and ARRAY_PARTITION factors"
```
```text
hardware_knob_contract.json:73:         "planner.policy.array_partition_mode",
```
```text
hardware_knob_contract.json:75:         "HLS ARRAY_PARTITION mode where supported"
```
```text
hardware_knob_contract.json:79:       "path": "optimization.parallel.array_partition_mode",
```
```text
hardware_knob_contract.json:113:         "planner dense tile selection",
```
```text
hardware_knob_contract.json:114:         "layer_plan.architecture.tiling",
```
```text
hardware_knob_contract.json:121:       "note": "Layer-specific tiling can override global dense tiling.",
```
```text
hardware_knob_contract.json:122:       "path": "optimization.tiling.dense",
```
```text
hardware_knob_contract.json:133:         "planner conv tile selection",
```
```text
hardware_knob_contract.json:134:         "layer_plan.architecture.tiling",
```
```text
hardware_knob_contract.json:138:       "note": "Layer-specific tiling can override global conv tiling.",
```
```text
hardware_knob_contract.json:139:       "path": "optimization.tiling.conv",
```
```text
hardware_knob_contract.json:150:         "planner layer-specific tile selection",
```
```text
hardware_knob_contract.json:151:         "layer_plan.architecture.tiling for matching layer names"
```
```text
hardware_knob_contract.json:154:       "note": "Manual layer entries have priority over global tiling defaults.",
```
```text
hardware_knob_contract.json:155:       "path": "optimization.tiling.layers",
```
```text
hardware_knob_contract.json:166:       "effective": "bram",
```
```text
hardware_knob_contract.json:169:       "requested": "bram",
```
```text
hardware_knob_contract.json:179:         "BRAM",
```
```text
hardware_knob_contract.json:180:         "URAM",
```
```text
hardware_knob_contract.json:195:         "BRAM",
```
```text
hardware_knob_contract.json:196:         "URAM",
```
### `reports/resource_prediction.json`
```text
resource_prediction.json:37:           "activation_region": "BRAM",
```
```text
resource_prediction.json:44:           "weight_region": "BRAM"
```
```text
resource_prediction.json:62:         "tiling": {
```
```text
resource_prediction.json:101:           "activation_region": "BRAM",
```
```text
resource_prediction.json:108:           "weight_region": "BRAM"
```
```text
resource_prediction.json:126:         "tiling": {},
```
```text
resource_prediction.json:160:           "activation_region": "BRAM",
```
```text
resource_prediction.json:167:           "weight_region": "BRAM"
```
```text
resource_prediction.json:185:         "tiling": {
```
```text
resource_prediction.json:224:           "activation_region": "BRAM",
```
```text
resource_prediction.json:231:           "weight_region": "BRAM"
```
```text
resource_prediction.json:249:         "tiling": {},
```
```text
resource_prediction.json:309:           "activation_region": "BRAM",
```
```text
resource_prediction.json:316:           "weight_region": "BRAM"
```
```text
resource_prediction.json:334:         "tiling": {
```
```text
resource_prediction.json:407:       "resource_components": {
```
```text
resource_prediction.json:465:           "activation_region": "BRAM",
```
```text
resource_prediction.json:472:           "weight_region": "BRAM"
```
```text
resource_prediction.json:490:         "tiling": {},
```
```text
resource_prediction.json:551:       "resource_components": {
```
```text
resource_prediction.json:600:           "activation_region": "BRAM",
```
```text
resource_prediction.json:607:           "weight_region": "BRAM"
```
```text
resource_prediction.json:625:         "tiling": {
```
```text
resource_prediction.json:698:       "resource_components": {
```
```text
resource_prediction.json:756:           "activation_region": "BRAM",
```
```text
resource_prediction.json:763:           "weight_region": "BRAM"
```
```text
resource_prediction.json:781:         "tiling": {},
```
```text
resource_prediction.json:843:       "resource_components": {
```
```text
resource_prediction.json:859:   "prediction_kind": "pre_hls_resource_estimate",
```
```text
resource_prediction.json:933:         "activation_region": "BRAM",
```
```text
resource_prediction.json:940:         "weight_region": "BRAM"
```
```text
resource_prediction.json:958:       "tiling": {
```
```text
resource_prediction.json:1031:     "resource_components": {
```
```text
resource_prediction.json:1088:         "activation_region": "BRAM",
```
```text
resource_prediction.json:1095:         "weight_region": "BRAM"
```
```text
resource_prediction.json:1113:       "tiling": {
```
```text
resource_prediction.json:1186:     "resource_components": {
```
```text
resource_prediction.json:1243:         "activation_region": "BRAM",
```
```text
resource_prediction.json:1250:         "weight_region": "BRAM"
```
```text
resource_prediction.json:1268:       "tiling": {
```
### `runtime_package/hls/hls_artifact_metadata.json`
```text
hls_artifact_metadata.json:46:         "tiling": {
```
```text
hls_artifact_metadata.json:59:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:60:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:104:         "tiling": {
```
```text
hls_artifact_metadata.json:114:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:115:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:160:         "tiling": {
```
```text
hls_artifact_metadata.json:173:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:174:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:218:         "tiling": {
```
```text
hls_artifact_metadata.json:228:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:229:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:1728:       "path": "hls/reports/tiling_resource_estimate.json",
```
```text
hls_artifact_metadata.json:1878:       "path": "reports/resource_prediction.json",
```
## `kv260_tiling_medium`
### `analysis/architecture_capabilities.json`
```text
architecture_capabilities.json:109:       "feature": "tiling",
```
```text
architecture_capabilities.json:123:       "detail": "Dense/Conv code generation emits tiled call sites for the requested tile sizes.",
```
```text
architecture_capabilities.json:148:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:149:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:155:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:156:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:252:       "feature": "tiling",
```
```text
architecture_capabilities.json:260:       "detail": "No tiling was requested.",
```
```text
architecture_capabilities.json:285:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:286:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:292:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:293:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:395:       "feature": "tiling",
```
```text
architecture_capabilities.json:409:       "detail": "Dense/Conv code generation emits tiled call sites for the requested tile sizes.",
```
```text
architecture_capabilities.json:434:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:435:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:441:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:442:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:538:       "feature": "tiling",
```
```text
architecture_capabilities.json:546:       "detail": "No tiling was requested.",
```
```text
architecture_capabilities.json:571:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:572:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:578:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:579:         "activation_region": "BRAM",
```
### `calibration/calibrated_model.json`
```text
calibrated_model.json:3:     "bram": 1.0,
```
```text
calibrated_model.json:14:     "bram",
```
### `calibration/compile_plan_for_calibration.json`
```text
compile_plan_for_calibration.json:11:   "global_resource_budget": {
```
```text
compile_plan_for_calibration.json:27:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:30:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:66:         "tiling": {
```
```text
compile_plan_for_calibration.json:100:       "tile": {
```
```text
compile_plan_for_calibration.json:121:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:124:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:159:         "tiling": {
```
```text
compile_plan_for_calibration.json:190:       "tile": {},
```
```text
compile_plan_for_calibration.json:207:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:210:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:246:         "tiling": {
```
```text
compile_plan_for_calibration.json:280:       "tile": {
```
```text
compile_plan_for_calibration.json:301:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:304:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:339:         "tiling": {
```
```text
compile_plan_for_calibration.json:370:       "tile": {},
```
```text
compile_plan_for_calibration.json:389:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:392:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:428:         "tiling": {
```
```text
compile_plan_for_calibration.json:462:       "tile": {
```
```text
compile_plan_for_calibration.json:483:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:486:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:521:         "tiling": {
```
```text
compile_plan_for_calibration.json:552:       "tile": {},
```
```text
compile_plan_for_calibration.json:569:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:572:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:608:         "tiling": {
```
```text
compile_plan_for_calibration.json:642:       "tile": {
```
```text
compile_plan_for_calibration.json:663:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:666:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:701:         "tiling": {
```
```text
compile_plan_for_calibration.json:732:       "tile": {},
```
```text
compile_plan_for_calibration.json:748:       "BRAM",
```
```text
compile_plan_for_calibration.json:749:       "URAM",
```
```text
compile_plan_for_calibration.json:753:     "array_partition_mode": "cyclic",
```
```text
compile_plan_for_calibration.json:773:     "policy_resource_awareness": {
```
```text
compile_plan_for_calibration.json:803:       "BRAM",
```
```text
compile_plan_for_calibration.json:804:       "URAM",
```
```text
compile_plan_for_calibration.json:807:     "weight_storage": "bram"
```
### `calibration/estimate_vs_hls.json`
```text
estimate_vs_hls.json:4:       "bram": 1.0,
```
```text
estimate_vs_hls.json:15:       "bram",
```
```text
estimate_vs_hls.json:26:       "bram": 0.0,
```
```text
estimate_vs_hls.json:33:       "bram": null,
```
```text
estimate_vs_hls.json:41:       "bram": 0.0,
```
### `design_space/results.json`
```text
results.json:5:     "resources": "operator_structural_v2",
```
```text
results.json:177:     "resource_estimation_mode": "analytical",
```
```text
results.json:199:         "optimization.parallel.array_partition_mode",
```
```text
results.json:202:         "optimization.tiling.dense",
```
```text
results.json:203:         "optimization.tiling.conv",
```
```text
results.json:212:         "resource_prediction",
```
```text
results.json:216:         "resource_score",
```
```text
results.json:221:       "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:385:     "resource_estimation_mode": "analytical",
```
```text
results.json:407:         "optimization.parallel.array_partition_mode",
```
```text
results.json:410:         "optimization.tiling.dense",
```
```text
results.json:411:         "optimization.tiling.conv",
```
```text
results.json:420:         "resource_prediction",
```
```text
results.json:424:         "resource_score",
```
```text
results.json:429:       "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:593:     "resource_estimation_mode": "analytical",
```
```text
results.json:615:         "optimization.parallel.array_partition_mode",
```
```text
results.json:618:         "optimization.tiling.dense",
```
```text
results.json:619:         "optimization.tiling.conv",
```
```text
results.json:628:         "resource_prediction",
```
```text
results.json:632:         "resource_score",
```
```text
results.json:637:       "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:802:       "resource_estimation_mode": "analytical",
```
```text
results.json:824:           "optimization.parallel.array_partition_mode",
```
```text
results.json:827:           "optimization.tiling.dense",
```
```text
results.json:828:           "optimization.tiling.conv",
```
```text
results.json:837:           "resource_prediction",
```
```text
results.json:841:           "resource_score",
```
```text
results.json:846:         "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:1010:       "resource_estimation_mode": "analytical",
```
```text
results.json:1032:           "optimization.parallel.array_partition_mode",
```
```text
results.json:1035:           "optimization.tiling.dense",
```
```text
results.json:1036:           "optimization.tiling.conv",
```
```text
results.json:1045:           "resource_prediction",
```
```text
results.json:1049:           "resource_score",
```
```text
results.json:1054:         "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:1218:       "resource_estimation_mode": "analytical",
```
```text
results.json:1240:           "optimization.parallel.array_partition_mode",
```
```text
results.json:1243:           "optimization.tiling.dense",
```
```text
results.json:1244:           "optimization.tiling.conv",
```
### `estimate_vs_hls/layer_validation/results.json`
```text
results.json:4:   "resource_model": "operator_structural_v4_inference_hls_sharing_training_problem_shared",
```
### `estimate_vs_hls/modules/results.json`
```text
results.json:44:   "top_resources": {
```
```text
results.json:50:   "unassigned_top_resources": {
```
```text
results.json:508:   "aggregation_note": "Primary operator totals exclude generated pipeline and loop helper reports. Helper resources are hierarchical subsets and must not be added to their parent function resources."
```
### `estimate_vs_hls/results.json`
```text
results.json:168:     "resource_estimation_mode": "analytical",
```
```text
results.json:190:         "optimization.parallel.array_partition_mode",
```
```text
results.json:193:         "optimization.tiling.dense",
```
```text
results.json:194:         "optimization.tiling.conv",
```
```text
results.json:203:         "resource_prediction",
```
```text
results.json:207:         "resource_score",
```
```text
results.json:212:       "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:320:     "resources": {
```
```text
results.json:331:     "resources": {
```
```text
results.json:344:             "resources": {
```
### `hls/codegen_meta.json`
```text
codegen_meta.json:35:   "tiling_resource_estimate": {
```
```text
codegen_meta.json:38:     "format": "fpgai.tiling_resource_model.v1",
```
```text
codegen_meta.json:39:     "path": "reports/tiling_resource_estimate.json",
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/a.g.ld.0.bc.clang.reflow.diag.yml`
```text
a.g.ld.0.bc.clang.reflow.diag.yml:852:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:872:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:892:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:910:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:934:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:954:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:974:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:992:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1010:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1030:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1041:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1052:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1063:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1074:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1085:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1096:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1107:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1118:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1129:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1140:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1151:   - Name:            Resource/Bind_Storage
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/activations.pp.0.cpp`
```text
activations.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
activations.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
activations.pp.0.cpp:6048: #pragma HLS ARRAY_PARTITION variable=temporary cyclic factor=4
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/autopilot.rtl.models.tcl`
```text
autopilot.rtl.models.tcl:55:       {MODELNAME deeplearn_layer_in_RAM_1P_BRAM_1R1W RTLNAME deeplearn_layer_in_RAM_1P_BRAM_1R1W BINDTYPE storage TYPE ram_1p IMPL bram LATENCY 2 ALLOW_PRAGMA 1}
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/batchnorm.pp.0.cpp`
```text
batchnorm.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
batchnorm.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
batchnorm.pp.0.cpp:15799: #pragma HLS array_partition variable=swap_table complete
```
```text
batchnorm.pp.0.cpp:15800: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15801: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15802: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15803: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K3 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15804: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K4 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15834: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15835: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15836: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15837: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K3 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15838: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K4 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15839: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15840: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15841: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15842: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K3 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15843: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K4 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15892: #pragma HLS array_partition variable=swap_table complete
```
```text
batchnorm.pp.0.cpp:15893: #pragma HLS BIND_STORAGE variable=second_order_float::sin_cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15894: #pragma HLS BIND_STORAGE variable=second_order_float::sin_cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15895: #pragma HLS BIND_STORAGE variable=second_order_float::sin_cos_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15922: #pragma HLS BIND_STORAGE variable=second_order_float::cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15923: #pragma HLS BIND_STORAGE variable=second_order_float::cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15924: #pragma HLS BIND_STORAGE variable=second_order_float::cos_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15925: #pragma HLS BIND_STORAGE variable=second_order_float::sin_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15926: #pragma HLS BIND_STORAGE variable=second_order_float::sin_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15927: #pragma HLS BIND_STORAGE variable=second_order_float::sin_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15963: #pragma HLS BIND_STORAGE variable=first_order_fixed_16::sin_cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15964: #pragma HLS BIND_STORAGE variable=first_order_fixed_16::sin_cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15993: #pragma HLS array_partition variable=swap_table complete
```
```text
batchnorm.pp.0.cpp:15994: #pragma HLS array_partition variable=neg_sin_table complete
```
```text
batchnorm.pp.0.cpp:15995: #pragma HLS array_partition variable=neg_cos_table complete
```
```text
batchnorm.pp.0.cpp:16065: #pragma HLS array_partition variable=swap_table complete
```
```text
batchnorm.pp.0.cpp:16066: #pragma HLS array_partition variable=neg_sin_table complete
```
```text
batchnorm.pp.0.cpp:16067: #pragma HLS array_partition variable=neg_cos_table complete
```
```text
batchnorm.pp.0.cpp:16136: #pragma HLS array_partition variable=neg_sin_table complete
```
```text
batchnorm.pp.0.cpp:16137: #pragma HLS array_partition variable=neg_cos_table complete
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml`
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:50:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:55:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:60:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:65:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:70:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:75:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:80:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:85:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:90:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:95:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:100:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:105:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:110:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:115:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:120:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:125:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:130:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:135:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:140:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:145:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:150:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:155:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:160:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:165:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:170:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:175:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/conv.pp.0.cpp`
```text
conv.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
conv.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
conv.pp.0.cpp:5792: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:5793: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:5794: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.pp.0.cpp:5817: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
conv.pp.0.cpp:6016: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6017: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6018: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6221: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6222: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6223: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6395: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6396: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/deeplearn.compgen.tcl`
```text
deeplearn.compgen.tcl:4: 	::AP::rtl_comp_handler deeplearn_layer_in_RAM_1P_BRAM_1R1W BINDTYPE {storage} TYPE {ram_1p} IMPL {bram} LATENCY 2 ALLOW_PRAGMA 1
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/deeplearn.pp.0.cpp`
```text
deeplearn.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
deeplearn.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
deeplearn.pp.0.cpp:184: #pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:185: #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:186: #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:190: #pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:202: #pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:203: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:204: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2
```
```text
deeplearn.pp.0.cpp:285: #pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:286: #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:287: #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=2
```
```text
deeplearn.pp.0.cpp:291: #pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:303: #pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:304: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:305: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2
```
```text
deeplearn.pp.0.cpp:9299: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9300: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9301: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9310: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
deeplearn.pp.0.cpp:9429: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9430: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9431: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9531: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9532: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9586: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9587: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9588: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9596: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
deeplearn.pp.0.cpp:9723: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9724: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9783: #pragma HLS ARRAY_PARTITION variable=B cyclic factor=PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9784: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9862: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9863: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9864: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9887: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
deeplearn.pp.0.cpp:10086: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:10087: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:10088: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml`
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:869:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:874:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:879:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:884:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:889:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:894:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:899:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:904:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:909:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:914:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:919:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:924:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:929:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:934:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:939:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:944:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:949:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:954:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:959:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:964:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:969:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:974:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:979:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:984:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:989:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:994:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2205:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2228:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2233:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2238:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2243:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/dense.pp.0.cpp`
```text
dense.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
dense.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
dense.pp.0.cpp:5785: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5786: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5787: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5796: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
dense.pp.0.cpp:5915: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5916: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5917: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6017: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6018: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6072: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6073: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6074: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6082: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
dense.pp.0.cpp:6209: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=PARTITION dim=1
```
```text
dense.pp.0.cpp:6210: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=PARTITION dim=1
```
```text
dense.pp.0.cpp:6269: #pragma HLS ARRAY_PARTITION variable=B cyclic factor=PARTITION dim=1
```
```text
dense.pp.0.cpp:6270: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=PARTITION dim=1
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/fpgai_params.pp.0.cpp`
```text
fpgai_params.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
fpgai_params.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/pool.pp.0.cpp`
```text
pool.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
pool.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
### `hls/fpgai_hls_proj/sol1/impl/ip/run_ippack.tcl`
```text
run_ippack.tcl:103: deeplearn_layer_in_RAM_1P_BRAM_1R1W
```
```text
run_ippack.tcl:483:         bram {
```
```text
run_ippack.tcl:484:             dict set properties bus_type_vlnv "xilinx.com:interface:bram:1.0"
```
```text
run_ippack.tcl:1559:         bram {
```
```text
run_ippack.tcl:1583:                 set current_bus_interface [add_bus_interface $core ${interface_name}_PORT$suffix bram master]
```
```text
run_ippack.tcl:1630:                 set current_bus_interface [add_bus_interface $core ${interface_name}_PORT$suffix bram master]
```
### `hls/fpgai_hls_proj/sol1/sol1_data.json`
```text
sol1_data.json:107:       "impl\/vhdl\/deeplearn_layer_in_RAM_1P_BRAM_1R1W.vhd",
```
```text
sol1_data.json:145:       "impl\/verilog\/deeplearn_layer_in_RAM_1P_BRAM_1R1W.v",
```
```text
sol1_data.json:772:           "URAM": "0",
```
```text
sol1_data.json:811:           "URAM": "0",
```
```text
sol1_data.json:850:           "URAM": "0",
```
```text
sol1_data.json:889:           "URAM": "0",
```
```text
sol1_data.json:928:           "URAM": "0",
```
```text
sol1_data.json:967:           "URAM": "0",
```
```text
sol1_data.json:999:           "URAM": "0",
```
```text
sol1_data.json:1031:           "URAM": "0",
```
```text
sol1_data.json:1063:           "URAM": "0",
```
```text
sol1_data.json:1102:           "URAM": "0",
```
```text
sol1_data.json:1141:           "URAM": "0",
```
```text
sol1_data.json:1173:           "URAM": "0",
```
```text
sol1_data.json:1205:           "URAM": "0",
```
### `hls/include/layers/activations.h`
```text
activations.h:300: #pragma HLS ARRAY_PARTITION variable=temporary cyclic factor=FPGAI_ACT_UNROLL
```
### `hls/include/layers/conv.h`
```text
conv.h:48: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.h:49: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.h:50: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.h:73: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
conv.h:272: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.h:273: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.h:274: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.h:477: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.h:478: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.h:479: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.h:651: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.h:652: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
### `hls/include/layers/dense.h`
```text
dense.h:41: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.h:42: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:43: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.h:52: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
dense.h:171: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.h:172: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:173: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.h:273: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:274: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:328: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:329: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.h:330: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.h:338: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
dense.h:465: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=PARTITION dim=1
```
```text
dense.h:466: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=PARTITION dim=1
```
```text
dense.h:525: #pragma HLS ARRAY_PARTITION variable=B cyclic factor=PARTITION dim=1
```
```text
dense.h:526: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=PARTITION dim=1
```
### `hls/reports/tiling_analysis.json`
```text
tiling_analysis.json:29:       "tile": {
```
```text
tiling_analysis.json:40:       "detail": "Tiling analysis is implemented for Dense and Conv layers.",
```
```text
tiling_analysis.json:44:       "tile": {
```
```text
tiling_analysis.json:73:       "tile": {
```
```text
tiling_analysis.json:84:       "detail": "Tiling analysis is implemented for Dense and Conv layers.",
```
```text
tiling_analysis.json:88:       "tile": {
```
### `hls/reports/tiling_performance_estimate.json`
```text
tiling_performance_estimate.json:5:     "note": "This is an analytical estimate for comparing tile choices. Final achieved II and latency should be taken from HLS reports.",
```
### `hls/reports/tiling_resource_estimate.json`
```text
tiling_resource_estimate.json:8:     "note": "This estimates local tile buffer storage only. It does not replace full HLS resource reports."
```
```text
tiling_resource_estimate.json:10:   "format": "fpgai.tiling_resource_model.v1",
```
### `hls/src/deeplearn.cpp`
```text
deeplearn.cpp:24: // FPGAI real dense tiling helper.
```
```text
deeplearn.cpp:50: #pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
```
```text
deeplearn.cpp:51: #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1
```
```text
deeplearn.cpp:52: #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.cpp:56: #pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
```
```text
deeplearn.cpp:68: #pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
```
```text
deeplearn.cpp:69: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
```
```text
deeplearn.cpp:70: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2
```
```text
deeplearn.cpp:151: #pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
```
```text
deeplearn.cpp:152: #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1
```
```text
deeplearn.cpp:153: #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=2
```
```text
deeplearn.cpp:157: #pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
```
```text
deeplearn.cpp:169: #pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
```
```text
deeplearn.cpp:170: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
```
```text
deeplearn.cpp:171: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2
```
```text
deeplearn.cpp:230: //   0: layer_0 (Dense) ii=2 pe=4 simd=4 part_in=? part_out=? part_w=? tile={'sizes': {'in': 8, 'out': 4}} memory={'weight_mode': 'embedded', 'activation_mode': 'buffer', 'weight_region': 'BRAM', 'activation_region': 'BRAM', 'gradient_region': None} sig=19d5c1026e6172b78199490fb19ed2f4609276bcb58ffb3247f4cd73ccb14b1b
```
```text
deeplearn.cpp:231: //   1: layer_1 (Relu) ii=2 pe=4 simd=1 part_in=? part_out=? part_w=? tile={'sizes': {}} memory={'weight_mode': 'embedded', 'activation_mode': 'stream', 'weight_region': 'BRAM', 'activation_region': 'BRAM', 'gradient_region': None} sig=09bc099ed45f804c1a3df2bbd5f907af96944d4c5fede1242504710155c9d925
```
```text
deeplearn.cpp:232: //   2: layer_2 (Dense) ii=2 pe=4 simd=4 part_in=? part_out=? part_w=? tile={'sizes': {'in': 4, 'out': 2}} memory={'weight_mode': 'embedded', 'activation_mode': 'buffer', 'weight_region': 'BRAM', 'activation_region': 'BRAM', 'gradient_region': None} sig=888ba83c641c72204d227163f339c47de6aa22bcdc4c26f18d74e1c1583a37be
```
```text
deeplearn.cpp:233: //   3: layer_3 (Softmax) ii=2 pe=4 simd=1 part_in=? part_out=? part_w=? tile={'sizes': {}} memory={'weight_mode': 'embedded', 'activation_mode': 'stream', 'weight_region': 'BRAM', 'activation_region': 'BRAM', 'gradient_region': None} sig=f21e31cd34d79fb173904136638eea3644adf2d088771862838d6e00b9bacbbb
```
```text
deeplearn.cpp:315: #pragma HLS BIND_STORAGE variable=layer_in type=ram_1p impl=bram
```
```text
deeplearn.cpp:337:     // tile: {'in': 8, 'out': 4}
```
```text
deeplearn.cpp:339:     // output placement: BRAM / size=16 bytes
```
```text
deeplearn.cpp:342: #pragma HLS BIND_STORAGE variable=layer_0_out type=ram_1p impl=bram
```
```text
deeplearn.cpp:344:     // Weight cache placement: BRAM / size=weight_cache bytes
```
```text
deeplearn.cpp:345:     // FPGAI storage binding requested for embedded parameter W0/B0; top-level BIND_STORAGE is disabled for initialized W/B arrays because Vitis HLS rejects initialized global/static parameter arrays bound this way. Use runtime-loaded URAM buffers for real URAM parameter storage.
```
```text
deeplearn.cpp:358:     // tile: {}
```
```text
deeplearn.cpp:360:     // output placement: BRAM / size=16 bytes
```
```text
deeplearn.cpp:363: #pragma HLS BIND_STORAGE variable=layer_1_out type=ram_1p impl=bram
```
```text
deeplearn.cpp:376:     // tile: {'in': 4, 'out': 2}
```
```text
deeplearn.cpp:378:     // output placement: BRAM / size=8 bytes
```
```text
deeplearn.cpp:381: #pragma HLS BIND_STORAGE variable=layer_2_out type=ram_1p impl=bram
```
```text
deeplearn.cpp:383:     // Weight cache placement: BRAM / size=weight_cache bytes
```
```text
deeplearn.cpp:384:     // FPGAI storage binding requested for embedded parameter W1/B1; top-level BIND_STORAGE is disabled for initialized W/B arrays because Vitis HLS rejects initialized global/static parameter arrays bound this way. Use runtime-loaded URAM buffers for real URAM parameter storage.
```
```text
deeplearn.cpp:397:     // tile: {}
```
```text
deeplearn.cpp:402: #pragma HLS BIND_STORAGE variable=layer_3_out type=ram_1p impl=bram
```
### `hls/src/fpgai_params.cpp`
```text
fpgai_params.cpp:18: // FPGAI storage binding: parameter arrays requested for BRAM.
```
```text
fpgai_params.cpp:19: // FPGAI note: file-scope BIND_STORAGE pragmas are disabled because Vitis HLS csynth rejects them on global const arrays.
```
```text
fpgai_params.cpp:20: // FPGAI storage binding: bram requested for W0; file-scope BIND_STORAGE disabled.
```
```text
fpgai_params.cpp:21: // FPGAI storage binding: bram requested for B0; file-scope BIND_STORAGE disabled.
```
```text
fpgai_params.cpp:22: // FPGAI storage binding: bram requested for W1; file-scope BIND_STORAGE disabled.
```
```text
fpgai_params.cpp:23: // FPGAI storage binding: bram requested for B1; file-scope BIND_STORAGE disabled.
```
### `hls_artifact_metadata.json`
```text
hls_artifact_metadata.json:46:         "tiling": {
```
```text
hls_artifact_metadata.json:59:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:60:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:104:         "tiling": {
```
```text
hls_artifact_metadata.json:114:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:115:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:160:         "tiling": {
```
```text
hls_artifact_metadata.json:173:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:174:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:218:         "tiling": {
```
```text
hls_artifact_metadata.json:228:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:229:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:1728:       "path": "hls/reports/tiling_resource_estimate.json",
```
```text
hls_artifact_metadata.json:1878:       "path": "reports/resource_prediction.json",
```
### `ir/comm_plan.json`
```text
comm_plan.json:52:         "region": "BRAM",
```
```text
comm_plan.json:82:         "region": "BRAM",
```
```text
comm_plan.json:112:         "region": "BRAM",
```
```text
comm_plan.json:142:         "region": "BRAM",
```
```text
comm_plan.json:172:         "region": "BRAM",
```
```text
comm_plan.json:202:         "region": "BRAM",
```
```text
comm_plan.json:232:         "region": "BRAM",
```
### `ir/compile_plan.json`
```text
compile_plan.json:18:       "tile": {
```
```text
compile_plan.json:87:         "tiling": {
```
```text
compile_plan.json:100:           "weight_region": "BRAM",
```
```text
compile_plan.json:101:           "activation_region": "BRAM",
```
```text
compile_plan.json:113:       "tile": {},
```
```text
compile_plan.json:177:         "tiling": {
```
```text
compile_plan.json:187:           "weight_region": "BRAM",
```
```text
compile_plan.json:188:           "activation_region": "BRAM",
```
```text
compile_plan.json:200:       "tile": {
```
```text
compile_plan.json:269:         "tiling": {
```
```text
compile_plan.json:282:           "weight_region": "BRAM",
```
```text
compile_plan.json:283:           "activation_region": "BRAM",
```
```text
compile_plan.json:295:       "tile": {},
```
```text
compile_plan.json:359:         "tiling": {
```
```text
compile_plan.json:369:           "weight_region": "BRAM",
```
```text
compile_plan.json:370:           "activation_region": "BRAM",
```
```text
compile_plan.json:377:   "global_resource_budget": {
```
```text
compile_plan.json:386:     "weight_storage": "bram",
```
```text
compile_plan.json:397:     "policy_resource_awareness": {
```
```text
compile_plan.json:430:       "BRAM",
```
```text
compile_plan.json:431:       "URAM",
```
```text
compile_plan.json:435:       "BRAM",
```
```text
compile_plan.json:436:       "URAM",
```
```text
compile_plan.json:443:     "array_partition_mode": "cyclic",
```
### `ir/memory_plan.json`
```text
memory_plan.json:21:       "region": "BRAM",
```
```text
memory_plan.json:39:       "region": "BRAM",
```
```text
memory_plan.json:57:       "region": "BRAM",
```
```text
memory_plan.json:66:         "tile": {
```
```text
memory_plan.json:82:       "region": "BRAM",
```
```text
memory_plan.json:91:         "tile": {},
```
```text
memory_plan.json:103:       "region": "BRAM",
```
```text
memory_plan.json:121:       "region": "BRAM",
```
```text
memory_plan.json:139:       "region": "BRAM",
```
```text
memory_plan.json:148:         "tile": {
```
```text
memory_plan.json:179:     "BRAM": 224
```
```text
memory_plan.json:186:       "BRAM",
```
```text
memory_plan.json:187:       "URAM",
```
```text
memory_plan.json:191:       "BRAM",
```
```text
memory_plan.json:192:       "URAM",
```
### `manifest.json`
```text
manifest.json:137:         "feature": "tiling",
```
```text
manifest.json:151:         "detail": "Dense/Conv code generation emits tiled call sites for the requested tile sizes.",
```
```text
manifest.json:176:           "weight_region": "BRAM",
```
```text
manifest.json:177:           "activation_region": "BRAM",
```
```text
manifest.json:183:           "weight_region": "BRAM",
```
```text
manifest.json:184:           "activation_region": "BRAM",
```
```text
manifest.json:280:         "feature": "tiling",
```
```text
manifest.json:288:         "detail": "No tiling was requested.",
```
```text
manifest.json:313:           "weight_region": "BRAM",
```
```text
manifest.json:314:           "activation_region": "BRAM",
```
```text
manifest.json:320:           "weight_region": "BRAM",
```
```text
manifest.json:321:           "activation_region": "BRAM",
```
```text
manifest.json:423:         "feature": "tiling",
```
```text
manifest.json:437:         "detail": "Dense/Conv code generation emits tiled call sites for the requested tile sizes.",
```
```text
manifest.json:462:           "weight_region": "BRAM",
```
```text
manifest.json:463:           "activation_region": "BRAM",
```
```text
manifest.json:469:           "weight_region": "BRAM",
```
```text
manifest.json:470:           "activation_region": "BRAM",
```
```text
manifest.json:566:         "feature": "tiling",
```
```text
manifest.json:574:         "detail": "No tiling was requested.",
```
```text
manifest.json:599:           "weight_region": "BRAM",
```
```text
manifest.json:600:           "activation_region": "BRAM",
```
```text
manifest.json:606:           "weight_region": "BRAM",
```
```text
manifest.json:607:           "activation_region": "BRAM",
```
```text
manifest.json:618:     "resource_prediction_json": "/home/umutcanaltin/Desktop/github_projects/fpgai_compiler/paper_experiments/full_pipeline_gate/sprint26_paper_matrix/runs/kv260_tiling_medium/reports/resource_prediction.json",
```
```text
manifest.json:690:     "BRAM": 224
```
```text
manifest.json:827:       "resources": "operator_structural_v2",
```
```text
manifest.json:857:         "optimization.parallel.array_partition_mode",
```
```text
manifest.json:860:         "optimization.tiling.dense",
```
```text
manifest.json:861:         "optimization.tiling.conv",
```
```text
manifest.json:870:         "resource_prediction",
```
```text
manifest.json:874:         "resource_score",
```
```text
manifest.json:1039:       "resource_estimation_mode": "analytical",
```
```text
manifest.json:1061:           "optimization.parallel.array_partition_mode",
```
```text
manifest.json:1064:           "optimization.tiling.dense",
```
```text
manifest.json:1065:           "optimization.tiling.conv",
```
```text
manifest.json:1074:           "resource_prediction",
```
```text
manifest.json:1078:           "resource_score",
```
```text
manifest.json:1083:         "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
manifest.json:1247:       "resource_estimation_mode": "analytical",
```
### `reports/board_fit.json`
```text
board_fit.json:6:     "limiting_resource": "target_clock_mhz",
```
```text
board_fit.json:9:     "resources": {
```
```text
board_fit.json:52:       "uram": {
```
```text
board_fit.json:64:   "normalized_resources": {
```
### `reports/hardware_knob_contract.json`
```text
hardware_knob_contract.json:62:         "generated HLS template args and ARRAY_PARTITION factors"
```
```text
hardware_knob_contract.json:73:         "planner.policy.array_partition_mode",
```
```text
hardware_knob_contract.json:75:         "HLS ARRAY_PARTITION mode where supported"
```
```text
hardware_knob_contract.json:79:       "path": "optimization.parallel.array_partition_mode",
```
```text
hardware_knob_contract.json:113:         "planner dense tile selection",
```
```text
hardware_knob_contract.json:114:         "layer_plan.architecture.tiling",
```
```text
hardware_knob_contract.json:121:       "note": "Layer-specific tiling can override global dense tiling.",
```
```text
hardware_knob_contract.json:122:       "path": "optimization.tiling.dense",
```
```text
hardware_knob_contract.json:133:         "planner conv tile selection",
```
```text
hardware_knob_contract.json:134:         "layer_plan.architecture.tiling",
```
```text
hardware_knob_contract.json:138:       "note": "Layer-specific tiling can override global conv tiling.",
```
```text
hardware_knob_contract.json:139:       "path": "optimization.tiling.conv",
```
```text
hardware_knob_contract.json:150:         "planner layer-specific tile selection",
```
```text
hardware_knob_contract.json:151:         "layer_plan.architecture.tiling for matching layer names"
```
```text
hardware_knob_contract.json:154:       "note": "Manual layer entries have priority over global tiling defaults.",
```
```text
hardware_knob_contract.json:155:       "path": "optimization.tiling.layers",
```
```text
hardware_knob_contract.json:166:       "effective": "bram",
```
```text
hardware_knob_contract.json:169:       "requested": "bram",
```
```text
hardware_knob_contract.json:179:         "BRAM",
```
```text
hardware_knob_contract.json:180:         "URAM",
```
```text
hardware_knob_contract.json:195:         "BRAM",
```
```text
hardware_knob_contract.json:196:         "URAM",
```
### `reports/resource_prediction.json`
```text
resource_prediction.json:37:           "activation_region": "BRAM",
```
```text
resource_prediction.json:44:           "weight_region": "BRAM"
```
```text
resource_prediction.json:62:         "tiling": {
```
```text
resource_prediction.json:101:           "activation_region": "BRAM",
```
```text
resource_prediction.json:108:           "weight_region": "BRAM"
```
```text
resource_prediction.json:126:         "tiling": {},
```
```text
resource_prediction.json:160:           "activation_region": "BRAM",
```
```text
resource_prediction.json:167:           "weight_region": "BRAM"
```
```text
resource_prediction.json:185:         "tiling": {
```
```text
resource_prediction.json:224:           "activation_region": "BRAM",
```
```text
resource_prediction.json:231:           "weight_region": "BRAM"
```
```text
resource_prediction.json:249:         "tiling": {},
```
```text
resource_prediction.json:309:           "activation_region": "BRAM",
```
```text
resource_prediction.json:316:           "weight_region": "BRAM"
```
```text
resource_prediction.json:334:         "tiling": {
```
```text
resource_prediction.json:407:       "resource_components": {
```
```text
resource_prediction.json:465:           "activation_region": "BRAM",
```
```text
resource_prediction.json:472:           "weight_region": "BRAM"
```
```text
resource_prediction.json:490:         "tiling": {},
```
```text
resource_prediction.json:551:       "resource_components": {
```
```text
resource_prediction.json:600:           "activation_region": "BRAM",
```
```text
resource_prediction.json:607:           "weight_region": "BRAM"
```
```text
resource_prediction.json:625:         "tiling": {
```
```text
resource_prediction.json:698:       "resource_components": {
```
```text
resource_prediction.json:756:           "activation_region": "BRAM",
```
```text
resource_prediction.json:763:           "weight_region": "BRAM"
```
```text
resource_prediction.json:781:         "tiling": {},
```
```text
resource_prediction.json:843:       "resource_components": {
```
```text
resource_prediction.json:859:   "prediction_kind": "pre_hls_resource_estimate",
```
```text
resource_prediction.json:933:         "activation_region": "BRAM",
```
```text
resource_prediction.json:940:         "weight_region": "BRAM"
```
```text
resource_prediction.json:958:       "tiling": {
```
```text
resource_prediction.json:1031:     "resource_components": {
```
```text
resource_prediction.json:1088:         "activation_region": "BRAM",
```
```text
resource_prediction.json:1095:         "weight_region": "BRAM"
```
```text
resource_prediction.json:1113:       "tiling": {
```
```text
resource_prediction.json:1186:     "resource_components": {
```
```text
resource_prediction.json:1243:         "activation_region": "BRAM",
```
```text
resource_prediction.json:1250:         "weight_region": "BRAM"
```
```text
resource_prediction.json:1268:       "tiling": {
```
### `runtime_package/hls/hls_artifact_metadata.json`
```text
hls_artifact_metadata.json:46:         "tiling": {
```
```text
hls_artifact_metadata.json:59:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:60:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:104:         "tiling": {
```
```text
hls_artifact_metadata.json:114:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:115:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:160:         "tiling": {
```
```text
hls_artifact_metadata.json:173:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:174:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:218:         "tiling": {
```
```text
hls_artifact_metadata.json:228:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:229:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:1728:       "path": "hls/reports/tiling_resource_estimate.json",
```
```text
hls_artifact_metadata.json:1878:       "path": "reports/resource_prediction.json",
```
## `kv260_tiling_large`
### `analysis/architecture_capabilities.json`
```text
architecture_capabilities.json:109:       "feature": "tiling",
```
```text
architecture_capabilities.json:123:       "detail": "Dense/Conv code generation emits tiled call sites for the requested tile sizes.",
```
```text
architecture_capabilities.json:148:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:149:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:155:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:156:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:252:       "feature": "tiling",
```
```text
architecture_capabilities.json:260:       "detail": "No tiling was requested.",
```
```text
architecture_capabilities.json:285:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:286:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:292:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:293:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:395:       "feature": "tiling",
```
```text
architecture_capabilities.json:409:       "detail": "Dense/Conv code generation emits tiled call sites for the requested tile sizes.",
```
```text
architecture_capabilities.json:434:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:435:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:441:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:442:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:538:       "feature": "tiling",
```
```text
architecture_capabilities.json:546:       "detail": "No tiling was requested.",
```
```text
architecture_capabilities.json:571:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:572:         "activation_region": "BRAM",
```
```text
architecture_capabilities.json:578:         "weight_region": "BRAM",
```
```text
architecture_capabilities.json:579:         "activation_region": "BRAM",
```
### `calibration/calibrated_model.json`
```text
calibrated_model.json:3:     "bram": 1.0,
```
```text
calibrated_model.json:14:     "bram",
```
### `calibration/compile_plan_for_calibration.json`
```text
compile_plan_for_calibration.json:11:   "global_resource_budget": {
```
```text
compile_plan_for_calibration.json:27:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:30:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:66:         "tiling": {
```
```text
compile_plan_for_calibration.json:100:       "tile": {
```
```text
compile_plan_for_calibration.json:121:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:124:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:159:         "tiling": {
```
```text
compile_plan_for_calibration.json:190:       "tile": {},
```
```text
compile_plan_for_calibration.json:207:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:210:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:246:         "tiling": {
```
```text
compile_plan_for_calibration.json:280:       "tile": {
```
```text
compile_plan_for_calibration.json:301:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:304:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:339:         "tiling": {
```
```text
compile_plan_for_calibration.json:370:       "tile": {},
```
```text
compile_plan_for_calibration.json:389:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:392:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:428:         "tiling": {
```
```text
compile_plan_for_calibration.json:462:       "tile": {
```
```text
compile_plan_for_calibration.json:483:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:486:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:521:         "tiling": {
```
```text
compile_plan_for_calibration.json:552:       "tile": {},
```
```text
compile_plan_for_calibration.json:569:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:572:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:608:         "tiling": {
```
```text
compile_plan_for_calibration.json:642:       "tile": {
```
```text
compile_plan_for_calibration.json:663:           "activation_region": "BRAM",
```
```text
compile_plan_for_calibration.json:666:           "weight_region": "BRAM"
```
```text
compile_plan_for_calibration.json:701:         "tiling": {
```
```text
compile_plan_for_calibration.json:732:       "tile": {},
```
```text
compile_plan_for_calibration.json:748:       "BRAM",
```
```text
compile_plan_for_calibration.json:749:       "URAM",
```
```text
compile_plan_for_calibration.json:753:     "array_partition_mode": "cyclic",
```
```text
compile_plan_for_calibration.json:773:     "policy_resource_awareness": {
```
```text
compile_plan_for_calibration.json:803:       "BRAM",
```
```text
compile_plan_for_calibration.json:804:       "URAM",
```
```text
compile_plan_for_calibration.json:807:     "weight_storage": "bram"
```
### `calibration/estimate_vs_hls.json`
```text
estimate_vs_hls.json:4:       "bram": 1.0,
```
```text
estimate_vs_hls.json:15:       "bram",
```
```text
estimate_vs_hls.json:26:       "bram": 0.0,
```
```text
estimate_vs_hls.json:33:       "bram": null,
```
```text
estimate_vs_hls.json:41:       "bram": 0.0,
```
### `design_space/results.json`
```text
results.json:5:     "resources": "operator_structural_v2",
```
```text
results.json:177:     "resource_estimation_mode": "analytical",
```
```text
results.json:199:         "optimization.parallel.array_partition_mode",
```
```text
results.json:202:         "optimization.tiling.dense",
```
```text
results.json:203:         "optimization.tiling.conv",
```
```text
results.json:212:         "resource_prediction",
```
```text
results.json:216:         "resource_score",
```
```text
results.json:221:       "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:385:     "resource_estimation_mode": "analytical",
```
```text
results.json:407:         "optimization.parallel.array_partition_mode",
```
```text
results.json:410:         "optimization.tiling.dense",
```
```text
results.json:411:         "optimization.tiling.conv",
```
```text
results.json:420:         "resource_prediction",
```
```text
results.json:424:         "resource_score",
```
```text
results.json:429:       "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:593:     "resource_estimation_mode": "analytical",
```
```text
results.json:615:         "optimization.parallel.array_partition_mode",
```
```text
results.json:618:         "optimization.tiling.dense",
```
```text
results.json:619:         "optimization.tiling.conv",
```
```text
results.json:628:         "resource_prediction",
```
```text
results.json:632:         "resource_score",
```
```text
results.json:637:       "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:802:       "resource_estimation_mode": "analytical",
```
```text
results.json:824:           "optimization.parallel.array_partition_mode",
```
```text
results.json:827:           "optimization.tiling.dense",
```
```text
results.json:828:           "optimization.tiling.conv",
```
```text
results.json:837:           "resource_prediction",
```
```text
results.json:841:           "resource_score",
```
```text
results.json:846:         "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:1010:       "resource_estimation_mode": "analytical",
```
```text
results.json:1032:           "optimization.parallel.array_partition_mode",
```
```text
results.json:1035:           "optimization.tiling.dense",
```
```text
results.json:1036:           "optimization.tiling.conv",
```
```text
results.json:1045:           "resource_prediction",
```
```text
results.json:1049:           "resource_score",
```
```text
results.json:1054:         "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:1218:       "resource_estimation_mode": "analytical",
```
```text
results.json:1240:           "optimization.parallel.array_partition_mode",
```
```text
results.json:1243:           "optimization.tiling.dense",
```
```text
results.json:1244:           "optimization.tiling.conv",
```
### `estimate_vs_hls/layer_validation/results.json`
```text
results.json:4:   "resource_model": "operator_structural_v4_inference_hls_sharing_training_problem_shared",
```
### `estimate_vs_hls/modules/results.json`
```text
results.json:44:   "top_resources": {
```
```text
results.json:50:   "unassigned_top_resources": {
```
```text
results.json:508:   "aggregation_note": "Primary operator totals exclude generated pipeline and loop helper reports. Helper resources are hierarchical subsets and must not be added to their parent function resources."
```
### `estimate_vs_hls/results.json`
```text
results.json:168:     "resource_estimation_mode": "analytical",
```
```text
results.json:190:         "optimization.parallel.array_partition_mode",
```
```text
results.json:193:         "optimization.tiling.dense",
```
```text
results.json:194:         "optimization.tiling.conv",
```
```text
results.json:203:         "resource_prediction",
```
```text
results.json:207:         "resource_score",
```
```text
results.json:212:       "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
results.json:320:     "resources": {
```
```text
results.json:331:     "resources": {
```
```text
results.json:344:             "resources": {
```
### `hls/codegen_meta.json`
```text
codegen_meta.json:35:   "tiling_resource_estimate": {
```
```text
codegen_meta.json:38:     "format": "fpgai.tiling_resource_model.v1",
```
```text
codegen_meta.json:39:     "path": "reports/tiling_resource_estimate.json",
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/a.g.ld.0.bc.clang.reflow.diag.yml`
```text
a.g.ld.0.bc.clang.reflow.diag.yml:852:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:872:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:892:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:910:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:934:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:954:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:974:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:992:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1010:   - String:          array_partition
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1030:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1041:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1052:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1063:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1074:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1085:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1096:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1107:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1118:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1129:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1140:   - Name:            Resource/Bind_Storage
```
```text
a.g.ld.0.bc.clang.reflow.diag.yml:1151:   - Name:            Resource/Bind_Storage
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/activations.pp.0.cpp`
```text
activations.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
activations.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
activations.pp.0.cpp:6048: #pragma HLS ARRAY_PARTITION variable=temporary cyclic factor=4
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/autopilot.rtl.models.tcl`
```text
autopilot.rtl.models.tcl:55:       {MODELNAME deeplearn_layer_in_RAM_1P_BRAM_1R1W RTLNAME deeplearn_layer_in_RAM_1P_BRAM_1R1W BINDTYPE storage TYPE ram_1p IMPL bram LATENCY 2 ALLOW_PRAGMA 1}
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/batchnorm.pp.0.cpp`
```text
batchnorm.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
batchnorm.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
batchnorm.pp.0.cpp:15799: #pragma HLS array_partition variable=swap_table complete
```
```text
batchnorm.pp.0.cpp:15800: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15801: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15802: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15803: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K3 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15804: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_cos_K4 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15834: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15835: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15836: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15837: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K3 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15838: #pragma HLS BIND_STORAGE variable=fourth_order_double::cos_K4 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15839: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15840: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15841: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15842: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K3 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15843: #pragma HLS BIND_STORAGE variable=fourth_order_double::sin_K4 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15892: #pragma HLS array_partition variable=swap_table complete
```
```text
batchnorm.pp.0.cpp:15893: #pragma HLS BIND_STORAGE variable=second_order_float::sin_cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15894: #pragma HLS BIND_STORAGE variable=second_order_float::sin_cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15895: #pragma HLS BIND_STORAGE variable=second_order_float::sin_cos_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15922: #pragma HLS BIND_STORAGE variable=second_order_float::cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15923: #pragma HLS BIND_STORAGE variable=second_order_float::cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15924: #pragma HLS BIND_STORAGE variable=second_order_float::cos_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15925: #pragma HLS BIND_STORAGE variable=second_order_float::sin_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15926: #pragma HLS BIND_STORAGE variable=second_order_float::sin_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15927: #pragma HLS BIND_STORAGE variable=second_order_float::sin_K2 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15963: #pragma HLS BIND_STORAGE variable=first_order_fixed_16::sin_cos_K0 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15964: #pragma HLS BIND_STORAGE variable=first_order_fixed_16::sin_cos_K1 type=ROM_1P impl=LUTRAM
```
```text
batchnorm.pp.0.cpp:15993: #pragma HLS array_partition variable=swap_table complete
```
```text
batchnorm.pp.0.cpp:15994: #pragma HLS array_partition variable=neg_sin_table complete
```
```text
batchnorm.pp.0.cpp:15995: #pragma HLS array_partition variable=neg_cos_table complete
```
```text
batchnorm.pp.0.cpp:16065: #pragma HLS array_partition variable=swap_table complete
```
```text
batchnorm.pp.0.cpp:16066: #pragma HLS array_partition variable=neg_sin_table complete
```
```text
batchnorm.pp.0.cpp:16067: #pragma HLS array_partition variable=neg_cos_table complete
```
```text
batchnorm.pp.0.cpp:16136: #pragma HLS array_partition variable=neg_sin_table complete
```
```text
batchnorm.pp.0.cpp:16137: #pragma HLS array_partition variable=neg_cos_table complete
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml`
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:50:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:55:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:60:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:65:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:70:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:75:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:80:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:85:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:90:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:95:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:100:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:105:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:110:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:115:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:120:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:125:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:130:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:135:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:140:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:145:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:150:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:155:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:160:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:165:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:170:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
batchnorm.pp.0.cpp.clang-tidy.loop-label.diag.yml:175:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/conv.pp.0.cpp`
```text
conv.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
conv.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
conv.pp.0.cpp:5792: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:5793: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:5794: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.pp.0.cpp:5817: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
conv.pp.0.cpp:6016: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6017: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6018: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6221: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6222: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6223: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6395: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.pp.0.cpp:6396: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/deeplearn.compgen.tcl`
```text
deeplearn.compgen.tcl:4: 	::AP::rtl_comp_handler deeplearn_layer_in_RAM_1P_BRAM_1R1W BINDTYPE {storage} TYPE {ram_1p} IMPL {bram} LATENCY 2 ALLOW_PRAGMA 1
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/deeplearn.pp.0.cpp`
```text
deeplearn.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
deeplearn.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
deeplearn.pp.0.cpp:184: #pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:185: #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:186: #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:190: #pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:202: #pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:203: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:204: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2
```
```text
deeplearn.pp.0.cpp:285: #pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:286: #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:287: #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=2
```
```text
deeplearn.pp.0.cpp:291: #pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:303: #pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:304: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
```
```text
deeplearn.pp.0.cpp:305: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2
```
```text
deeplearn.pp.0.cpp:9299: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9300: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9301: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9310: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
deeplearn.pp.0.cpp:9429: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9430: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9431: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9531: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9532: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9586: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9587: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9588: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9596: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
deeplearn.pp.0.cpp:9723: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9724: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9783: #pragma HLS ARRAY_PARTITION variable=B cyclic factor=PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9784: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9862: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9863: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9864: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:9887: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
deeplearn.pp.0.cpp:10086: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:10087: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.pp.0.cpp:10088: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml`
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:869:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:874:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:879:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:884:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:889:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:894:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:899:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:904:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:909:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:914:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:919:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:924:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:929:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:934:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:939:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:944:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:949:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:954:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:959:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:964:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:969:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:974:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:979:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:984:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:989:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:994:     Message:         'Invalid Directive: for current device, ROM_1P + LUTRAM is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2205:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2228:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2233:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2238:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
```text
deeplearn.pp.0.cpp.clang-tidy.loop-label.diag.yml:2243:     Message:         'Invalid Directive: for current device, ram_1p + bram is invalid combination for BIND_STORAGE''s option ''type + impl'''
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/dense.pp.0.cpp`
```text
dense.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
dense.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
dense.pp.0.cpp:5785: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5786: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5787: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5796: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
dense.pp.0.cpp:5915: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5916: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:5917: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6017: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6018: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6072: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6073: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6074: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.pp.0.cpp:6082: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
dense.pp.0.cpp:6209: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=PARTITION dim=1
```
```text
dense.pp.0.cpp:6210: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=PARTITION dim=1
```
```text
dense.pp.0.cpp:6269: #pragma HLS ARRAY_PARTITION variable=B cyclic factor=PARTITION dim=1
```
```text
dense.pp.0.cpp:6270: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=PARTITION dim=1
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/fpgai_params.pp.0.cpp`
```text
fpgai_params.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
fpgai_params.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
### `hls/fpgai_hls_proj/sol1/.autopilot/db/pool.pp.0.cpp`
```text
pool.pp.0.cpp:93:     void _ssdm_op_SpecResource(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
```text
pool.pp.0.cpp:94:     void _ssdm_op_SpecResourceLimit(...) __attribute__ ((nothrow)) __attribute__((overloadable));
```
### `hls/fpgai_hls_proj/sol1/impl/ip/run_ippack.tcl`
```text
run_ippack.tcl:103: deeplearn_layer_in_RAM_1P_BRAM_1R1W
```
```text
run_ippack.tcl:483:         bram {
```
```text
run_ippack.tcl:484:             dict set properties bus_type_vlnv "xilinx.com:interface:bram:1.0"
```
```text
run_ippack.tcl:1559:         bram {
```
```text
run_ippack.tcl:1583:                 set current_bus_interface [add_bus_interface $core ${interface_name}_PORT$suffix bram master]
```
```text
run_ippack.tcl:1630:                 set current_bus_interface [add_bus_interface $core ${interface_name}_PORT$suffix bram master]
```
### `hls/fpgai_hls_proj/sol1/sol1_data.json`
```text
sol1_data.json:107:       "impl\/vhdl\/deeplearn_layer_in_RAM_1P_BRAM_1R1W.vhd",
```
```text
sol1_data.json:145:       "impl\/verilog\/deeplearn_layer_in_RAM_1P_BRAM_1R1W.v",
```
```text
sol1_data.json:772:           "URAM": "0",
```
```text
sol1_data.json:811:           "URAM": "0",
```
```text
sol1_data.json:850:           "URAM": "0",
```
```text
sol1_data.json:889:           "URAM": "0",
```
```text
sol1_data.json:928:           "URAM": "0",
```
```text
sol1_data.json:967:           "URAM": "0",
```
```text
sol1_data.json:999:           "URAM": "0",
```
```text
sol1_data.json:1031:           "URAM": "0",
```
```text
sol1_data.json:1063:           "URAM": "0",
```
```text
sol1_data.json:1102:           "URAM": "0",
```
```text
sol1_data.json:1141:           "URAM": "0",
```
```text
sol1_data.json:1173:           "URAM": "0",
```
```text
sol1_data.json:1205:           "URAM": "0",
```
### `hls/include/layers/activations.h`
```text
activations.h:300: #pragma HLS ARRAY_PARTITION variable=temporary cyclic factor=FPGAI_ACT_UNROLL
```
### `hls/include/layers/conv.h`
```text
conv.h:48: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.h:49: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.h:50: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.h:73: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
conv.h:272: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.h:273: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.h:274: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.h:477: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
conv.h:478: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.h:479: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
conv.h:651: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
conv.h:652: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
### `hls/include/layers/dense.h`
```text
dense.h:41: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.h:42: #pragma HLS ARRAY_PARTITION variable=y cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:43: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.h:52: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
dense.h:171: #pragma HLS ARRAY_PARTITION variable=x cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.h:172: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:173: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.h:273: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:274: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:328: #pragma HLS ARRAY_PARTITION variable=dY cyclic factor=OUTPUT_PARTITION dim=1
```
```text
dense.h:329: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=WEIGHT_PARTITION dim=1
```
```text
dense.h:330: #pragma HLS ARRAY_PARTITION variable=dX cyclic factor=INPUT_PARTITION dim=1
```
```text
dense.h:338: #pragma HLS ARRAY_PARTITION variable=accumulators complete
```
```text
dense.h:465: #pragma HLS ARRAY_PARTITION variable=W cyclic factor=PARTITION dim=1
```
```text
dense.h:466: #pragma HLS ARRAY_PARTITION variable=dW cyclic factor=PARTITION dim=1
```
```text
dense.h:525: #pragma HLS ARRAY_PARTITION variable=B cyclic factor=PARTITION dim=1
```
```text
dense.h:526: #pragma HLS ARRAY_PARTITION variable=dB cyclic factor=PARTITION dim=1
```
### `hls/reports/tiling_analysis.json`
```text
tiling_analysis.json:29:       "tile": {
```
```text
tiling_analysis.json:40:       "detail": "Tiling analysis is implemented for Dense and Conv layers.",
```
```text
tiling_analysis.json:44:       "tile": {
```
```text
tiling_analysis.json:73:       "tile": {
```
```text
tiling_analysis.json:84:       "detail": "Tiling analysis is implemented for Dense and Conv layers.",
```
```text
tiling_analysis.json:88:       "tile": {
```
### `hls/reports/tiling_performance_estimate.json`
```text
tiling_performance_estimate.json:5:     "note": "This is an analytical estimate for comparing tile choices. Final achieved II and latency should be taken from HLS reports.",
```
### `hls/reports/tiling_resource_estimate.json`
```text
tiling_resource_estimate.json:8:     "note": "This estimates local tile buffer storage only. It does not replace full HLS resource reports."
```
```text
tiling_resource_estimate.json:10:   "format": "fpgai.tiling_resource_model.v1",
```
### `hls/src/deeplearn.cpp`
```text
deeplearn.cpp:24: // FPGAI real dense tiling helper.
```
```text
deeplearn.cpp:50: #pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
```
```text
deeplearn.cpp:51: #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1
```
```text
deeplearn.cpp:52: #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=1
```
```text
deeplearn.cpp:56: #pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
```
```text
deeplearn.cpp:68: #pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
```
```text
deeplearn.cpp:69: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
```
```text
deeplearn.cpp:70: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2
```
```text
deeplearn.cpp:151: #pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
```
```text
deeplearn.cpp:152: #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1
```
```text
deeplearn.cpp:153: #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=2
```
```text
deeplearn.cpp:157: #pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
```
```text
deeplearn.cpp:169: #pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
```
```text
deeplearn.cpp:170: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
```
```text
deeplearn.cpp:171: #pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2
```
```text
deeplearn.cpp:230: //   0: layer_0 (Dense) ii=2 pe=4 simd=4 part_in=? part_out=? part_w=? tile={'sizes': {'in': 8, 'out': 4}} memory={'weight_mode': 'embedded', 'activation_mode': 'buffer', 'weight_region': 'BRAM', 'activation_region': 'BRAM', 'gradient_region': None} sig=19d5c1026e6172b78199490fb19ed2f4609276bcb58ffb3247f4cd73ccb14b1b
```
```text
deeplearn.cpp:231: //   1: layer_1 (Relu) ii=2 pe=4 simd=1 part_in=? part_out=? part_w=? tile={'sizes': {}} memory={'weight_mode': 'embedded', 'activation_mode': 'stream', 'weight_region': 'BRAM', 'activation_region': 'BRAM', 'gradient_region': None} sig=09bc099ed45f804c1a3df2bbd5f907af96944d4c5fede1242504710155c9d925
```
```text
deeplearn.cpp:232: //   2: layer_2 (Dense) ii=2 pe=4 simd=4 part_in=? part_out=? part_w=? tile={'sizes': {'in': 4, 'out': 2}} memory={'weight_mode': 'embedded', 'activation_mode': 'buffer', 'weight_region': 'BRAM', 'activation_region': 'BRAM', 'gradient_region': None} sig=888ba83c641c72204d227163f339c47de6aa22bcdc4c26f18d74e1c1583a37be
```
```text
deeplearn.cpp:233: //   3: layer_3 (Softmax) ii=2 pe=4 simd=1 part_in=? part_out=? part_w=? tile={'sizes': {}} memory={'weight_mode': 'embedded', 'activation_mode': 'stream', 'weight_region': 'BRAM', 'activation_region': 'BRAM', 'gradient_region': None} sig=f21e31cd34d79fb173904136638eea3644adf2d088771862838d6e00b9bacbbb
```
```text
deeplearn.cpp:315: #pragma HLS BIND_STORAGE variable=layer_in type=ram_1p impl=bram
```
```text
deeplearn.cpp:337:     // tile: {'in': 8, 'out': 4}
```
```text
deeplearn.cpp:339:     // output placement: BRAM / size=16 bytes
```
```text
deeplearn.cpp:342: #pragma HLS BIND_STORAGE variable=layer_0_out type=ram_1p impl=bram
```
```text
deeplearn.cpp:344:     // Weight cache placement: BRAM / size=weight_cache bytes
```
```text
deeplearn.cpp:345:     // FPGAI storage binding requested for embedded parameter W0/B0; top-level BIND_STORAGE is disabled for initialized W/B arrays because Vitis HLS rejects initialized global/static parameter arrays bound this way. Use runtime-loaded URAM buffers for real URAM parameter storage.
```
```text
deeplearn.cpp:358:     // tile: {}
```
```text
deeplearn.cpp:360:     // output placement: BRAM / size=16 bytes
```
```text
deeplearn.cpp:363: #pragma HLS BIND_STORAGE variable=layer_1_out type=ram_1p impl=bram
```
```text
deeplearn.cpp:376:     // tile: {'in': 4, 'out': 2}
```
```text
deeplearn.cpp:378:     // output placement: BRAM / size=8 bytes
```
```text
deeplearn.cpp:381: #pragma HLS BIND_STORAGE variable=layer_2_out type=ram_1p impl=bram
```
```text
deeplearn.cpp:383:     // Weight cache placement: BRAM / size=weight_cache bytes
```
```text
deeplearn.cpp:384:     // FPGAI storage binding requested for embedded parameter W1/B1; top-level BIND_STORAGE is disabled for initialized W/B arrays because Vitis HLS rejects initialized global/static parameter arrays bound this way. Use runtime-loaded URAM buffers for real URAM parameter storage.
```
```text
deeplearn.cpp:397:     // tile: {}
```
```text
deeplearn.cpp:402: #pragma HLS BIND_STORAGE variable=layer_3_out type=ram_1p impl=bram
```
### `hls/src/fpgai_params.cpp`
```text
fpgai_params.cpp:18: // FPGAI storage binding: parameter arrays requested for BRAM.
```
```text
fpgai_params.cpp:19: // FPGAI note: file-scope BIND_STORAGE pragmas are disabled because Vitis HLS csynth rejects them on global const arrays.
```
```text
fpgai_params.cpp:20: // FPGAI storage binding: bram requested for W0; file-scope BIND_STORAGE disabled.
```
```text
fpgai_params.cpp:21: // FPGAI storage binding: bram requested for B0; file-scope BIND_STORAGE disabled.
```
```text
fpgai_params.cpp:22: // FPGAI storage binding: bram requested for W1; file-scope BIND_STORAGE disabled.
```
```text
fpgai_params.cpp:23: // FPGAI storage binding: bram requested for B1; file-scope BIND_STORAGE disabled.
```
### `hls_artifact_metadata.json`
```text
hls_artifact_metadata.json:46:         "tiling": {
```
```text
hls_artifact_metadata.json:59:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:60:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:104:         "tiling": {
```
```text
hls_artifact_metadata.json:114:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:115:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:160:         "tiling": {
```
```text
hls_artifact_metadata.json:173:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:174:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:218:         "tiling": {
```
```text
hls_artifact_metadata.json:228:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:229:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:1728:       "path": "hls/reports/tiling_resource_estimate.json",
```
```text
hls_artifact_metadata.json:1878:       "path": "reports/resource_prediction.json",
```
### `ir/comm_plan.json`
```text
comm_plan.json:52:         "region": "BRAM",
```
```text
comm_plan.json:82:         "region": "BRAM",
```
```text
comm_plan.json:112:         "region": "BRAM",
```
```text
comm_plan.json:142:         "region": "BRAM",
```
```text
comm_plan.json:172:         "region": "BRAM",
```
```text
comm_plan.json:202:         "region": "BRAM",
```
```text
comm_plan.json:232:         "region": "BRAM",
```
### `ir/compile_plan.json`
```text
compile_plan.json:18:       "tile": {
```
```text
compile_plan.json:87:         "tiling": {
```
```text
compile_plan.json:100:           "weight_region": "BRAM",
```
```text
compile_plan.json:101:           "activation_region": "BRAM",
```
```text
compile_plan.json:113:       "tile": {},
```
```text
compile_plan.json:177:         "tiling": {
```
```text
compile_plan.json:187:           "weight_region": "BRAM",
```
```text
compile_plan.json:188:           "activation_region": "BRAM",
```
```text
compile_plan.json:200:       "tile": {
```
```text
compile_plan.json:269:         "tiling": {
```
```text
compile_plan.json:282:           "weight_region": "BRAM",
```
```text
compile_plan.json:283:           "activation_region": "BRAM",
```
```text
compile_plan.json:295:       "tile": {},
```
```text
compile_plan.json:359:         "tiling": {
```
```text
compile_plan.json:369:           "weight_region": "BRAM",
```
```text
compile_plan.json:370:           "activation_region": "BRAM",
```
```text
compile_plan.json:377:   "global_resource_budget": {
```
```text
compile_plan.json:386:     "weight_storage": "bram",
```
```text
compile_plan.json:397:     "policy_resource_awareness": {
```
```text
compile_plan.json:430:       "BRAM",
```
```text
compile_plan.json:431:       "URAM",
```
```text
compile_plan.json:435:       "BRAM",
```
```text
compile_plan.json:436:       "URAM",
```
```text
compile_plan.json:443:     "array_partition_mode": "cyclic",
```
### `ir/memory_plan.json`
```text
memory_plan.json:21:       "region": "BRAM",
```
```text
memory_plan.json:39:       "region": "BRAM",
```
```text
memory_plan.json:57:       "region": "BRAM",
```
```text
memory_plan.json:66:         "tile": {
```
```text
memory_plan.json:82:       "region": "BRAM",
```
```text
memory_plan.json:91:         "tile": {},
```
```text
memory_plan.json:103:       "region": "BRAM",
```
```text
memory_plan.json:121:       "region": "BRAM",
```
```text
memory_plan.json:139:       "region": "BRAM",
```
```text
memory_plan.json:148:         "tile": {
```
```text
memory_plan.json:179:     "BRAM": 224
```
```text
memory_plan.json:186:       "BRAM",
```
```text
memory_plan.json:187:       "URAM",
```
```text
memory_plan.json:191:       "BRAM",
```
```text
memory_plan.json:192:       "URAM",
```
### `manifest.json`
```text
manifest.json:137:         "feature": "tiling",
```
```text
manifest.json:151:         "detail": "Dense/Conv code generation emits tiled call sites for the requested tile sizes.",
```
```text
manifest.json:176:           "weight_region": "BRAM",
```
```text
manifest.json:177:           "activation_region": "BRAM",
```
```text
manifest.json:183:           "weight_region": "BRAM",
```
```text
manifest.json:184:           "activation_region": "BRAM",
```
```text
manifest.json:280:         "feature": "tiling",
```
```text
manifest.json:288:         "detail": "No tiling was requested.",
```
```text
manifest.json:313:           "weight_region": "BRAM",
```
```text
manifest.json:314:           "activation_region": "BRAM",
```
```text
manifest.json:320:           "weight_region": "BRAM",
```
```text
manifest.json:321:           "activation_region": "BRAM",
```
```text
manifest.json:423:         "feature": "tiling",
```
```text
manifest.json:437:         "detail": "Dense/Conv code generation emits tiled call sites for the requested tile sizes.",
```
```text
manifest.json:462:           "weight_region": "BRAM",
```
```text
manifest.json:463:           "activation_region": "BRAM",
```
```text
manifest.json:469:           "weight_region": "BRAM",
```
```text
manifest.json:470:           "activation_region": "BRAM",
```
```text
manifest.json:566:         "feature": "tiling",
```
```text
manifest.json:574:         "detail": "No tiling was requested.",
```
```text
manifest.json:599:           "weight_region": "BRAM",
```
```text
manifest.json:600:           "activation_region": "BRAM",
```
```text
manifest.json:606:           "weight_region": "BRAM",
```
```text
manifest.json:607:           "activation_region": "BRAM",
```
```text
manifest.json:618:     "resource_prediction_json": "/home/umutcanaltin/Desktop/github_projects/fpgai_compiler/paper_experiments/full_pipeline_gate/sprint26_paper_matrix/runs/kv260_tiling_large/reports/resource_prediction.json",
```
```text
manifest.json:690:     "BRAM": 224
```
```text
manifest.json:827:       "resources": "operator_structural_v2",
```
```text
manifest.json:857:         "optimization.parallel.array_partition_mode",
```
```text
manifest.json:860:         "optimization.tiling.dense",
```
```text
manifest.json:861:         "optimization.tiling.conv",
```
```text
manifest.json:870:         "resource_prediction",
```
```text
manifest.json:874:         "resource_score",
```
```text
manifest.json:1039:       "resource_estimation_mode": "analytical",
```
```text
manifest.json:1061:           "optimization.parallel.array_partition_mode",
```
```text
manifest.json:1064:           "optimization.tiling.dense",
```
```text
manifest.json:1065:           "optimization.tiling.conv",
```
```text
manifest.json:1074:           "resource_prediction",
```
```text
manifest.json:1078:           "resource_score",
```
```text
manifest.json:1083:         "truth_note": "DSE recommends among configured YAML candidates only. Resource/timing values are pre-HLS estimates; no exhaustive search or measured HLS/Vivado optimization is claimed."
```
```text
manifest.json:1247:       "resource_estimation_mode": "analytical",
```
### `reports/board_fit.json`
```text
board_fit.json:6:     "limiting_resource": "target_clock_mhz",
```
```text
board_fit.json:9:     "resources": {
```
```text
board_fit.json:52:       "uram": {
```
```text
board_fit.json:64:   "normalized_resources": {
```
### `reports/hardware_knob_contract.json`
```text
hardware_knob_contract.json:62:         "generated HLS template args and ARRAY_PARTITION factors"
```
```text
hardware_knob_contract.json:73:         "planner.policy.array_partition_mode",
```
```text
hardware_knob_contract.json:75:         "HLS ARRAY_PARTITION mode where supported"
```
```text
hardware_knob_contract.json:79:       "path": "optimization.parallel.array_partition_mode",
```
```text
hardware_knob_contract.json:113:         "planner dense tile selection",
```
```text
hardware_knob_contract.json:114:         "layer_plan.architecture.tiling",
```
```text
hardware_knob_contract.json:121:       "note": "Layer-specific tiling can override global dense tiling.",
```
```text
hardware_knob_contract.json:122:       "path": "optimization.tiling.dense",
```
```text
hardware_knob_contract.json:133:         "planner conv tile selection",
```
```text
hardware_knob_contract.json:134:         "layer_plan.architecture.tiling",
```
```text
hardware_knob_contract.json:138:       "note": "Layer-specific tiling can override global conv tiling.",
```
```text
hardware_knob_contract.json:139:       "path": "optimization.tiling.conv",
```
```text
hardware_knob_contract.json:150:         "planner layer-specific tile selection",
```
```text
hardware_knob_contract.json:151:         "layer_plan.architecture.tiling for matching layer names"
```
```text
hardware_knob_contract.json:154:       "note": "Manual layer entries have priority over global tiling defaults.",
```
```text
hardware_knob_contract.json:155:       "path": "optimization.tiling.layers",
```
```text
hardware_knob_contract.json:166:       "effective": "bram",
```
```text
hardware_knob_contract.json:169:       "requested": "bram",
```
```text
hardware_knob_contract.json:179:         "BRAM",
```
```text
hardware_knob_contract.json:180:         "URAM",
```
```text
hardware_knob_contract.json:195:         "BRAM",
```
```text
hardware_knob_contract.json:196:         "URAM",
```
### `reports/resource_prediction.json`
```text
resource_prediction.json:37:           "activation_region": "BRAM",
```
```text
resource_prediction.json:44:           "weight_region": "BRAM"
```
```text
resource_prediction.json:62:         "tiling": {
```
```text
resource_prediction.json:101:           "activation_region": "BRAM",
```
```text
resource_prediction.json:108:           "weight_region": "BRAM"
```
```text
resource_prediction.json:126:         "tiling": {},
```
```text
resource_prediction.json:160:           "activation_region": "BRAM",
```
```text
resource_prediction.json:167:           "weight_region": "BRAM"
```
```text
resource_prediction.json:185:         "tiling": {
```
```text
resource_prediction.json:224:           "activation_region": "BRAM",
```
```text
resource_prediction.json:231:           "weight_region": "BRAM"
```
```text
resource_prediction.json:249:         "tiling": {},
```
```text
resource_prediction.json:309:           "activation_region": "BRAM",
```
```text
resource_prediction.json:316:           "weight_region": "BRAM"
```
```text
resource_prediction.json:334:         "tiling": {
```
```text
resource_prediction.json:407:       "resource_components": {
```
```text
resource_prediction.json:465:           "activation_region": "BRAM",
```
```text
resource_prediction.json:472:           "weight_region": "BRAM"
```
```text
resource_prediction.json:490:         "tiling": {},
```
```text
resource_prediction.json:551:       "resource_components": {
```
```text
resource_prediction.json:600:           "activation_region": "BRAM",
```
```text
resource_prediction.json:607:           "weight_region": "BRAM"
```
```text
resource_prediction.json:625:         "tiling": {
```
```text
resource_prediction.json:698:       "resource_components": {
```
```text
resource_prediction.json:756:           "activation_region": "BRAM",
```
```text
resource_prediction.json:763:           "weight_region": "BRAM"
```
```text
resource_prediction.json:781:         "tiling": {},
```
```text
resource_prediction.json:843:       "resource_components": {
```
```text
resource_prediction.json:859:   "prediction_kind": "pre_hls_resource_estimate",
```
```text
resource_prediction.json:933:         "activation_region": "BRAM",
```
```text
resource_prediction.json:940:         "weight_region": "BRAM"
```
```text
resource_prediction.json:958:       "tiling": {
```
```text
resource_prediction.json:1031:     "resource_components": {
```
```text
resource_prediction.json:1088:         "activation_region": "BRAM",
```
```text
resource_prediction.json:1095:         "weight_region": "BRAM"
```
```text
resource_prediction.json:1113:       "tiling": {
```
```text
resource_prediction.json:1186:     "resource_components": {
```
```text
resource_prediction.json:1243:         "activation_region": "BRAM",
```
```text
resource_prediction.json:1250:         "weight_region": "BRAM"
```
```text
resource_prediction.json:1268:       "tiling": {
```
### `runtime_package/hls/hls_artifact_metadata.json`
```text
hls_artifact_metadata.json:46:         "tiling": {
```
```text
hls_artifact_metadata.json:59:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:60:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:104:         "tiling": {
```
```text
hls_artifact_metadata.json:114:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:115:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:160:         "tiling": {
```
```text
hls_artifact_metadata.json:173:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:174:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:218:         "tiling": {
```
```text
hls_artifact_metadata.json:228:           "weight_region": "BRAM",
```
```text
hls_artifact_metadata.json:229:           "activation_region": "BRAM",
```
```text
hls_artifact_metadata.json:1728:       "path": "hls/reports/tiling_resource_estimate.json",
```
```text
hls_artifact_metadata.json:1878:       "path": "reports/resource_prediction.json",
```
