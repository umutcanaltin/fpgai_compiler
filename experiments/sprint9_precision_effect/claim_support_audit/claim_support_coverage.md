# FPGAI claim support coverage audit

Experiment: `/home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect`

## Experiment-level checks

| Status | Check | Detail |
|---|---|---|
| OK | passed design records | passed 4/4 |
| FAIL | traceable estimate-vs-HLS evidence | design failures=4 |
| OK | unsupported/partial claim areas | warnings=0 |

## Claim status

| Claim area | Status |
|---|---|
| `precision_policy_materialization` | `incomplete` |
| `per_design_estimate_vs_hls_traceability` | `incomplete` |
| `weight_storage_memory_strategy` | `supported` |
| `broad_all_knobs_claim` | `not_safe` |

## Design checks

### precision_fx8_3_balanced

| Status | Check | Detail |
|---|---|---|
| OK | materialized YAML exists | /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/configs/precision_fx8_3_balanced.yml |
| OK | precision knob recorded | fx8_3 |
| OK | policy knob recorded | balanced |
| OK | board knob recorded | kv260 |
| OK | precision materialized | {"board": "targets.platform.board", "policy": "optimization.parallel_policy", "precision_mode": "numerics.defaults", "project_out_dir": "project.out_dir"} |
| OK | policy materialized | {"board": "targets.platform.board", "policy": "optimization.parallel_policy", "precision_mode": "numerics.defaults", "project_out_dir": "project.out_dir"} |
| OK | board materialized | {"board": "targets.platform.board", "policy": "optimization.parallel_policy", "precision_mode": "numerics.defaults", "project_out_dir": "project.out_dir"} |
| OK | generated C/C++ artifacts preserved | 56 files under /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/artifacts/precision_fx8_3_balanced |
| OK | HLS reports preserved | 231 report files under /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/artifacts/precision_fx8_3_balanced |
| OK | estimate_vs_hls dataset exists | /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/artifacts/precision_fx8_3_balanced/build/calibration/estimate_vs_hls.json |
| FAIL | estimated and real HLS metrics exist | no estimate_vs_hls samples found |
| OK | ap_fixed widths match precision config | config=['16,6', '8,3'] artifacts=['104,7', '11,2', '11,6', '119,7', '12,4', '13,7', '14,5', '15,0', '16,16', '16,6', '17,9', '19,2'] |
| OK | weight/memory strategy knob coverage | optimization.parallel.partition_factor |
### precision_fx8_3_latency_first

| Status | Check | Detail |
|---|---|---|
| OK | materialized YAML exists | /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/configs/precision_fx8_3_latency_first.yml |
| OK | precision knob recorded | fx8_3 |
| OK | policy knob recorded | latency_first |
| OK | board knob recorded | kv260 |
| OK | precision materialized | {"board": "targets.platform.board", "policy": "optimization.parallel_policy", "precision_mode": "numerics.defaults", "project_out_dir": "project.out_dir"} |
| OK | policy materialized | {"board": "targets.platform.board", "policy": "optimization.parallel_policy", "precision_mode": "numerics.defaults", "project_out_dir": "project.out_dir"} |
| OK | board materialized | {"board": "targets.platform.board", "policy": "optimization.parallel_policy", "precision_mode": "numerics.defaults", "project_out_dir": "project.out_dir"} |
| OK | generated C/C++ artifacts preserved | 56 files under /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/artifacts/precision_fx8_3_latency_first |
| OK | HLS reports preserved | 231 report files under /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/artifacts/precision_fx8_3_latency_first |
| OK | estimate_vs_hls dataset exists | /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/artifacts/precision_fx8_3_latency_first/build/calibration/estimate_vs_hls.json |
| FAIL | estimated and real HLS metrics exist | no estimate_vs_hls samples found |
| OK | ap_fixed widths match precision config | config=['16,6', '8,3'] artifacts=['104,7', '11,2', '11,6', '119,7', '12,4', '13,7', '14,5', '15,0', '16,16', '16,6', '17,9', '19,2'] |
| OK | weight/memory strategy knob coverage | optimization.parallel.partition_factor |
### precision_fx16_6_balanced

| Status | Check | Detail |
|---|---|---|
| OK | materialized YAML exists | /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/configs/precision_fx16_6_balanced.yml |
| OK | precision knob recorded | fx16_6 |
| OK | policy knob recorded | balanced |
| OK | board knob recorded | kv260 |
| OK | precision materialized | {"board": "targets.platform.board", "policy": "optimization.parallel_policy", "precision_mode": "numerics.defaults", "project_out_dir": "project.out_dir"} |
| OK | policy materialized | {"board": "targets.platform.board", "policy": "optimization.parallel_policy", "precision_mode": "numerics.defaults", "project_out_dir": "project.out_dir"} |
| OK | board materialized | {"board": "targets.platform.board", "policy": "optimization.parallel_policy", "precision_mode": "numerics.defaults", "project_out_dir": "project.out_dir"} |
| OK | generated C/C++ artifacts preserved | 56 files under /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/artifacts/precision_fx16_6_balanced |
| OK | HLS reports preserved | 231 report files under /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/artifacts/precision_fx16_6_balanced |
| OK | estimate_vs_hls dataset exists | /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/artifacts/precision_fx16_6_balanced/build/calibration/estimate_vs_hls.json |
| FAIL | estimated and real HLS metrics exist | no estimate_vs_hls samples found |
| OK | ap_fixed widths match precision config | config=['16,6', '24,10'] artifacts=['104,7', '11,2', '11,6', '119,7', '12,4', '13,7', '14,5', '15,0', '16,16', '16,6', '17,9', '19,2'] |
| OK | weight/memory strategy knob coverage | optimization.parallel.partition_factor |
### precision_fx16_6_latency_first

| Status | Check | Detail |
|---|---|---|
| OK | materialized YAML exists | /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/configs/precision_fx16_6_latency_first.yml |
| OK | precision knob recorded | fx16_6 |
| OK | policy knob recorded | latency_first |
| OK | board knob recorded | kv260 |
| OK | precision materialized | {"board": "targets.platform.board", "policy": "optimization.parallel_policy", "precision_mode": "numerics.defaults", "project_out_dir": "project.out_dir"} |
| OK | policy materialized | {"board": "targets.platform.board", "policy": "optimization.parallel_policy", "precision_mode": "numerics.defaults", "project_out_dir": "project.out_dir"} |
| OK | board materialized | {"board": "targets.platform.board", "policy": "optimization.parallel_policy", "precision_mode": "numerics.defaults", "project_out_dir": "project.out_dir"} |
| OK | generated C/C++ artifacts preserved | 56 files under /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/artifacts/precision_fx16_6_latency_first |
| OK | HLS reports preserved | 231 report files under /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/artifacts/precision_fx16_6_latency_first |
| OK | estimate_vs_hls dataset exists | /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint9_precision_effect/artifacts/precision_fx16_6_latency_first/build/calibration/estimate_vs_hls.json |
| FAIL | estimated and real HLS metrics exist | no estimate_vs_hls samples found |
| OK | ap_fixed widths match precision config | config=['16,6', '24,10'] artifacts=['104,7', '11,2', '11,6', '119,7', '12,4', '13,7', '14,5', '15,0', '16,16', '16,6', '17,9', '19,2'] |
| OK | weight/memory strategy knob coverage | optimization.parallel.partition_factor |
