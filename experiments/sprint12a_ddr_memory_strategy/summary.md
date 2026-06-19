# FPGAI Experiment Summary

Experiment directory: `/home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint12a_ddr_memory_strategy`

## Status

- Total design points: 4
- Passed: 4
- Failed: 0
- Skipped/resumed: 0

## Results

| # | Design | Status | Board | Config | Duration (s) |
|---:|---|---|---|---|---:|
| 0 | `memory_on_chip_balanced` | passed | kv260 | `experiments/sprint12a_ddr_memory_strategy/configs/memory_on_chip_balanced.yml` | 144.8 |
| 1 | `memory_streaming_balanced` | passed | kv260 | `experiments/sprint12a_ddr_memory_strategy/configs/memory_streaming_balanced.yml` | 153.8 |
| 2 | `memory_external_ddr_balanced` | passed | kv260 | `experiments/sprint12a_ddr_memory_strategy/configs/memory_external_ddr_balanced.yml` | 157.3 |
| 3 | `memory_external_ddr_latency_first` | passed | kv260 | `experiments/sprint12a_ddr_memory_strategy/configs/memory_external_ddr_latency_first.yml` | 218.6 |

## Reproducibility metadata

Every row in `results.json` and `results.csv` includes commit hash, config path, model path, tool version, and board target when available.
