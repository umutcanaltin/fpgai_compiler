# FPGAI Experiment Summary

Experiment directory: `/home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint12b_memory_binding_strategy`

## Status

- Total design points: 4
- Passed: 4
- Failed: 0
- Skipped/resumed: 0

## Results

| # | Design | Status | Board | Config | Duration (s) |
|---:|---|---|---|---|---:|
| 0 | `memory_on_chip_bram_baseline` | passed | kv260 | `experiments/sprint12b_memory_binding_strategy/configs/memory_on_chip_bram_baseline.yml` | 118.5 |
| 1 | `memory_bram_saver_balanced` | passed | kv260 | `experiments/sprint12b_memory_binding_strategy/configs/memory_bram_saver_balanced.yml` | 134.2 |
| 2 | `memory_uram_first_balanced` | passed | kv260 | `experiments/sprint12b_memory_binding_strategy/configs/memory_uram_first_balanced.yml` | 133.2 |
| 3 | `memory_uram_first_latency_first` | passed | kv260 | `experiments/sprint12b_memory_binding_strategy/configs/memory_uram_first_latency_first.yml` | 193.3 |

## Reproducibility metadata

Every row in `results.json` and `results.csv` includes commit hash, config path, model path, tool version, and board target when available.
