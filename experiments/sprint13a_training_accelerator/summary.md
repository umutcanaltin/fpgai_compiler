# FPGAI Experiment Summary

Experiment directory: `/home/umutcanaltin/Desktop/github_projects/fpgai_compiler/experiments/sprint13a_training_accelerator`

## Status

- Total design points: 4
- Passed: 4
- Failed: 0
- Skipped/resumed: 0

## Results

| # | Design | Status | Board | Config | Duration (s) |
|---:|---|---|---|---|---:|
| 0 | `training_cnn_embedded_balanced` | passed | kv260 | `experiments/sprint13a_training_accelerator/configs/training_cnn_embedded_balanced.yml` | 128.9 |
| 1 | `training_cnn_stream_balanced` | passed | kv260 | `experiments/sprint13a_training_accelerator/configs/training_cnn_stream_balanced.yml` | 22.45 |
| 2 | `training_cnn_embedded_latency_first` | passed | kv260 | `experiments/sprint13a_training_accelerator/configs/training_cnn_embedded_latency_first.yml` | 152.6 |
| 3 | `training_cnn_stream_latency_first` | passed | kv260 | `experiments/sprint13a_training_accelerator/configs/training_cnn_stream_latency_first.yml` | 22.49 |

## Reproducibility metadata

Every row in `results.json` and `results.csv` includes commit hash, config path, model path, tool version, and board target when available.
