# FPGAI paper experiment setup

- status: `ready`
- experiment_count: `19`
- error_count: `0`
- warning_count: `0`

## Sections
- `inference`: `10`
- `training`: `9`

## Expected claim levels
- `level_2_vivado_implementation`: `15`
- `level_3_bitstream_package`: `2`
- `level_4_board_execution`: `2`

## Configured stages
- `bitstream_package`: `4`
- `vivado_implementation`: `15`

## Tables
- `setup_rows_csv`: `paper_results/experiment_setup/paper_experiment_setup_rows.csv`
- `setup_rows_md`: `paper_results/experiment_setup/paper_experiment_setup_rows.md`
- `compile_command_plan_csv`: `paper_results/experiment_setup/compile_command_plan.csv`
- `compile_command_plan_md`: `paper_results/experiment_setup/compile_command_plan.md`
- `compile_command_plan_sh_txt`: `paper_results/experiment_setup/compile_command_plan.sh.txt`
- `compile_selected_smoke_sh_txt`: `paper_results/experiment_setup/compile_selected_smoke.sh.txt`
- `compile_inference_matrix_sh_txt`: `paper_results/experiment_setup/compile_inference_matrix.sh.txt`
- `compile_training_matrix_sh_txt`: `paper_results/experiment_setup/compile_training_matrix.sh.txt`
- `compile_bitstream_candidates_sh_txt`: `paper_results/experiment_setup/compile_bitstream_candidates.sh.txt`
- `compile_board_runtime_candidates_sh_txt`: `paper_results/experiment_setup/compile_board_runtime_candidates.sh.txt`
- `regenerate_plots_sh_txt`: `paper_results/experiment_setup/regenerate_plots.sh.txt`

## Next command

```bash
python -m fpgai.paper.plots build --output-dir paper_results/plots
```
