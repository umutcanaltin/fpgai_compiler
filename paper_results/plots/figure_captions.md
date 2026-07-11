# Figure captions

| figure | status | path | caption | pending_reason |
| --- | --- | --- | --- | --- |
| figure_00_plot_status | created | paper_results/plots/figures/figure_00_plot_status.svg | Artifact-status overview showing which paper figures are generated and which require real board-runtime measurements. | — |
| figure_01_inference_hls_latency | created | paper_results/plots/figures/figure_01_inference_hls_latency.svg | Inference HLS latency across the frozen inference subset, generated from Vitis HLS synthesis reports. | — |
| figure_02_inference_vivado_lut | created | paper_results/plots/figures/figure_02_inference_vivado_lut.svg | Implemented inference LUT usage across the frozen inference subset, generated from Vivado implementation reports. | — |
| figure_03_inference_real_latency_ms | pending | — | Real KV260 inference latency; generated only after board-runtime measurements are imported. | pending real inference board-runtime measurements |
| figure_04_inference_energy_mj | pending | — | Estimated real inference energy per inference using board-runtime latency and Vivado power. | pending real inference latency/power measurements |
| figure_05_training_hls_latency | created | paper_results/plots/figures/figure_05_training_hls_latency.svg | Training HLS latency across the frozen training subset, generated from Vitis HLS synthesis reports. | — |
| figure_06_training_vivado_lut | created | paper_results/plots/figures/figure_06_training_vivado_lut.svg | Implemented training LUT usage across the frozen training subset, generated from Vivado implementation reports. | — |
| figure_07_training_step_ms | pending | — | Real KV260 training-step latency; generated only after board-runtime training measurements are imported. | pending real training board-runtime measurements |
| figure_08_training_curve_loss | pending | — | Real FPGA training loss curve; generated only from board-runtime training curve rows. | pending real FPGA training-curve rows |
| figure_09_knob_status_counts | created | paper_results/plots/figures/figure_09_knob_status_counts.svg | Coverage of YAML/hardware knob application status across generated hardware knob contracts. | — |
| figure_10_training_vivado_power | created | paper_results/plots/figures/figure_10_training_vivado_power.svg | Implemented training design power reported by Vivado for the frozen training subset. | — |
| figure_11_training_vivado_dsp | created | paper_results/plots/figures/figure_11_training_vivado_dsp.svg | Implemented training DSP usage reported by Vivado for the frozen training subset. | — |
| figure_12_all_hls_lut | created | paper_results/plots/figures/figure_12_all_hls_lut.svg | HLS LUT comparison across all frozen paper designs with HLS reports. | — |
| figure_13_all_vivado_power | created | paper_results/plots/figures/figure_13_all_vivado_power.svg | Vivado power comparison across all frozen paper designs with implementation reports. | — |
| figure_14_all_vivado_wns | created | paper_results/plots/figures/figure_14_all_vivado_wns.svg | Vivado timing slack comparison across all frozen paper designs with implementation reports. | — |
