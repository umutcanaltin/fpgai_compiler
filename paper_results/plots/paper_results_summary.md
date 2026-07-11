# FPGAI paper results summary

- compile_output_count: `18`
- created_figures: `11`
- pending_figures: `4`

## Current paper story

The current frozen subset supports an inference-first and training-second results section. HLS/Vivado figures are generated from existing synthesis/implementation artifacts. Real latency, energy, and FPGA training-curve figures are intentionally pending until board-runtime measurements are imported.

## Key computed comparisons
- HLS latency cycles: I1_precision_fx8_embedded increased by 0.51% versus I0_baseline_fx16_embedded (delta 1673).
- HLS LUT: I1_precision_fx8_embedded decreased by 6.70% versus I0_baseline_fx16_embedded (delta -1312).
- HLS DSP: I1_precision_fx8_embedded decreased by 42.86% versus I0_baseline_fx16_embedded (delta -3).
- HLS BRAM: I1_precision_fx8_embedded decreased by 60.00% versus I0_baseline_fx16_embedded (delta -69).
- Vivado LUT: I1_precision_fx8_embedded decreased by 36.90% versus I0_baseline_fx16_embedded (delta -4689).
- Vivado DSP: I1_precision_fx8_embedded decreased by 20.00% versus I0_baseline_fx16_embedded (delta -1).
- Vivado BRAM: I1_precision_fx8_embedded decreased by 66.40% versus I0_baseline_fx16_embedded (delta -41.5).
- Power W: I1_precision_fx8_embedded decreased by 1.97% versus I0_baseline_fx16_embedded (delta -0.056).
- WNS ns: I1_precision_fx8_embedded decreased by 28.11% versus I0_baseline_fx16_embedded (delta -0.725).
- HLS latency cycles: I3_parallel_pe2 decreased by 29.67% versus I0_baseline_fx16_embedded (delta -9.77e+04).
- HLS LUT: I3_parallel_pe2 increased by 91.24% versus I0_baseline_fx16_embedded (delta 1.786e+04).
- HLS DSP: I3_parallel_pe2 increased by 157.14% versus I0_baseline_fx16_embedded (delta 11).
- HLS BRAM: I3_parallel_pe2 increased by 21.74% versus I0_baseline_fx16_embedded (delta 25).
- Vivado LUT: I3_parallel_pe2 increased by 212.42% versus I0_baseline_fx16_embedded (delta 2.699e+04).
- Vivado DSP: I3_parallel_pe2 increased by 140.00% versus I0_baseline_fx16_embedded (delta 7).
- Vivado BRAM: I3_parallel_pe2 increased by 13.60% versus I0_baseline_fx16_embedded (delta 8.5).

## Use in paper

- Use `figure_captions.md` and `table_captions.md` for caption drafting.
- Use `paper_claims_from_artifacts.md` to avoid over-claiming runtime behavior.
- Open `plot_gallery.html` locally to inspect all generated SVG plots in one page.
