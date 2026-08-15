# HLS synthesis characterization

FPGAI records Vitis HLS synthesis results separately from declared implementation-package metrics and compiler estimates.

The report `reports/hls_synthesis_characterization.json` records the generated top-level design's measured HLS resources, latency, initiation interval, estimated clock period, estimated Fmax, and target-clock status.

For a mixed graph, these measurements are explicitly scoped as `mixed_graph_top`. FPGAI does not attribute whole-design LUT/FF/DSP/BRAM/URAM or latency to an individual external operator without a separately synthesized module-level experiment.

The report also records participating external implementation packages and their declared metrics for provenance. Declared package metrics and whole-design synthesis metrics are therefore visible together but are not treated as directly comparable measurements.

A successful `csynth` result establishes the `hls_synthesized` validation level for the generated design artifact. It does not imply Vivado synthesis, implementation, bitstream generation, or hardware validation.
