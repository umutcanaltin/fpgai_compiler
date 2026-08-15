# Board-aware clocks and residual-CNN validation

## Board clock planning

FPGAI separates a requested frequency from the frequency that the processing-system clock generator actually produces. `scripts/plan_vivado_clocks.py` creates a lightweight Vivado PS-only probe for each request, reads the generated PL-clock constraint, and reports the unique realizable frequencies before an expensive HLS/Vivado implementation sweep is started.

Clock selection is explicit. `exact_only` rejects a request without a matching probed clock, `nearest` selects the closest probed clock, and `nearest_below` selects the greatest probed clock not above the request. No policy silently changes a requested clock.

`scripts/run_vivado_adaptive_fmax.py` first probes candidate requests, deduplicates requests that map to the same physical PL clock, then forwards only the unique realizable targets to the existing constraint-verified implementation sweep.

## Residual CNN validation ladder

The maintained residual CNN now has separate configurations for numeric C simulation, HLS synthesis, and Vivado implementation:

- `mixed_external_residual_cnn_csim.yml`
- `mixed_external_residual_cnn_synth.yml`
- `mixed_external_residual_cnn_vivado.yml`

The mixed-graph reference executor supports the constrained Conv profile used by this maintained model: NCHW activations, OIHW constant weights, two-dimensional pads/strides/dilations, and `group=1`. Unsupported Conv profiles are rejected rather than approximated.

## VHDL tool result

The external VHDL runner accepts the maintained simulation pass marker from either Vivado stdout or an emitted XSim log. This keeps `rtl_simulated` and `vivado_synthesized` validation levels distinct from project generation.
