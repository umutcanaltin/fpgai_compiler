# Maintained mixed external HLS validation

E4D maintains a reproducible graph:

`Input -> Relu -> community.fpgai::ScaleBias -> Sigmoid -> Output`

Generate the model:

```bash
python scripts/make_mixed_external_hls_example.py
```

Compile and prepare reference artifacts:

```bash
python -m fpgai.cli compile --config configs/examples/mixed_external_relu_scale_bias_sigmoid.yml
```

The compiler writes `input.bin`, `reference_output.bin`, a real HLS testbench, and
`reports/mixed_external_hls_validation.json`. To execute Vitis C simulation in the
same compile, set `ecosystem.validation.run_vitis_csim: true`. Optional synthesis
remains controlled by `build.stages.hls_synthesis`.

The validation report distinguishes prepared, C-simulation, synthesis, and numeric
comparison statuses. FPGAI does not claim hardware validation unless the local
Vitis result and output comparison pass.
