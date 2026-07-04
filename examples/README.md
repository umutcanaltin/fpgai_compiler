# FPGAI examples

These examples are part of the compiler contract. Production examples should use canonical YAML keys and should compile to real FPGAI artifacts. Reference and paper examples are templates and should not be mistaken for completed HLS/Vivado/FPGA evidence.

## Production compile examples

Inference:

- `examples/inference/mnist_mlp_embedded.yml` — embedded BRAM weights, m_axi input/output.
- `examples/inference/mnist_mlp_import_weights.yml` — runtime m_axi weight import/export.
- `examples/inference/cnn_stream_input.yml` — CNN model with AXI-stream/DMA input and output.
- `examples/inference/cnn_m_axi_input.yml` — CNN model with m_axi input and output.

Training:

- `examples/training/mnist_mlp_training_sgd.yml`
- `examples/training/mnist_mlp_training_momentum.yml`
- `examples/training/mnist_mlp_training_adam.yml`
- `examples/training/mnist_mlp_cross_entropy.yml`
- `examples/training/batch_accumulation.yml`
- `examples/training/tiled_training_m_axi.yml`
- `examples/training/tiled_training_axi_stream.yml`

Boards/build stages:

- `examples/boards/pynq_z2_inference.yml`
- `examples/boards/kv260_inference.yml`
- `examples/boards/kr260_training.yml`
- `examples/build/cpp_only.yml`
- `examples/build/hls_project.yml`
- `examples/build/hls_synthesis.yml`
- `examples/build/vivado_project.yml`
- `examples/build/vivado_bitstream.yml`


## Training example communication contract

The non-tiled training examples use AXI-stream input, label, and output movement because the current generated training top consumes training samples through `in`, `aux`, and `out` streams. DDR/m_axi training I/O is represented by the tiled examples where the generated HLS top materializes explicit `input_mem`, `label_mem`, and `output_mem` ports.

## Reference/template examples

- `examples/reference/full_options_reference.yml` documents options and alternatives.
- `examples/paper/*.yml` are sweep templates. They are not direct single compiler configs until materialized into design-point YAMLs.

## Batch-2 artifact smoke

After tests, compile selected examples and audit generated artifacts:

```bash
python -m fpgai.cli compile --config examples/inference/mnist_mlp_embedded.yml
python -m fpgai.cli compile --config examples/inference/mnist_mlp_import_weights.yml
python -m fpgai.cli compile --config examples/training/mnist_mlp_training_sgd.yml
python -m fpgai.cli compile --config examples/build/cpp_only.yml
python -m fpgai.cli compile --config examples/build/hls_project.yml
python -m fpgai.cli compile --config examples/build/vivado_project.yml
python -m fpgai.reporting.artifact_smoke   build/examples/mnist_mlp_embedded   build/examples/mnist_mlp_import_weights   build/examples/mnist_mlp_training_sgd   build/examples/cpp_only   build/examples/hls_project   build/examples/vivado_project   --output build/examples/artifact_smoke_suite_batch2.json
```

The smoke report is evidence-only. It should show compiler-estimated evidence for these examples. HLS/Vivado/bitstream/FPGA truth remains false or not_requested unless those tools/stages are actually run.
