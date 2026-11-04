# Examples

Examples are intentionally **synthetic**, small and suitable for local, non-native compilation. Do not place private ONNX weights, datasets or generated Vitis/Vivado outputs in this directory.

## Generate and compile a tiny neural graph

```bash
python -m pip install -e ".[onnx]"
mkdir -p build
python examples/make_toy_onnx.py build/toy.onnx
fpgai-research build/toy.onnx -o build/example-basic
```

Input: a static 2×3-by-3×2 MatMul followed by ReLU. Output: HLS-oriented `model.cpp` and `manifest.json`. No board or vendor tool is invoked.

## Request per-operator architecture

```bash
fpgai-research build/toy.onnx \
  --architecture examples/architecture.json \
  -o build/example-mapped
```

The JSON example specifies a model default and operator-local mapping for node `mm`.

## Explore scoped requests

```bash
fpgai-research build/toy.onnx \
  --architecture examples/architecture_scopes.json \
  -o build/example-scoped
```

The scoped configuration demonstrates model, region, operator and loop syntax, **not** a promise that any arbitrary loop selector is legal. If a scoped request is rejected, treat it as a capability/semantic restriction; consult [the source architecture guide](../docs/USING_FPGAI.md) and simplify the contract.

## Model IR export

```bash
fpgai-research build/toy.onnx --export-ir -o build/example-ir
```

**Warning:** the exported reconstructible IR contains tensor constants. Do not distribute this output for private or proprietary models.

## Complete CNN and dense examples

The [neural-network gallery](NETWORK_GALLERY.md) contains three complete
synthetic networks: convolution–pooling–dense, strided convolution–dense,
and two convolutions with a Softmax output, plus a standalone two-layer MLP. It includes expected tensor
shapes, a canonical neural-chain compilation command, independent
operator-local mappings and optional GNU C++ numerical verification of
every intermediate tensor.

```bash
mkdir -p build/gallery
python examples/make_neural_network_onnx.py \
  cnn_pool_dense build/gallery/cnn_pool_dense.onnx
python examples/compile_neural_network.py \
  build/gallery/cnn_pool_dense.onnx \
  --output build/gallery/cnn_pool_dense_hls
```

This is HLS C++ generation; it does not invoke vendor tools.

## Preparing an explicit Vitis HLS study input

After generating an ONNX model, create a deterministic synthetic NumPy input with [make_sample_input.py](make_sample_input.py):

```bash
python examples/make_sample_input.py \
  build/gallery/cnn_pool_dense.onnx \
  build/gallery/cnn_pool_dense_input.npy
```

For vendor installation and an explicitly executed HLS testbench, follow the [AMD toolchain guide](../docs/AMD_TOOLCHAIN_SETUP.md). For **real board execution**, follow [FPGA deployment](../docs/FPGA_DEPLOYMENT.md).

## Next steps

Read [Supported features](../docs/SUPPORTED_FEATURES.md), [Architecture](../docs/ARCHITECTURE.md) and [Contributing](../CONTRIBUTING.md). Original source/native research experiments are separate from these onboarding examples.
