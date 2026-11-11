# Neural-network example gallery

This gallery contains **three complete, deterministic, synthetic static ONNX networks**. It demonstrates the real compiler's existing convolution, pooling, flattening, matrix-product and activation paths without relying on any training dataset or external model downloads.

None of these networks is pretrained; numerical reference agreement is **not classification accuracy**. Generating ONNX or HLS C++ does not invoke Vitis/Vivado, infer FPGA throughput or create a bitstream.

## Network architectures

| Example | Operators in order | Input | Output |
|---|---|---|---|
| **Convolution–pooling–dense** | Conv2D → ReLU → MaxPool2D → Conv2D → Flatten → MatMul → ReLU → Gemm | NCHW **1 × 1 × 6 × 6** | **1 × 2** |
| **Strided convolution–dense** | stride-2 Conv2D → ReLU → Flatten → Gemm → ReLU → MatMul | NCHW **2 × 1 × 7 × 8** | **2 × 3** |
| **Two-convolution classifier** | Conv2D → Conv2D → ReLU → Flatten → Gemm → Softmax | NCHW **1 × 2 × 5 × 7** | **1 × 3** |

### 1. Convolution, pooling and two dense stages

```text
Input [1,1,6,6]
    │
3×3 Conv (2 channels, pad 1) → ReLU  [1,2,6,6]
    │
2×2 MaxPool (stride 2)                 [1,2,3,3]
    │
1×1 Conv (3 channels)                  [1,3,3,3]
    │
Flatten                                [1,27]
    │
MatMul → ReLU                          [1,5]
    │
Gemm (transposed weight + bias)        [1,2]
```

This is the recommended first multi-operator CNN example.

### 2. Strided convolution and two dense stages

```text
Input [2,1,7,8]
    │
3×3 Conv (3 channels, stride 2, pad 1) [2,3,4,4]
    │
ReLU → Flatten                         [2,48]
    │
Gemm (+ bias) → ReLU                   [2,4]
    │
MatMul                                 [2,3]
```

This example demonstrates an NCHW strided Conv2D with a non-square input and a batch size of two. **Important:** in a separate research HLS experiment, related strided CNN/Dense configurations passed C simulation but missed their strict effective synthesis clock budget. Source example generation is **not** a claim that this network meets a 9 ns FPGA timing target.

### 3. Two convolutions with a Softmax output

```text
Input [1,2,5,7]
    │
3×2 Conv (3 channels, asymmetric pad)  [1,3,5,6]
    │
2×3 Conv (2 channels, asymmetric pad)  [1,2,4,6]
    │
ReLU → Flatten                         [1,48]
    │
Gemm (transposed weight + bias)        [1,3]
    │
Softmax                                [1,3]
```

These are small numerical probes with reproducible synthetic weights. A Softmax output is not evidence of a trained or accurate classifier.

## Additional example: two-layer fully connected network

A separate [standalone synthetic MLP generator](make_mlp_onnx.py) exercises
the general compiler route, which supports dense-only networks independently
of the strict rank-four CNN-to-dense neural-chain entrypoint:

```text
Input [1,4] → Gemm [1,6] → ReLU [1,6]
            → Gemm [1,3] → Softmax [1,3]
```

```bash
python examples/make_mlp_onnx.py build/gallery/tiny_mlp.onnx
fpgai-research build/gallery/tiny_mlp.onnx \
  -o build/gallery/tiny_mlp_hls
```

As in the CNN examples, the weights are synthetic and untrained.
The source compiler does not perform hardware synthesis or establish model
accuracy.

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[onnx]"
mkdir -p build/gallery
```

## Generate ONNX, then compile the complete neural chain

```bash
# Example 1: convolution–pooling–dense
python examples/make_neural_network_onnx.py \
  cnn_pool_dense build/gallery/cnn_pool_dense.onnx
python examples/compile_neural_network.py \
  build/gallery/cnn_pool_dense.onnx \
  --output build/gallery/cnn_pool_dense_scalar

# Example 2: strided convolution–dense
python examples/make_neural_network_onnx.py \
  cnn_strided_dense build/gallery/cnn_strided_dense.onnx
python examples/compile_neural_network.py \
  build/gallery/cnn_strided_dense.onnx \
  --output build/gallery/cnn_strided_dense_scalar

# Example 3: two convolutions with Softmax
python examples/make_neural_network_onnx.py \
  cnn_two_conv_softmax build/gallery/cnn_two_conv_softmax.onnx
python examples/compile_neural_network.py \
  build/gallery/cnn_two_conv_softmax.onnx \
  --output build/gallery/cnn_two_conv_softmax_scalar
```

The specialized example compiler deliberately uses `compile_neural_chain`: it verifies **all layers were preserved** and that the supported sequential graph has explicit intermediate boundaries. The general command `fpgai-research` also emits source from supported ONNX models, but it does not perform the same strict sequential-neural-chain checks.

Outputs from each compile include:

| File | Interpretation |
|---|---|
| `model.cpp` | Generated source from the canonical HLS C++ emitter |
| `manifest.json` | Canonical source/compiler metadata |
| `network_contract.json` | Bounded source neural-chain contract receipt |
| `example_summary.json` | Input/output, operator ordering and source-only evidence status |

The generated C++ and contract files may embed tensor constants. These gallery weights are synthetic, but **do not share equivalent outputs for private ONNX models** without reviewing the contents.

## Convenient Makefile targets

The repository provides source-only shortcuts for the four demonstration
networks. They require the same `.[onnx]` installation as the commands above:

```bash
make example-cnn
make example-strided
make example-two-conv
make example-mlp
```

These targets deliberately refuse overwriting existing example outputs.
Use a new output path with the Python scripts for a repeat experiment.
No Makefile example target launches Vitis or Vivado.

## Independent operator mapping versus scalar mapping

The gallery can compile each network using a complete scalar architecture control or legal operator-local convolution and dense mappings. The latter exercises the same mapping parameters as the existing bounded research implementation, not a second source generator.

```bash
python examples/compile_neural_network.py \
  build/gallery/cnn_pool_dense.onnx \
  --mapping independent \
  --output build/gallery/cnn_pool_dense_independent
```

The independent mapping requests PEs, SIMD lanes, temporal serialization and tiles for the eligible Conv/Dense operators, leaving scalar-only activations and flattening unchanged. Generated source differences are **not a claim of better latency, area, power or physical memory reuse**.

## Optional host C++ numerical verification

If `g++` is installed, `--verify-host` compiles and runs the actual generated C++ against an independent NumPy reference for **every intermediate layer**:

```bash
python examples/compile_neural_network.py \
  build/gallery/cnn_two_conv_softmax.onnx \
  --mapping independent \
  --verify-host \
  --output build/gallery/cnn_two_conv_softmax_verified
```

This additionally writes `host_reference_verification.json`. It checks the bounded host-float source execution, **not Vitis CSim, synthesis or hardware**.

## Safe reruns

Example files and compilation directories are **create-only**. Use a new filename/output path for another run; never delete the original research studies to make these commands succeed.

See [Using FPGAI](../docs/USING_FPGAI.md), [supported features](../docs/SUPPORTED_FEATURES.md), and [scientific reproducibility](../docs/SCIENTIFIC_REPRODUCIBILITY.md).

## Source-pinned cross-network comparison experiment (11 November 2026)

The repository now includes a maintainer experiment entry point for the same four untrained synthetic gallery networks (the 3 CNNs and standalone MLP). It requests scalar/operator-local mapping, 16/6 and 12/5 fixed-point formats and two target periods, yielding **32 prepared source design points**, without asserting any hardware measurements:

```bash
python -m scripts.network_benchmark plan
python -m scripts.network_benchmark prepare --output build/network_matrix_20261111_01
python -m scripts.network_benchmark collect --root build/network_matrix_20261111_01 --output build/network_matrix_report_20261111_01
```

The initial collector reports `TBD`/null for native cycles, estimated resources and routed timing because no licensed native tools have been run. In later **explicit** opt-in local runs, use `python -m scripts.network_benchmark run --root build/network_matrix_20261111_01 --stage vitis --first 0 --limit 1 --execute`, then explicitly run `--stage vivado` only after E3 synthesis and fixed-point CSim qualify. The source-only results are not trained-model accuracy, board inference latency or measured Fmax. See internal manuscript protocol at `paper/compiler_ir/CROSS_NETWORK_FPGAI_EXPERIMENT_PROTOCOL_20261111.md` (excluded from curated public preview).

## Native tool not found during cross-network execution?

A prepared source case does not require Vitis/Vivado, but `run --execute` does. If the CLI reports `vitis_hls executable not found`, source the **actual installed** AMD `settings64.sh` in your current terminal (or use the explicit path option), then run:

```bash
python -m scripts.network_benchmark doctor --stage vitis
python -m scripts.network_benchmark run --root build/network_matrix_20261111_01 --stage vitis --first 0 --limit 1 --execute
```

`doctor` never launches a vendor tool. The Python virtual environment does not set the AMD toolchain path. Existing prepared cases need no regeneration, and a missing executable is not an HLS synthesis failure. See the internal experimental protocol at `paper/compiler_ir/CROSS_NETWORK_FPGAI_EXPERIMENT_PROTOCOL_20261111.md` for reproducible source/metric requirements.
