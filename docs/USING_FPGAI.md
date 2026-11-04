# Using FPGAI

FPGAI is a research compiler for **static ONNX → typed Model IR → generated HLS C++**. This guide covers the non-native, locally executable workflow. No AMD toolchain is required to generate source.

## Install

Python **3.10 or later** is required. Use an isolated virtual environment:

```bash
git clone https://github.com/umutcanaltin/fpgai_research.git
cd fpgai_research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[onnx]"
```

To run repository tests, also install `python -m pip install -e ".[test]"`. Some research-only tests need additional optional dependencies; the normal compiler CLI requires no Gym or PyTorch.

## First example: a tiny MatMul and ReLU

```bash
mkdir -p build
python examples/make_toy_onnx.py build/toy.onnx
fpgai-research build/toy.onnx -o build/toy-hls
```

This creates two principal files:

- `build/toy-hls/model.cpp`: generated C++ targeting the supported HLS dialect.
- `build/toy-hls/manifest.json`: model/architecture and source-generation metadata.

The graph is synthetic and contains no private training assets. This command **does not call Vitis**, verify synthesizability or measure timing and numerical accuracy.

An equivalent programmatic pipeline is:

```python
from fpgai_research.frontend.onnx import load_onnx
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler import compile_architecture

model = load_onnx("build/toy.onnx")
architecture = resolve(model, ArchitectureRequest())
artifact = compile_architecture(architecture)
print(artifact.emission.manifest)
```

## Compile multi-layer convolutional networks

For realistic (but small) sequential neural-network structures, use the
[synthetic neural-network gallery](../examples/NETWORK_GALLERY.md).
It provides Conv2D, ReLU, MaxPool2D, Flatten, MatMul, Gemm and Softmax
examples built with public ONNX APIs and deterministic local weights.

```bash
mkdir -p build/gallery
python examples/make_neural_network_onnx.py \
  cnn_pool_dense build/gallery/cnn_pool_dense.onnx
python examples/compile_neural_network.py \
  build/gallery/cnn_pool_dense.onnx \
  --mapping independent \
  --output build/gallery/cnn_pool_dense_compiled
```

To independently check the generated C++ against NumPy intermediates,
add `--verify-host` (requires GNU g++). The stricter neural-chain
entrypoint checks that every stage is preserved. It does not invoke
Vitis/Vivado or imply a successful FPGA implementation.

## Request an architecture

Pass a JSON architecture contract:

```bash
fpgai-research build/toy.onnx \
  --architecture examples/architecture.json \
  -o build/toy-mapped
```

The configuration uses complete operator-level requests; the compiler never silently treats an unsupported setting as a successfully realized design.

```json
{
  "default": {"precision_bits": 12, "integer_bits": 5},
  "operators": {
    "mm": {
      "precision_bits": 12,
      "integer_bits": 5,
      "parallelism": 2,
      "pes": 2,
      "simd": 2,
      "serialization": 2,
      "tile": [1, 2, 2],
      "pipeline_ii": 2
    }
  }
}
```

For an example with `regions`, `operators`, and `loops` see `examples/architecture_scopes.json`. Not all operators support every knob, loop override, precision or target capability. Supported attributes and shapes are validated before source emission; see [the support matrix](SUPPORTED_FEATURES.md).

### Optional target limits

JSON may include a `target` object, for example:

```json
{"target": {"max_parallel_units": 4}}
```

This is an **abstract source architecture constraint**, not a model of LUT, DSP, memory-port or routed capacity on a specific FPGA. A source-legal design may fail HLS synthesis or routing.

### Exporting all IR levels

```bash
fpgai-research build/toy.onnx --export-ir -o build/toy-inspection
```

This additionally writes `progressive_model_ir.json` and refuses to overwrite a populated target directory. The IR artifact **includes model constants/weights**; never commit or share it for private ONNX models.

## Common errors

| Symptom | Explanation / next step |
|---|---|
| ONNX dependency missing | Reinstall `python -m pip install -e ".[onnx]"` inside the activated venv |
| Unsupported operator, attribute or shape | Check [Supported features](SUPPORTED_FEATURES.md); trim the demonstration to a supported static graph |
| Unsupported architecture or capacity | Simplify the operator request or increase only justified abstract limits; do not treat it as hardware qualification |
| Source generated but no bitstream | This is expected: source emission does not invoke Vitis/Vivado or program a board |
| A source artifact cannot be shared | The output may embed model weights or private source identity; use only synthetic models in documentation |

## Native hardware studies (advanced)

Native Vitis HLS, Vivado, RTL and physical observations are **separate opt-in research workflows**. They require compatible proprietary software, target definitions, explicit native execution flags, numerical verification and provenance checks. Original research experiments are create-only and must not be rerun into existing study directories.

Read [AMD toolchain setup](AMD_TOOLCHAIN_SETUP.md) for actual Vitis/Vivado settings, environment variables and the supported native workflow. To turn HLS RTL into a program executable on a board, read [FPGA deployment](FPGA_DEPLOYMENT.md). For evidence tiers, see [Scientific reproducibility](SCIENTIFIC_REPRODUCIBILITY.md). The core compiler API is not a guarantee of board integration, bitstream generation, timing closure or physical inference performance.
