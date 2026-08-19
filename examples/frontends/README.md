# Python framework to FPGAI examples

These examples start from framework-native Python model definitions and export into interchange formats already owned by FPGAI's existing frontends. They do not add framework-specific compiler branches.

All four examples implement the same deterministic `1x4 -> 1x3` linear computation with identical weights and bias so source-framework provenance can be separated from normalized FPGAI IR semantics.

## JAX -> StableHLO -> FPGAI IR

```bash
python examples/frontends/jax_linear.py \
  --out build/examples/frontends/jax_linear.mlir

python -m fpgai.cli frontend import \
  --input build/examples/frontends/jax_linear.mlir \
  --format stablehlo \
  --framework jax \
  --out build/examples/frontends/jax_import
```

The JAX model uses JAX's StableHLO export path, which FPGAI imports through its existing StableHLO/MLIR frontend. The exporter also writes a deterministic `fpgai.frontend-reference/v1` bundle beside the MLIR file (`jax_linear.fpgai-reference/reference_manifest.json`). FPGAI auto-discovers that bundle, so numeric validation compares the original JAX output with the functional FPGAI IR while MLIR/StableHLO remains the compiler ingress path.

The same reference-bundle contract can be produced by TensorFlow, PyTorch, custom Python/MLIR exporters, or other frontends without adding framework-specific branches to FPGAI. An explicit bundle may also be selected in YAML:

```yaml
validation:
  numeric:
    enabled: true
    policy: enforce
    levels: [model, layer, intermediate, state]
    reference:
      source: framework
      compare_ir: true
      bundle: build/examples/frontends/jax_linear.fpgai-reference/reference_manifest.json
```

A model-only bundle validates the model boundary. If `layer`/`intermediate` is requested, the producer must also provide the corresponding captured intermediate tensors; FPGAI reports insufficient reference coverage instead of silently claiming layerwise validation.

## PyTorch -> ONNX -> FPGAI IR

```bash
python examples/frontends/pytorch_linear.py \
  --out build/examples/frontends/pytorch_linear.onnx

python -m fpgai.cli frontend import \
  --input build/examples/frontends/pytorch_linear.onnx \
  --format onnx \
  --framework pytorch \
  --out build/examples/frontends/pytorch_import
```

PyTorch is represented as an export-adapter route through ONNX. This is not claimed as a native PyTorch dialect frontend.

## TensorFlow -> ONNX -> FPGAI IR

```bash
python examples/frontends/tensorflow_linear.py \
  --out build/examples/frontends/tensorflow_linear.onnx

python -m fpgai.cli frontend import \
  --input build/examples/frontends/tensorflow_linear.onnx \
  --format onnx \
  --framework tensorflow \
  --out build/examples/frontends/tensorflow_import
```

The TensorFlow example uses the optional upstream `tf2onnx` converter. FPGAI owns the ONNX-to-FPGAI stage; it does not claim native TensorFlow-dialect import.

## Direct ONNX -> FPGAI IR

```bash
python examples/frontends/onnx_linear.py \
  --out build/examples/frontends/onnx_linear.onnx

python -m fpgai.cli frontend import \
  --input build/examples/frontends/onnx_linear.onnx \
  --format onnx \
  --framework onnx \
  --out build/examples/frontends/onnx_import
```

## What to compare

Each import writes `frontend_import_result.json`, including framework/route provenance and the canonical FPGAI IR fingerprints. Equivalent exported models should converge to the same canonical structure after frontend normalization even though their framework provenance differs.

After frontend validation, compile the exported model with a normal FPGAI YAML configuration so the same IR can proceed through architecture resolution, generated HLS/VHDL artifacts, and hardware validation.
