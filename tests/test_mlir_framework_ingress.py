from __future__ import annotations

from pathlib import Path

import pytest

from fpgai.frontend.mlir import (
    MLIRImportError,
    detect_mlir_dialect,
    framework_mlir_routes,
    import_mlir_program,
    import_stablehlo_mlir,
)


_STABLEHLO_LINEAR = r'''
module {
  func.func @main(
    %image: tensor<28x28xf32>,
    %weights: tensor<784x10xf32>,
    %bias: tensor<1x10xf32>
  ) -> tensor<1x10xf32> {
    %0 = "stablehlo.reshape"(%image) : (tensor<28x28xf32>) -> tensor<1x784xf32>
    %1 = "stablehlo.dot"(%0, %weights) : (tensor<1x784xf32>, tensor<784x10xf32>) -> tensor<1x10xf32>
    %2 = "stablehlo.add"(%1, %bias) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %3 = "stablehlo.constant"() {value = dense<0.0> : tensor<1x10xf32>} : () -> tensor<1x10xf32>
    %4 = "stablehlo.maximum"(%2, %3) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    "func.return"(%4): (tensor<1x10xf32>) -> ()
  }
}
'''


def test_framework_routes_expose_jax_pytorch_stablehlo_and_tensorflow_boundary() -> None:
    routes = framework_mlir_routes()
    assert routes["jax"]["preferred_dialect"] == "stablehlo"
    assert routes["jax"]["accepted_by_fpgai"] is True
    assert routes["pytorch"]["preferred_dialect"] == "stablehlo"
    assert "onnx" in routes["pytorch"]["accepted_formats"]
    assert routes["tensorflow"]["preferred_dialect"] == "tf"
    assert routes["tensorflow"]["accepted_by_fpgai"] is False
    assert routes["tensorflow"]["legalization_required"] is True
    assert "stablehlo" in routes["tensorflow"]["accepted_formats"]


def test_import_official_style_stablehlo_subset_into_fpgai_ir() -> None:
    graph = import_stablehlo_mlir(_STABLEHLO_LINEAR, source_framework="jax", target_board="kv260")
    assert graph.name == "main"
    assert graph.schema == "fpgai.ir/v2"
    assert graph.semantics.source_ir == "stablehlo"
    assert graph.semantics.source_metadata["framework"] == "jax"
    assert graph.semantics.target_board == "kv260"
    assert [op.op_type for op in graph.ops] == ["Reshape", "MatMul", "Add", "Maximum"]
    assert graph.get_tensor("1").shape == (1, 10)
    assert graph.outputs == ["4"]
    assert "3" in graph.constants


def test_unified_mlir_import_dispatches_stablehlo() -> None:
    assert detect_mlir_dialect(_STABLEHLO_LINEAR) == "stablehlo"
    graph = import_mlir_program(_STABLEHLO_LINEAR, source_framework="pytorch")
    assert graph.semantics.source_metadata["framework"] == "pytorch"
    assert any(op.op_type == "MatMul" for op in graph.ops)


def test_tensorflow_dialect_requires_upstream_legalization() -> None:
    tf_mlir = 'module { func.func @main() { %0 = "tf.Const"() : () -> tensor<1xf32> return } }'
    with pytest.raises(MLIRImportError, match="legalization to StableHLO"):
        import_mlir_program(tf_mlir, source_framework="tensorflow")


def test_native_mlir_dialect_scaffold_has_attention_and_transport_ops() -> None:
    root = Path(__file__).resolve().parents[1]
    ops = (root / "native/mlir/include/fpgai/Dialect/FPGAI/FPGAIOps.td").read_text(encoding="utf-8")
    dialect = (root / "native/mlir/include/fpgai/Dialect/FPGAI/FPGAIDialect.td").read_text(encoding="utf-8")
    assert 'let name = "fpgai"' in dialect
    assert 'FPGAI_Op<"matmul"' in ops
    assert 'FPGAI_Op<"softmax"' in ops
    assert 'FPGAI_Op<"layer_norm"' in ops
    assert 'FPGAI_Op<"transport"' in ops
    assert 'FPGAI_Op<"buffer"' in ops


def test_same_stablehlo_from_frameworks_has_same_canonical_fpgai_ir_fingerprint() -> None:
    from fpgai.frontend.mlir import canonical_ir_equivalence_manifest, compare_canonical_ir
    graphs = {name: import_stablehlo_mlir(_STABLEHLO_LINEAR, source_framework=name) for name in ("jax", "tensorflow", "pytorch")}
    report = compare_canonical_ir(graphs)
    assert report["equivalent"] is True
    assert report["status"] == "equivalent"
    assert len(set(report["fingerprints"].values())) == 1
    assert canonical_ir_equivalence_manifest(graphs["jax"])["schema"] == "fpgai.frontend-equivalence/v1"
