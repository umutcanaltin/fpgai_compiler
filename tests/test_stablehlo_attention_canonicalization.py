from __future__ import annotations

import numpy as np

from fpgai.frontend.mlir.stablehlo import import_stablehlo_mlir


JAX_SOFTMAX_STABLEHLO = r'''module @jit_f {
  func.func public @main(%arg0: tensor<1x4x4xf32>) -> (tensor<1x4x4xf32>) {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [2] : (tensor<1x4x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %2 = stablehlo.maximum %1, %0 : tensor<1x4xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x4xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x4x4xf32>
    %6 = stablehlo.exponential %5 : tensor<1x4x4xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x4x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x4xf32>
    %10 = stablehlo.divide %6, %9 : tensor<1x4x4xf32>
    return %10 : tensor<1x4x4xf32>
  }
}'''

JAX_RMSNORM_STABLEHLO = r'''module @jit_rms {
  func.func public @main(%arg0: tensor<1x4x8xf32>, %arg1: tensor<8xf32>) -> (tensor<1x4x8xf32>) {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<1x4x8xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x4x8xf32>, tensor<f32>) -> tensor<1x4xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %cst_0 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4x1xf32>
    %4 = stablehlo.divide %2, %3 : tensor<1x4x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x4x1xf32>
    %6 = stablehlo.add %4, %5 : tensor<1x4x1xf32>
    %7 = stablehlo.rsqrt %6 : tensor<1x4x1xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x8xf32>
    %9 = stablehlo.multiply %arg0, %8 : tensor<1x4x8xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<8xf32>) -> tensor<1x1x8xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<1x1x8xf32>) -> tensor<1x4x8xf32>
    %12 = stablehlo.multiply %9, %11 : tensor<1x4x8xf32>
    return %12 : tensor<1x4x8xf32>
  }
}'''


def test_jax_softmax_decomposition_canonicalizes_to_single_fpgai_softmax():
    graph = import_stablehlo_mlir(JAX_SOFTMAX_STABLEHLO, source_framework="jax")
    assert len(graph.ops) == 1
    op = graph.ops[0]
    assert op.op_type == "Softmax"
    assert op.inputs == ["arg0"]
    assert op.attrs["axis"] == 2
    assert graph.metadata["canonicalizations"][0]["pass"] == "stablehlo_softmax"


def test_jax_rmsnorm_decomposition_canonicalizes_to_rmsnorm():
    graph = import_stablehlo_mlir(JAX_RMSNORM_STABLEHLO, source_framework="jax")
    assert len(graph.ops) == 1
    op = graph.ops[0]
    assert op.op_type == "RMSNorm"
    assert op.inputs == ["arg0", "arg1"]
    assert op.attrs["axis"] == 2
    assert np.isclose(op.attrs["epsilon"], 1e-5, rtol=1e-5)


def test_batch1_dot_layout_canonicalizer_accepts_extra_reshape_views():
    from fpgai.frontend.mlir.stablehlo import import_stablehlo_mlir

    text = r'''
module {
  func.func @main(%q: tensor<1x4x8xf32>, %kt: tensor<1x8x4xf32>) -> tensor<1x4x4xf32> {
    %0 = stablehlo.reshape %q : (tensor<1x4x8xf32>) -> tensor<4x8xf32>
    %1 = stablehlo.reshape %kt : (tensor<1x8x4xf32>) -> tensor<8x4xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    %3 = stablehlo.reshape %2 : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
    return %3 : tensor<1x4x4xf32>
  }
}
'''
    graph = import_stablehlo_mlir(text, source_framework="jax")
    assert [op.op_type for op in graph.ops] == ["MatMul"]
    assert graph.ops[0].inputs == ["q", "kt"]
    assert tuple(graph.get_tensor(graph.ops[0].outputs[0]).shape) == (1, 4, 4)


def test_batch1_dot_layout_canonicalizer_accepts_jax011_batch_restoring_broadcast():
    text = r'''
module {
  func.func @main(%q: tensor<1x4x8xf32>, %kt: tensor<1x8x4xf32>) -> tensor<1x4x4xf32> {
    %0 = stablehlo.reshape %q : (tensor<1x4x8xf32>) -> tensor<4x8xf32>
    %1 = stablehlo.reshape %kt : (tensor<1x8x4xf32>) -> tensor<8x4xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
    return %3 : tensor<1x4x4xf32>
  }
}
'''
    graph = import_stablehlo_mlir(text, source_framework="jax")
    assert [op.op_type for op in graph.ops] == ["MatMul"]
    assert graph.ops[0].inputs == ["q", "kt"]
    assert tuple(graph.get_tensor(graph.ops[0].outputs[0]).shape) == (1, 4, 4)
    reports = graph.metadata.get("canonicalizations", [])
    assert any(r.get("pass") == "stablehlo_batch1_dot_layout" and r.get("count") == 1 for r in reports)


def test_batch1_dot_layout_does_not_strip_semantic_broadcast():
    text = r'''
module {
  func.func @main(%q: tensor<1x4x8xf32>, %kt: tensor<1x8x4xf32>) -> tensor<2x4x4xf32> {
    %0 = stablehlo.reshape %q : (tensor<1x4x8xf32>) -> tensor<4x8xf32>
    %1 = stablehlo.reshape %kt : (tensor<1x8x4xf32>) -> tensor<8x4xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [1, 2] : (tensor<4x4xf32>) -> tensor<2x4x4xf32>
    return %3 : tensor<2x4x4xf32>
  }
}
'''
    graph = import_stablehlo_mlir(text, source_framework="jax")
    assert [op.op_type for op in graph.ops] == ["Reshape", "Reshape", "MatMul", "Broadcast"]


def test_jax011_full_attention_layout_canonicalizes_to_logical_attention_core():
    text = r'''
module {
  func.func @main(%q: tensor<1x4x8xf32>, %k: tensor<1x4x8xf32>, %v: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
    %kt = stablehlo.transpose %k, dims = [0, 2, 1] : (tensor<1x4x8xf32>) -> tensor<1x8x4xf32>
    %q2 = stablehlo.reshape %q : (tensor<1x4x8xf32>) -> tensor<4x8xf32>
    %k2 = stablehlo.reshape %kt : (tensor<1x8x4xf32>) -> tensor<8x4xf32>
    %score2 = stablehlo.dot_general %q2, %k2, contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    %score = stablehlo.broadcast_in_dim %score2, dims = [1, 2] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
    %scale = stablehlo.constant dense<0.3535533905932738> : tensor<f32>
    %scaled = stablehlo.multiply %score, %scale : tensor<1x4x4xf32>
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %mx = stablehlo.reduce(%scaled init: %neg_inf) applies stablehlo.maximum across dimensions = [2] : (tensor<1x4x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %mxb = stablehlo.broadcast_in_dim %mx, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %mxb2 = stablehlo.broadcast_in_dim %mxb, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x4xf32>
    %sub = stablehlo.subtract %scaled, %mxb2 : tensor<1x4x4xf32>
    %exp = stablehlo.exponential %sub : tensor<1x4x4xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %sum = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [2] : (tensor<1x4x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %sumb = stablehlo.broadcast_in_dim %sum, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %sumb2 = stablehlo.broadcast_in_dim %sumb, dims = [0, 1, 2] : (tensor<1x4x1xf32>) -> tensor<1x4x4xf32>
    %prob = stablehlo.divide %exp, %sumb2 : tensor<1x4x4xf32>
    %p2 = stablehlo.reshape %prob : (tensor<1x4x4xf32>) -> tensor<4x4xf32>
    %v2 = stablehlo.reshape %v : (tensor<1x4x8xf32>) -> tensor<4x8xf32>
    %ctx2 = stablehlo.dot_general %p2, %v2, contracting_dims = [1] x [0] : (tensor<4x4xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    %ctx = stablehlo.broadcast_in_dim %ctx2, dims = [1, 2] : (tensor<4x8xf32>) -> tensor<1x4x8xf32>
    return %ctx : tensor<1x4x8xf32>
  }
}
'''
    graph = import_stablehlo_mlir(text, source_framework="jax")
    assert [op.op_type for op in graph.ops] == ["Transpose", "MatMul", "Mul", "Softmax", "MatMul"]
    assert graph.ops[1].inputs == ["q", "kt"]
    assert graph.ops[-1].inputs[0] == graph.ops[3].outputs[0]
    assert graph.ops[-1].inputs[1] == "v"
    assert tuple(graph.get_tensor(graph.ops[-1].outputs[0]).shape) == (1, 4, 8)
