from __future__ import annotations

from pathlib import Path

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from fpgai.backends.hls.project import emit_hls_project
from fpgai.benchmark.graph_reference import deterministic_graph_inputs, execute_graph_reference
from fpgai.frontend.mlir.stablehlo import import_stablehlo_mlir
from fpgai.ir.passes.attention_lowering import plan_attention_lowering


def _stablehlo(fn, *specs) -> str:
    exported = jax.export.export(jax.jit(fn))(*specs)
    return str(exported.mlir_module())


def test_real_jax_attention_stablehlo_reaches_existing_hls_backend(tmp_path: Path):
    def attention(q, k, v):
        scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * (1.0 / jnp.sqrt(8.0))
        probs = jax.nn.softmax(scores, axis=-1)
        return jnp.matmul(probs, v)

    spec = jax.ShapeDtypeStruct((1, 4, 8), jnp.float32)
    graph = import_stablehlo_mlir(_stablehlo(attention, spec, spec, spec), source_framework="jax")
    assert [op.op_type for op in graph.ops] == ["Transpose", "MatMul", "Mul", "Softmax", "MatMul"]
    plans = plan_attention_lowering(graph, tile_m=2, tile_n=2, tile_k=4)
    assert len(plans) == 1
    inputs = deterministic_graph_inputs(graph)
    expected = execute_graph_reference(graph, inputs)
    build = emit_hls_project(graph, tmp_path / "jax_attention", input_values=inputs, expected_output=expected, artifact_namespace="attention", result_schema="fpgai.attention-hls-result/v1", reference_schema="fpgai.attention-hls-reference/v1", pass_token="FPGAI_ATTENTION_HLS_PASS")
    cpp = build.project.top_cpp.read_text(encoding="utf-8")
    assert cpp.count("matmul_tiled<") == 2
    assert cpp.count("softmax_rows<") == 1
    assert cpp.count("scale_vector<") == 1
    assert build.reference_report.exists()


def test_real_jax_layernorm_and_rmsnorm_canonicalize():
    x = jax.ShapeDtypeStruct((1, 4, 8), jnp.float32)
    scale = jax.ShapeDtypeStruct((8,), jnp.float32)

    def rms(v, s):
        return v * jax.lax.rsqrt(jnp.mean(v * v, axis=-1, keepdims=True) + 1e-5) * s

    def layer(v, s, b):
        mean = jnp.mean(v, axis=-1, keepdims=True)
        centered = v - mean
        variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
        return centered * jax.lax.rsqrt(variance + 1e-5) * s + b

    rms_graph = import_stablehlo_mlir(_stablehlo(rms, x, scale), source_framework="jax")
    layer_graph = import_stablehlo_mlir(_stablehlo(layer, x, scale, scale), source_framework="jax")
    assert [op.op_type for op in rms_graph.ops] == ["RMSNorm"]
    assert [op.op_type for op in layer_graph.ops] == ["LayerNormalization"]


def test_real_jax_attention_plus_rmsnorm_reaches_same_hls_project(tmp_path: Path):
    def model(q, k, v, scale):
        scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * (1.0 / jnp.sqrt(8.0))
        probs = jax.nn.softmax(scores, axis=-1)
        context = jnp.matmul(probs, v)
        return context * jax.lax.rsqrt(jnp.mean(context * context, axis=-1, keepdims=True) + 1e-5) * scale

    spec = jax.ShapeDtypeStruct((1, 4, 8), jnp.float32)
    scale = jax.ShapeDtypeStruct((8,), jnp.float32)
    graph = import_stablehlo_mlir(_stablehlo(model, spec, spec, spec, scale), source_framework="jax")
    assert [op.op_type for op in graph.ops] == ["Transpose", "MatMul", "Mul", "Softmax", "MatMul", "RMSNorm"]
    assert len(plan_attention_lowering(graph, tile_m=2, tile_n=2, tile_k=4)) == 1
    inputs = deterministic_graph_inputs(graph)
    expected = execute_graph_reference(graph, inputs)
    build = emit_hls_project(graph, tmp_path / "jax_attention_rmsnorm", input_values=inputs, expected_output=expected, artifact_namespace="attention", result_schema="fpgai.attention-hls-result/v1", reference_schema="fpgai.attention-hls-reference/v1", pass_token="FPGAI_ATTENTION_HLS_PASS")
    cpp = build.project.top_cpp.read_text(encoding="utf-8")
    assert cpp.count("matmul_tiled<") == 2
    assert "softmax_rows<" in cpp
    assert "rms_norm_rows<" in cpp
