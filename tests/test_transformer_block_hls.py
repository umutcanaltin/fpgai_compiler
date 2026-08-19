from __future__ import annotations

from pathlib import Path

from fpgai.backends.hls.project import emit_hls_project
from fpgai.benchmark.graph_reference import deterministic_graph_inputs, execute_graph_reference
from fpgai.benchmark.model_graphs import build_demo_transformer_block_graph


def test_transformer_block_reference_executes_and_is_finite():
    g = build_demo_transformer_block_graph()
    inputs = deterministic_graph_inputs(g)
    out = execute_graph_reference(g, inputs)
    assert out.shape == (1, 4, 8)
    assert bool((out == out).all())


def test_transformer_block_hls_codegen_contains_projection_rope_and_serialized_mha(tmp_path: Path):
    g = build_demo_transformer_block_graph()
    inputs = deterministic_graph_inputs(g)
    expected = execute_graph_reference(g, inputs)
    build = emit_hls_project(g, tmp_path / "build", input_values=inputs, expected_output=expected, artifact_namespace="transformer_block", result_schema="fpgai.transformer-block-hls-result/v1", reference_schema="fpgai.transformer-block-hls-reference/v1", pass_token="FPGAI_TRANSFORMER_BLOCK_HLS_PASS")
    src = build.project.top_cpp.read_text(encoding="utf-8")
    assert src.count("matmul_tiled<") >= 4
    assert src.count("rotary_embedding_pairs<") == 2
    assert "multi_head_attention_serialized<4, 8, 2" in src
    assert "rms_norm_rows<" in src
    assert "fpgai_matmul_right_0" in src
    assert build.reference_report.is_file()
