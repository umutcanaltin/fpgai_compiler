from pathlib import Path

from fpgai.backends.hls.project import emit_hls_project
from fpgai.benchmark.graph_reference import deterministic_graph_inputs, execute_graph_reference
from fpgai.benchmark.model_graphs import build_demo_attention_graph


def test_attention_reference_and_project_emit_real_testbench(tmp_path: Path) -> None:
    graph = build_demo_attention_graph(sequence_length=4, head_dimension=8)
    inputs = deterministic_graph_inputs(graph)
    expected = execute_graph_reference(graph, inputs)
    assert expected.shape == (1, 4, 8)
    build = emit_hls_project(graph, tmp_path / "build", input_values=inputs, expected_output=expected, artifact_namespace="attention", result_schema="fpgai.attention-hls-result/v1", reference_schema="fpgai.attention-hls-reference/v1", pass_token="FPGAI_ATTENTION_HLS_PASS")
    top = build.project.top_cpp.read_text(encoding="utf-8")
    tb = build.project.tb_cpp.read_text(encoding="utf-8")
    assert "matmul_tiled<4, 8, 4" in top
    assert "softmax_rows<4, 4" in top
    assert "FPGAI_ATTENTION_HLS_PASS" in tb
    assert build.reference_report.exists()
