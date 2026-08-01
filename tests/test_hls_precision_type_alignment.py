from pathlib import Path

from fpgai.backends.hls.codegen import emit_hls_stub
from fpgai.ir.graph import Graph


def test_hls_project_uses_precision_aware_types_matching_axis_width(tmp_path: Path) -> None:
    graph = Graph("precision_alignment")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1, 4), "float32")
    graph.add_tensor("output", (1, 4), "float32")
    graph.add_op("Relu", ["input"], ["output"], "relu_0")

    raw_cfg = {
        "numerics": {
            "kind": "fixed",
            "defaults": {
                "activation": {
                    "type": "ap_fixed",
                    "total_bits": 16,
                    "int_bits": 6,
                }
            },
        }
    }

    emit_hls_stub(
        graph=graph,
        out_dir=tmp_path,
        top_name="deeplearn",
        hls_options={
            "pipeline_mode": "inference",
            "weights_mode": "embedded",
            "part": "xck26-sfvc784-2LV-c",
            "clk_mhz": 200,
            "run_csim": False,
            "run_csynth": False,
            "raw_cfg": raw_cfg,
        },
    )

    types_source = (tmp_path / "hls/include/fpgai_types.h").read_text(encoding="utf-8")
    top_source = (tmp_path / "hls/src/deeplearn.cpp").read_text(encoding="utf-8")

    assert "typedef ap_fixed<16,6> op0_act_t;" in types_source
    assert "typedef float op0_act_t;" not in types_source
    assert "static const int FPGAI_ACT_BITS = 16;" in top_source
    assert "fpgai_unpack_axis_value<op0_act_t, FPGAI_ACT_BITS>" in top_source
