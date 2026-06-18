from __future__ import annotations

from types import SimpleNamespace

from fpgai.backends.hls.codegen import emit_hls_stub


def test_emit_hls_stub_uses_real_inference_top_emitter_when_graph_is_complete(
    tmp_path,
    monkeypatch,
) -> None:
    calls = {}

    def fake_emit_top_cpp(**kwargs):
        calls.update(kwargs)
        return """
#include <hls_stream.h>
extern "C" void deeplearn() {
    dense_out_in_tiled<16, 8, 8, 4, float, float, float, float, float>(
        nullptr, nullptr, nullptr, nullptr
    );
}
"""

    monkeypatch.setattr(
        "fpgai.backends.hls.codegen.emit_top_cpp",
        fake_emit_top_cpp,
    )

    graph = SimpleNamespace(
        inputs=["input"],
        outputs=["output"],
        ops=[
            SimpleNamespace(
                name="dense0",
                op_type="Dense",
                inputs=["input", "W", "B"],
                outputs=["output"],
                attrs={},
            )
        ],
    )
    compile_plan = SimpleNamespace(layer_plans=[])

    project = emit_hls_stub(
        graph=graph,
        out_dir=tmp_path,
        top_name="deeplearn",
        hls_options={
            "pipeline_mode": "inference",
            "weights_mode": "embedded",
            "run_csim": False,
            "run_csynth": False,
        },
        compile_plan=compile_plan,
    )

    source = project.top_cpp.read_text(encoding="utf-8")

    assert "dense_out_in_tiled" in source
    assert calls["graph"] is graph
    assert calls["top_name"] == "deeplearn"
    assert calls["weights_mode"] == "embedded"
    assert calls["compile_plan"] is compile_plan


def test_emit_hls_stub_keeps_stub_fallback_for_incomplete_graph(tmp_path) -> None:
    project = emit_hls_stub(
        graph=SimpleNamespace(),
        out_dir=tmp_path,
        top_name="deeplearn",
        hls_options={
            "pipeline_mode": "inference",
            "run_csim": False,
            "run_csynth": False,
        },
    )

    source = project.top_cpp.read_text(encoding="utf-8")

    assert source == 'extern "C" void deeplearn() {}\n'
