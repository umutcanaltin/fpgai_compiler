from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fpgai.backends.hls.emit.dense_tiling_codegen import (
    apply_dense_tiling_to_top_source,
    dense_tile_for_layer,
    rewrite_dense_calls_with_tiling,
)


def test_dense_tile_for_layer_reads_legacy_tile_keys() -> None:
    assert dense_tile_for_layer(
        {
            "tile": {
                "in": 16,
                "out": 8,
            }
        }
    ) == (16, 8)


def test_dense_tile_for_layer_reads_typed_architecture_keys() -> None:
    assert dense_tile_for_layer(
        {
            "architecture": {
                "tiling": {
                    "input": 32,
                    "output": 4,
                }
            }
        }
    ) == (32, 4)


def test_rewrite_dense_call_inserts_tile_template_arguments() -> None:
    source = (
        "dense_out_in<128, 64, in_t, out_t, w_t, b_t, acc_t>"
        "(x, y, W0, B0);"
    )

    rewritten = rewrite_dense_calls_with_tiling(
        source,
        [(16, 8)],
    )

    assert (
        "dense_out_in_tiled<128, 64, 16, 8, "
        "in_t, out_t, w_t, b_t, acc_t>"
    ) in rewritten
    assert "FPGAI real dense tiling helper" in rewritten
    assert (
        "for (int out_base = 0; out_base < OUT; "
        "out_base += TILE_OUT)"
    ) in rewritten
    assert (
        "for (int in_base = 0; in_base < IN; "
        "in_base += TILE_IN)"
    ) in rewritten


def test_apply_dense_tiling_to_top_source_matches_graph_order() -> None:
    graph = SimpleNamespace(
        ops=[
            SimpleNamespace(
                name="dense0",
                op_type="Dense",
            ),
            SimpleNamespace(
                name="relu0",
                op_type="Relu",
            ),
            SimpleNamespace(
                name="dense1",
                op_type="Dense",
            ),
        ]
    )

    source = "\n".join(
        [
            "dense_out_in<4, 3>(a, b, W0, B0);",
            "relu_inplace<3>(b);",
            "dense_out_in<3, 2>(b, c, W1, B1);",
        ]
    )

    plan = {
        "layer_plans": [
            {
                "node_name": "dense0",
                "tile": {
                    "in": 2,
                    "out": 1,
                },
            },
            {
                "node_name": "dense1",
                "architecture": {
                    "tiling": {
                        "input": 3,
                        "output": 2,
                    }
                },
            },
        ]
    }

    rewritten = apply_dense_tiling_to_top_source(
        source,
        graph,
        plan,
    )

    assert "dense_out_in_tiled<4, 3, 2, 1>" in rewritten
    assert "dense_out_in_tiled<3, 2, 3, 2>" in rewritten


def test_top_cpp_is_wrapped_for_dense_tiling() -> None:
    source = Path(
        "fpgai/backends/hls/emit/top_cpp.py"
    ).read_text(encoding="utf-8")

    assert "apply_dense_tiling_to_top_source" in source
    assert "_fpgai_dense_tiling_original_emit_top_cpp" in source


def test_dense_tile_for_layer_reads_nested_sizes() -> None:
    assert dense_tile_for_layer(
        {
            "architecture": {
                "tiling": {
                    "sizes": {
                        "in": 8,
                        "out": 4,
                    }
                }
            }
        }
    ) == (8, 4)


def test_rewrite_dense_call_handles_multiline_call_and_explicit_types() -> None:
    source = """
void top() {
    dense_out_in<676, 10, op3_act_t, op4_act_t, op4_wgt_t, op4_bias_t, op4_acc_t, 1, 1, 1, 1, 1, 1>(
        layer_3_out,
        layer_4_out,
        W1,
        B1
    );
}
"""

    rewritten = rewrite_dense_calls_with_tiling(source, [(64, 10)])

    assert "dense_out_in_tiled<676, 10, 64, 10, op3_act_t, op4_act_t, op4_wgt_t" in rewritten
    assert "typename IN_T = act_t" not in rewritten
    assert "const W_T weights[OUT * IN]" in rewritten
    assert "const W_T weights[OUT][IN]" in rewritten
