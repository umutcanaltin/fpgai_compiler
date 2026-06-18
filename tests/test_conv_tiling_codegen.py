from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fpgai.backends.hls.emit.conv_tiling_codegen import (
    apply_conv_tiling_to_top_source,
    conv_tile_for_layer,
    rewrite_conv_calls_with_tiling,
)


def test_conv_tile_for_layer_reads_legacy_tile_keys() -> None:
    assert conv_tile_for_layer(
        {
            "tile": {
                "oc": 4,
                "oh": 8,
                "ow": 8,
                "ic": 2,
            }
        }
    ) == (4, 8, 8, 2)


def test_conv_tile_for_layer_reads_typed_architecture_keys() -> None:
    assert conv_tile_for_layer(
        {
            "architecture": {
                "tiling": {
                    "output_channels": 6,
                    "output_height": 5,
                    "output_width": 7,
                    "input_channels": 3,
                }
            }
        }
    ) == (6, 5, 7, 3)


def test_rewrite_conv_call_inserts_tile_template_arguments() -> None:
    source = (
        "conv2d<28, 28, 1, 26, 26, 8, 3, 1, 0, "
        "in_t, out_t, w_t, b_t, acc_t>(x, y, W0, B0);"
    )

    rewritten = rewrite_conv_calls_with_tiling(
        source,
        [(4, 13, 13, 1)],
    )

    assert (
        "conv2d_tiled<28, 28, 1, 26, 26, 8, 3, 1, 0, "
        "4, 13, 13, 1, in_t, out_t, w_t, b_t, acc_t>"
    ) in rewritten
    assert "FPGAI real convolution tiling helper" in rewritten
    assert "for (int oc_base = 0; oc_base < OUT_C; oc_base += TILE_OC)" in rewritten
    assert "for (int oh_base = 0; oh_base < OUT_H; oh_base += TILE_OH)" in rewritten
    assert "for (int ow_base = 0; ow_base < OUT_W; ow_base += TILE_OW)" in rewritten
    assert "for (int ic_base = 0; ic_base < IN_C; ic_base += TILE_IC)" in rewritten


def test_apply_conv_tiling_to_top_source_matches_graph_order() -> None:
    graph = SimpleNamespace(
        ops=[
            SimpleNamespace(name="conv0", op_type="Conv"),
            SimpleNamespace(name="relu0", op_type="Relu"),
            SimpleNamespace(name="conv1", op_type="Conv"),
        ]
    )

    source = "\n".join(
        [
            "conv2d<28, 28, 1, 26, 26, 8, 3, 1, 0>(a, b, W0, B0);",
            "relu_inplace<5408>(b);",
            "conv2d<26, 26, 8, 24, 24, 16, 3, 1, 0>(b, c, W1, B1);",
        ]
    )

    plan = {
        "layer_plans": [
            {
                "node_name": "conv0",
                "tile": {
                    "oc": 4,
                    "oh": 13,
                    "ow": 13,
                    "ic": 1,
                },
            },
            {
                "node_name": "conv1",
                "architecture": {
                    "tiling": {
                        "output_channels": 8,
                        "output_height": 12,
                        "output_width": 12,
                        "input_channels": 4,
                    }
                },
            },
        ]
    }

    rewritten = apply_conv_tiling_to_top_source(
        source,
        graph,
        plan,
    )

    assert "conv2d_tiled<28, 28, 1, 26, 26, 8, 3, 1, 0, 4, 13, 13, 1>" in rewritten
    assert "conv2d_tiled<26, 26, 8, 24, 24, 16, 3, 1, 0, 8, 12, 12, 4>" in rewritten


def test_top_cpp_is_wrapped_for_conv_tiling() -> None:
    source = Path(
        "fpgai/backends/hls/emit/top_cpp.py"
    ).read_text(encoding="utf-8")

    assert "apply_conv_tiling_to_top_source" in source
    assert "_fpgai_conv_tiling_original_emit_top_cpp" in source
