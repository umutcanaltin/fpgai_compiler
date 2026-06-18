from __future__ import annotations

import json
from types import SimpleNamespace

from fpgai.analysis.tiling_analysis import (
    analyze_tiling,
    write_tiling_analysis_json,
)


class _Section(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)


def _layer(
    name: str,
    op_type: str,
    sizes: dict,
    **dims,
):
    return SimpleNamespace(
        node_name=name,
        op_type=op_type,
        architecture=SimpleNamespace(
            tiling=_Section(sizes=sizes),
        ),
        **dims,
    )


def test_dense_tiling_analysis_reports_tile_counts_and_buffers() -> None:
    compile_plan = SimpleNamespace(
        layer_plans=[
            _layer(
                "dense0",
                "Dense",
                {"in": 16, "out": 8},
                input_features=64,
                output_features=32,
            )
        ]
    )

    report = analyze_tiling(compile_plan)
    layer = report["layers"][0]

    assert report["format"] == "fpgai.tiling_analysis.v1"
    assert report["totals"]["tiled_layer_count"] == 1
    assert report["totals"]["implemented_tiled_layer_count"] == 1
    assert layer["op_type"] == "Dense"
    assert layer["tile"] == {
        "input_features": 16,
        "output_features": 8,
    }
    assert layer["tile_counts"] == {
        "input_tiles": 4,
        "output_tiles": 4,
        "total_tiles": 16,
    }
    assert layer["local_buffers"]["input_tile_elements"] == 16
    assert layer["local_buffers"]["weight_tile_elements"] == 128
    assert layer["local_buffers"]["accumulator_tile_elements"] == 8
    assert layer["estimated_traffic"]["macs"] == 2048
    assert layer["estimated_traffic"]["activation_reads"] == 256


def test_conv_tiling_analysis_reports_spatial_channel_counts_and_buffers() -> None:
    compile_plan = SimpleNamespace(
        layer_plans=[
            _layer(
                "conv0",
                "Conv",
                {
                    "output_channels": 4,
                    "output_height": 7,
                    "output_width": 7,
                    "input_channels": 2,
                },
                input_height=28,
                input_width=28,
                input_channels=8,
                output_height=14,
                output_width=14,
                output_channels=16,
                kernel=3,
                stride=1,
                padding=1,
            )
        ]
    )

    report = analyze_tiling(compile_plan)
    layer = report["layers"][0]

    assert layer["op_type"] == "Conv"
    assert layer["tile"] == {
        "output_channels": 4,
        "output_height": 7,
        "output_width": 7,
        "input_channels": 2,
    }
    assert layer["tile_counts"]["output_channel_tiles"] == 4
    assert layer["tile_counts"]["output_height_tiles"] == 2
    assert layer["tile_counts"]["output_width_tiles"] == 2
    assert layer["tile_counts"]["input_channel_tiles"] == 4
    assert layer["tile_counts"]["total_tiles"] == 64
    assert layer["local_buffers"]["accumulator_tile_elements"] == 196
    assert layer["local_buffers"]["weight_tile_elements"] == 72
    assert layer["estimated_traffic"]["output_writes"] == 3136
    assert layer["estimated_traffic"]["macs"] == 225792


def test_unsupported_tiled_operator_is_marked_planning_only() -> None:
    compile_plan = SimpleNamespace(
        layer_plans=[
            _layer(
                "pool0",
                "MaxPool",
                {"height": 2, "width": 2},
            )
        ]
    )

    report = analyze_tiling(compile_plan)
    layer = report["layers"][0]

    assert report["totals"]["tiled_layer_count"] == 1
    assert report["totals"]["implemented_tiled_layer_count"] == 0
    assert report["totals"]["planning_only_tiled_layer_count"] == 1
    assert layer["status"] == "planning_only"


def test_write_tiling_analysis_json(tmp_path) -> None:
    compile_plan = SimpleNamespace(
        layer_plans=[
            _layer(
                "dense0",
                "Dense",
                {"in": 8, "out": 4},
                input_features=16,
                output_features=8,
            )
        ]
    )
    output = tmp_path / "tiling_analysis.json"

    report = write_tiling_analysis_json(
        output,
        compile_plan,
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == report
    assert loaded["totals"]["local_buffer_elements"] == 44
