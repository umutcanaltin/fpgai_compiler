from __future__ import annotations

import json
from types import SimpleNamespace

from fpgai.analysis.tiling_resource_model import (
    analyze_tiling_resource_overhead,
    attach_tiling_resource_estimate_to_manifest,
    estimate_layer_tile_buffer_bits,
    estimate_tiling_resource_overhead,
    write_tiling_resource_estimate_json,
)


class _Section(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)


def test_estimate_layer_tile_buffer_bits_uses_precision_by_layer() -> None:
    layer_report = {
        "layer_name": "dense0",
        "op_type": "Dense",
        "local_buffers": {
            "input_tile_elements": 8,
            "weight_tile_elements": 32,
            "accumulator_tile_elements": 4,
        },
    }

    estimate = estimate_layer_tile_buffer_bits(
        layer_report,
        precision_by_layer={
            "dense0": {
                "activation_bits": 16,
                "weight_bits": 8,
                "accumulator_bits": 32,
            }
        },
    )

    assert estimate["tile_buffer_bits"] == {
        "activation": 128,
        "weight": 256,
        "accumulator": 128,
        "total": 512,
    }
    assert estimate["estimated_bram18"] == 1
    assert estimate["estimated_bram36"] == 1


def test_estimate_tiling_resource_overhead_sums_dense_and_conv_layers() -> None:
    tiling_report = {
        "format": "fpgai.tiling_analysis.v1",
        "layers": [
            {
                "layer_name": "dense0",
                "op_type": "Dense",
                "local_buffers": {
                    "input_tile_elements": 8,
                    "weight_tile_elements": 32,
                    "accumulator_tile_elements": 4,
                },
            },
            {
                "layer_name": "conv0",
                "op_type": "Conv",
                "local_buffers": {
                    "input_tile_elements": 200,
                    "weight_tile_elements": 72,
                    "accumulator_tile_elements": 196,
                },
            },
            {
                "layer_name": "pool0",
                "op_type": "MaxPool",
                "status": "planning_only",
            },
        ],
    }

    report = estimate_tiling_resource_overhead(
        tiling_report,
        precision_by_layer={
            "dense0": {
                "activation_bits": 16,
                "weight_bits": 8,
                "accumulator_bits": 32,
            },
            "conv0": {
                "activation_bits": 12,
                "weight_bits": 10,
                "accumulator_bits": 24,
            },
        },
    )

    dense_bits = 512
    conv_bits = 200 * 12 + 72 * 10 + 196 * 24
    assert report["format"] == "fpgai.tiling_resource_model.v1"
    assert report["totals"]["tiled_layer_count"] == 2
    assert report["totals"]["tile_buffer_bits"] == dense_bits + conv_bits
    assert report["totals"]["tile_buffer_elements"] == 8 + 32 + 4 + 200 + 72 + 196
    assert len(report["layers"]) == 2


def test_analyze_tiling_resource_overhead_from_compile_plan() -> None:
    compile_plan = SimpleNamespace(
        layer_plans=[
            SimpleNamespace(
                node_name="dense0",
                op_type="Dense",
                architecture=SimpleNamespace(
                    tiling=_Section(
                        sizes={
                            "in": 8,
                            "out": 4,
                        }
                    ),
                ),
                input_features=16,
                output_features=8,
            )
        ]
    )

    report = analyze_tiling_resource_overhead(
        compile_plan,
        precision_by_layer={
            "dense0": {
                "activation_bits": 16,
                "weight_bits": 8,
                "accumulator_bits": 32,
            }
        },
    )

    assert report["totals"]["tiled_layer_count"] == 1
    assert report["totals"]["tile_buffer_elements"] == 44
    assert report["totals"]["tile_buffer_bits"] == 512


def test_write_tiling_resource_estimate_json_and_manifest_entry(tmp_path) -> None:
    tiling_report = {
        "format": "fpgai.tiling_analysis.v1",
        "layers": [
            {
                "layer_name": "dense0",
                "op_type": "Dense",
                "local_buffers": {
                    "input_tile_elements": 8,
                    "weight_tile_elements": 32,
                    "accumulator_tile_elements": 4,
                },
            },
        ],
    }

    output = tmp_path / "tiling_resource_estimate.json"
    report = write_tiling_resource_estimate_json(
        output,
        tiling_report,
        precision_by_layer={
            "dense0": {
                "activation_bits": 16,
                "weight_bits": 8,
                "accumulator_bits": 32,
            }
        },
        input_is_tiling_report=True,
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == report

    manifest = {
        "format": "fpgai.compile_manifest.v1",
    }
    updated = attach_tiling_resource_estimate_to_manifest(
        manifest,
        report,
    )

    assert updated is manifest
    assert manifest["tiling_resource_estimate"] == {
        "format": "fpgai.tiling_resource_model.v1",
        "path": "tiling_resource_estimate.json",
        "tiled_layer_count": 1,
        "tile_buffer_elements": 44,
        "tile_buffer_bits": 512,
        "estimated_bram18": 1,
        "estimated_bram36": 1,
    }
