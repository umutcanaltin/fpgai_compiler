from __future__ import annotations

import json
from types import SimpleNamespace

from fpgai.analysis.tiling_performance_model import (
    analyze_tiling_performance,
    attach_tiling_performance_estimate_to_manifest,
    estimate_layer_tiling_performance,
    estimate_tiling_performance,
    write_tiling_performance_estimate_json,
)


class _Section(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)


def test_estimate_layer_tiling_performance_uses_parallelism_and_ii() -> None:
    layer_report = {
        "layer_name": "dense0",
        "op_type": "Dense",
        "tile_counts": {
            "total_tiles": 16,
        },
        "estimated_traffic": {
            "activation_reads": 256,
            "weight_reads": 2048,
            "output_writes": 32,
            "macs": 2048,
        },
    }

    estimate = estimate_layer_tiling_performance(
        layer_report,
        parallelism_by_layer={
            "dense0": {
                "pe": 4,
                "simd": 2,
            }
        },
        ii_by_layer={
            "dense0": 2,
        },
        clock_mhz=200,
        memory_words_per_cycle=4,
        tile_overhead_cycles=3,
    )

    assert estimate["cycle_estimate"]["compute_cycles"] == 512
    assert estimate["cycle_estimate"]["memory_cycles"] == 584
    assert estimate["cycle_estimate"]["tile_overhead_cycles"] == 48
    assert estimate["cycle_estimate"]["overlapped_total_cycles"] == 632
    assert estimate["cycle_estimate"]["non_overlapped_total_cycles"] == 1144
    assert estimate["latency_estimate"]["overlapped_latency_us"] == 3.16
    assert estimate["bottleneck"] == "memory"


def test_estimate_tiling_performance_sums_layers() -> None:
    tiling_report = {
        "format": "fpgai.tiling_analysis.v1",
        "layers": [
            {
                "layer_name": "dense0",
                "op_type": "Dense",
                "tile_counts": {
                    "total_tiles": 4,
                },
                "estimated_traffic": {
                    "activation_reads": 32,
                    "weight_reads": 128,
                    "output_writes": 8,
                    "macs": 128,
                },
            },
            {
                "layer_name": "pool0",
                "op_type": "MaxPool",
                "status": "planning_only",
            },
        ],
    }

    report = estimate_tiling_performance(
        tiling_report,
        parallelism_by_layer={
            "dense0": {
                "pe": 2,
                "simd": 2,
            }
        },
        clock_mhz=100,
        memory_words_per_cycle=2,
        tile_overhead_cycles=1,
    )

    assert report["format"] == "fpgai.tiling_performance_model.v1"
    assert report["totals"]["tiled_layer_count"] == 1
    assert report["totals"]["compute_cycles"] == 32
    assert report["totals"]["memory_cycles"] == 84
    assert report["totals"]["overlapped_total_cycles"] == 88
    assert report["totals"]["overlapped_latency_us"] == 0.88


def test_analyze_tiling_performance_from_compile_plan() -> None:
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

    report = analyze_tiling_performance(
        compile_plan,
        parallelism_by_layer={
            "dense0": {
                "pe": 2,
                "simd": 2,
            }
        },
        clock_mhz=200,
        memory_words_per_cycle=4,
        tile_overhead_cycles=1,
    )

    assert report["totals"]["tiled_layer_count"] == 1
    assert report["totals"]["estimated_macs"] == 128
    assert report["totals"]["estimated_memory_words"] == 168
    assert report["totals"]["compute_cycles"] == 32
    assert report["totals"]["memory_cycles"] == 42


def test_write_tiling_performance_json_and_manifest_entry(tmp_path) -> None:
    tiling_report = {
        "format": "fpgai.tiling_analysis.v1",
        "layers": [
            {
                "layer_name": "dense0",
                "op_type": "Dense",
                "tile_counts": {
                    "total_tiles": 4,
                },
                "estimated_traffic": {
                    "activation_reads": 32,
                    "weight_reads": 128,
                    "output_writes": 8,
                    "macs": 128,
                },
            },
        ],
    }

    output = tmp_path / "tiling_performance_estimate.json"
    report = write_tiling_performance_estimate_json(
        output,
        tiling_report,
        parallelism_by_layer={
            "dense0": {
                "pe": 2,
                "simd": 2,
            }
        },
        memory_words_per_cycle=2,
        input_is_tiling_report=True,
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == report

    manifest = {
        "format": "fpgai.compile_manifest.v1",
    }
    updated = attach_tiling_performance_estimate_to_manifest(
        manifest,
        report,
    )

    assert updated is manifest
    assert manifest["tiling_performance_estimate"]["path"] == "tiling_performance_estimate.json"
    assert manifest["tiling_performance_estimate"]["tiled_layer_count"] == 1
    assert manifest["tiling_performance_estimate"]["estimated_macs"] == 128
    assert manifest["tiling_performance_estimate"]["estimated_memory_words"] == 168
