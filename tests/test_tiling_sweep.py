from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from fpgai.analysis.tiling_sweep import (
    conv_tile_candidates,
    dense_tile_candidates,
    score_tile_candidate,
    sweep_compile_plan_tiles,
    sweep_layer_tiles,
    write_tiling_sweep_json,
)


class _Section(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)


def _dense_plan():
    return SimpleNamespace(
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


def test_dense_tile_candidates_filter_by_layer_dimensions() -> None:
    candidates = dense_tile_candidates(
        input_features=16,
        output_features=8,
        input_tiles=[4, 8, 32],
        output_tiles=[2, 4, 16],
    )

    assert candidates == [
        {"in": 4, "out": 2},
        {"in": 4, "out": 4},
        {"in": 8, "out": 2},
        {"in": 8, "out": 4},
    ]


def test_conv_tile_candidates_filter_by_layer_dimensions() -> None:
    candidates = conv_tile_candidates(
        input_channels=4,
        output_channels=8,
        output_height=7,
        output_width=7,
        input_channel_tiles=[1, 8],
        output_channel_tiles=[4, 16],
        output_height_tiles=[7, 14],
        output_width_tiles=[7],
    )

    assert candidates == [
        {
            "input_channels": 1,
            "output_channels": 4,
            "output_height": 7,
            "output_width": 7,
        }
    ]


def test_score_tile_candidate_does_not_mutate_original_plan() -> None:
    plan = _dense_plan()

    result = score_tile_candidate(
        plan,
        layer_name="dense0",
        tile_sizes={
            "in": 4,
            "out": 4,
        },
        parallelism_by_layer={
            "dense0": {
                "pe": 2,
                "simd": 2,
            }
        },
        memory_words_per_cycle=2,
        tile_overhead_cycles=1,
    )

    assert result["layer_name"] == "dense0"
    assert result["tile"] == {
        "in": 4,
        "out": 4,
    }
    assert result["performance"]["estimated_macs"] == 128
    assert result["resources"]["tiled_layer_count"] == 1

    original_sizes = plan.layer_plans[0].architecture.tiling.sizes
    assert original_sizes == {
        "in": 8,
        "out": 4,
    }


def test_score_tile_candidate_rejects_missing_layer() -> None:
    with pytest.raises(ValueError, match="Layer not found"):
        score_tile_candidate(
            _dense_plan(),
            layer_name="missing",
            tile_sizes={
                "in": 4,
                "out": 4,
            },
        )


def test_sweep_layer_tiles_ranks_candidates_and_keeps_best() -> None:
    report = sweep_layer_tiles(
        _dense_plan(),
        layer_name="dense0",
        candidates=[
            {
                "in": 4,
                "out": 2,
            },
            {
                "in": 8,
                "out": 4,
            },
        ],
        parallelism_by_layer={
            "dense0": {
                "pe": 2,
                "simd": 2,
            }
        },
        memory_words_per_cycle=2,
        tile_overhead_cycles=1,
        bram_weight=0,
    )

    assert report["format"] == "fpgai.tiling_sweep.v1"
    assert report["candidate_count"] == 2
    assert len(report["ranking"]) == 2
    assert report["best"] == report["ranking"][0]
    assert report["ranking"][0]["score"] <= report["ranking"][1]["score"]


def test_sweep_compile_plan_tiles_collects_best_by_layer() -> None:
    report = sweep_compile_plan_tiles(
        _dense_plan(),
        candidates_by_layer={
            "dense0": [
                {
                    "in": 4,
                    "out": 2,
                },
                {
                    "in": 8,
                    "out": 4,
                },
            ]
        },
        top_k=1,
    )

    assert report["format"] == "fpgai.tiling_sweep_collection.v1"
    assert report["layer_count"] == 1
    assert "dense0" in report["best_by_layer"]
    assert len(report["layers"][0]["ranking"]) == 1


def test_write_tiling_sweep_json(tmp_path) -> None:
    report = sweep_layer_tiles(
        _dense_plan(),
        layer_name="dense0",
        candidates=[
            {
                "in": 8,
                "out": 4,
            }
        ],
    )
    output = tmp_path / "tiling_sweep.json"

    written = write_tiling_sweep_json(
        output,
        report,
    )

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == written
    assert loaded["best"]["tile"] == {
        "in": 8,
        "out": 4,
    }
