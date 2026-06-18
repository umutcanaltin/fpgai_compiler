from __future__ import annotations

import json
from types import SimpleNamespace

from fpgai.analysis.tiling_reports import (
    attach_tiling_sweep_to_manifest,
    collect_layer_architecture_assumptions,
    tiling_sweep_manifest_entry,
    write_tiling_report_artifacts,
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
                    precision=_Section(
                        activation_bits=16,
                        weight_bits=8,
                        accumulator_bits=32,
                    ),
                    parallelism=_Section(
                        pe=2,
                        simd=2,
                    ),
                    pipeline=_Section(
                        ii=1,
                    ),
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


def test_collect_layer_architecture_assumptions_from_typed_objects() -> None:
    assumptions = collect_layer_architecture_assumptions(
        _dense_plan(),
    )

    assert assumptions["precision_by_layer"]["dense0"] == {
        "activation_bits": 16,
        "weight_bits": 8,
        "accumulator_bits": 32,
    }
    assert assumptions["parallelism_by_layer"]["dense0"] == {
        "pe": 2,
        "simd": 2,
    }
    assert assumptions["ii_by_layer"]["dense0"] == {
        "ii": 1,
    }


def test_write_tiling_report_artifacts_writes_three_reports_and_manifest(tmp_path) -> None:
    manifest = {
        "format": "fpgai.compile_manifest.v1",
    }

    reports, updated = write_tiling_report_artifacts(
        tmp_path,
        _dense_plan(),
        manifest=manifest,
        clock_mhz=200,
        memory_words_per_cycle=4,
        tile_overhead_cycles=1,
    )

    assert updated is manifest
    assert set(reports) == {
        "tiling_analysis",
        "tiling_resource_estimate",
        "tiling_performance_estimate",
    }

    for filename in (
        "tiling_analysis.json",
        "tiling_resource_estimate.json",
        "tiling_performance_estimate.json",
    ):
        path = tmp_path / filename
        assert path.exists()
        assert json.loads(path.read_text(encoding="utf-8"))

    assert manifest["tiling_analysis"]["path"] == "tiling_analysis.json"
    assert manifest["tiling_resource_estimate"]["path"] == "tiling_resource_estimate.json"
    assert manifest["tiling_performance_estimate"]["path"] == "tiling_performance_estimate.json"
    assert manifest["tiling_resource_estimate"]["tile_buffer_bits"] == 512
    assert manifest["tiling_performance_estimate"]["estimated_macs"] == 128


def test_write_tiling_report_artifacts_can_include_sweep(tmp_path) -> None:
    reports, manifest = write_tiling_report_artifacts(
        tmp_path,
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
        sweep_top_k=1,
    )

    assert "tiling_sweep" in reports
    assert (tmp_path / "tiling_sweep.json").exists()
    assert manifest["tiling_sweep"]["path"] == "tiling_sweep.json"
    assert manifest["tiling_sweep"]["layer_count"] == 1
    assert manifest["tiling_sweep"]["best_layer_count"] == 1
    assert len(reports["tiling_sweep"]["layers"][0]["ranking"]) == 1


def test_tiling_sweep_manifest_entry_for_single_layer_report() -> None:
    entry = tiling_sweep_manifest_entry(
        {
            "format": "fpgai.tiling_sweep.v1",
            "layer_name": "dense0",
            "candidate_count": 3,
            "best": {
                "score": 1.0,
            },
        },
        path="sweep.json",
    )

    assert entry == {
        "format": "fpgai.tiling_sweep.v1",
        "path": "sweep.json",
        "layer_name": "dense0",
        "candidate_count": 3,
        "has_best": True,
    }


def test_attach_tiling_sweep_to_manifest() -> None:
    manifest = {}
    updated = attach_tiling_sweep_to_manifest(
        manifest,
        {
            "format": "fpgai.tiling_sweep_collection.v1",
            "layer_count": 1,
            "best_by_layer": {
                "dense0": {
                    "score": 1.0,
                },
            },
        },
    )

    assert updated is manifest
    assert manifest["tiling_sweep"]["layer_count"] == 1
    assert manifest["tiling_sweep"]["best_layer_count"] == 1
