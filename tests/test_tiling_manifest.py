from __future__ import annotations

import json
from types import SimpleNamespace

from fpgai.analysis.tiling_manifest import (
    attach_tiling_analysis_to_manifest,
    tiling_analysis_manifest_entry,
    write_tiling_analysis_artifact,
)


class _Section(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)


def _dense_layer():
    return SimpleNamespace(
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


def test_tiling_analysis_manifest_entry_is_compact() -> None:
    report = {
        "format": "fpgai.tiling_analysis.v1",
        "totals": {
            "tiled_layer_count": 2,
            "implemented_tiled_layer_count": 1,
            "planning_only_tiled_layer_count": 1,
            "local_buffer_elements": 44,
            "estimated_activation_reads": 32,
            "estimated_weight_reads": 128,
            "estimated_output_writes": 8,
            "estimated_macs": 128,
        },
        "layers": [
            {
                "layer_name": "dense0",
            }
        ],
    }

    entry = tiling_analysis_manifest_entry(
        report,
        path="reports/tiling_analysis.json",
    )

    assert entry == {
        "format": "fpgai.tiling_analysis.v1",
        "path": "reports/tiling_analysis.json",
        "tiled_layer_count": 2,
        "implemented_tiled_layer_count": 1,
        "planning_only_tiled_layer_count": 1,
        "local_buffer_elements": 44,
        "estimated_activation_reads": 32,
        "estimated_weight_reads": 128,
        "estimated_output_writes": 8,
        "estimated_macs": 128,
    }


def test_attach_tiling_analysis_to_existing_manifest() -> None:
    manifest = {
        "format": "fpgai.compile_manifest.v1",
        "artifacts": {},
    }
    report = {
        "format": "fpgai.tiling_analysis.v1",
        "totals": {
            "tiled_layer_count": 1,
            "implemented_tiled_layer_count": 1,
        },
    }

    updated = attach_tiling_analysis_to_manifest(
        manifest,
        report,
    )

    assert updated is manifest
    assert updated["tiling_analysis"]["path"] == "tiling_analysis.json"
    assert updated["tiling_analysis"]["tiled_layer_count"] == 1
    assert updated["tiling_analysis"]["implemented_tiled_layer_count"] == 1


def test_write_tiling_analysis_artifact_writes_json_and_updates_manifest(tmp_path) -> None:
    compile_plan = SimpleNamespace(
        layer_plans=[
            _dense_layer(),
        ]
    )
    manifest = {
        "format": "fpgai.compile_manifest.v1",
    }

    report, updated = write_tiling_analysis_artifact(
        tmp_path,
        compile_plan,
        manifest=manifest,
    )

    output = tmp_path / "tiling_analysis.json"
    assert output.exists()

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == report
    assert updated is manifest
    assert updated["tiling_analysis"]["path"] == "tiling_analysis.json"
    assert updated["tiling_analysis"]["tiled_layer_count"] == 1
    assert updated["tiling_analysis"]["implemented_tiled_layer_count"] == 1
    assert updated["tiling_analysis"]["local_buffer_elements"] == 44


def test_write_tiling_analysis_artifact_can_create_new_manifest(tmp_path) -> None:
    compile_plan = SimpleNamespace(
        layer_plans=[
            _dense_layer(),
        ]
    )

    _, manifest = write_tiling_analysis_artifact(
        tmp_path,
        compile_plan,
        filename="reports_tiling.json",
    )

    assert manifest["tiling_analysis"]["path"] == "reports_tiling.json"
    assert manifest["tiling_analysis"]["estimated_macs"] == 128
