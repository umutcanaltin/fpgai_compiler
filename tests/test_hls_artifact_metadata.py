from __future__ import annotations

import json
from pathlib import Path

from fpgai.analysis.hls_artifact_metadata import emit_hls_artifact_metadata
from fpgai.engine.models import CompilePlan


def test_emit_hls_artifact_metadata_collects_files_and_layers(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    inc_dir = tmp_path / "include"
    src_dir.mkdir()
    inc_dir.mkdir()

    (src_dir / "top.cpp").write_text("void top() {}\n", encoding="utf-8")
    (inc_dir / "dense.h").write_text("#pragma once\n", encoding="utf-8")
    (tmp_path / "ignore.bin").write_bytes(b"binary")

    plan = {
        "architecture_signature": "arch-test",
        "layer_plans": [
            {
                "name": "dense0",
                "op_type": "Dense",
                "precision": "ap_fixed<16,6>",
                "pipeline_ii": 2,
                "tile": {},
                "unroll": {"in": 4, "out": 2},
                "architecture": {
                    "pipeline": {"ii": 2},
                    "parallelism": {"pe": 2, "simd": 4},
                },
                "architecture_signature": "layer-arch-test",
            }
        ],
    }

    result = emit_hls_artifact_metadata(
        tmp_path,
        plan,
        schedule_summary={
            "path": "hls_schedule_summary.json",
            "summary": {
                "report_count": 1,
                "loop_count": 2,
            },
        },
    )

    assert result["path"] == "hls_artifact_metadata.json"
    assert result["architecture_signature"] == "arch-test"
    assert result["layer_count"] == 1
    assert result["file_count"] == 2

    data = json.loads(
        (tmp_path / "hls_artifact_metadata.json").read_text(encoding="utf-8")
    )

    assert data["schema_version"] == 1
    assert data["architecture_signature"] == "arch-test"
    assert data["layers"][0]["name"] == "dense0"
    assert data["hls_schedule_summary"]["summary"]["loop_count"] == 2

    file_paths = {entry["path"] for entry in data["files"]}
    assert "src/top.cpp" in file_paths
    assert "include/dense.h" in file_paths
    assert "ignore.bin" not in file_paths


def test_compiler_records_hls_artifact_metadata_in_manifest_source() -> None:
    source = Path("fpgai/engine/compiler.py").read_text(encoding="utf-8")

    assert "emit_hls_artifact_metadata" in source
    assert "hls_artifact_metadata=hls_artifact_metadata" in source
    assert 'manifest["hls_artifact_metadata"] = hls_artifact_metadata' in source
