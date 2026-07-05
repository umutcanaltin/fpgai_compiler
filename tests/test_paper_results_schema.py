from __future__ import annotations

import csv
import json
from pathlib import Path

from fpgai.reporting.paper_results import MASTER_RESULT_FIELDS, build_master_result_row, write_master_results, write_schema_json


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _example(tmp_path: Path, name: str = "demo") -> Path:
    out = tmp_path / name
    _write_json(out / "manifest.json", {"pipeline_mode": "inference", "model_path": "models/demo.onnx", "build_stages": {"cpp": True, "hls_project": True, "runtime_package": True}})
    _write_json(out / "reports" / "model_profile.json", {"graph_name": "demo_graph", "model_path": "models/demo.onnx", "pipeline_mode": "inference", "parameter_bytes": 128})
    _write_json(out / "reports" / "resource_prediction.json", {"totals": {"LUT": 10, "FF": 20, "BRAM18": 1, "DSP": 2, "URAM": 0}})
    _write_json(out / "reports" / "timing_prediction.json", {"clock_mhz": 100, "predicted_latency_ms": 0.25, "predicted_cycles": 25000, "predicted_throughput_fps": 4000, "predicted_parallel_macs": 4})
    _write_json(out / "reports" / "board_fit.json", {"status": "fits", "board": "kv260", "part": "xck26", "limiting_dimension": "none", "vivado_implementation_allowed": True, "bitstream_allowed": True})
    _write_json(out / "reports" / "generated_cpp_validation.json", {"status": "passed"})
    _write_json(out / "reports" / "movement_contract_validation.json", {"status": "passed", "passed": True})
    _write_json(out / "reports" / "runtime_sequence.json", {"memory_semantics_mode": "embedded", "sequence": [{"command": "run_inference"}]})
    _write_json(out / "runtime_package" / "buffer_plan.json", {"buffers": [{"name": "input"}, {"name": "output"}]})
    _write_json(out / "runtime_package" / "package_manifest.json", {"status": "created", "board": "kv260", "pipeline_mode": "inference", "build_stages": {"cpp": True, "hls_project": True, "runtime_package": True}})
    return out


def test_build_master_result_row_reads_existing_reports(tmp_path: Path) -> None:
    out = _example(tmp_path)
    row = build_master_result_row(out)
    assert set(MASTER_RESULT_FIELDS).issubset(row.keys())
    assert row["design_id"] == "demo"
    assert row["model"] == "demo_graph"
    assert row["board"] == "kv260"
    assert row["estimated_lut"] == 10
    assert row["estimated_ff"] == 20
    assert row["estimated_bram18"] == 1
    assert row["estimated_dsp"] == 2
    assert row["estimated_latency_ms"] == 0.25
    assert row["generated_cpp_status"] == "passed"
    assert row["movement_validation_status"] == "passed"
    assert row["support_status"] == "static_validation_passed"
    assert row["hls_status"] == "not_run"
    assert "hls_synthesis" in row["required_validation"]


def test_write_master_results_outputs_json_csv_md_and_schema(tmp_path: Path) -> None:
    out = _example(tmp_path)
    results = write_master_results(
        [out],
        output_json=tmp_path / "paper_results" / "master_results.json",
        output_csv=tmp_path / "paper_results" / "master_results.csv",
        output_md=tmp_path / "paper_results" / "master_results.md",
        schema_json=tmp_path / "paper_results" / "schema" / "master_result_schema.json",
        schema_md=tmp_path / "paper_results" / "schema" / "master_result_schema.md",
    )
    assert results["summary"]["rows"] == 1
    assert results["summary"]["with_hls_synthesis_result"] == 0
    assert results["summary"]["with_vivado_implementation_result"] == 0
    assert results["summary"]["with_runtime_result"] == 0
    assert results["summary"]["with_static_validation"] == 1
    data = json.loads((tmp_path / "paper_results" / "master_results.json").read_text(encoding="utf-8"))
    assert data["artifact_kind"] == "paper_master_results"
    with (tmp_path / "paper_results" / "master_results.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["design_id"] == "demo"
    schema = json.loads((tmp_path / "paper_results" / "schema" / "master_result_schema.json").read_text(encoding="utf-8"))
    assert schema["artifact_kind"] == "paper_master_result_schema"
    assert "estimated_lut" in [field["name"] for field in schema["fields"]]


def test_schema_writer_has_all_master_fields(tmp_path: Path) -> None:
    path = write_schema_json(tmp_path / "schema.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    names = [field["name"] for field in data["fields"]]
    assert names == MASTER_RESULT_FIELDS
