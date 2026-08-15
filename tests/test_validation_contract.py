from __future__ import annotations

import json
from pathlib import Path

from fpgai.benchmark.verification import emit_validation_summary_artifacts
from fpgai.validation.numeric import emit_numeric_validation_report


def test_numeric_validation_inference_is_conservative_without_outputs(tmp_path: Path) -> None:
    artifacts = emit_numeric_validation_report(
        tmp_path,
        pipeline_mode="inference",
        source_generated=True,
        hls_ran=False,
    )
    payload = json.loads(artifacts["numeric_validation_json"].read_text(encoding="utf-8"))

    assert payload["status"] == "not_run"
    assert payload["passed"] is False
    assert payload["validation_claim_allowed"]["numeric_correctness"] is False
    assert artifacts["numeric_validation_md"].exists()


def test_validation_summary_requires_numeric_validation_for_validation_ready_inference(tmp_path: Path) -> None:
    numeric = emit_numeric_validation_report(
        tmp_path,
        pipeline_mode="inference",
        source_generated=True,
        hls_ran=False,
    )
    benchmark = emit_validation_summary_artifacts(
        tmp_path,
        pipeline_mode="inference",
        source_generated=True,
        numeric_validation_json=numeric["numeric_validation_json"],
        hls_ran=False,
        build_stages={"cpp": True, "hls_synthesis": False},
    )

    payload = json.loads(benchmark["validation_summary_json"].read_text(encoding="utf-8"))
    row = json.loads(benchmark["benchmark_row_json"].read_text(encoding="utf-8"))

    assert payload["validation_flags"]["source_generated"] is True
    assert payload["validation_flags"]["numeric_validated"] is False
    assert payload["validated_capabilities"]["source_generation"] is True
    assert payload["validated_capabilities"]["numeric_correctness"] is False
    assert payload["validation_ready"] is False
    assert row["validation_ready"] is False
