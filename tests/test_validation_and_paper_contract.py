from __future__ import annotations

import json
from pathlib import Path

from fpgai.paper.verification import emit_paper_verification_artifacts
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
    assert payload["paper_claim_allowed"]["numeric_correctness"] is False
    assert artifacts["numeric_validation_md"].exists()


def test_paper_verification_requires_numeric_validation_for_paper_safe_inference(tmp_path: Path) -> None:
    numeric = emit_numeric_validation_report(
        tmp_path,
        pipeline_mode="inference",
        source_generated=True,
        hls_ran=False,
    )
    paper = emit_paper_verification_artifacts(
        tmp_path,
        pipeline_mode="inference",
        source_generated=True,
        numeric_validation_json=numeric["numeric_validation_json"],
        hls_ran=False,
        build_stages={"cpp": True, "hls_synthesis": False},
    )

    payload = json.loads(paper["paper_verification_json"].read_text(encoding="utf-8"))
    row = json.loads(paper["paper_row_json"].read_text(encoding="utf-8"))

    assert payload["verification_flags"]["source_generated"] is True
    assert payload["verification_flags"]["numeric_validated"] is False
    assert payload["allowed_claims"]["source_generation"] is True
    assert payload["allowed_claims"]["numeric_correctness"] is False
    assert payload["paper_safe"] is False
    assert row["paper_safe"] is False
