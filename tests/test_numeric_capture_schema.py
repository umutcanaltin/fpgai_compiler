from __future__ import annotations

import json
from pathlib import Path

import pytest

from fpgai.validation.capture_schema import (
    NumericCaptureContract,
    compare_capture_contracts,
    default_training_capture_requirements,
    write_capture_contract,
)


def _contract(kind: str, workload: str = "same") -> dict:
    captures = default_training_capture_requirements(optimizer_type="adam", export_gradients=True)
    return NumericCaptureContract(
        workload_fingerprint_sha256=workload,
        implementation_stack_fingerprint_sha256=f"impl-{kind}",
        producer_kind=kind,
        producer_id=f"producer.{kind}",
        captures=captures,
        metadata={"mechanism": "fused_update"},
    ).to_dict()


def test_capture_contract_is_backend_neutral_and_fingerprinted(tmp_path: Path) -> None:
    payload = _contract("hls_csim")
    assert payload["artifact_kind"] == "fpgai_numeric_capture_contract"
    assert payload["producer"]["kind"] == "hls_csim"
    assert payload["captures"]["optimizer_m_after"]["required"] is True
    assert payload["captures"]["parameter_gradients"]["status"] == "capture_pending"
    path = write_capture_contract(
        tmp_path / "capture.json",
        NumericCaptureContract(
            workload_fingerprint_sha256="w",
            implementation_stack_fingerprint_sha256="i",
            producer_kind="rtl_simulation",
            producer_id="community.dense.vhdl",
            captures=default_training_capture_requirements(optimizer_type="sgd", export_gradients=False),
            metadata={},
        ),
    )
    assert json.loads(path.read_text())["producer"]["kind"] == "rtl_simulation"


def test_capture_comparison_rejects_workload_mismatch() -> None:
    result = compare_capture_contracts(_contract("python_reference", "a"), _contract("hls_csim", "b"))
    assert result["status"] == "workload_mismatch"
    assert result["same_workload"] is False


def test_capture_comparison_remains_pending_until_required_artifacts_exist() -> None:
    result = compare_capture_contracts(_contract("python_reference"), _contract("hls_csim"))
    assert result["status"] == "capture_pending"
    assert "weights_after" in result["missing_required_captures"]


def test_capture_contract_rejects_unknown_producer_kind() -> None:
    with pytest.raises(ValueError):
        NumericCaptureContract(
            workload_fingerprint_sha256="w",
            implementation_stack_fingerprint_sha256="i",
            producer_kind="unknown",
            producer_id="x",
            captures={},
            metadata={},
        ).to_dict()
