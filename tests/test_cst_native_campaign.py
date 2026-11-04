"""Native CST experiment preparation/audit tests: no Vitis executable needed."""
from __future__ import annotations

import json

import pytest

from fpgai_research.experiments.cst_native_campaign import (
    audit, execute, prepare, verify_prepared,
)


def test_native_triplet_prepare_is_create_only_and_has_source_provenance(tmp_path):
    root = tmp_path / "hls_cases"
    manifest = prepare(output=root, max_cases=3, clock_ns=32.0,
                       clock_uncertainty_ns=8.64)
    assert manifest["schema"] == "fpgai-research/cst-ir-original-vitis-experiment/v1"
    assert not manifest["original_native_tool_executed_by_preparation"]
    assert [c["variant"] for c in manifest["cases"]] == [
        "whole_tensor", "column_major", "row_lifetime"]
    assert len({c["cpp_sha256"] for c in manifest["cases"]}) == 3
    assert len({c["functional_sha256"] for c in manifest["cases"]}) == 1
    for row in manifest["cases"]:
        case = root / row["case"]
        assert (case / "model.cpp").is_file()
        assert (case / "tb.cpp").is_file()
        assert (case / "run_hls.tcl").is_file()
        assert (case / "manifest.json").is_file()
        assert not (case / "vitis_invocation.json").exists()
    _, checked = verify_prepared(root)
    assert len(checked) == 3
    dry = execute(root, execute_native=False, max_calls=3)
    assert dry["original_vitis_calls"] == 0
    assert dry["requires_explicit_execute_native"]
    results = audit(root)
    assert results["cases_with_native_csim_and_bounded_csynth_cycles"] == 0
    assert all(not x["hls_csim_attested_with_bounded_latency"]
               and x["requested_period_scaled_hls_latency_ns"] is None
               for x in results["cases"])
    with pytest.raises(FileExistsError, match="Create-only"):
        prepare(output=root, max_cases=3)


def test_native_prepare_rejects_unpaired_or_invalid_clock(tmp_path):
    with pytest.raises(ValueError, match="multiple of three"):
        prepare(output=tmp_path / "invalid", max_cases=4)
    with pytest.raises(ValueError, match="Clock and uncertainty"):
        prepare(output=tmp_path / "invalid2", max_cases=3,
                clock_ns=8.0, clock_uncertainty_ns=8.64)
    with pytest.raises(ValueError, match="max_cases"):
        prepare(output=tmp_path / "invalid3", max_cases=2)


def test_native_audit_fails_closed_on_source_tamper(tmp_path):
    root = tmp_path / "tamper"
    prepare(output=root, max_cases=3)
    target = root / "case_000" / "model.cpp"
    target.write_text(target.read_text() + "// malicious mutation\n")
    with pytest.raises(ValueError, match="Prepared HLS input changed"):
        audit(root)


def test_native_run_only_if_explicitly_requested(tmp_path, monkeypatch):
    root = tmp_path / "safe"
    prepare(output=root, max_cases=3)
    def prohibit(*args, **kwargs):
        raise AssertionError("Original Vitis must never run in dry mode")
    monkeypatch.setattr(
        "fpgai_research.experiments.cst_native_campaign.run_vitis", prohibit)
    assert execute(root, execute_native=False, max_calls=1)["original_vitis_calls"] == 0
    with pytest.raises(ValueError, match="max_calls must be 1..24"):
        execute(root, execute_native=True, max_calls=0)


def test_native_historical_audit_allows_source_revision_to_change(tmp_path, monkeypatch):
    root = tmp_path / "historical"
    prepare(output=root, max_cases=3)
    monkeypatch.setattr(
        "fpgai_research.experiments.cst_native_campaign.revision",
        lambda: "another-revision")
    assert len(audit(root)["cases"]) == 3
    with pytest.raises(ValueError, match="Original source revision mismatch"):
        verify_prepared(root)
