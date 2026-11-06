"""R4y prospective 24 ns source-only four-arm checks; no vendor tools."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import (
    cst_paper_r4y_24ns_pipeline_source_prepare as sut,
)


def _original_good():
    known = sut.r4k.plan()
    return {
        "schema": sut.r4l.SCHEMA,
        "paired_HLS_source_arms": 2,
        "E2_qualified_arms": 2,
        "E3_qualified_arms": 2,
        "original_vendor_calls_during_audit": 0,
        "package_hashes": {
            "full_storage": {
                "model.cpp": known["source_full_storage_sha256"]},
            "row_storage": {
                "model.cpp": known["source_row_original_sha256"]},
        },
        # Mirror actual R4l.audit_native: its R4k E1 flag is not exposed.
        "E1_matched_pair_accepted": True,
        "both_E2_qualified": True,
        "both_E3_qualified": True,
        "source_revision": "a" * 40,
        "new_vendor_calls_during_read_only_audit": 0,
        "original_sources_modified": False,
    }


@pytest.fixture
def originals(tmp_path, monkeypatch):
    roots = [tmp_path / p for p in ("old_r4k", "old_r4d", "old_r4a")]
    for root in roots:
        root.mkdir()
    monkeypatch.setattr(sut.r4l, "audit_native", lambda **_: _original_good())
    monkeypatch.setattr(sut.r4a, "_git_revision", lambda: "b" * 40)
    return {
        "r4k_study": roots[0], "original_r4d": roots[1],
        "original_r4a": roots[2],
    }


def test_source_plan_four_arms_same_graph_schedule_and_clock():
    p = sut.plan()
    assert p["clock_period_ns"] == 24.0
    assert p["clock_uncertainty_ns"] == 1.5
    assert p["numeric_format"] == [32, 6]
    assert len(p["cases"]) == 4
    assert len(set(p["sources_sha256"].values())) == 4
    assert not p["E2_CSim_qualified"]
    assert not p["E3_HLS_qualified"]
    assert not p["board_qualified"]


def test_directive_is_single_loop_scoped_with_reverse_identity():
    _, graph, cases = sut._emissions()
    for storage in sut.STORAGE:
        baseline = cases[f"{storage}__baseline"]
        candidate = cases[f"{storage}__producer_mac_ii1_requested"]
        assert candidate.cpp.count(sut.DIRECTIVE) == 1
        assert candidate.cpp.replace(sut.DIRECTIVE, "", 1) == baseline.cpp
        assert candidate.manifest["source_event_order_sha256"] == baseline.manifest[
            "source_event_order_sha256"]
        assert candidate.manifest["functional_sha256"] == baseline.manifest[
            "functional_sha256"]
        assert candidate.manifest["requested_hls_directive_count"] == 1
        assert candidate.manifest["actual_hls_ii_verified"] is None


def test_fail_closed_when_original_loop_scaffold_changed():
    from fpgai_research.codegen.hls.emitter import Emission
    with pytest.raises(ValueError, match="canonical producer MAC loop"):
        sut._pipeline_request(Emission("not a loop", {}), k=3)


def test_four_packages_are_create_only_and_byte_matched(originals, tmp_path):
    output = tmp_path / "new_r4y"
    receipt = sut.prepare(output=output, **originals)
    assert receipt["status"] == "PREPARED_FOUR_SOURCE_ONLY_NEW_VITIS_PACKAGES"
    assert receipt["native_vendor_calls"] == 0
    assert receipt["original_R4k_source_revision"] == "a" * 40
    for case in sut.CASES:
        root = output / case
        model = (root / "vitis_source" / "model.cpp").read_bytes()
        assert sha256(model).hexdigest() == receipt["sources_sha256"][case]
        tcl = (root / "vitis_source" / "run_hls.tcl").read_text()
        assert "create_clock -period 24" in tcl
        assert "set_clock_uncertainty 1.500" in tcl
    assert sut.audit(study=output, **originals)["matched_testbench_and_tcl"]
    with pytest.raises(FileExistsError, match="create-only"):
        sut.prepare(output=output, **originals)


@pytest.mark.parametrize("touched", [
    "full_storage__baseline/source.cpp",
    "row_storage__producer_mac_ii1_requested/vitis_source/model.cpp",
    "full_storage__baseline/vitis_source/run_hls.tcl",
    "row_storage__baseline/vitis_source/tb.cpp",
    "full_storage__baseline/source_manifest.json",
])
def test_source_tcl_or_oracle_tamper_rejected(originals, tmp_path, touched):
    root = tmp_path / "fresh_r4y"
    sut.prepare(output=root, **originals)
    p = root / touched
    p.write_text(p.read_text() + "\nBAD\n")
    with pytest.raises(ValueError):
        sut.audit(study=root, **originals)


def test_fake_manifest_cannot_override_a_changed_source(originals, tmp_path):
    root = tmp_path / "fresh_r4y"
    sut.prepare(output=root, **originals)
    path = root / "full_storage__baseline/source.cpp"
    path.write_text(path.read_text() + " ")
    manifest = json.loads((root / "artifact_sha256.json").read_text())
    manifest["full_storage__baseline/source.cpp"] = sha256(
        path.read_bytes()).hexdigest()
    (root / "artifact_sha256.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source or package model drift"):
        sut.audit(study=root, **originals)


def test_original_provenance_failure_fails_before_allocating(originals, tmp_path,
                                                            monkeypatch):
    monkeypatch.setattr(sut.r4l, "audit_native",
                        lambda **_: {**_original_good(), "both_E3_qualified": False})
    root = tmp_path / "not_created"
    with pytest.raises(ValueError, match="preflight"):
        sut.prepare(output=root, **originals)
    assert not root.exists()


def test_original_folder_overlap_rejected(originals):
    bad = originals["r4k_study"] / "bad"
    with pytest.raises(ValueError, match="overlap"):
        sut.prepare(output=bad, **originals)


def test_invalid_extra_file_and_symlink_rejected(originals, tmp_path):
    root = tmp_path / "new_r4y"
    sut.prepare(output=root, **originals)
    p = root / "full_storage__baseline" / "vitis_source" / "injected.v"
    p.write_text("module injected; endmodule")
    with pytest.raises(ValueError, match="unexpected file"):
        sut.audit(study=root, **originals)


def test_plan_and_audit_only_never_call_vendor(originals, tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("Vendor execution unexpectedly requested")
    from fpgai_research.toolchains.vitis_hls import runner as hlsrunner
    from fpgai_research.toolchains.vivado import runner as vivadorunner
    monkeypatch.setattr(hlsrunner, "run_vitis", forbidden)
    monkeypatch.setattr(vivadorunner, "run_vivado", forbidden)
    assert sut.plan()["native_vendor_calls"] == 0
    dest = tmp_path / "run"
    sut.prepare(output=dest, **originals)
    assert sut.audit(study=dest, **originals)[
        "new_vendor_calls_during_audit"] == 0


def test_original_source_revision_not_enough_if_cpp_sha_mismatches(
        originals, tmp_path, monkeypatch):
    wrong = _original_good()
    wrong["package_hashes"]["row_storage"]["model.cpp"] = "0" * 64
    monkeypatch.setattr(sut.r4l, "audit_native", lambda **_: wrong)
    output = tmp_path / "should_not_exist"
    with pytest.raises(ValueError, match="source SHAs"):
        sut.prepare(output=output, **originals)
    assert not output.exists()


def test_symlinked_package_directory_rejected(originals, tmp_path):
    output = tmp_path / "prepared"
    sut.prepare(output=output, **originals)
    pkg = output / "row_storage__baseline" / "vitis_source"
    moved = output / "symlink_target"
    pkg.rename(moved)
    pkg.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        sut.audit(study=output, **originals)


def test_source_manifest_roundtrip_tuples_are_not_an_evidence_failure(
        originals, tmp_path):
    """The canonical source-slot map contains tuples; stored JSON uses lists."""
    root = tmp_path / "fresh_four_cases"
    sut.prepare(output=root, **originals)
    _, _, emitted = sut._emissions()
    case = "row_storage__baseline"
    stored = json.loads((root / case / "source_manifest.json").read_text())
    assert stored == json.loads(json.dumps(emitted[case].manifest))
    assert sut.audit(study=root, **originals)[
        "status"] == "SOURCE_ONLY_FOUR_ARM_PACKAGE_AUDITED_NOT_NATIVE"


def test_real_r4l_audit_schema_without_r4k_source_only_key_passes(
        originals, tmp_path):
    # R4l.audit_native does not include the R4k summary-only E1 flag.
    case = _original_good()
    assert "source_only_E1_verified" not in case
    assert sut._original(**originals)["schema"] == sut.r4l.SCHEMA
    study = tmp_path / "schema_matched"
    sut.prepare(output=study, **originals)
    assert sut.audit(study=study, **originals)["matched_testbench_and_tcl"]


@pytest.mark.parametrize("field,invalid", [
    ("schema", "incorrect"),
    ("paired_HLS_source_arms", 1),
    ("E2_qualified_arms", 1),
    ("E3_qualified_arms", 1),
    ("original_vendor_calls_during_audit", 1),
    ("new_vendor_calls_during_read_only_audit", 1),
])
def test_invalid_original_native_audit_metadata_fails_closed(
        originals, tmp_path, monkeypatch, field, invalid):
    old = {**_original_good(), field: invalid}
    monkeypatch.setattr(sut.r4l, "audit_native", lambda **_: old)
    destination = tmp_path / "absent_source"
    with pytest.raises(ValueError, match="preflight"):
        sut.prepare(output=destination, **originals)
    assert not destination.exists()


@pytest.mark.parametrize("field,value", [
    ("status", "PROSPECTIVE_SOURCE_ONLY_NOT_EXECUTED"),
    ("clock_period_ns", 20.0),
    ("E2_CSim_qualified", True),
])
def test_prepared_lifecycle_and_science_fields_cannot_be_forged(
        originals, tmp_path, field, value):
    root = tmp_path / "forged_summary"
    sut.prepare(output=root, **originals)
    p = root / "summary.json"
    data = json.loads(p.read_text())
    data[field] = value
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="prepared source plan changed|provenance"):
        sut.audit(study=root, **originals)
