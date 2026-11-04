"""R2o original multi-reader C3 GCC and read-only receipt controls."""
from __future__ import annotations

from copy import deepcopy
import json
import shutil

import pytest

from fpgai_research.experiments import (
    cst_r2o_multi_reader_action_audit as audit_module,
    cst_r2o_multi_reader_action_study as original,
)


@pytest.fixture(scope="module")
def one_original_gcc_study(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("Original host GCC required")
    destination = tmp_path_factory.mktemp("r2o") / "original"
    summary = original.run(output=destination, graphs=1)
    return destination, summary


def _copy_original(one_original_gcc_study, tmp_path):
    source, _ = one_original_gcc_study
    dest = tmp_path / "copy"
    shutil.copytree(source, dest)
    return dest


def test_real_source_pinned_case_has_three_actual_GCC_and_one_rejection(
        one_original_gcc_study):
    dest, summary = one_original_gcc_study
    assert summary["graph_count"] == 1
    assert summary["original_action_denominator"] == 4
    assert summary["status_denominators"] == {
        "E1_GCC_MULTI_READER_QUALIFIED": 3,
        "REJECTED": 1,
    }
    assert summary["original_GCC_qualified_actions"] == 3
    assert summary["original_GCC_source_qualification_attempts"] == 3
    assert summary["original_Vitis_calls"] == 0
    assert summary["original_Vivado_calls"] == 0
    assert summary["trained_RL_policies"] == 0
    evidence = json.loads((dest / "graph_000" / "receipt.json").read_text())
    assert evidence["fanout_readers"] == 2
    assert evidence["evidence"]["candidate_count"] == 4
    assert len(evidence["evidence"]["rows"]) == 4
    assert [r["status"] for r in evidence["evidence"]["rows"]] == [
        "E1_GCC_MULTI_READER_QUALIFIED",
        "E1_GCC_MULTI_READER_QUALIFIED",
        "REJECTED",
        "E1_GCC_MULTI_READER_QUALIFIED",
    ]
    actual = audit_module.audit(dest)
    assert actual["original_actions"] == 4
    assert actual["source_GCC_e1_attempts"] == 3
    assert actual["source_GCC_qualified_actions"] == 3
    assert actual["original_vendor_calls_during_audit"] == 0
    assert len(actual["per_graph"][0]["gcc_address_witness_counts"]) == 3
    with pytest.raises(FileExistsError, match="create-only"):
        original.run(output=dest, graphs=1)


def test_invalid_original_cohort_and_create_only_preflight(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 3"):
        original.run(output=tmp_path/"out0", graphs=0)
    with pytest.raises(ValueError, match="between 1 and 3"):
        original.run(output=tmp_path/"out4", graphs=4)
    with pytest.raises(ValueError, match="between 1 and 3"):
        original.run(output=tmp_path/"outbool", graphs=True)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="create-only"):
        original.run(output=existing, graphs=1)


def test_corrupt_original_stdout_is_rejected_without_rerunning_GCC(
        one_original_gcc_study, tmp_path):
    copied = _copy_original(one_original_gcc_study, tmp_path)
    trace = copied / "graph_000" / "case_000_original_stdout.txt"
    trace.write_text(trace.read_text()+"\nFPGAI_MUTATED_WRONG_SOURCE\n")
    with pytest.raises(ValueError, match="SHA or layout mismatch"):
        audit_module.audit(copied)


def test_corrupt_original_source_is_rejected_without_GCC(
        one_original_gcc_study, tmp_path):
    copied = _copy_original(one_original_gcc_study, tmp_path)
    source = copied / "graph_000" / "case_001_source.cpp"
    source.write_text(source.read_text()+"\n// forged source\n")
    with pytest.raises(ValueError, match="SHA or layout mismatch"):
        audit_module.audit(copied)


def test_forged_e1_status_without_e1_receipt_rejected(
        one_original_gcc_study, tmp_path):
    copied = _copy_original(one_original_gcc_study, tmp_path)
    p = copied / "graph_000" / "receipt.json"
    d = json.loads(p.read_text())
    original_row = d["evidence"]["rows"][2]
    original_row["status"] = "E1_GCC_MULTI_READER_QUALIFIED"
    original_row["source_admissible"] = True
    original_row["e0_certified"] = True
    original_row["e1_qualified"] = True
    d["evidence"]["statuses"][2] = "E1_GCC_MULTI_READER_QUALIFIED"
    d["evidence"]["e0_certified"][2] = True
    d["evidence"]["e1_qualified"][2] = True
    p.write_text(json.dumps(d))
    # The forged row has no state/source/certificate SHA. The auditor must
    # reject the missing E0 identity *before* considering the absent E1
    # receipt; it must never promote this fabricated action to E1.
    with pytest.raises(
            ValueError, match="E0-admitted original action lacks certificate"):
        audit_module.audit(copied)


def test_forged_e1_status_with_fake_e0_but_no_gcc_receipt_rejected(
        one_original_gcc_study, tmp_path):
    copied = _copy_original(one_original_gcc_study, tmp_path)
    p = copied / "graph_000" / "receipt.json"
    d = json.loads(p.read_text())
    # Give the forged producer-PE action syntactically valid E0 digests,
    # so the independent auditor reaches its separate actual-GCC-receipt
    # check. Digest presence is *not* proof of executed source.
    row = d["evidence"]["rows"][2]
    source = d["evidence"]["rows"][0]
    row["status"] = "E1_GCC_MULTI_READER_QUALIFIED"
    row["source_admissible"] = True
    row["e0_certified"] = True
    row["e1_qualified"] = True
    for key in ("state_sha256", "cpp_sha256", "certificate_sha256"):
        row[key] = source[key]
    d["evidence"]["statuses"][2] = "E1_GCC_MULTI_READER_QUALIFIED"
    d["evidence"]["e0_certified"][2] = True
    d["evidence"]["e1_qualified"][2] = True
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="original GCC receipt absent"):
        audit_module.audit(copied)


def test_original_nine_action_denominator_cannot_be_faked(
        one_original_gcc_study, tmp_path):
    copied = _copy_original(one_original_gcc_study, tmp_path)
    p = copied / "summary.json"
    d = json.loads(p.read_text())
    d["original_action_denominator"] += 1
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="population/status mismatch"):
        audit_module.audit(copied)


def test_unknown_actions_not_misreported_as_qualified(
        one_original_gcc_study, tmp_path):
    copied = _copy_original(one_original_gcc_study, tmp_path)
    p = copied / "graph_000" / "receipt.json"
    d = json.loads(p.read_text())
    case = d["evidence"]["rows"][0]
    case["status"] = "UNKNOWN_E1_NOT_RUN"
    case["e1_qualified"] = False
    d["evidence"]["statuses"][0] = "UNKNOWN_E1_NOT_RUN"
    d["evidence"]["e1_qualified"][0] = False
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="unknown or failed GCC action promoted|per-graph"):
        audit_module.audit(copied)


def test_original_input_sha_cannot_be_changed(
        one_original_gcc_study, tmp_path):
    copied = _copy_original(one_original_gcc_study, tmp_path)
    p = copied / "graph_000" / "receipt.json"
    d = json.loads(p.read_text())
    d["input_float32_sha256"] = "a"*64
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="input SHA mismatch"):
        audit_module.audit(copied)


def test_symlinked_original_evidence_refused(
        one_original_gcc_study, tmp_path):
    copied = _copy_original(one_original_gcc_study, tmp_path)
    trace = copied / "graph_000" / "case_000_original_stdout.txt"
    trace.unlink()
    trace.symlink_to(copied / "graph_000" / "case_001_original_stdout.txt")
    with pytest.raises(ValueError, match="symlinked"):
        audit_module.audit(copied)
