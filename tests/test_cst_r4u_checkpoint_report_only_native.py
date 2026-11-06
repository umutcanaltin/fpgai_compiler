"""R4u checkpoint-report-only staging/native audit regression (NO real Vivado).

Mocked report files/checkpoints are SYNTHETIC. Never claim that testing
these cases measures physical E4 or proves original checkpoint provenance.
"""
from __future__ import annotations

from hashlib import sha256
import json
import sys

import pytest

from fpgai_research.experiments import cst_r4u_checkpoint_report_only_native as sut


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def originals(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in (
        "r4t_study", "study", "r4o_study", "r4k_study",
        "original_r4d", "original_r4a")}
    for root in paths.values():
        root.mkdir()
    arm_meta = {}
    r4s_arms = []
    for arm, wns, count in (
        ("full_storage", -2.170, 74),
        ("row_storage", -3.718, 71),
    ):
        original = paths["study"] / arm / "vivado" / "outputs" / "routed.dcp"
        original_sha = write(original, ("SYNTHETIC-ROUTED-DCP-" + arm).encode())
        root = paths["r4t_study"] / arm
        staged = root / sut.CHECKPOINT
        write(staged, original.read_bytes())
        write(root / sut.SCRIPT, sut.r4t._script())
        arm_meta[arm] = {
            "original_checkpoint_path": str(original),
            "posthoc_original_checkpoint_sha256": original_sha,
            "checkpoint_bytes": original.stat().st_size,
            "original_signed_timing_summary_sha256": "a"*64,
            "original_signed_drc_report_sha256": "b"*64,
            "original_routed_wns_ns": wns,
            "original_timing_path_detail_status":
                "UNKNOWN_NO_COMPLETE_PATH_DETAILS_IN_ORIGINAL_TIMING_SUMMARY",
            "original_dcp_sha_contemporaneous_in_R4p_vendor_receipt": False,
        }
        rules = {
            "DPIP-2": 29, "DPOP-3": 20,
            "DPOP-4": 24 if arm == "full_storage" else 21,
            "RTSTAT-10": 1,
        }
        assert sum(rules.values()) == count
        r4s_arms.append({
            "arm": arm,
            "original_drc_total_violations": count,
            "original_drc_rule_counts": rules,
        })
    meta = {
        "schema": sut.r4t.SCHEMA,
        "status": "ORIGINAL_R4P_ROUTED_DCP_POSTHOC_PINNED_REPORT_ONLY_READY",
        "original_source_revision": sut.r4t.SOURCE_REVISION,
        "original_report_sha_and_vendor_receipts_verified": True,
        "original_checkpoint_provenance_strength": "Synthetic posthoc checkpoint",
        "arms": arm_meta,
        "planned_new_report_only_vivado_calls": 2,
        "new_native_vivado_calls": 0,
        "original_vendor_artifacts_modified": False,
        "strict_physical_E4": None,
        "board_qualified": False,
    }
    source = {
        **meta,
        "status": "R4T_ISOLATED_CHECKPOINT_REPORT_ONLY_SOURCE_PREPARED",
    }
    write(paths["r4t_study"] / "summary.json",
          json.dumps(source, sort_keys=True))
    inventory = {
        str(p.relative_to(paths["r4t_study"])): sha256(p.read_bytes()).hexdigest()
        for p in paths["r4t_study"].rglob("*") if p.is_file()
    }
    write(paths["r4t_study"] / "artifact_sha256.json",
          json.dumps(inventory, sort_keys=True))
    monkeypatch.setattr(sut.r4t, "inspect", lambda **kwargs: meta)
    monkeypatch.setattr(
        sut.r4s, "inspect",
        lambda **kwargs: {"arms": r4s_arms},
    )
    return paths, meta, r4s_arms


@pytest.fixture
def prepared(originals, tmp_path):
    roots, meta, r4s_arms = originals
    dest = tmp_path / "new_r4u_report_only"
    assert sut.prepare(**roots, output=dest)["status"] == (
        "R4U_ISOLATED_REPORT_ONLY_NATIVE_INPUTS_PREPARED")
    return roots, dest, meta, r4s_arms


def kwargs(prepared):
    return {"native_study": prepared[1], **prepared[0]}


def test_prepared_source_only_sha_and_no_vendor_calls(prepared):
    original, root, meta, arms = prepared
    result = sut.audit_inputs(**kwargs(prepared))
    assert result["source_gate"] == (
        "R4P_R4S_R4T_R4U_POSTHOC_COPIED_DCP_SHA_ATTESTED")
    assert result["actual_vendor_calls_during_audit"] == 0
    assert result["strict_physical_E4"] is None
    manifest = json.loads((root / "artifact_sha256.json").read_text())
    assert len(manifest) == 5
    for rel, digest in manifest.items():
        assert sha256((root / rel).read_bytes()).hexdigest() == digest
    for arm in sut.ARMS:
        assert (root / arm / sut.CHECKPOINT).read_bytes() == (
            original["study"] / arm / "vivado" / "outputs" / "routed.dcp"
        ).read_bytes()
        assert (root / arm / sut.SCRIPT).read_text() == sut.r4t._script()
        assert not (root / arm / "logs").exists()
        assert not (root / arm / "reports").exists()
    dry = sut.execute(**kwargs(prepared))
    assert dry["status"] == "R4U_DRY_RUN_NO_VENDOR_CALLS"
    assert dry["planned_report_only_calls"] == 2
    assert dry["actual_vendor_calls"] == 0
    assert not (root / "r4u_native_attempt.json").exists()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.prepare(**original, output=root)


def test_original_r4t_staged_checkpoint_drift_refused(originals, tmp_path):
    paths, _, _ = originals
    source = paths["r4t_study"] / "row_storage" / sut.CHECKPOINT
    source.write_bytes(source.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="SHA changed"):
        sut.prepare(**paths, output=tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_native_r4u_staged_checkpoint_mutation_refused(prepared):
    roots, dest, _, _ = prepared
    path = dest / "full_storage" / sut.CHECKPOINT
    path.write_bytes(b"tamper")
    with pytest.raises(ValueError, match="SHA changed"):
        sut.execute(**kwargs(prepared), execute_native=False)


def test_original_report_only_script_mutation_refused(prepared):
    roots, dest, _, _ = prepared
    script = roots["r4t_study"] / "full_storage" / sut.SCRIPT
    script.write_text(script.read_text() + "route_design\n")
    with pytest.raises(ValueError, match="SHA changed"):
        sut.audit_inputs(**kwargs(prepared))


def test_native_requires_two_explicit_flags_timeout_and_tool(prepared):
    with pytest.raises(ValueError, match="second confirmation"):
        sut.execute(**kwargs(prepared), execute_native=True)
    with pytest.raises(ValueError, match="timeout invalid"):
        sut.execute(
            **kwargs(prepared), execute_native=True,
            confirm_report_only_native=True,
            executable=sys.executable, timeout_seconds=0)
    assert not (prepared[1] / "r4u_native_attempt.json").exists()


def test_partial_case_and_permanent_attempt_lock_prevent_rerun(
    prepared, monkeypatch,
):
    _, root, _, _ = prepared
    calls = []
    def fake_run(folder, exe, timeout):
        calls.append(folder.name)
        write(folder / "logs" / "vivado.stdout.log", "synthetic failed\n")
        return {"completed_successfully": False, "returncode": 1}
    monkeypatch.setattr(sut, "_run_one", fake_run)
    result = sut.execute(
        **kwargs(prepared), execute_native=True,
        confirm_report_only_native=True, executable=sys.executable)
    assert result["actual_vendor_calls"] == 1
    assert calls == ["full_storage"]
    assert (root / "r4u_native_attempt.json").is_file()
    assert not (root / "row_storage" / "logs").exists()
    with pytest.raises(FileExistsError, match="native report lock exists"):
        sut.execute(
            **kwargs(prepared), execute_native=True,
            confirm_report_only_native=True, executable=sys.executable)


def test_existing_partial_report_prevents_initial_native_attempt(
    prepared, monkeypatch,
):
    _, root, _, _ = prepared
    (root / "row_storage" / "reports").mkdir()
    monkeypatch.setattr(
        sut, "_run_one", lambda *args: (_ for _ in ()).throw(
            AssertionError("no native calls are permitted")))
    with pytest.raises(FileExistsError, match="already attempted"):
        sut.execute(
            **kwargs(prepared), execute_native=True,
            confirm_report_only_native=True, executable=sys.executable)
    assert not (root / "r4u_native_attempt.json").exists()


def mock_report_only_success(prepared, *, change_wns=None,
                             change_drc=None, omit_paths=False):
    original, root, meta, old_arms = prepared
    write(root / "r4u_native_attempt.json", json.dumps({
        "schema": sut.SCHEMA + "/permanent-attempt-lock",
        "source_revision": sut.r4t.SOURCE_REVISION,
        "arms": list(sut.ARMS),
        "never_retry_this_study": True,
    }))
    for arm, original_wns in (
        ("full_storage", -2.170),
        ("row_storage", -3.718),
    ):
        case = root / arm
        reports = case / "reports"
        wns = change_wns if arm == "row_storage" and change_wns is not None \
            else original_wns
        drc_rules = next(x["original_drc_rule_counts"]
                         for x in old_arms if x["arm"] == arm)
        if arm == "row_storage" and change_drc is not None:
            drc_rules = {**drc_rules, "DPOP-4": change_drc}
        drc = "Violations found: " + str(sum(drc_rules.values())) + "\n"
        for code, value in drc_rules.items():
            drc += f"| {code} | Warning | mock rule context | {value} |\n"
        write(reports / "drc.rpt", drc)
        write(reports / "timing_summary.rpt",
              f"Design Timing Summary\nWNS (ns): {wns:.3f}\n")
        timing = ("Max Delay Paths\n"
                  "Startpoint: demo/q\n"
                  "Endpoint: demo/d\n"
                  "Path Group: ap_clk\n"
                  f"Slack (VIOLATED) : {wns:.3f} ns\n")
        if omit_paths and arm == "row_storage":
            timing = "Design Timing Summary\nWNS (ns): -3.718\n"
        write(reports / "max_delay_paths.rpt", timing)
        stdout = write(case / "logs" / "vivado.stdout.log",
                       "SYNTHETIC VIVADO MOCK\n")
        stderr = write(case / "logs" / "vivado.stderr.log", b"")
        before = {
            sut.CHECKPOINT: sut._digest(case / sut.CHECKPOINT),
            sut.SCRIPT: sut._digest(case / sut.SCRIPT),
        }
        receipt = {
            "schema": sut.SCHEMA + "/one-time-report-only-invocation",
            "executable": "/tools/mock/vivado",
            "argv": ["/tools/mock/vivado", "-mode", "batch",
                     "-source", sut.SCRIPT, "-notrace"],
            "cwd": str(case.resolve()),
            "started_utc": "2026-11-06T12:00:00+00:00",
            "duration_seconds": 3.0,
            "returncode": 0,
            "timed_out": False,
            "input_sha256_before": before,
            "input_sha256_after": before,
            "input_stable": True,
            "stdout_sha256": stdout,
            "stderr_sha256": stderr,
            "fresh_report_sha256": {
                n: sut._digest(reports / n) for n in sut.REPORT_NAMES
            },
            "completed_successfully": True,
            "strict_physical_E4": None,
            "board_qualified": False,
        }
        write(case / "report_only_invocation.json", json.dumps(receipt))


def test_mock_two_arm_new_paths_can_be_reported_without_e4(prepared):
    mock_report_only_success(prepared)
    result = sut.audit_native(**kwargs(prepared))
    assert result["native_attempt_lock_verified"] is True
    assert result["reported_complete_path_arms"] == 2
    assert result["new_vendor_calls_during_audit"] == 0
    assert result["strict_physical_E4"] is None
    assert all(x["status"] ==
               "SOURCE_ATTRIBUTED_REPORT_ONLY_PATHS_AVAILABLE_NOT_E4"
               for x in result["arms"])
    assert result["arms"][1]["complete_detailed_path_count"] == 1
    assert result["arms"][1]["sample_detailed_paths"][0][
        "original_destination"] == "demo/d"
    assert result["arms"][1]["original_to_new_WNS_consistency"] is True
    assert result["arms"][1]["new_drc_counts_consistent_with_original"] is True


@pytest.mark.parametrize("change_wns,change_drc,omit_paths", [
    (-4.1, None, False),
    (None, 22, False),
    (None, None, True),
])
def test_mock_mismatched_wns_drc_or_absent_paths_are_not_qualified(
    prepared, change_wns, change_drc, omit_paths,
):
    mock_report_only_success(
        prepared, change_wns=change_wns,
        change_drc=change_drc, omit_paths=omit_paths)
    result = sut.audit_native(**kwargs(prepared))
    row = result["arms"][1]
    assert row["status"] == (
        "POSTHOC_CHECKPOINT_REPORT_ONLY_OBSERVATION_INCOMPLETE_OR_DIFFERENT")
    assert result["reported_complete_path_arms"] == 1
    assert result["strict_physical_E4"] is None
    if change_wns is not None:
        assert row["original_to_new_WNS_consistency"] is False
    if change_drc is not None:
        assert row["new_drc_counts_consistent_with_original"] is False
    if omit_paths:
        assert row["complete_detailed_path_count"] == 0


def test_missing_new_report_or_unsigned_log_never_qualifies(prepared):
    mock_report_only_success(prepared)
    case = prepared[1] / "row_storage"
    path = case / "reports" / "max_delay_paths.rpt"
    path.write_text("modified after synthetic tool invocation\n")
    with pytest.raises(ValueError, match="SHA"):
        sut.audit_native(**kwargs(prepared))


def test_no_native_receipts_does_not_create_paths(prepared):
    result = sut.audit_native(**kwargs(prepared))
    assert result["native_attempt_lock_verified"] is False
    assert result["reported_complete_path_arms"] == 0
    assert result["strict_physical_E4"] is None
    assert result["arms"][0]["sample_detailed_paths"] is None
