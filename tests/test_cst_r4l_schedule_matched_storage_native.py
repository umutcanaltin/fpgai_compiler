"""R4l isolated NEW native pair guards; tests launch NO vendor executables.

Synthetic source packages and mock E2/E3 summaries exercise dry runs,
two-arm create-once checks, SHA tamper resistance and qualification.
The real R4k original file never gets rewritten by these tests.
"""
from __future__ import annotations

from hashlib import sha256
import json
import sys

import pytest

from fpgai_research.experiments import cst_r4l_schedule_matched_storage_native as sut


def _digest(data):
    return sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return _digest(text)


@pytest.fixture
def case(tmp_path, monkeypatch):
    study = tmp_path / "new_r4k"
    old_d = tmp_path / "old_r4d"
    old_a = tmp_path / "old_r4a"
    old_d.mkdir()
    old_a.mkdir()
    study.mkdir()
    plain = "FPGAI_R2G OUTPUT 0 1.0\n"
    traces = {
        "full_storage": "FPGAI_SLOT READ h 0 0\n" + plain,
        "row_storage": "FPGAI_SLOT READ h 0 1\n" + plain,
    }
    cpp = {
        "full_storage": "void full_storage_cpp(){}\n",
        "row_storage": "void row_storage_cpp(){}\n",
    }
    for arm in sut.r4k.ARMS:
        pkg = study / arm / "vitis_source"
        _write(study / (arm + ".host_stdout.txt"), traces[arm])
        _write(study / arm / "source.cpp", cpp[arm])
        _write(study / arm / "manifest.json", "{}\n")
        _write(pkg / "model.cpp", cpp[arm])
        _write(pkg / "tb.cpp", "same-testbench\n")
        _write(pkg / "run_hls.tcl", "same-clock\n")
        _write(pkg / "manifest.json", json.dumps({
            "cpp_sha256": _digest(cpp[arm]),
            "precision_bits": 32,
            "integer_bits": 6,
            "part": "xck26-sfvc784-2LV-c",
            "clock_ns": 12.0,
            "explicit_clock_uncertainty_ns": 1.5,
            "numerical_tolerance": 0.02,
            "csim_sample_count": 1,
            "csim_outputs_per_sample": 128,
            "testbench_sha256": _digest("same-testbench\n"),
            "functional_sha256": "f" * 64,
            "architecture_sha256": "a" * 64,
        }))
    config = {
        "source_schedule_sha256": "e" * 64,
        "source_full_storage_sha256": _digest(cpp["full_storage"]),
        "source_row_original_sha256": _digest(cpp["row_storage"]),
        "part": "xck26-sfvc784-2LV-c",
        "clock_ns": 12.0,
        "uncertainty_ns": 1.5,
        "csim_abs_tolerance": 0.02,
    }
    base = {
        "source_revision": "b" * 40,
        "original_R4a_source_revision": "c" * 40,
        "original_tcl_sha256": _digest("same-clock\n"),
    }
    e1 = {
        "status": "MATCHED_ROW_SOURCE_SCHEDULE_REAL_HOST_E1_ACCEPTED",
        "original_host_gcc_compilations": 2,
        "original_host_gcc_executions": 2,
        "new_native_Vitis_calls": 0,
        "all_intermediates_numerically_equal": True,
        "all_final_outputs_numerically_equal": True,
        "source_schedule_sha256": "e" * 64,
        "source_scratch_elements_saved": 240,
        "full_source_sha256": _digest(cpp["full_storage"]),
        "row_source_sha256": _digest(cpp["row_storage"]),
        "source_logic_trace_without_addresses_sha256": _digest(plain),
    }
    report = {
        "schema": sut.r4k.SCHEMA,
        "configuration": config,
        "source_only_E1_verified": True,
        "source_revision": "d" * 40,
        "original_preflight": base,
        "HLS_E2_run": False,
        "HLS_E3_run": False,
        "physical_E4_run": False,
        "new_vendor_Vitis_calls": 0,
        "new_vendor_Vivado_calls": 0,
        "E1": e1,
    }
    _write(study / "summary.json", json.dumps(report))
    inventory = {
        str(file.relative_to(study)): _digest(file.read_bytes())
        for file in study.rglob("*") if file.is_file()
    }
    _write(study / "artifact_sha256.json", json.dumps(inventory))
    monkeypatch.setattr(sut.r4k, "plan", lambda: config)
    monkeypatch.setattr(
        sut.r4k, "_read_only_original_preflight", lambda *args: base)
    return study, old_d, old_a, config


def _call(case, *, execute_native=False, confirmation=False,
          executable=sys.executable):
    study, old_d, old_a, _ = case
    return sut.execute(
        study=study,
        original_r4d=old_d,
        original_r4a=old_a,
        execute_native=execute_native,
        confirm_isolated_native=confirmation,
        vitis_executable=executable,
    )


def test_read_only_preflight_uses_original_signed_packages(case):
    study, old_d, old_a, config = case
    evidence = sut.audit_prepared(
        study=study, original_r4d=old_d, original_r4a=old_a)
    assert evidence["schema"] == sut.SCHEMA
    assert evidence["E1_matched_pair_accepted"] is True
    assert evidence["source_only_storage_delta_elements"] == 240
    assert evidence["source_revision"] == "d" * 40
    assert evidence["paired_HLS_source_arms"] == 2
    assert evidence["package_hashes"]["row_storage"]["model.cpp"] == (
        config["source_row_original_sha256"])
    assert evidence["physical_E4"] is None
    dry = _call(case)
    assert dry["status"] == "DRY_RUN_NO_VENDOR_CALLS"
    assert dry["planned_new_vendor_calls"] == 2
    assert dry["new_vendor_calls"] == 0
    assert not (study / "full_storage" / "vitis_source" /
                "vitis_invocation.json").exists()


def test_new_source_or_tcl_tamper_fails_before_vendor(case):
    study, old_d, old_a, _ = case
    pkg = study / "row_storage" / "vitis_source"
    for file in (pkg / "model.cpp",):
        file.write_bytes(file.read_bytes() + b"// tamper\n")
        with pytest.raises(ValueError, match="SHA inventory mismatch"):
            _call(case)
        file.write_bytes(file.read_bytes().replace(b"// tamper\n", b""))
    _write(pkg / "run_hls.tcl", "changed-clock\n")
    with pytest.raises(ValueError, match="SHA inventory mismatch"):
        _call(case)


def test_source_trace_and_e1_status_tamper_refused(case):
    study, _, _, _ = case
    trace = study / "row_storage.host_stdout.txt"
    trace.write_text(trace.read_text() + "FPGAI_R2G OUTPUT 7 7.0\n")
    with pytest.raises(ValueError, match="SHA inventory mismatch"):
        _call(case)


def test_missing_explicit_confirmation_or_invalid_timeout_refused(case):
    with pytest.raises(ValueError, match="confirmation"):
        _call(case, execute_native=True, confirmation=False)
    study, old_d, old_a, _ = case
    with pytest.raises(ValueError, match="timeout"):
        sut.execute(
            study=study, original_r4d=old_d, original_r4a=old_a,
            execute_native=True, confirm_isolated_native=True,
            vitis_executable=sys.executable, timeout_seconds=0)


def test_partial_previous_attempt_in_second_arm_blocks_both(case, monkeypatch):
    study, _, _, _ = case
    (study / "row_storage" / "vitis_source" / "logs").mkdir()
    monkeypatch.setattr(
        sut, "run_vitis",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Vitis should never run")))
    with pytest.raises(FileExistsError, match="already attempted"):
        _call(case, execute_native=True, confirmation=True)


def test_vendor_execution_happens_before_hls_qualification(
    case, monkeypatch,
):
    study, _, _, _ = case
    calls = []
    def run_vitis(pkg, **kwargs):
        calls.append(("vitis", pkg.parent.name))
        (pkg / "vitis_invocation.json").write_text('{"completed": true}\n')
        return {"completed_successfully": True}
    def qualify(pkg, **kwargs):
        if not (pkg / "vitis_invocation.json").exists():
            raise AssertionError("R4l qualified vendor results BEFORE Vitis")
        calls.append(("qualify", pkg.parent.name))
        return {
            "arm": pkg.parent.name, "E2_qualified": True,
            "E3_qualified": True,
        }
    monkeypatch.setattr(sut, "run_vitis", run_vitis)
    monkeypatch.setattr(sut, "_native_observation",
                        lambda pkg, **kwargs: qualify(pkg))
    result = _call(case, execute_native=True, confirmation=True)
    assert result["original_native_vendor_calls_made"] == 2
    assert calls == [
        ("vitis", "full_storage"), ("qualify", "full_storage"),
        ("vitis", "row_storage"), ("qualify", "row_storage"),
    ]
    with pytest.raises(FileExistsError, match="already attempted"):
        _call(case, execute_native=True, confirmation=True)


def test_native_audit_never_imputes_e3_when_e2_failed(case, monkeypatch):
    monkeypatch.setattr(
        sut, "_native_observation", lambda pkg, **kw: {
            "arm": kw["arm"],
            "native_invocation_file_present": False,
            "E2_qualified": kw["arm"] == "row_storage",
            "E3_qualified": False,
            "HLS_estimated_resources": {key: None for key in sut._RESOURCES},
            "HLS_worst_cycles": None,
            "HLS_estimated_clock_ns": None,
        })
    study, old_d, old_a, _ = case
    audit = sut.audit_native(
        study=study, original_r4d=old_d, original_r4a=old_a)
    assert audit["both_E2_qualified"] is False
    assert audit["both_E3_qualified"] is False
    assert audit["estimated_resource_delta_row_minus_full"]["bram_18k"] is None
    assert audit["HLS_worst_cycles_delta_row_minus_full"] is None
    assert audit["new_vendor_calls_during_read_only_audit"] == 0


def test_qualified_native_deltas_are_original_hls_estimates_only(
    case, monkeypatch,
):
    def observation(pkg, **kwargs):
        arm = kwargs["arm"]
        return {
            "arm": arm, "native_invocation_file_present": True,
            "E2_qualified": True, "E3_qualified": True,
            "HLS_estimated_resources": {
                "bram_18k": 13, "dsp": 23 if arm == "full_storage" else 19,
                "lut": 4990 if arm == "full_storage" else 4755,
                "ff": 1798 if arm == "full_storage" else 1647, "uram": 0,
            },
            "HLS_worst_cycles": 22192 if arm == "full_storage" else 22602,
            "HLS_estimated_clock_ns": 10.302 if arm == "full_storage" else 9.5,
        }
    monkeypatch.setattr(sut, "_native_observation", observation)
    study, old_d, old_a, _ = case
    audit = sut.audit_native(
        study=study, original_r4d=old_d, original_r4a=old_a)
    assert audit["both_E2_qualified"] is True
    assert audit["both_E3_qualified"] is True
    assert audit["estimated_resource_delta_row_minus_full"] == {
        "bram_18k": 0, "dsp": -4, "lut": -235, "ff": -151, "uram": 0}
    assert audit["HLS_worst_cycles_delta_row_minus_full"] == 410
    assert audit["physical_E4"] is None
    assert audit["new_vendor_calls_during_read_only_audit"] == 0
