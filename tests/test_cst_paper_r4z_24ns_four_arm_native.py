"""R4z four-arm new-Vitis runner regression; NEVER run vendor tools."""
from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fpgai_research.experiments import (
    cst_paper_r4z_24ns_four_arm_native as sut,
)


def _digest(text: str) -> str:
    return sha256(text.encode()).hexdigest()


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    src = tmp_path / "original_r4y"
    src.mkdir()
    (src / "summary.json").write_text('{"original":true}\n')
    (src / "artifact_sha256.json").write_text('{"original":"sha"}\n')
    originals = [tmp_path / name for name in ("old_r4k", "old_r4d", "old_r4a")]
    for old in originals:
        old.mkdir()
    hashes = {}
    for case in sut.r4y.CASES:
        pkg = src / case / "vitis_source"
        pkg.mkdir(parents=True)
        model = "// source for " + case + "\n"
        hashes[case] = _digest(model)
        (pkg / "model.cpp").write_text(model)
        (pkg / "tb.cpp").write_text("// original common testbench\n")
        (pkg / "run_hls.tcl").write_text(
            "create_clock -period 24\nset_clock_uncertainty 1.500\n")
        (pkg / "manifest.json").write_text(json.dumps({
            "cpp_sha256": hashes[case],
            "source_cpp_sha256": hashes[case],
            "clock_ns": 24.0,
            "explicit_clock_uncertainty_ns": 1.5,
            "part": sut.r4y.PART,
            "numerical_tolerance": 0.02,
        }, sort_keys=True) + "\n")

    audit = {
        "schema": sut.r4y.SCHEMA,
        "status": "SOURCE_ONLY_FOUR_ARM_PACKAGE_AUDITED_NOT_NATIVE",
        "cases": list(sut.r4y.CASES),
        "clock_period_ns": 24.0,
        "matched_testbench_and_tcl": True,
        "new_vendor_calls_during_audit": 0,
        "originals_modified": False,
        "source_revision": "d" * 40,
        "original_R4k_source_revision": "7" * 40,
        "source_sha256": hashes,
    }
    monkeypatch.setattr(sut.r4y, "audit", lambda **_: audit)
    kw = {
        "source_study": src,
        "r4k_study": originals[0],
        "original_r4d": originals[1],
        "original_r4a": originals[2],
    }
    return kw, audit


def _prepared(tmp_path, cohort):
    kw, _ = cohort
    dest = tmp_path / "r4z_new_native"
    summary = sut.prepare(output=dest, **kw)
    return dest, kw, summary


def test_four_arm_fresh_native_source_copy_and_read_only_audit(cohort, tmp_path):
    dest, kw, original = _prepared(tmp_path, cohort)
    assert original["clock_period_ns"] == 24.0
    assert original["native_vendor_calls_at_preparation"] == 0
    assert original["measured_latency_ns"] is None
    inventory = json.loads((dest / "input_sha256.json").read_text())
    assert len(inventory) == 17
    assert (sut.audit_prepared(study=dest, **kw)[
        "verified_input_sha256_count"]) == 17
    for case in sut.r4y.CASES:
        assert (dest / case / "model.cpp").read_bytes() == (
            kw["source_study"] / case / "vitis_source" / "model.cpp").read_bytes()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.prepare(output=dest, **kw)


def test_source_root_overlap_prevented_before_any_write(cohort):
    kw, _ = cohort
    with pytest.raises(ValueError, match="overlaps"):
        sut.prepare(output=kw["source_study"] / "new_vendor", **kw)


def test_source_root_or_native_case_tamper_fails_preflight(cohort, tmp_path):
    dest, kw, _ = _prepared(tmp_path, cohort)
    altered = dest / "row_storage__baseline" / "model.cpp"
    altered.write_text(altered.read_text() + "// tampered\n")
    with pytest.raises(ValueError, match="source SHA inventory"):
        sut.audit_prepared(study=dest, **kw)


def test_source_origin_drift_fails_native_preflight(cohort, tmp_path):
    dest, kw, _ = _prepared(tmp_path, cohort)
    origin = kw["source_study"] / "full_storage__baseline" / "vitis_source" / "tb.cpp"
    origin.write_text(origin.read_text() + "// modified old root\n")
    with pytest.raises(ValueError, match="no longer matches R4y"):
        sut.audit_prepared(study=dest, **kw)


@pytest.mark.parametrize("filename", ["logs", "fpgai_hls_project",
                                       "vitis_invocation.json"])
def test_unexpected_native_attempt_blocks_before_lock(cohort, tmp_path, filename):
    dest, kw, _ = _prepared(tmp_path, cohort)
    (dest / "row_storage__baseline" / filename).mkdir()
    with pytest.raises(ValueError, match="without attempt lock"):
        sut.audit_prepared(study=dest, **kw)


def test_default_execute_and_audit_are_vendor_free(cohort, tmp_path, monkeypatch):
    dest, kw, _ = _prepared(tmp_path, cohort)
    def forbidden(*args, **kwargs):
        raise AssertionError("Vitis must never execute by default")
    monkeypatch.setattr(sut, "run_vitis", forbidden)
    dry = sut.execute(study=dest, **kw)
    assert dry["status"] == "DRY_RUN_NO_VENDOR_CALLS"
    assert dry["planned_vendor_calls"] == 4
    assert not (dest / sut.LOCK).exists()
    observations = sut.audit_native(study=dest, **kw)
    assert observations["native_receipts_present"] == 0
    assert observations["E3_qualified_cases"] == 0
    assert all(c["real_measured_latency_ns"] is None
               for c in observations["observations"])
    assert not (dest / sut.LOCK).exists()


def test_missing_confirmation_timeout_and_executable_rejected(
        cohort, tmp_path, monkeypatch):
    dest, kw, _ = _prepared(tmp_path, cohort)
    with pytest.raises(ValueError, match="confirm-isolated-native"):
        sut.execute(study=dest, **kw, execute_native=True)
    with pytest.raises(ValueError, match="timeout"):
        sut.execute(study=dest, **kw, execute_native=True,
                    confirm_isolated_native=True, timeout_seconds=0)
    monkeypatch.setattr(sut.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="executable"):
        sut.execute(study=dest, **kw, execute_native=True,
                    confirm_isolated_native=True)
    assert not (dest / sut.LOCK).exists()


def test_native_requires_two_flags_and_writes_lock_before_any_call(
        cohort, tmp_path, monkeypatch):
    dest, kw, _ = _prepared(tmp_path, cohort)
    calls = []
    monkeypatch.setattr(sut.shutil, "which", lambda _: "/fake/vitis_hls")
    def native(pkg, **kwargs):
        assert (dest / sut.LOCK).exists()
        calls.append(pkg.name)
        (pkg / "vitis_invocation.json").write_text('{"fake":"receipt"}\n')
        return {"completed_successfully": True}
    monkeypatch.setattr(sut, "run_vitis", native)
    out = sut.execute(study=dest, **kw, execute_native=True,
                      confirm_isolated_native=True)
    assert out["native_vendor_calls"] == 4
    assert calls == list(sut.r4y.CASES)
    assert sut.audit_prepared(study=dest, **kw)[
        "one_time_attempt_lock_present"]
    with pytest.raises(FileExistsError, match="already attempted"):
        sut.execute(study=dest, **kw, execute_native=True,
                    confirm_isolated_native=True)


def test_exception_during_second_case_persists_lock_and_never_retries(
        cohort, tmp_path, monkeypatch):
    dest, kw, _ = _prepared(tmp_path, cohort)
    calls = []
    monkeypatch.setattr(sut.shutil, "which", lambda _: "/fake/vitis_hls")
    def partial(pkg, **kwargs):
        calls.append(pkg.name)
        if len(calls) == 2:
            raise RuntimeError("simulated host interruption")
        (pkg / "vitis_invocation.json").write_text('{"attempt":true}\n')
        return {"completed_successfully": True}
    monkeypatch.setattr(sut, "run_vitis", partial)
    with pytest.raises(RuntimeError, match="interruption"):
        sut.execute(study=dest, **kw, execute_native=True,
                    confirm_isolated_native=True)
    assert len(calls) == 2
    assert sut.audit_prepared(study=dest, **kw)[
        "one_time_attempt_lock_present"]
    with pytest.raises(FileExistsError, match="already attempted"):
        sut.execute(study=dest, **kw, execute_native=True,
                    confirm_isolated_native=True)


def test_modified_attempt_lock_does_not_waive_original_hash(cohort, tmp_path):
    dest, kw, _ = _prepared(tmp_path, cohort)
    lock = dest / sut.LOCK
    lock.write_text('{"forged":"true"}\n')
    with pytest.raises(ValueError, match="attempt lock drift"):
        sut.audit_prepared(study=dest, **kw)


def test_read_only_hls_reports_separate_e2_e3_and_never_board_latency(
        cohort, tmp_path, monkeypatch):
    dest, kw, summary = _prepared(tmp_path, cohort)
    (dest / sut.LOCK).write_text(json.dumps(sut._lock_expected(summary)))
    for case in sut.r4y.CASES:
        (dest / case / "vitis_invocation.json").write_text("{}\n")
    def summarize(folder):
        case = folder.split("/")[-1]
        return {
            "source_sha256": summary["source_sha256"][case],
            "csim_receipt": {"max_absolute_error": 0.001},
            "metrics": {"ii_max": 2},
        }
    def qualify(folder, expected, *, summary):
        return {
            "status": "QUALIFIED_HLS_ESTIMATES_NOT_PHYSICAL",
            "native_e2_accepted": True,
            "native_e3_qualified": True,
            "hls_worst_cycles": 100,
            "hls_report_sha256": "f" * 64,
            "hls_estimated_clock_ns": 18.5,
            "estimated_resources": {k: 3 for k in sut.RESOURCES},
        }
    monkeypatch.setattr(sut, "summarize_case", summarize)
    monkeypatch.setattr(sut, "qualify_case", qualify)
    output = sut.audit_native(study=dest, **kw)
    assert output["E2_qualified_cases"] == 4
    assert output["E3_qualified_cases"] == 4
    assert output["native_receipts_present"] == 4
    assert output["board_measured_latency_ns"] is None
    assert all(a["E3_requested_clock_cycle_product_ns_NOT_measured"] == 2400
               and a["real_measured_latency_ns"] is None
               for a in output["observations"])
    assert len(output["same_clock_storage_pairs"]) == 2
    assert output["four_arm_factorial_interaction"]["all_four_E3_qualified"]
    assert output["four_arm_factorial_interaction"][
        "pipeline_x_storage_interaction_HLS_cycles"] == 0
    assert all(x["HLS_worst_cycles_delta_requested_minus_baseline"] == 0
               and x["HLS_resource_delta_requested_minus_baseline"]["dsp"] == 0
               for x in output["within_storage_pipeline_effects"])
    assert all(x["both_E3_HLS_qualified"]
               and x["HLS_cycle_product_ns_delta_NOT_measured"] == 0
               and x["measured_FPGA_latency_delta_ns"] is None
               for x in output["same_clock_storage_pairs"])


def test_unqualified_e2_never_promotes_e3_even_with_fake_synthesis(
        cohort, tmp_path, monkeypatch):
    dest, kw, summary = _prepared(tmp_path, cohort)
    (dest / sut.LOCK).write_text(json.dumps(sut._lock_expected(summary)))
    name = "row_storage__baseline"
    (dest / name / "vitis_invocation.json").write_text("{}\n")
    monkeypatch.setattr(sut, "summarize_case", lambda _: {
        "source_sha256": summary["source_sha256"][name],
        "csim_receipt": {"failures": 1}, "metrics": {"latency_worst": 99},
    })
    monkeypatch.setattr(sut, "qualify_case",
                        lambda *a, **k: {
                            "status": "FAILED_ORIGINAL_CSIM",
                            "native_e2_accepted": False,
                            "native_e3_qualified": True,
                            "hls_worst_cycles": 99,
                            "estimated_resources": {k: 1 for k in sut.RESOURCES},
                        })
    output = sut.audit_native(study=dest, **kw)
    row = next(x for x in output["observations"] if x["case"] == name)
    assert not row["E2_numerical_qualified"]
    assert not row["E3_HLS_qualified"]
    assert row["HLS_worst_cycles"] is None
    assert row["E3_requested_clock_cycle_product_ns_NOT_measured"] is None
    assert output["E3_qualified_cases"] == 0
    assert output["four_arm_factorial_interaction"][
        "pipeline_x_storage_interaction_HLS_cycles"] is None
    assert all(p["HLS_worst_cycles_delta_requested_minus_baseline"] is None
               for p in output["within_storage_pipeline_effects"])
    assert not output["same_clock_storage_pairs"][0]["both_E3_HLS_qualified"]


def test_pipeline_factorial_effects_keep_negative_hls_causal_result(
        cohort, tmp_path, monkeypatch):
    dest, kw, summary = _prepared(tmp_path, cohort)
    (dest / sut.LOCK).write_text(json.dumps(sut._lock_expected(summary)))
    cycles = {
        "full_storage__baseline": 100,
        "full_storage__producer_mac_ii1_requested": 110,
        "row_storage__baseline": 94,
        "row_storage__producer_mac_ii1_requested": 105,
    }
    for case in sut.r4y.CASES:
        (dest / case / "vitis_invocation.json").write_text("{}\n")
    monkeypatch.setattr(sut, "summarize_case", lambda folder: {
        "source_sha256": summary["source_sha256"][folder.split("/")[-1]],
        "csim_receipt": {"failures": 0, "max_absolute_error": 0.003},
        "metrics": {"ii_max": "3"},
    })
    def qualify(folder, expected, *, summary):
        case = folder.name
        return {
            "status": "QUALIFIED_HLS_ESTIMATES_NOT_PHYSICAL",
            "native_e2_accepted": True,
            "native_e3_qualified": True,
            "hls_worst_cycles": cycles[case],
            "hls_report_sha256": "f" * 64,
            "hls_estimated_clock_ns": 16.0,
            "estimated_resources": {key: 4 for key in sut.RESOURCES},
        }
    monkeypatch.setattr(sut, "qualify_case", qualify)
    out = sut.audit_native(study=dest, **kw)
    assert [r["HLS_worst_cycles_delta_requested_minus_baseline"]
            for r in out["within_storage_pipeline_effects"]] == [10, 11]
    assert out["four_arm_factorial_interaction"][
        "pipeline_x_storage_interaction_HLS_cycles"] == 1
    assert out["four_arm_factorial_interaction"][
        "pipeline_x_storage_interaction_HLS_clock_product_ns_NOT_measured"] == 24
    assert out["observations"][0]["HLS_top_level_II_max_cycles"] == 3
    assert all(r["real_measured_latency_ns"] is None for r in out["observations"])
