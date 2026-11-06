"""R4d supported-format numerical sweep: source-only, fail closed.

No HLS/Vivado/GitHub Actions calls. The costly test executes real host GCC
on NEW temporary E1 cases and verifies the original R4a root is unchanged.
"""
from __future__ import annotations

from hashlib import sha256
import json
import shutil

import pytest

from fpgai_research.experiments import cst_r4a_row_retention_study as r4a
from fpgai_research.experiments import cst_r4d_matched_numeric_formats as sut


def test_only_supported_preregistered_formats_and_new_case_003():
    p = sut.plan()
    assert p["original_R4a_case"] == "case_003"
    assert p["predeclared_formats_total_integer_bits"] == [
        [16, 6], [32, 6], [32, 8]]
    assert p["matched_storage_pairs"] == 3
    assert p["source_GCC_arms"] == 6
    assert p["absolute_tolerance"] == 0.02
    assert p["native_Vitis_calls"] == 0
    assert p["native_Vivado_calls"] == 0
    assert p["E1_host_float_only"] is True
    assert p["original_failure_retained"] is True
    assert p["graph"] == [2, [16, 4, 16, 8], "mixed"]


def test_reject_unsupported_20_or_24_bit_formats(monkeypatch):
    for invalid in ((20, 6), (24, 8)):
        monkeypatch.setattr(sut, "FORMATS", ((16, 6), invalid))
        with pytest.raises(ValueError, match="PRECISION|Unsupported"):
            sut.plan()


def test_canonical_baseline_16_6_is_unchanged():
    arch_default = r4a._case(r4a.CASES[3])
    arch_explicit = r4a._case(
        r4a.CASES[3], precision_bits=16, integer_bits=6)
    for emitter in (sut.emit_fanout_cpp, sut.emit_row_retained_fanout_cpp):
        control = emitter(arch_default)
        explicit = emitter(arch_explicit)
        assert control.cpp == explicit.cpp
        assert control.manifest["source_cpp_sha256"] == (
            explicit.manifest["source_cpp_sha256"])
    for width, integer in ((16, 6), (32, 6), (32, 8)):
        arch = r4a._case(
            r4a.CASES[3], precision_bits=width, integer_bits=integer)
        assert {(p.effective.precision_bits, p.effective.integer_bits)
                for p in arch.points} == {(width, integer)}


@pytest.mark.skipif(shutil.which("g++") is None, reason="Host g++ unavailable")
def test_original_study_hash_unchanged_and_new_e1_packages_create_only(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(r4a, "_git_revision", lambda: "d" * 40)
    original = tmp_path / "original_paper"
    r4a.run(output=original, profile="paper")
    old_digests = {
        str(p.relative_to(original)): sha256(p.read_bytes()).hexdigest()
        for p in original.rglob("*") if p.is_file()
    }
    root = tmp_path / "new_r4d"
    outcome = sut.run(original_study=original, output=root)
    assert outcome["original_R4a_case"] == "case_003"
    assert outcome["completed_original_GCC_pairs"] == 3
    assert outcome["new_GCC_executions"] == 6
    assert outcome["new_native_Vitis_calls"] == 0
    assert outcome["new_Vitis_E2_results"] is None
    assert outcome["new_HLS_E3_qualified_pairs"] is None
    assert outcome["new_physical_E4_results"] is None
    rows = outcome["rows"]
    assert [r["format"] for r in rows] == [[16, 6], [32, 6], [32, 8]]
    baseline = json.loads((original / "summary.json").read_text())["rows"][3]
    for idx, row in enumerate(rows):
        name = ["format_16_06", "format_32_06", "format_32_08"][idx]
        assert all(row[arm]["E1_GCC_passed"] for arm in sut.ARMS)
        for arm in sut.ARMS:
            manifest = json.loads((root / name / arm / "vitis_source"
                                   / "manifest.json").read_text())
            assert [manifest["precision_bits"], manifest["integer_bits"]] == (
                row["format"])
            assert manifest["numerical_tolerance"] == 0.02
            assert manifest["csim_sample_count"] == 1
            assert manifest["csim_outputs_per_sample"] == 128
            assert row[arm]["Vitis_E2_passed"] is None
            assert row[arm]["HLS_E3_qualified"] is None
            if idx == 0:
                assert row[arm]["E1_source_cpp_sha256"] == (
                    baseline[arm]["source_sha256"])
                assert row[arm]["HLS_package_sha256"] == (
                    baseline[arm]["source_vitis_package_sha256"])
    assert (root / "artifact_sha256.json").is_file()
    digests = json.loads((root / "artifact_sha256.json").read_text())
    assert digests["summary.json"] == sha256(
        (root / "summary.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="create-only"):
        sut.run(original_study=original, output=root)
    with pytest.raises(ValueError, match="cannot write inside"):
        sut.run(original_study=original, output=original / "forbidden")
    assert old_digests == {
        str(p.relative_to(original)): sha256(p.read_bytes()).hexdigest()
        for p in original.rglob("*") if p.is_file()
    }


@pytest.mark.skipif(shutil.which("g++") is None, reason="Host g++ unavailable")
def test_tampered_original_refused_before_creating_new_output(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(r4a, "_git_revision", lambda: "c" * 40)
    original = tmp_path / "r4a_quick"
    r4a.run(output=original, profile="quick")
    new = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="original R4a paper case_003|population"):
        sut.run(original_study=original, output=new)
    assert not new.exists()
