"""First-paper R4a original E1 matched storage study.

Unit checks keep the population and HLS source attribution stable.
The optional quick GCC test runs only host C++ and source preparation.
"""
from __future__ import annotations

import json
import shutil

import pytest

from fpgai_research.experiments import cst_r4a_row_retention_study as sut


def test_prospective_population_and_vendor_free_plan():
    quick = sut.plan(profile="quick")
    paper = sut.plan(profile="paper")
    assert quick["prespecified_graphs"] == 3
    assert paper["prespecified_graphs"] == 5
    assert quick["matched_arms"] == 6
    assert paper["matched_arms"] == 10
    assert quick["native_vendor_calls"] == 0
    assert paper["HLS_package_only"] is True
    assert len(set((r, tuple(shape), p) for r, shape, p in sut.CASES)) == 5
    assert sut.PART == "xck26-sfvc784-2LV-c"
    assert sut.CLOCK_NS > sut.UNCERTAINTY_NS > 0
    with pytest.raises(ValueError):
        sut.plan(profile="invented")


@pytest.mark.parametrize("config", sut.CASES)
def test_each_case_has_both_source_arms_and_preserves_metadata(config):
    arch = sut._case(config)
    full = sut.emit_fanout_cpp(arch)
    row = sut.emit_row_retained_fanout_cpp(arch)
    full_package = sut._vitis_emission(arch, full)
    row_package = sut._vitis_emission(arch, row)
    assert full.manifest["source_cpp_sha256"] == (
        full_package.manifest["cpp_sha256"])
    assert row.manifest["source_cpp_sha256"] == (
        row_package.manifest["cpp_sha256"])
    assert full.manifest["functional_sha256"] == row.manifest["functional_sha256"]
    assert full.manifest["architecture_sha256"] == row.manifest["architecture_sha256"]
    assert row.manifest["source_scratch_array_elements"] < (
        full.manifest["source_scratch_array_elements"])
    assert row.manifest["source_H_row_buffer_elements"] == config[1][2]
    assert full_package.manifest["precision_bits"] == (
        row_package.manifest["precision_bits"])
    assert full_package.manifest["integer_bits"] == (
        row_package.manifest["integer_bits"])


@pytest.mark.skipif(shutil.which("g++") is None, reason="Host g++ missing")
def test_quick_create_only_original_GCC_and_vitis_source_prep(
    tmp_path, monkeypatch,
):
    # The standalone campaign requires an original source revision;
    # the actual workstation run records its clean git SHA.
    monkeypatch.setattr(sut, "_git_revision", lambda: "c"*40)
    result = sut.run(output=tmp_path / "r4a_new", profile="quick")
    assert result["completed_original_GCC_matched_pairs"] == 3
    assert result["original_GCC_compilations"] == 6
    assert result["native_Vitis_calls"] == 0
    assert result["native_Vivado_calls"] == 0
    assert result["observed_HLS_E3_qualified"] == 0
    assert len(result["rows"]) == 3
    assert all(row["source_scratch_elements_saved"] > 0
               for row in result["rows"])
    assert all(row["HLS_BRAM18K_delta"] is None for row in result["rows"])
    path = tmp_path / "r4a_new"
    assert (path / "artifact_sha256.json").is_file()
    assert (path / "case_000" / "row_H" / "host_receipt.json").is_file()
    assert (path / "case_000" / "row_H" / "vitis_source" / "run_hls.tcl").is_file()
    saved = json.loads((path / "summary.json").read_text())
    assert saved["source_revision"] == "c" * 40
    with pytest.raises(FileExistsError, match="create-only"):
        sut.run(output=path, profile="quick")


def test_report_wont_impute_hardware_from_source_data():
    report = sut.plan(profile="quick")
    assert report["source_conformance"] == "E1_HOST_GCC_FLOAT32"
    assert report["HLS_package_only"] is True
    assert report["native_vendor_calls"] == 0
