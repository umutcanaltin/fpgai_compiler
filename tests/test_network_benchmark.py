"""Source-only regression for diverse synthetic-network benchmarking."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from scripts import network_benchmark as study


def test_balanced_default_matrix_has_complete_matched_denominators():
    rows = study.plan()
    assert len(rows) == 32
    assert len({r["case"] for r in rows}) == 32
    assert len({(r["network"], r["mapping"], r["precision"], r["clock_ns"])
                for r in rows}) == 32
    assert {r["network"] for r in rows} == set(study.NETWORKS)
    assert {r["mapping"] for r in rows} == {"scalar", "independent"}
    assert {r["precision"] for r in rows} == {"fx16_6", "fx12_5"}
    assert {r["clock_ns"] for r in rows} == {12.0, 20.0}
    assert all(row["part"] == "xck26-sfvc784-2LV-c" for row in rows)


@pytest.mark.parametrize("kwargs", [
    {"networks": ("tiny_mlp", "tiny_mlp")},
    {"mappings": ("scalar", "scalar")},
    {"precisions": ("unknown",)},
    {"clocks": (12.0, 12.0)},
    {"clocks": (0.0,)},
    {"clocks": (float("nan"),)},
    {"part": "xck26; rm -rf /"},
])
def test_invalid_or_duplicate_architecture_axes_fail_closed(kwargs):
    with pytest.raises(ValueError):
        study.plan(**kwargs)


@pytest.mark.parametrize("network", study.NETWORKS)
def test_prepare_real_onnx_and_source_attested_vendor_scripts(
        tmp_path, network):
    pytest.importorskip("onnx")
    root = tmp_path / "prepared"
    result = study.prepare(
        root, networks=(network,), mappings=("scalar",),
        precisions=("fx16_6",), clocks=(16.0,), samples=1)
    assert result["case_count"] == 1
    assert result["native_tools_executed"] is False
    assert result["latency_cycles_or_resources_measured"] is False
    checked = study._inventory(root)
    assert checked == result
    case = root / "cases/case_0000"
    data = json.loads((case / "network_case.json").read_text())
    assert data["network"] == network
    assert data["precision_bits"] == 16
    assert data["integer_bits"] == 6
    assert data["clock_ns"] == 16.0
    assert len(data["operator_kinds"]) >= 4
    manifest = json.loads((case / "manifest.json").read_text())
    assert manifest["cpp_sha256"] == data["cpp_sha256"]
    assert manifest["precision_bits"] == 16
    assert manifest["integer_bits"] == 6
    assert (case / "run_hls.tcl").is_file()
    assert (case / "tb.cpp").is_file()
    assert (case / "vivado/run_vivado.tcl").is_file()
    assert not (case / "vitis_invocation.json").exists()
    assert not (case / "vivado/vivado_invocation.json").exists()
    with pytest.raises(FileExistsError):
        study.prepare(root, networks=(network,), mappings=("scalar",),
                      precisions=("fx16_6",), clocks=(16.0,), samples=1)


def test_independent_fx12_mapping_does_not_restore_default_precision(tmp_path):
    pytest.importorskip("onnx")
    root = tmp_path / "fx12"
    study.prepare(root, networks=("cnn_pool_dense",), mappings=("independent",),
                  precisions=("fx12_5",), clocks=(20.0,), samples=1)
    case = root / "cases/case_0000"
    manifest = json.loads((case / "manifest.json").read_text())
    assert manifest["precision_bits"] == 12
    assert manifest["integer_bits"] == 5
    assert study._inventory(root)["case_count"] == 1


def _prepare_pair(tmp_path):
    pytest.importorskip("onnx")
    root = tmp_path / "matrix"
    study.prepare(root, networks=("tiny_mlp",),
                  mappings=("scalar", "independent"),
                  precisions=("fx16_6",), clocks=(20.0,), samples=1)
    return root


def _fake_native(root, *, do_e4=True):
    def case_summary(folder):
        name = folder.rsplit("/", 1)[-1]
        meta = json.loads((root / "cases" / name / "network_case.json").read_text())
        scalar = meta["mapping"] == "scalar"
        result = {
            "source_sha256": meta["cpp_sha256"],
            "evidence": "E4-Vivado-implementation" if do_e4 else "E1-HLS-source",
            "csim_passed": True if do_e4 else None,
            "hls_report_sha256": "a" * 64 if do_e4 else None,
            "metrics": {
                "latency_best": "180" if scalar else "90",
                "latency_worst": "200" if scalar else "110",
                "ii_min": "1", "ii_max": "1",
                "lut": "80" if scalar else "100",
                "ff": "200", "dsp": "2", "bram_18k": "1", "uram": "0",
            } if do_e4 else {},
            "vivado_implementation_mode": "out_of_context" if do_e4 else None,
            "vivado_metrics": {
                "routed_nets_complete": True,
                "timing": {"wns_ns": 0.5, "timing_met": True},
                "drc": {"summary_verified": True, "violations": 0,
                        "clean": True, "critical_warnings": 0,
                        "unconstrained_io": False},
                "utilization": {"lut": 100 if scalar else 90, "ff": 240,
                                "dsp": 2, "bram": 1, "uram": 0},
            } if do_e4 else None,
        }
        return result
    return case_summary


def test_source_only_results_report_nulls_not_invented_hardware(tmp_path, monkeypatch):
    from fpgai_research.experiments import results as origin
    root = _prepare_pair(tmp_path)
    monkeypatch.setattr(origin, "summarize_case",
                        _fake_native(root, do_e4=False))
    report = study.collect(root, tmp_path / "report")
    assert report["prepared_cases"] == 2
    assert report["hls_qualified_cases"] == 0
    assert report["strict_e4_qualified_cases"] == 0
    assert report["matched_strict_e4_pairs"] == 0
    assert all(row["hls_latency_worst_cycles"] is None for row in report["rows"])
    assert all(row["postroute_wns_ns"] is None for row in report["rows"])
    assert all(row["measured_board_latency_us"] is None for row in report["rows"])
    assert all(row["measured_fmax_mhz"] is None for row in report["rows"])
    assert report["within_network_comparisons"][0][
        "delta_worst_hls_cycles_mapped_minus_scalar"] is None
    assert "TBD" in (tmp_path / "report/results.csv").read_text()
    assert "TBD" in (tmp_path / "report/results.md").read_text()
    with pytest.raises(FileExistsError):
        study.collect(root, tmp_path / "report")


def test_matched_strict_e4_pair_keeps_estimates_distinct_from_board_results(
        tmp_path, monkeypatch):
    from fpgai_research.experiments import results as origin
    root = _prepare_pair(tmp_path)
    monkeypatch.setattr(origin, "summarize_case", _fake_native(root))
    report = study.collect(root, tmp_path / "results")
    assert report["strict_e4_qualified_cases"] == 2
    assert report["matched_strict_e4_pairs"] == 1
    pair = report["within_network_comparisons"][0]
    assert pair["both_strict_e4_qualified"] is True
    assert pair["delta_worst_hls_cycles_mapped_minus_scalar"] == -90
    assert pair["delta_postroute_lut_mapped_minus_scalar"] == -10
    assert all(row["hls_nominal_worst_latency_us"] is not None
               for row in report["rows"])
    assert all(row["measured_board_latency_us"] is None
               for row in report["rows"])
    assert all(row["measured_fmax_mhz"] is None
               for row in report["rows"])
    assert all(x["highest_qualified_tested_target_mhz"] == 50.0
               for x in report["best_qualified_tested_clock_targets"])


@pytest.mark.parametrize("mutation", ["drc_warning", "timing_fail", "csim_fail",
                                        "missing_source_hash", "hls_not_attested"])
def test_negative_native_evidence_is_preserved_and_not_qualified(
        tmp_path, monkeypatch, mutation):
    from fpgai_research.experiments import results as origin
    root = _prepare_pair(tmp_path)
    original = _fake_native(root)

    def changed(folder):
        result = deepcopy(original(folder))
        if folder.endswith("case_0001"):
            if mutation == "drc_warning":
                result["vivado_metrics"]["drc"]["violations"] = 1
                result["vivado_metrics"]["drc"]["clean"] = False
                result["vivado_metrics"]["drc"]["critical_warnings"] = 0
            elif mutation == "timing_fail":
                result["vivado_metrics"]["timing"] = {
                    "wns_ns": -1.2, "timing_met": False}
            elif mutation == "csim_fail":
                result["csim_passed"] = False
            elif mutation == "missing_source_hash":
                result["source_sha256"] = "f" * 64
            elif mutation == "hls_not_attested":
                result["hls_report_sha256"] = None
        return result

    monkeypatch.setattr(origin, "summarize_case", changed)
    if mutation == "missing_source_hash":
        with pytest.raises(ValueError, match="does not match"):
            study.collect(root, tmp_path / "invalid")
    else:
        report = study.collect(root, tmp_path / "invalid")
        if mutation == "hls_not_attested":
            # E4 qualifier itself checks physical route and CSim, whereas
            # the table's HLS metric population requires independent E3
            # report attestation.
            assert report["rows"][1]["hls_qualified"] is False
        else:
            assert report["rows"][1]["e4_qualified"] is False
        assert report["matched_strict_e4_pairs"] == 0
        assert report["within_network_comparisons"][0][
            "delta_postroute_lut_mapped_minus_scalar"] is None


def test_source_hash_mutation_rejected_before_vendor_plan(tmp_path):
    root = _prepare_pair(tmp_path)
    case = root / "cases/case_0000"
    (case / "model.cpp").write_text((case / "model.cpp").read_text() + "\n// tampered")
    with pytest.raises(ValueError, match="Changed emitted"):
        study.vendor(root, stage="vitis", execute=False)


def test_native_plan_never_uses_vendor_and_vivado_needs_qualified_e3(
        tmp_path, monkeypatch):
    root = _prepare_pair(tmp_path)
    result = study.vendor(root, stage="vitis", execute=False, first=0, limit=2)
    assert result["executed"] is False
    assert len(result["cases"]) == 2
    assert all(item["status"] == "DRY_RUN_NOT_EXECUTED"
               for item in result["cases"])
    assert not any((root/"cases"/row["case"]/"vitis_invocation.json").exists()
                   for row in study._inventory(root)["cases"])
    with pytest.raises(ValueError, match="requires source-attested E3"):
        study.vendor(root, stage="vivado", execute=False)


@pytest.mark.parametrize("path", ["tb.cpp", "run_hls.tcl", "vivado/run_vivado.tcl"])
def test_prepared_native_inputs_remain_immutable(tmp_path, path):
    root = _prepare_pair(tmp_path)
    case = root / "cases/case_0000"
    target = case / path
    target.write_bytes(target.read_bytes() + b"\n# tamper")
    with pytest.raises(ValueError, match="Changed source-attested native input"):
        study._inventory(root)


def test_experiment_manifest_clock_and_precision_tampering_rejected(tmp_path):
    root = _prepare_pair(tmp_path)
    file = root / "cases/case_0000/manifest.json"
    old = json.loads(file.read_text())
    old["clock_ns"] = 13.0
    file.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="numeric/clock/source manifest mismatch"):
        study._inventory(root)


def test_native_reinvocation_is_refused_even_as_a_dry_run(tmp_path):
    root = _prepare_pair(tmp_path)
    case = root / "cases/case_0000"
    (case / "vitis_invocation.json").write_text('{"status":"previous run"}')
    with pytest.raises(FileExistsError, match="never rerun"):
        study.vendor(root, stage="vitis", execute=False)
