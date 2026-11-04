"""R2m negative controls; source-only, no vendor or original study writes."""
from __future__ import annotations

from copy import deepcopy
import math

import pytest

import fpgai_research.experiments.cst_r2m_original_physical_paper as r2m

POSITIVES = (20, 23, 28, 31, 32, 35)
WNS = {
    20: (26.356, 27.304),
    23: (26.941, 27.008),
    28: (25.491, 27.133),
    31: (26.359, 27.054),
    32: (27.205, 26.863),
    35: (27.119, 26.121),
}


def _fixture():
    ledger_rows = []
    effects_cases = []
    original_cases = []
    followup_cases = []
    e3_indices = set(range(18)) | set(POSITIVES)
    assert len(e3_indices) == 24
    for pair in range(36):
        pos = pair in POSITIVES
        a, b = f"case_{2 * pair:03d}", f"case_{2 * pair + 1:03d}"
        left_bram = right_bram = 1.5 if pair == 20 else 2.0
        lut_a, lut_b = (649, 684) if pair == 20 else (400, 398)
        ff_a, ff_b = (396, 407) if pair == 20 else (300, 299)
        wa, wb = WNS.get(pair, (1.0, 1.0))
        selected = ({
            "both_routed_observation_valid": True,
            "both_strict_clean_DRC_E4": False,
            "routed_resource_delta_disjoint_minus_optimal": {
                "bram": left_bram - right_bram,
                "lut": lut_a - lut_b, "ff": ff_a - ff_b, "dsp": 0,
            },
            "optimized_minus_disjoint_WNS_ns": wb - wa,
        } if pos else None)
        ledger_rows.append({
            "pair": pair, "cases": [a, b], "graph": f"graph{pair}",
            "mapping": "scalar",
            "both_original_E3_qualified": pair in e3_indices,
            "HLS_estimated_delta_BRAM_18K_disjoint_minus_optimal":
                (1 if pos else 0) if pair in e3_indices else None,
            "selected_physical_observation": selected,
        })
        effects_cases.append({
            "source_scalar_slot_savings": 2,
            "HLS_worst_cycle_delta_disjoint_minus_optimal":
                25 if pair in e3_indices else None,
        })
        for variant in (0, 1):
            index = pair * 2 + variant
            row = {"case": f"case_{index:03d}"}
            if pos:
                row.update({
                    "routed_observation_valid": True,
                    "routed_timing_met": True,
                    "routed_wns_ns": wa if variant == 0 else wb,
                    "physical_resources": {
                        "bram": left_bram if variant == 0 else right_bram,
                        "lut": lut_a if variant == 0 else lut_b,
                        "ff": ff_a if variant == 0 else ff_b,
                        "dsp": 5,
                    },
                    "routed_qualified": False,
                    "routed_warning_observation": True,
                    "drc_verified_clean": False,
                    "drc_warning_summary": {
                        "summary_verified": True, "violations": 1,
                        "rules": {"DPIP-2": {"violations": 1}},
                    },
                    "original_routed_report_sha256": {
                        name: "a" * 64 for name in r2m.CASE_REPORTS
                    },
                })
            original_cases.append(deepcopy(row))
            followup_cases.append(deepcopy(row))
    ledger = {
        "rows": ledger_rows,
        "original_HLS_source_revision": "b2d5dcc4f18fad7cb1cd2c342101dd33185e817f",
        "original_physical_observation_pair_indices": [20],
        "separate_followup_observation_pair_indices": [23, 28, 31, 32, 35],
        "original_HLS_positive_pairs_with_routed_observations": 6,
        "HLS_positive_unobserved_pairs": 0,
        "original_both_E3_qualified_pairs": 24,
        "original_E3_unknown_pairs": 12,
    }
    return ledger, {"cases": effects_cases}, {"cases": original_cases}, {"cases": followup_cases}


def _run(monkeypatch, data):
    ledger, effects, original, followup = data
    monkeypatch.setattr(r2m, "build_completion_ledger", lambda **_: ledger)
    return r2m.compile_paper_results(
        native={}, effects=effects, original=original, followup=followup,
    )


def test_final_six_pairs_keep_entire_denominator_and_real_tiers(monkeypatch):
    report = _run(monkeypatch, _fixture())
    assert report["schema"] == r2m.SCHEMA
    assert len(report["rows"]) == 36
    assert report["both_original_E3_pairs"] == 24
    assert report["E3_unknown_pairs"] == 12
    assert report["selected_original_routed_pairs"] == 6
    assert report["physical_descriptive_BRAM_tie_pairs"] == 6
    assert report["physical_descriptive_BRAM_saving_pairs"] == 0
    assert report["strict_clean_DRC_E4_qualified_pairs"] == 0
    assert report["warning_only_descriptive_pairs"] == 6
    assert report["total_unobserved_physical_matched_pairs"] == 30
    assert report["descriptive_optimized_WNS_increase_pairs"] == 4
    assert report["descriptive_optimized_WNS_decrease_pairs"] == 2
    assert report["descriptive_mean_optimized_minus_disjoint_WNS_ns"] == pytest.approx(
        2.012 / 6)
    assert report["descriptive_optimized_LUT_increase_pairs"] == 1
    assert report["descriptive_optimized_LUT_decrease_pairs"] == 5
    assert report["descriptive_optimized_FF_increase_pairs"] == 1
    assert report["descriptive_optimized_FF_decrease_pairs"] == 5
    assert report["descriptive_mean_optimized_minus_disjoint_LUT"] == pytest.approx((35+5*(-2))/6)
    assert report["new_original_GCC_Vitis_Vivado_calls_during_export"] == 0
    assert report["rows"][20]["physical_LUT_saved_descriptive"] == -35
    assert report["rows"][20]["physical_FF_saved_descriptive"] == -11
    assert all(report["rows"][p]["physical_BRAM_tiles_saved_descriptive"] is None
               for p in range(36) if p not in POSITIVES)
    assert report["rows"][19]["HLS_worst_cycle_reduction"] is None
    table = r2m._md(report)
    assert "30 other matched" in table
    assert "warning-only" in table
    assert "**0/6 strict clean-DRC E4**." in table
    assert "DSP disjoint" in table
    assert "Descriptive routed LUT changes" in table


def test_original_and_followup_selection_fail_closed(monkeypatch):
    data = _fixture()
    data[0]["separate_followup_observation_pair_indices"] = [23, 28, 31, 32]
    with pytest.raises(ValueError, match="six completed prespecified pairs"):
        _run(monkeypatch, data)


def test_missing_original_routed_report_sha_rejected(monkeypatch):
    data = _fixture()
    del data[3]["cases"][46]["original_routed_report_sha256"]["drc.rpt"]
    with pytest.raises(ValueError, match="digest provenance"):
        _run(monkeypatch, data)


def test_invalid_original_drc_verification_rejected(monkeypatch):
    data = _fixture()
    data[2]["cases"][40]["drc_warning_summary"]["summary_verified"] = False
    with pytest.raises(ValueError, match="not independently verified"):
        _run(monkeypatch, data)


def test_missing_rule_cannot_be_relabelled_warning_only(monkeypatch):
    data = _fixture()
    data[3]["cases"][57]["drc_warning_summary"]["rules"] = {}
    with pytest.raises(ValueError, match="must retain DRC violations"):
        _run(monkeypatch, data)


def test_wrong_routed_resource_delta_cannot_be_published(monkeypatch):
    data = _fixture()
    data[0]["rows"][28]["selected_physical_observation"][
        "routed_resource_delta_disjoint_minus_optimal"]["bram"] = 1
    with pytest.raises(ValueError, match="resource delta mismatch"):
        _run(monkeypatch, data)


def test_wrong_routed_wns_cannot_be_published(monkeypatch):
    data = _fixture()
    data[0]["rows"][32]["selected_physical_observation"][
        "optimized_minus_disjoint_WNS_ns"] = -0.12
    with pytest.raises(ValueError, match="WNS delta mismatch"):
        _run(monkeypatch, data)


def test_strict_e4_cannot_be_invented(monkeypatch):
    data = _fixture()
    data[0]["rows"][23]["selected_physical_observation"][
        "both_strict_clean_DRC_E4"] = True
    with pytest.raises(ValueError, match="strict pair qualification mismatch"):
        _run(monkeypatch, data)


def test_nonfinite_wns_cannot_be_reported(monkeypatch):
    data = _fixture()
    data[3]["cases"][71]["routed_wns_ns"] = math.nan
    with pytest.raises(ValueError, match="WNS absent or nonfinite"):
        _run(monkeypatch, data)


def test_nonpositive_unknown_physical_not_imputed(monkeypatch):
    data = _fixture()
    data[0]["rows"][0]["selected_physical_observation"] = {
        "both_routed_observation_valid": True}
    with pytest.raises(ValueError, match="unregistered pair"):
        _run(monkeypatch, data)


def test_creates_source_only_tables_and_refuses_overwrite(monkeypatch, tmp_path):
    import fpgai_research.experiments.cst_r2i_fanout_native_execute as nat
    import fpgai_research.experiments.cst_r2i_fanout_native_effects as fx
    import fpgai_research.experiments.cst_r2j_fanout_physical as phy
    ledger, effects, original, followup = _fixture()
    monkeypatch.setattr(r2m, "build_completion_ledger", lambda **_: ledger)
    monkeypatch.setattr(nat, "audit_native", lambda *a, **k: {})
    monkeypatch.setattr(fx, "compute_effects", lambda *a, **k: effects)
    incoming = iter((original, followup))
    monkeypatch.setattr(phy, "audit_routed_study", lambda *a, **k: next(incoming))
    path = tmp_path / "paper"
    report = r2m.export(hls_study="ignored", original_physical_study="ignored",
                        followup_study="ignored", output=path)
    assert len(report["rows"]) == 36
    assert (path / "results.json").is_file()
    assert len((path / "all_36_pairs.csv").read_text().splitlines()) == 37
    assert (path / "paper_table.md").is_file()
    with pytest.raises(FileExistsError, match="create-only"):
        r2m.export(hls_study="ignored", original_physical_study="ignored",
                   followup_study="ignored", output=path)
