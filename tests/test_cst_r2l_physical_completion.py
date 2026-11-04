"""R2l: prospective physical completion ledger, no vendor calls or writes."""
from __future__ import annotations
import json
from copy import deepcopy
import pytest

from fpgai_research.experiments.cst_r2l_physical_completion import (
    ORIGINAL_HLS_POSITIVE_PAIRS, SCHEMA, build_completion_ledger,
)


def _fixtures():
    native_cases = []
    pilot_cases = []
    followup_cases = []
    effects_cases = []
    for pair in range(36):
        positive = pair in ORIGINAL_HLS_POSITIVE_PAIRS
        e3 = pair < 18 or positive
        for j in range(2):
            index = 2 * pair + j
            name = f"case_{index:03d}"
            shared = {
                "case": name, "group": pair,
                "graph": f"graph_{pair}", "mapping": "scalar",
                "source_storage": "disjoint" if j == 0 else "per_element_optimal",
                "source_cpp_sha256": ("a" if j == 0 else "b") * 64,
            }
            native_cases.append({**shared,
                                 "original_CSynth_timing_resources_qualified": e3})
            physical = {**shared,
                        "original_hls_qualified": e3,
                        "routed_observation_valid": pair == 20,
                        "routed_qualified": False,
                        "routed_warning_observation": pair == 20,
                        "routed_wns_ns": (26.356 if j == 0 else 27.304)
                            if pair == 20 else None,
                        "physical_resources": (
                            {"bram": 1.5, "lut": (649 if j == 0 else 684),
                             "ff": (396 if j == 0 else 407), "dsp": 7}
                            if pair == 20 else None),
                        "drc_verified_clean": False if pair == 20 else None,
                        "drc_warning_summary": ({
                            "rules": {"DPIP-2": {"severity": "Warning", "violations": 3}}}
                            if pair == 20 else None),
                        "status": ("OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS"
                                   if pair == 20 else "PREPARED_NOT_RUN" if e3
                                   else "BLOCKED_NO_QUALIFIED_ORIGINAL_HLS"),
                        "requested_clock_ns": 32.0}
            pilot_cases.append(physical)
            followup_cases.append({**physical,
                                   "routed_observation_valid": False,
                                   "routed_wns_ns": None,
                                   "physical_resources": None,
                                   "drc_warning_summary": None,
                                   "routed_warning_observation": False,
                                   "drc_verified_clean": None,
                                   "status": "PREPARED_NOT_RUN" if e3
                                   else "BLOCKED_NO_QUALIFIED_ORIGINAL_HLS"})
        effects_cases.append({
            "group": pair,
            "cases": [f"case_{2*pair:03d}", f"case_{2*pair+1:03d}"],
            "both_original_E3_qualified": e3,
            "HLS_estimated_delta_disjoint_minus_optimal": {
                "bram_18k": (1 if positive else 0) if e3 else None,
            },
        })
    native = {
        "original_source_revision": "b2d5dcc4f18fad7cb1cd2c342101dd33185e817f",
        "original_CSim_qualified_cases": 72,
        "original_CSynth_timing_resource_qualified_cases": 48,
        "prespecified_source_cases": 72,
        "prespecified_matched_pairs": 36,
        "per_case_status": native_cases,
    }
    effects = {
        "original_source_revision": "b2d5dcc4f18fad7cb1cd2c342101dd33185e817f",
        "both_E3_qualified_matched_pairs": sum(r["both_original_E3_qualified"] for r in effects_cases),
        "not_both_E3_qualified_matched_pairs": sum(not r["both_original_E3_qualified"] for r in effects_cases),
        "cases": effects_cases,
    }
    original = {
        "prespecified_cases": 72, "original_hls_source_revision": "b2d5dcc4f18fad7cb1cd2c342101dd33185e817f",
        "cases": pilot_cases,
    }
    followup = deepcopy(original)
    followup["cases"] = followup_cases
    return native, effects, original, followup


def _run(native, effects, original, followup=None):
    return build_completion_ledger(native=native, effects=effects,
                                   original_physical=original,
                                   followup_physical=followup)


def test_original_readonly_ledger_keeps_denominators_and_negative_pilot():
    native, effects, original, followup = _fixtures()
    report = _run(native, effects, original)
    assert report["schema"] == SCHEMA
    assert report["original_prespecified_cases"] == 72
    assert report["original_both_E3_qualified_pairs"] == 24
    assert report["original_E3_unknown_pairs"] == 12
    assert report["original_prespecified_pairs"] == 36
    assert report["original_HLS_positive_pairs"] == [20, 23, 28, 31, 32, 35]
    assert report["original_HLS_positive_pairs_with_routed_observations"] == 1
    assert report["positive_pairs_with_routed_BRAM_tie_descriptive"] == 1
    assert report["positive_pairs_with_routed_BRAM_reduction_descriptive"] == 0
    assert report["original_HLS_positive_pairs_with_strict_E4"] == 0
    assert report["HLS_positive_unobserved_pairs"] == 5
    assert report["new_original_GCC_Vitis_Vivado_calls"] == 0
    assert report["rows"][20]["original_observation"][
        "routed_resource_delta_disjoint_minus_optimal"]["lut"] == -35
    assert report["rows"][20]["followup_execution_status"] == "ORIGINAL_PHYSICAL_PILOT_OBSERVED"
    assert all(report["rows"][i]["selected_physical_observation"] is None
               for i in (23, 28, 31, 32, 35))


def test_new_separate_followup_only_allows_remaining_five():
    native, effects, original, followup = _fixtures()
    report = _run(native, effects, original, followup)
    assert report["all_ready_original_pair_indices"] == [23, 28, 31, 32, 35]
    assert report["ready_for_next_original_pair_in_new_study"] == 23
    assert report["original_physical_observation_pair_indices"] == [20]
    assert report["separate_followup_observation_pair_indices"] == []
    assert report["new_original_GCC_Vitis_Vivado_calls"] == 0


def test_followup_routed_bram_saving_is_observation_only_without_clean_drc():
    native, effects, original, followup = _fixtures()
    for j in (46, 47):
        row = followup["cases"][j]
        row["routed_observation_valid"] = True
        row["physical_resources"] = {"bram": 2.5 if j == 46 else 1.5,
                                     "lut": 500, "ff": 500, "dsp": 4}
        row["routed_wns_ns"] = 1.25
        row["routed_warning_observation"] = True
        row["drc_warning_summary"] = {"rules": {"DPIP-2": {"severity": "Warning", "violations": 1}}}
    report = _run(native, effects, original, followup)
    assert report["original_HLS_positive_pairs_with_routed_observations"] == 2
    assert report["positive_pairs_with_routed_BRAM_reduction_descriptive"] == 1
    assert report["original_HLS_positive_pairs_with_strict_E4"] == 0
    assert report["HLS_positive_unobserved_pairs"] == 4
    assert report["ready_for_next_original_pair_in_new_study"] == 28


def test_duplicate_pilot_prohibited_in_followup():
    native, effects, original, followup = _fixtures()
    for j in (40, 41):
        followup["cases"][j]["routed_observation_valid"] = True
        followup["cases"][j]["physical_resources"] = dict(original["cases"][j]["physical_resources"])
        followup["cases"][j]["routed_wns_ns"] = original["cases"][j]["routed_wns_ns"]
    with pytest.raises(ValueError, match="duplicated"):
        _run(native, effects, original, followup)


def test_tampered_source_or_clock_fails_closed():
    native, effects, original, followup = _fixtures()
    followup["cases"][47]["requested_clock_ns"] = 20.0
    with pytest.raises(ValueError, match="changed original source or clock"):
        _run(native, effects, original, followup)
    followup["cases"][47]["requested_clock_ns"] = 32.0
    original["cases"][46]["source_cpp_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="exact-source identity"):
        _run(native, effects, original, followup)


def test_partial_failed_unknown_does_not_trigger_rerun():
    native, effects, original, followup = _fixtures()
    followup["cases"][46]["status"] = "REJECTED_VENDOR_PROVENANCE"
    report = _run(native, effects, original, followup)
    assert report["rows"][23]["followup_execution_status"] == (
        "PARTIAL_FAILED_OR_BLOCKED_NO_AUTOMATIC_RERUN")
    assert report["ready_for_next_original_pair_in_new_study"] == 28


def test_false_hls_positive_population_rejected():
    native, effects, original, followup = _fixtures()
    effects["cases"][23]["HLS_estimated_delta_disjoint_minus_optimal"]["bram_18k"] = 0
    with pytest.raises(ValueError, match="preregistration"):
        _run(native, effects, original, followup)


def test_unregistered_followup_pair_is_rejected():
    native, effects, original, followup = _fixtures()
    for j in (0, 1):
        row = followup["cases"][j]
        row["routed_observation_valid"] = True
        row["physical_resources"] = {"bram": 2, "lut": 300, "ff": 200, "dsp": 5}
        row["routed_wns_ns"] = 1.0
    with pytest.raises(ValueError, match="unregistered"):
        _run(native, effects, original, followup)


def test_strict_clean_drc_requires_clean_receipts():
    native, effects, original, followup = _fixtures()
    for j in (46, 47):
        row = followup["cases"][j]
        row["routed_qualified"] = True
        row["routed_observation_valid"] = True
        row["physical_resources"] = {"bram": 2, "lut": 300, "ff": 200, "dsp": 5}
        row["routed_wns_ns"] = 1.0
    with pytest.raises(ValueError, match="requires both clean original DRC"):
        _run(native, effects, original, followup)
    for j in (46, 47):
        followup["cases"][j]["drc_verified_clean"] = True
    report = _run(native, effects, original, followup)
    assert report["original_HLS_positive_pairs_with_strict_E4"] == 1


def test_frozen_original_sha_and_e3_population_cannot_change():
    native, effects, original, followup = _fixtures()
    native["original_source_revision"] = "f" * 40
    with pytest.raises(ValueError, match="frozen original R2i source SHA"):
        _run(native, effects, original, followup)
    native, effects, original, followup = _fixtures()
    effects["both_E3_qualified_matched_pairs"] = 25
    with pytest.raises(ValueError, match="frozen original R2i E2/E3 population"):
        _run(native, effects, original, followup)
