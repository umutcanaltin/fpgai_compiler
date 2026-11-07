"""Paper memory bridge: strict matched, source/RTL evidence accounting.

Pure fixtures test the inference logic. Native Vitis/Vivado and frozen
original evidence must never be run or rewritten by these tests.
"""
from __future__ import annotations

import json

import pytest

from fpgai_research.experiments import cst_paper_memory_bridge as sut


def fixture_analysis():
    cases = [{"case": f"case_{i:03d}"} for i in range(32)]
    outcomes = ["decrease"] * 2 + ["tie"] * 47 + ["increase"] * 8 + ["unknown"] * 7
    contrasts = []
    index = 0
    for factor in sut.FACTORS:
        for graph in (0, 1):
            for j in range(8):
                outcome = outcomes[index]
                qualified = outcome != "unknown"
                delta = {"decrease": -1, "tie": 0,
                         "increase": 1, "unknown": None}[outcome]
                contrasts.append({
                    "factor": factor,
                    "graph_index": graph,
                    "arm_a": f"case_{graph*16+j:03d}",
                    "arm_b": f"case_{graph*16+j+8:03d}",
                    "original_E3_both_qualified": qualified,
                    "original_RTL_both_SHA_attested": True,
                    "both_top_static_reachability_known": True,
                    "estimated_BRAM_delta_or_null": delta,
                    "reachable_memory_instance_count_delta": (
                        -2 if index % 3 == 0 else 0),
                    "structural_pair_status": "STATIC_TYPE_GRAPH_COMPARISON_ONLY",
                })
                index += 1
    return {
        "schema": sut.R2X_SCHEMA,
        "original_source_revision": sut.SOURCE_REVISION,
        "original_R2u_cases": 32,
        "original_R2v_conditional_pairs": 64,
        "original_R2u_E3_qualified_cases": 30,
        "original_R2v_E3_qualified_pairs": 57,
        "original_R2v_E3_unknown_pairs": 7,
        "original_RTL_SHA_attested_cases": 32,
        "original_vendor_calls_during_analysis": 0,
        "physical_BRAM": None,
        "achieved_routed_Fmax": None,
        "cases": cases,
        "contrasts": contrasts,
    }


def test_pairwise_bridge_preserves_unknowns_and_measurement_tiers():
    result = sut.summarize(fixture_analysis())
    assert result["original_matched_pairs"] == 64
    assert result["HLS_E3_qualified_pairs"] == 57
    assert result["HLS_E3_unknown_pairs"] == 7
    assert result["HLS_BRAM18K_response"] == {
        "decrease": 2, "tie": 47, "increase": 8, "unknown": 7}
    assert len(result["contingency"]) == 16
    assert sum(r["pair_count"] for r in result["contingency"]) == 64
    assert sum(r["planned_pairs"] for r in result["by_factor_and_graph"]) == 64
    assert result["jointly_interpretable_HLS_RTL_pairs"] == 57
    assert result["RTL_changed_HLS_tied_count"] > 0
    assert result["routed_FPGA_BRAM"] is None
    assert result["original_vendor_calls_during_analysis"] == 0


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(physical_BRAM=3),
    lambda d: d.update(original_R2v_E3_qualified_pairs=58),
    lambda d: d["contrasts"][0].update(
        original_RTL_both_SHA_attested=False),
    lambda d: d["contrasts"][0].update(estimated_BRAM_delta_or_null=None),
    lambda d: d["contrasts"][63].update(
        estimated_BRAM_delta_or_null=0),
    lambda d: d["contrasts"][1].update(
        arm_a=d["contrasts"][0]["arm_a"],
        arm_b=d["contrasts"][0]["arm_b"]),
    lambda d: d["contrasts"][1].update(
        both_top_static_reachability_known=False),
])
def test_inconsistent_or_misleading_evidence_is_rejected(mutation):
    raw = fixture_analysis()
    mutation(raw)
    with pytest.raises(ValueError):
        sut.summarize(raw)


def test_unknown_static_structure_is_explicit_not_zero():
    raw = fixture_analysis()
    raw["contrasts"][1].update(
        both_top_static_reachability_known=False,
        reachable_memory_instance_count_delta=None,
        structural_pair_status="UNKNOWN_UNSUPPORTED_STATIC_RTL",
    )
    result = sut.summarize(raw)
    assert result["jointly_interpretable_HLS_RTL_pairs"] == 56
    assert result["pairs"][1]["RTL_candidate_class"] == "unknown"


def test_create_only_export_does_not_modify_original_fixture(tmp_path):
    raw = fixture_analysis()
    before = json.dumps(raw, sort_keys=True)
    report = sut.summarize(raw)
    dest = tmp_path / "new_report"
    sut.export_create_only(report, output=dest)
    assert (dest / "summary.json").is_file()
    assert (dest / "paired_observations.csv").is_file()
    digests = json.loads((dest / "derived_sha256.json").read_text())
    assert len(digests) == 2
    assert before == json.dumps(raw, sort_keys=True)
    with pytest.raises(FileExistsError):
        sut.export_create_only(report, output=dest)
