"""R3c scientific publication table tests: no GCC/Vitis/Vivado calls."""
from __future__ import annotations

from copy import deepcopy
import csv
import json

import pytest

from fpgai_research.experiments import cst_r3c_cnn_publication_tables as r3c


def _original_six():
    names = ["cnn_pool_dense", "cnn_pool_dense", "cnn_strided_dense",
             "cnn_strided_dense", "cnn_two_conv_softmax",
             "cnn_two_conv_softmax"]
    cycles = (624, 911, None, None, 2365, 5436)
    bram = (1, 1, None, None, 0, 0)
    dsp = (45, 93, None, None, 365, 394)
    ff = (2913, 4709, None, None, 24781, 47598)
    lut = (10555, 19701, None, None, 37131, 77009)
    estimated_clock = (8.971, 8.997, 9.181, 9.181, 8.917, 8.973)
    records = []
    for i in range(6):
        e3 = i not in (2, 3)
        diagnostic = {
            "effective_budget_ns": 9.0,
            "estimated_clock_ns": estimated_clock[i],
            "margin_ns": 9.0 - estimated_clock[i],
            "feasible": e3,
            "original_HLS_report_sha256": f"{i+10:064x}",
        }
        records.append({
            "case": f"case_{i:03d}",
            "network": names[i],
            "functional_sha256": f"{i//2+1:064x}",
            "source_cpp_sha256": f"{i+11:064x}",
            "state": ("QUALIFIED_HLS_E3_ESTIMATES_ONLY" if e3
                      else "ATTEMPTED_BUT_NOT_E3_QUALIFIED"),
            "original_vitis_invocation_present": True,
            "E2_original_CSim": True,
            "E3_original_CSynth": e3,
            "qualification_status": ("QUALIFIED_HLS_ESTIMATES_NOT_PHYSICAL"
                                     if e3 else "FAILED_HLS_EFFECTIVE_CLOCK_BUDGET"),
            "attested_HLS_timing_diagnostic": diagnostic,
            "hls_worst_cycles": cycles[i],
            "hls_estimated_clock_ns": estimated_clock[i] if e3 else None,
            "HLS_estimated_resources": ({
                "bram_18k": bram[i], "dsp": dsp[i], "ff": ff[i],
                "lut": lut[i], "uram": 0} if e3 else None),
        })
    pairs = [
        {
            "network": names[i],
            "control": f"case_{i:03d}",
            "independent_Phi": f"case_{i+1:03d}",
            "both_E2": True,
            "both_E3": i != 2,
            "HLS_estimated_BRAM18k_delta": 0 if i != 2 else None,
            "HLS_estimated_DSP_delta": (48 if i == 0 else
                                          29 if i == 4 else None),
            "HLS_estimated_FF_delta": (1796 if i == 0 else
                                         22817 if i == 4 else None),
            "HLS_estimated_LUT_delta": (9146 if i == 0 else
                                          39878 if i == 4 else None),
            "HLS_worst_latency_cycles_delta": (287 if i == 0 else
                                                   3071 if i == 4 else None),
        } for i in (0, 2, 4)
    ]
    return {
        "schema": "fpgai-research/R3b-CNN-Dense-publication-handoff/v1",
        "source_revision": "f665b62ca81ee181a1447e3eddd23d7eba578186",
        "source_GCC_E1_qualified_inputs": 12,
        "original_source_packages": 6,
        "original_Vitis_invocation_receipts": 6,
        "original_CSim_E2_qualified": 6,
        "original_CSynth_E3_qualified": 4,
        "original_native_attempted_not_E3": 2,
        "matched_E3_qualified_pairs": 2,
        "matched_E3_unknown_or_rejected_pairs": 1,
        "cases": records, "contrasts": pairs,
    }


def test_complete_original_six_case_tables_are_explicitly_hls_only():
    data = r3c.analyze(_original_six(), requested_clock_ns=10.0)
    assert data["original_native_attempts"] == 6
    assert data["original_CSim_E2_qualified"] == 6
    assert data["original_CSynth_E3_qualified"] == 4
    assert data["original_CSynth_E3_rejected"] == 2
    assert data["matched_pairs_both_E3_qualified"] == 2
    assert data["qualified_pairs_independent_phi_more_cycles"] == 2
    assert data["qualified_pairs_estimated_BRAM_ties"] == 2
    assert data["cases"][0]["estimated_latency_ns"] == 6240.0
    assert data["cases"][1]["estimated_latency_ns"] == 9110.0
    assert data["cases"][2]["estimated_latency_ns"] is None
    assert data["cases"][3]["bram_18k"] is None
    assert data["pairs"][0]["cycles_delta"] == 287
    assert data["pairs"][0]["estimated_latency_delta_ns"] == 2870.0
    assert data["pairs"][1]["cycles_delta"] is None
    assert data["pairs"][1]["dsp_delta"] is None
    assert data["pairs"][2]["cycles_delta"] == 3071
    assert data["pairs"][2]["ff_delta"] == 22817
    assert data["physical_E4"] is None
    assert data["learning_improvement"] is None
    text = r3c.markdown(data)
    assert "REJECTED E3 / NO DELTA" in text
    assert "624 → 911" in text
    assert "2365 → 5436" in text
    assert "not achieved physical latency" in text


@pytest.mark.parametrize("change", [
    lambda r: r.update(original_Vitis_invocation_receipts=5),
    lambda r: r["cases"][0].update(original_vitis_invocation_present=False),
    lambda r: r["cases"][2].update(state="UNATTEMPTED_UNKNOWN"),
    lambda r: r["cases"][2].update(hls_worst_cycles=999),
    lambda r: r["cases"][2].update(HLS_estimated_resources={"bram_18k": 1}),
    lambda r: r["cases"][0]["HLS_estimated_resources"].update(dsp=-1),
    lambda r: r["cases"][0]["attested_HLS_timing_diagnostic"].update(margin_ns=-1),
    lambda r: r["contrasts"][0].update(HLS_estimated_DSP_delta=47),
    lambda r: r["contrasts"][1].update(HLS_estimated_BRAM18k_delta=0),
    lambda r: r["contrasts"][2].update(both_E3=False),
    lambda r: r.update(matched_E3_qualified_pairs=3),
])
def test_incomplete_fabricated_or_conflicting_hls_evidence_is_rejected(change):
    original = _original_six()
    modified = deepcopy(original)
    change(modified)
    with pytest.raises(ValueError, match="R3c:"):
        r3c.analyze(modified, requested_clock_ns=10.0)


def test_create_only_raw_denominator_export_with_digests(tmp_path):
    report = r3c.analyze(_original_six(), requested_clock_ns=10.0)
    output = tmp_path / "independent_publication_report"
    r3c.export_create_only(report, output=output)
    assert json.loads((output / "summary.json").read_text())["original_CSynth_E3_qualified"] == 4
    with (output / "case_population.csv").open(newline="") as fp:
        case_rows = list(csv.DictReader(fp))
    with (output / "matched_pair_population.csv").open(newline="") as fp:
        pair_rows = list(csv.DictReader(fp))
    assert len(case_rows) == 6
    assert len(pair_rows) == 3
    assert case_rows[2]["worst_hls_cycles"] == ""
    assert pair_rows[1]["cycles_delta"] == ""
    assert pair_rows[0]["bram_18k_delta"] == "0"
    digests = json.loads((output / "derived_export_sha256.json").read_text())
    assert len(digests) == 4
    assert set(digests) == {
        "summary.json", "paper_results.md", "case_population.csv",
        "matched_pair_population.csv"}
    with pytest.raises(FileExistsError):
        r3c.export_create_only(report, output=output)


def test_requested_hls_clock_alters_estimate_not_native_e3_gate():
    original = _original_six()
    a = r3c.analyze(original, requested_clock_ns=10.0)
    b = r3c.analyze(original, requested_clock_ns=11.0)
    # This calculation is a derived estimate only; it does not change any
    # original CSim/E3 receipt and is NOT a native re-synthesis at 11 ns.
    assert a["matched_pairs_both_E3_qualified"] == b["matched_pairs_both_E3_qualified"]
    assert b["pairs"][0]["estimated_latency_delta_ns"] == 287 * 11
    with pytest.raises(ValueError, match="R3c:"):
        r3c.analyze(original, requested_clock_ns=float("nan"))
