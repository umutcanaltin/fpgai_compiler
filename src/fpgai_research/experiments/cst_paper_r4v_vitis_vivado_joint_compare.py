"""R4v: original matched storage comparison AFTER Vitis HLS and Vivado routing.

READ ONLY. This is the missing engineering/scientific bridge from the SAME
SHA-attested full_storage / row_storage source schedule to observed
Vitis CSIM, CSynth HLS estimates, and original Vivado routed OOC reports.

IMPORTANT: Both original Vivado arms have negative WNS and nonclean DRC.
The source-level code does NOT claim a physically achievable Fmax, inferred
FPGA latency, board throughput or E4. A Vitis cycle count times a Vitis
estimated period is merely an E3 timing PROXY, not actual board latency.
No Vitis/Vivado/GCC/other vendor tool is launched.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from fpgai_research.experiments import cst_r4l_schedule_matched_storage_native as r4l
from fpgai_research.experiments import cst_r4q_original_r4p_physical_diagnostics as r4q
from fpgai_research.experiments import cst_r4r_original_routed_tradeoff_ledger as r4r

SCHEMA = "fpgai-research/R4v-original-source-pinned-Vitis-Vivado-paired-scorecard/v1"
ARMS = ("full_storage", "row_storage")
HLS_RESOURCE = ("bram_18k", "dsp", "lut", "ff", "uram")
VIVADO_RESOURCE = ("bram", "dsp", "lut", "ff", "uram")
ORIGINAL_REQUESTED_VIVADO_CLOCK_NS = 12.0


def _nonnegative(value, *, integer=False) -> bool:
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        return False
    return not integer or int(value) == value


def _sha64(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def compare(hls: dict, routed: dict) -> dict:
    """Join independently audited ORIGINAL vendor stages with fail-closed gates."""
    if (not isinstance(hls, dict)
            or hls.get("schema") != r4l.SCHEMA
            or hls.get("status") != "PAIRED_QUALIFIED_HLS_ESTIMATES_ONLY"
            or hls.get("E1_matched_pair_accepted") is not True
            or hls.get("both_E2_qualified") is not True
            or hls.get("both_E3_qualified") is not True
            or hls.get("E2_qualified_arms") != 2
            or hls.get("E3_qualified_arms") != 2
            or hls.get("new_vendor_calls_during_read_only_audit") != 0
            or hls.get("original_sources_modified") is not False
            or hls.get("physical_E4") is not None
            or not isinstance(hls.get("source_revision"), str)
            or len(hls["source_revision"]) != 40
            or any(c not in "0123456789abcdef"
                   for c in hls["source_revision"])):
        raise ValueError("R4v original Vitis source E1/E2/E3 qualification missing")
    if (not isinstance(routed, dict)
            or routed.get("schema") != r4r.SCHEMA
            or routed.get("status") !=
                "SIGNED_ORIGINAL_ROUTED_OOC_TRADEOFF_TIMING_AND_DRC_NOT_QUALIFIED"
            or routed.get("source_attested_original_routed_pair") is not True
            or routed.get("original_source_revision") != hls["source_revision"]
            or routed.get("strict_clean_DRC_E4") is not False
            or routed.get("physical_E4") is not None
            or routed.get("board_qualified") is not False
            or routed.get("native_vendor_calls_during_analysis") != 0
            or routed.get("original_source_and_reports_modified") is not False):
        raise ValueError("R4v original Vivado R4p/R4q/R4r source/route evidence mismatch")
    source_arms = hls.get("observations")
    routed_arms = routed.get("original_arms")
    if (not isinstance(source_arms, list) or len(source_arms) != 2
            or not isinstance(routed_arms, list) or len(routed_arms) != 2
            or [a.get("arm") for a in source_arms] != list(ARMS)
            or [a.get("arm") for a in routed_arms] != list(ARMS)):
        raise ValueError("R4v original two-arm source/physical order differs")
    source_pkgs = hls.get("package_hashes")
    if (not isinstance(source_pkgs, dict) or set(source_pkgs) != set(ARMS)):
        raise ValueError("R4v original Vitis source/ROM lineage missing")
    paired = {}
    for e3, e4 in zip(source_arms, routed_arms):
        arm = e3["arm"]
        if (e3.get("E2_qualified") is not True
                or e3.get("E3_qualified") is not True
                or e3.get("native_invocation_file_present") is not True
                or e3.get("physical_E4") is not None
                or not _sha64(e3.get("source_sha256"))
                or source_pkgs[arm].get("model.cpp") != e3["source_sha256"]
                or not _sha64(e3.get("HLS_CSynth_report_sha256"))):
            raise ValueError("R4v original Vitis E2/E3 arm not SHA qualified")
        cycles = e3.get("HLS_worst_cycles")
        hls_period = e3.get("HLS_estimated_clock_ns")
        if not _nonnegative(cycles, integer=True) or cycles == 0:
            raise ValueError("R4v missing original Vitis worst cycles")
        if not _nonnegative(hls_period) or hls_period == 0:
            raise ValueError("R4v missing original Vitis estimated clock")
        estimated = e3.get("HLS_estimated_resources")
        if (not isinstance(estimated, dict)
                or any(not _nonnegative(estimated.get(k), integer=True)
                       for k in HLS_RESOURCE)):
            raise ValueError("R4v Vitis estimated resource vector incomplete")
        csim_err = e3.get("CSim_max_absolute_error")
        if not _nonnegative(csim_err):
            raise ValueError("R4v original Vitis numerical CSIM missing/invalid")
        res = e4.get("resources")
        if (not isinstance(res, dict)
                or any(not _nonnegative(res.get(k), integer=True)
                       for k in VIVADO_RESOURCE)
                or e4.get("routed_timing_met") is not False
                or e4.get("strict_clean_DRC_E4") is not False
                or e4.get("board_qualified") is not False
                or not isinstance(e4.get("blocked_conditions"), list)
                or not {"NEGATIVE_ROUTED_WNS", "DRC_NONCLEAN"}.issubset(
                    set(e4["blocked_conditions"]))
                or not _nonnegative(e4.get("drc_total_violations"),integer=True)
                or e4["drc_total_violations"] == 0):
            raise ValueError("R4v routed descriptive results not fully qualified")
        wns = e4.get("routed_wns_ns")
        if type(wns) not in (int,float) or not isfinite(wns) or wns >= 0:
            raise ValueError("R4v original routed failing timing is missing")
        rpt = e4.get("source_report_sha256")
        if (not isinstance(rpt, dict)
                or set(rpt) != set(r4q.REPORTS)
                or any(not _sha64(v) for v in rpt.values())):
            raise ValueError("R4v original route/Drc/util/timing report SHA missing")
        paired[arm] = {
            "arm":arm, "source_cpp_sha256":e3["source_sha256"],
            "original_vitis_csim_qualified": True,
            "vitis_csim_max_absolute_error":csim_err,
            "vitis_hls_worst_cycles":cycles,
            "vitis_hls_estimated_clock_ns":hls_period,
            "vitis_hls_estimated_latency_proxy_ns": round(cycles*hls_period,6),
            "vitis_hls_requested_constraint_latency_proxy_ns":
                round(cycles*ORIGINAL_REQUESTED_VIVADO_CLOCK_NS,6),
            "vitis_hls_estimated_resources":{k:estimated[k] for k in HLS_RESOURCE},
            "vitis_csynth_report_sha256":e3["HLS_CSynth_report_sha256"],
            "vivado_requested_clock_ns":ORIGINAL_REQUESTED_VIVADO_CLOCK_NS,
            "vivado_routed_resources":{k:res[k] for k in VIVADO_RESOURCE},
            "vivado_routed_wns_ns":wns,
            "vivado_routed_timing_met":False,
            "vivado_drc_violations":e4["drc_total_violations"],
            "vivado_blocked_conditions":sorted(e4["blocked_conditions"]),
            "vivado_report_sha256":rpt,
            "timing_qualified_physical_latency_ns":None,
            "board_end_to_end_inference_latency_ns":None,
            "board_qualified":False,
        }
    full, row = (paired[arm] for arm in ARMS)
    deltas = {
        "vitis_worst_cycles": row["vitis_hls_worst_cycles"]-
            full["vitis_hls_worst_cycles"],
        "vitis_estimated_period_ns": round(
            row["vitis_hls_estimated_clock_ns"]-
            full["vitis_hls_estimated_clock_ns"],6),
        "vitis_estimated_latency_proxy_ns": round(
            row["vitis_hls_estimated_latency_proxy_ns"]-
            full["vitis_hls_estimated_latency_proxy_ns"],6),
        "vitis_estimated_resources": {
            key:row["vitis_hls_estimated_resources"][key]-
                full["vitis_hls_estimated_resources"][key]
            for key in HLS_RESOURCE},
        "vivado_routed_resources": {
            key:row["vivado_routed_resources"][key]-
                full["vivado_routed_resources"][key]
            for key in VIVADO_RESOURCE},
        "vivado_routed_wns_ns":round(
            row["vivado_routed_wns_ns"]-
            full["vivado_routed_wns_ns"],6),
        "vivado_routed_drc_violations":
            row["vivado_drc_violations"]-full["vivado_drc_violations"],
    }
    actual_routed = routed.get("physical_resource_delta_row_minus_full")
    if (not isinstance(actual_routed,dict) or any(
            deltas["vivado_routed_resources"][k] != actual_routed.get(k)
            for k in VIVADO_RESOURCE)):
        raise ValueError("R4v Vivado descriptive deltas disagree with signed original")
    if (not _nonnegative(abs(routed.get("routed_WNS_delta_row_minus_full_ns",float("nan"))))
            or abs(deltas["vivado_routed_wns_ns"]-
                   routed["routed_WNS_delta_row_minus_full_ns"])>1e-6):
        raise ValueError("R4v signed original routed WNS delta disagrees")
    return {
        "schema":SCHEMA,
        "status":"VERIFIED_ORIGINAL_VITIS_ESTIMATES_AND_VIVADO_ROUTED_TRADEOFF_NOT_E4",
        "original_source_revision":hls["source_revision"],
        "same_functional_architecture_schedule_readonly_attested":True,
        "stages":["E2-Vitis-CSim","E3-Vitis-HLS-estimates",
                  "routed-Vivado-OOC-descriptive-NOT-E4"],
        "arms":paired,
        "row_minus_full":deltas,
        "both_vitis_E2_E3_qualified":True,
        "both_vivado_routed_observations_attested":True,
        "both_vivado_timing_qualified":False,
        "both_vivado_drc_clean":False,
        "strict_physical_E4":None,
        "physically_achievable_inference_latency_ns":None,
        "board_inference_latency_ns":None,
        "board_qualified":False,
        "native_vitis_vivado_calls_during_analysis":0,
        "original_evidence_modified":False,
        "limitations":[
            "Vitis latency proxy (cycles × estimated period) is not routed or measured FPGA latency",
            "Vitis resource estimates and Vivado mapped resources are stage-specific and may differ",
            "Both original routed OOC arms have negative WNS and nonclean DRC",
            "Original same-graph storage contrast is one fixed 32/6 format, schedule and target part",
            "No clean physical E4, timing-feasible compute or on-board DMA throughput",
        ],
    }


def run(*, r4p_study: str|Path, r4o_study: str|Path,
        r4k_study: str|Path, original_r4d: str|Path,
        original_r4a: str|Path) -> dict:
    kwargs=dict(r4o_study=r4o_study,r4k_study=r4k_study,
                original_r4d=original_r4d,original_r4a=original_r4a)
    hls=r4l.audit_native(study=r4k_study, original_r4d=original_r4d,
                         original_r4a=original_r4a)
    routed=r4r.summarize(r4q.inspect(study=r4p_study, **kwargs))
    return compare(hls,routed)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r4p-study",required=True)
    p.add_argument("--r4o-study",required=True)
    p.add_argument("--r4k-study",required=True)
    p.add_argument("--original-r4d",required=True)
    p.add_argument("--original-r4a",required=True)
    args=p.parse_args(argv)
    print(json.dumps(run(**vars(args)),indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
