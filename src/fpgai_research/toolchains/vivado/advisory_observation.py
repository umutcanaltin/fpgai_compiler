"""Conservative OOC advisory-warning classification for local model studies.

No DRC waivers, Tcl changes, severity adjustments, netlist edits or board
claims. The original source/RTL-attested E4 evidence and report parser own
qualification; this only labels an optional, explicitly disclosed observation.
"""
from __future__ import annotations

# Exact narrow rules documented as advisory warnings in the local 2023.2 pilot.
# Do not expand this list to accept unknown/future DRC results automatically.
ALLOWED_ADVISORY_RULES = frozenset({
    "DPIP-2",   # DSP input pipelining recommendation
    "DPOP-3",   # DSP PREG output pipelining recommendation
    "DPOP-4",   # DSP MREG output pipelining recommendation
    "RTSTAT-10",  # net with no routable loads; not a route error by itself
})


def classify_ooc_advisory_observation(evidence: dict, vendor_receipt: dict) -> dict:
    """Decide whether E4 has *only* the explicitly recognized advisory DRCs.

    This is a run-status convenience, not strict physical signoff. It can
    qualify only after normal native receipt, original RTL, report, HLS and
    numerical source evidence have already been accepted by summarize_case.
    """
    observation = {
        "schema": "fpgai-research/ooc-advisory-observation/v1",
        "eligible": False,
        "strict_drc_clean": False,
        "board_deployment_qualified": False,
        "allowed_rules": sorted(ALLOWED_ADVISORY_RULES),
        "observed_rules": {},
        "reason": "NOT_E4_SOURCE_ATTESTED",
    }
    if not isinstance(evidence, dict) or not isinstance(vendor_receipt, dict):
        return observation
    if evidence.get("evidence") != "E4-Vivado-implementation":
        return observation
    if (vendor_receipt.get("completed_successfully") is not True
            or vendor_receipt.get("returncode") != 0
            or evidence.get("csim_passed") is not True):
        observation["reason"] = "VENDOR_OR_CSIM_NOT_QUALIFIED"
        return observation
    physical = evidence.get("vivado_metrics")
    if not isinstance(physical, dict) or physical.get("routed_nets_complete") is not True:
        observation["reason"] = "ROUTING_NOT_ATTESTED_COMPLETE"
        return observation
    timing = physical.get("timing") or {}
    if timing.get("timing_met") is not True:
        observation["reason"] = "TIMING_NOT_MET"
        return observation
    drc = physical.get("drc")
    if not isinstance(drc, dict) or drc.get("summary_verified") is not True:
        observation["reason"] = "DRC_SUMMARY_NOT_VERIFIED"
        return observation
    if drc.get("clean") is True:
        observation["reason"] = "ALREADY_STRICT_DRC_CLEAN"
        return observation
    if drc.get("unconstrained_io") is not False:
        observation["reason"] = "IO_STATUS_NOT_VERIFIED_SAFE"
        return observation
    if drc.get("critical_warnings") != 0:
        observation["reason"] = "CRITICAL_DRC_PRESENT"
        return observation
    rules = drc.get("rules")
    total = drc.get("violations")
    if not isinstance(rules, dict) or not rules or type(total) is not int or total < 1:
        observation["reason"] = "MISSING_DRC_RULE_COUNTS"
        return observation
    if not set(rules).issubset(ALLOWED_ADVISORY_RULES):
        observation["reason"] = "UNRECOGNIZED_DRC_RULE"
        observation["observed_rules"] = rules
        return observation
    counts = 0
    for row in rules.values():
        if (not isinstance(row, dict)
                or row.get("severity") != "Warning"
                or type(row.get("violations")) is not int
                or row["violations"] < 1):
            observation["reason"] = "INVALID_DRC_SEVERITY_OR_COUNT"
            observation["observed_rules"] = rules
            return observation
        counts += row["violations"]
    if counts != total:
        observation["reason"] = "DRC_COUNTS_DO_NOT_MATCH"
        observation["observed_rules"] = rules
        return observation
    observation.update(
        eligible=True,
        observed_rules=rules,
        observed_warning_count=total,
        reason="SOURCE_ATTESTED_ROUTED_TIMING_MET_WITH_RECOGNIZED_ADVISORIES",
    )
    return observation
