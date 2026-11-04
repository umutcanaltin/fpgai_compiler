"""Pure tests for a disclosed, narrowly scoped warning-only OOC observation.

These checks do not run Vivado, change DRC severity, or promote board results.
"""
from __future__ import annotations

from copy import deepcopy
import pytest

from fpgai_research.toolchains.vivado.advisory_observation import (
    ALLOWED_ADVISORY_RULES,
    classify_ooc_advisory_observation,
)


def _snapshot():
    drc_rules = {
        "DPIP-2": {"severity": "Warning", "violations": 2},
        "DPOP-3": {"severity": "Warning", "violations": 2},
        "DPOP-4": {"severity": "Warning", "violations": 2},
        "RTSTAT-10": {"severity": "Warning", "violations": 1},
    }
    evidence = {
        "evidence": "E4-Vivado-implementation",
        "csim_passed": True,
        "vivado_metrics": {
            "routed_nets_complete": True,
            "timing": {"timing_met": True, "wns_ns": 3.699},
            "drc": {
                "available": True, "summary_verified": True,
                "clean": False, "violations": 7, "critical_warnings": 0,
                "unconstrained_io": False, "rules": drc_rules,
            },
        },
    }
    receipt = {"completed_successfully": True, "returncode": 0}
    return evidence, receipt


def test_exact_user_observation_is_disclosed_and_not_drc_clean():
    evidence, receipt = _snapshot()
    outcome = classify_ooc_advisory_observation(evidence, receipt)
    assert outcome["eligible"] is True
    assert outcome["strict_drc_clean"] is False
    assert outcome["board_deployment_qualified"] is False
    assert outcome["observed_warning_count"] == 7
    assert set(outcome["observed_rules"]) == ALLOWED_ADVISORY_RULES
    assert outcome["reason"] == "SOURCE_ATTESTED_ROUTED_TIMING_MET_WITH_RECOGNIZED_ADVISORIES"


@pytest.mark.parametrize("mutation,reason", [
    ("not_e4", "NOT_E4_SOURCE_ATTESTED"),
    ("csim_false", "VENDOR_OR_CSIM_NOT_QUALIFIED"),
    ("vendor_failed", "VENDOR_OR_CSIM_NOT_QUALIFIED"),
    ("nonzero_exit", "VENDOR_OR_CSIM_NOT_QUALIFIED"),
    ("route_incomplete", "ROUTING_NOT_ATTESTED_COMPLETE"),
    ("timing_violated", "TIMING_NOT_MET"),
    ("drc_not_verified", "DRC_SUMMARY_NOT_VERIFIED"),
    ("drc_clean", "ALREADY_STRICT_DRC_CLEAN"),
    ("unconstrained", "IO_STATUS_NOT_VERIFIED_SAFE"),
    ("unknown_unconstrained", "IO_STATUS_NOT_VERIFIED_SAFE"),
    ("critical", "CRITICAL_DRC_PRESENT"),
    ("missing_rules", "MISSING_DRC_RULE_COUNTS"),
    ("unknown_rule", "UNRECOGNIZED_DRC_RULE"),
    ("invalid_severity", "INVALID_DRC_SEVERITY_OR_COUNT"),
    ("invalid_count", "INVALID_DRC_SEVERITY_OR_COUNT"),
    ("inconsistent_count", "DRC_COUNTS_DO_NOT_MATCH"),
])
def test_unapproved_or_unverified_observation_is_rejected(mutation, reason):
    evidence, receipt = _snapshot()
    drc = evidence["vivado_metrics"]["drc"]
    if mutation == "not_e4":
        evidence["evidence"] = "E3-HLS-synthesis-report"
    elif mutation == "csim_false":
        evidence["csim_passed"] = False
    elif mutation == "vendor_failed":
        receipt["completed_successfully"] = False
    elif mutation == "nonzero_exit":
        receipt["returncode"] = 1
    elif mutation == "route_incomplete":
        evidence["vivado_metrics"]["routed_nets_complete"] = False
    elif mutation == "timing_violated":
        evidence["vivado_metrics"]["timing"]["timing_met"] = False
    elif mutation == "drc_not_verified":
        drc["summary_verified"] = False
    elif mutation == "drc_clean":
        drc["clean"] = True
    elif mutation == "unconstrained":
        drc["unconstrained_io"] = True
    elif mutation == "unknown_unconstrained":
        drc["unconstrained_io"] = None
    elif mutation == "critical":
        drc["critical_warnings"] = 1
    elif mutation == "missing_rules":
        drc["rules"] = {}
    elif mutation == "unknown_rule":
        drc["rules"]["UNEXPECTED-42"] = {
            "severity": "Warning", "violations": 1,
        }
        drc["violations"] += 1
    elif mutation == "invalid_severity":
        drc["rules"]["DPOP-3"]["severity"] = "Critical Warning"
    elif mutation == "invalid_count":
        drc["rules"]["DPOP-4"]["violations"] = 0
    elif mutation == "inconsistent_count":
        drc["violations"] = 8
    result = classify_ooc_advisory_observation(evidence, receipt)
    assert result["eligible"] is False
    assert result["reason"] == reason
    assert result["strict_drc_clean"] is False
    assert result["board_deployment_qualified"] is False


def test_no_mutation_of_vendor_or_parser_inputs():
    evidence, receipt = _snapshot()
    e_before, r_before = deepcopy(evidence), deepcopy(receipt)
    classify_ooc_advisory_observation(evidence, receipt)
    assert evidence == e_before
    assert receipt == r_before
