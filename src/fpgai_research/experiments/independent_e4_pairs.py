"""Read-only matched E4 pair observations from independently qualified cases.

Only a same-model, fixed numerical/target/tool/mode context and two
source-attested, physically qualified OOC implementations permit comparison.
A delta in postroute WNS is not an inference-latency measurement.
"""
from __future__ import annotations

from decimal import Decimal
from math import isfinite

SCHEMA="fpgai-research/qualified-independent-e4-pairs/v1"
PAIR_SPECS=(
    ("mnist_cnn","case_0000","case_0006","cnn_baseline_simd"),
    ("mnist_cnn","case_0004","case_0005","cnn_parallelism_pes"),
    ("mnist_mlp","case_0004","case_0005","mlp_parallelism_pes"),
    ("mnist_mlp","case_0000","case_0006","mlp_baseline_simd"),
)
_RESOURCES=("lut","ff","dsp","bram","uram")


def _number(value):
    return type(value) in (int,float) and isfinite(value)


def compare_qualified_e4_pairs(cases, *, contexts=None):
    """Derive exact physical deltas; incomplete or unmatched pairs stay null.

    contexts is a PRIVATE tuple fingerprint by (model,case): functional
    model, part, clock, numeric contract, Vitis/Vivado identity and OOC mode.
    No context strings, tool banners or hashes are ever exported.
    """
    if not isinstance(cases,list) or len(cases)!=8:
        raise ValueError("Require full balanced eight-case E4 audit")
    indexed={}
    for row in cases:
        if not isinstance(row,dict):
            raise ValueError("Invalid E4 case evidence")
        key=(row.get("model"),row.get("case"))
        if key in indexed:
            raise ValueError("Duplicated E4 case identity")
        indexed[key]=row
    expected={(model,case) for model in ("mnist_cnn","mnist_mlp")
              for case in ("case_0000","case_0004","case_0005","case_0006")}
    if set(indexed)!=expected:
        raise ValueError("Incomplete or foreign E4 case inventory")
    if contexts is not None and not isinstance(contexts,dict):
        raise ValueError("Private physical context must be a mapping")
    report=[]
    for model,a,b,ident in PAIR_SPECS:
        first=indexed[(model,a)]
        second=indexed[(model,b)]
        qualified=(first.get("qualified_e4") is True
                   and second.get("qualified_e4") is True
                   and first.get("qualification_status")=="E4_QUALIFIED_MATCHED_OOC"
                   and second.get("qualification_status")=="E4_QUALIFIED_MATCHED_OOC")
        ctxa=contexts.get((model,a)) if contexts is not None else None
        ctxb=contexts.get((model,b)) if contexts is not None else None
        comparable=qualified and isinstance(ctxa,tuple) and isinstance(ctxb,tuple) and (
            len(ctxa)==len(ctxb) and len(ctxa)>=8 and all(v is not None for v in ctxa+ctxb)
            and ctxa==ctxb)
        status=("AWAITING_TWO_QUALIFIED_E4_CASES" if not qualified else
                "MATCHED_CONTEXT_UNVERIFIED_OR_DIFFERENT" if not comparable else
                "QUALIFIED_MATCHED_PHYSICAL_COMPARISON")
        wa=wb=delta_wns=None
        resource_a=resource_b=resource_delta=None
        if comparable:
            wa=first.get("postroute_wns_ns")
            wb=second.get("postroute_wns_ns")
            utila=first.get("postroute_utilization")
            utilb=second.get("postroute_utilization")
            if (not _number(wa) or not _number(wb)
                    or not isinstance(utila,dict) or not isinstance(utilb,dict)
                    or any(not _number(utila.get(k)) or not _number(utilb.get(k))
                           for k in _RESOURCES)
                    or wa<0 or wb<0 or
                    any(utila[k]<0 or utilb[k]<0 for k in _RESOURCES)):
                raise ValueError("Qualified E4 case has inconsistent physical numerics")
            resource_a={k:utila[k] for k in _RESOURCES}
            resource_b={k:utilb[k] for k in _RESOURCES}
            resource_delta={k:utilb[k]-utila[k] for k in _RESOURCES}
            # Vivado reports WNS as decimal nanoseconds. Convert parsed
            # finite values through decimal text before subtraction to avoid
            # binary artifacts such as 0.25799999999999995 in scientific
            # JSON/Markdown. This does not imply extra measurement precision.
            delta_wns=float(Decimal(str(wb))-Decimal(str(wa)))
        report.append({
            "pair_id":ident,"model":model,
            "case_a":a,"case_b":b,
            "axis_a":first["axis"],"axis_b":second["axis"],
            "both_e4_qualified":qualified,
            "matched_physical_context_verified":comparable if qualified else None,
            "status":status,
            "postroute_wns_a_ns":wa,
            "postroute_wns_b_ns":wb,
            "postroute_wns_delta_b_minus_a_ns":delta_wns,
            "postroute_utilization_a":resource_a,
            "postroute_utilization_b":resource_b,
            "postroute_utilization_delta_b_minus_a":resource_delta,
            "achieved_operator_loop_ii_verified":False,
            "application_or_board_latency_measured":False,
        })
    return {
        "schema":SCHEMA,
        "planned_within_model_pair_count":len(report),
        "qualified_matched_e4_pair_count":sum(
            row["status"]=="QUALIFIED_MATCHED_PHYSICAL_COMPARISON" for row in report),
        "blocked_context_pair_count":sum(
            row["status"]=="MATCHED_CONTEXT_UNVERIFIED_OR_DIFFERENT" for row in report),
        "pairs":report,
        "native_vendor_tools_executed":False,
        "source_or_tool_fingerprints_exported":False,
        "private_weights_exported":False,
        "limitations":(
            "Values compare independently source-attested, OOC, routed "
            "implementations with matching model/part/clock/number format and "
            "native tool versions. Postroute WNS is timing slack margin, "
            "not achieved neural inference latency. Resource deltas describe "
            "implemented OOC utilization, not power or on-device throughput. "
            "A pair is one study contrast, not a statistically repeated "
            "physical timing distribution."
        ),
    }
