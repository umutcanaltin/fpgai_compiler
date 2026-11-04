"""Evidence-conditioned Pareto selection from real routed, numerically accepted results."""
from __future__ import annotations
from math import isfinite
from typing import Iterable, Mapping

SUPPORTED_OBJECTIVES=("hls_latency_worst","postroute_lut","postroute_ff","postroute_dsp","postroute_bram","postroute_uram")
DEFAULT_OBJECTIVES=("hls_latency_worst","postroute_lut","postroute_dsp")

def _metric(value, name):
    if value is None or isinstance(value,bool) or isinstance(value,str) and not value.strip():
        raise ValueError("Invalid or absent "+name)
    try:
        number=float(value)
    except (TypeError,ValueError) as exc:
        raise ValueError("Invalid "+name) from exc
    if not isfinite(number) or number<0:
        raise ValueError("Nonfinite/negative "+name)
    return number

def analyze_pareto(rows: Iterable[Mapping[str,object]],
                   objectives:tuple[str,...]=DEFAULT_OBJECTIVES) -> dict:
    """Minimize declared costs; never equate equal costs with equivalent architectures."""
    if not objectives or len(set(objectives))!=len(objectives) or set(objectives)-set(SUPPORTED_OBJECTIVES):
        raise ValueError("Objectives must be distinct supported measurements")
    entries=list(rows)
    seen=set()
    eligible=[]
    excluded=[]
    for row in entries:
        case=row.get("case")
        if not isinstance(case,str) or not case or case in seen:
            raise ValueError("Missing or duplicate Pareto case identity")
        seen.add(case)
        reason=None
        if row.get("status")!="READY":
            reason="blocked-or-unprepared-case"
        elif row.get("evidence")!="E4-Vivado-implementation":
            reason="no-attested-routed-evidence"
        elif row.get("numerical_pass") is not True:
            reason="numerical-validation-not-passed"
        elif row.get("postroute_timing_met") is not True:
            reason="timing-closure-not-established"
        else:
            try:
                costs=tuple(_metric(row.get(key),key) for key in objectives)
                wns=row.get("postroute_wns_ns")
                if wns is None or isinstance(wns,bool) or not isfinite(float(wns)) or float(wns)<0:
                    reason="nonnegative-finite-wns-required"
            except (ValueError,TypeError):
                reason="invalid-or-missing-cost-or-wns"
        if reason:
            excluded.append({"case":case,"reason":reason})
        else:
            eligible.append({"case":case,"values":costs,"costs":dict(zip(objectives,costs))})
    frontier=[]
    dominated=[]
    for point in eligible:
        dominators=[other["case"] for other in eligible if other["case"]!=point["case"]
                    and all(x<=y for x,y in zip(other["values"],point["values"]))
                    and any(x<y for x,y in zip(other["values"],point["values"]))]
        if dominators:
            dominated.append({"case":point["case"],"dominated_by":sorted(dominators),"costs":point["costs"]})
        else:
            frontier.append({"case":point["case"],"costs":point["costs"]})
    return {"schema":"fpgai-research/evidence-pareto/v1",
            "objectives_minimized":list(objectives),"total_cases":len(entries),
            "eligible_cases":len(eligible),"excluded":excluded,"frontier":frontier,
            "dominated":dominated,
            "evidence_required":"E4-Vivado-implementation with numerical pass and nonnegative WNS",
            "note":"Only compare matched model/input/target/clock; equal cost does not imply architecture equivalence."}
