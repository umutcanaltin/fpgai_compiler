"""Campaign evidence audit; missing physical results remain missing."""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import re
from fpgai_research.experiments.results import summarize_case
from fpgai_research.experiments.suite_native_batch import _qualify as _qualify_e4

AXES=("precision_bits","parallelism","pes","simd","serialization","tile","pipeline_ii")

def audit_campaign(directory, required_evidence="E1"):
    if required_evidence not in ("E1","E2","E3","E4"):
        raise ValueError("Evidence requirement must be E1, E2, E3 or E4")
    root=Path(directory)
    index_file=root/"campaign.json"
    campaign=json.loads(index_file.read_text())
    if campaign.get("schema")!="fpgai-research/paper-campaign/v1":
        raise ValueError("Invalid paper campaign schema")
    rows=campaign.get("cases")
    if not isinstance(rows,list) or not rows:
        raise ValueError("Campaign must have at least one case")
    ready=[]
    blocked=[]
    seen=set()
    levels={"E1":1,"E2":2,"E3":3,"E4":4}
    for row in rows:
        name=row.get("case","")
        if not isinstance(name,str) or not re.fullmatch(r"case_[0-9]{4,8}",name) or name in seen:
            raise ValueError(f"Unsafe/duplicate campaign case identity: {name!r}")
        seen.add(name)
        if row.get("status")=="BLOCKED":
            if not row.get("reason"):
                raise ValueError(f"Missing reason for blocked case {name}")
            blocked.append({"case":name,"reason":row["reason"]})
            continue
        if row.get("status")!="READY":
            raise ValueError(f"Unknown case status for {name}")
        report=summarize_case(str(root/name))
        if row.get("cpp_sha256")!=report["source_sha256"]:
            raise ValueError(f"Campaign index source hash mismatch: {name}")
        physical=report.get("vivado_metrics") or {}
        timing_met=(physical.get("timing") or {}).get("timing_met")
        # Reuse the canonical strict source/RTL-attested E4 qualifier:
        # passing CSim, OOC mode, complete route, WNS, utilization, and
        # an independently verified DRC summary with zero total violations.
        # Attested E4 reports with advisory DRC warnings are NOT qualified.
        e4_blockers=list(_qualify_e4(report))
        strict_e4=not e4_blockers
        requirement_met=levels.get(report["evidence"][:2],0)>=levels[required_evidence]
        if required_evidence=="E4":
            requirement_met=strict_e4
        ready.append({"case":name,"axis":row.get("dimension"),
                      "evidence":report["evidence"],
                      "requirement_met":requirement_met,
                      "strict_e4_qualified":strict_e4,
                      "e4_qualification_blockers":e4_blockers,
                      "csim_passed":report.get("csim_passed"),
                      "timing_met":timing_met,
                      "warnings":report.get("warnings",[])})
    covered={row["axis"] for row in ready}
    missing=sorted(set(AXES)-covered)
    baseline_present="baseline" in covered
    campaign_complete=bool(ready and not blocked and baseline_present and not missing
        and all(row["requirement_met"] for row in ready))
    return {
        "schema":"fpgai-research/reproduction-audit/v1",
        "campaign_sha256":sha256(index_file.read_bytes()).hexdigest(),
        "required_evidence":required_evidence,
        "total_cases":len(rows),"ready_cases":len(ready),"blocked_cases":len(blocked),
        "meeting_required_evidence":sum(row["requirement_met"] for row in ready),
        "observed_evidence_counts":dict(sorted(Counter(row["evidence"] for row in ready).items())),
        "missing_single_axis_variations":missing,
        "baseline_present":baseline_present,
        "campaign_complete_for_required_evidence":campaign_complete,
        "paper_physical_results_complete":bool(ready and not blocked and baseline_present and not missing and all(
            row["strict_e4_qualified"] for row in ready)),
        "strict_e4_qualified_cases":sum(row["strict_e4_qualified"] for row in ready),
        "cases":ready,"blockers":blocked,
        "note":"E1 code generation is not E2 numerical simulation, E3 HLS synthesis or E4 routed results; strict E4 requires complete routing, valid fixed-point CSim, OOC timing and zero verified DRC violations",
    }
