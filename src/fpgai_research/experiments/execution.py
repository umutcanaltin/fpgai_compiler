"""Explicit, bounded, evidence-gated vendor execution for prepared campaigns.

Neither preparation nor dry-run executes AMD tools. Vivado cannot run until
source-attested E3 synthesis results exist for the selected case.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import re

from fpgai_research.experiments.results import summarize_case
from fpgai_research.toolchains.vitis_hls.runner import run_vitis
from fpgai_research.toolchains.vivado.runner import run_vivado

def run_campaign(directory, *, stages=("vitis",), execute=False, first=0,
                 limit=8, continue_on_error=False, require_csim_pass=False,
                 vitis_executable="vitis_hls", vivado_executable="vivado",
                 vitis_timeout=7200, vivado_timeout=14400):
    root=Path(directory)
    index_file=root/"campaign.json"
    campaign=json.loads(index_file.read_text())
    if campaign.get("schema")!="fpgai-research/paper-campaign/v1":
        raise ValueError("Expected prepared paper campaign v1")
    if not stages or len(set(stages))!=len(stages) or any(s not in ("vitis","vivado") for s in stages):
        raise ValueError("stages must uniquely select vitis and/or vivado")
    if stages==("vivado","vitis"):
        raise ValueError("Cannot run Vivado before Vitis")
    if type(require_csim_pass) is not bool:
        raise ValueError("require_csim_pass must be a boolean")
    if require_csim_pass and "vivado" not in stages:
        raise ValueError("require_csim_pass requires a Vivado stage")
    if not isinstance(execute,bool) or type(first)!=int or type(limit)!=int or first<0 or not 1<=limit<=256:
        raise ValueError("Run must be explicit; first>=0 and limit 1..256")
    indices=campaign.get("cases",[])
    identities=set()
    ready=[]
    for row in indices:
        name=row.get("case","")
        if not re.fullmatch(r"case_[0-9]{4,8}",name) or name in identities:
            raise ValueError("Invalid/duplicate case identity")
        identities.add(name)
        if row.get("status")!="READY":
            if row.get("status")!="BLOCKED" or not row.get("reason"):
                raise ValueError("Unexpected campaign case status or missing blocker")
            continue
        ready.append(row)
    selected=ready[first:first+limit]
    if not selected:
        raise ValueError("No READY cases in requested range")
    checked=[]
    for row in selected:
        case=root/row["case"]
        summary=summarize_case(str(case))
        if row.get("cpp_sha256")!=summary["source_sha256"]:
            raise ValueError(f'Prepared case/source mismatch: {row["case"]}')
        checked.append((row,case,summary))
    outcomes=[]
    for row,case,prior in checked:
        outcome={"case":row["case"],"stages":{},"initial_evidence":prior["evidence"]}
        for stage in stages:
            if stage=="vivado":
                evidence=summarize_case(str(case)) if execute else prior
                if evidence["evidence"] not in ("E3-HLS-synthesis-report","E4-Vivado-implementation"):
                    outcome["stages"][stage]={"status":"BLOCKED",
                        "reason":"Attested E3 HLS result required before Vivado"}
                    if execute and not continue_on_error:
                        break
                    continue
                # In opt-in numerical campaigns, only an explicitly
                # source-attributed *passing* fixed-point C simulation
                # qualifies a case for physical implementation. Attested E3
                # by itself proves synthesis, not numerical acceptance.
                if require_csim_pass and evidence.get("csim_passed") is not True:
                    outcome["stages"][stage]={"status":"BLOCKED",
                        "reason":"Passing source-attested fixed-point C simulation required before Vivado"}
                    if execute and not continue_on_error:
                        break
                    continue
            if not execute:
                outcome["stages"][stage]={"status":"PLANNED-NOT-EXECUTED"}
                continue
            try:
                if stage=="vitis":
                    receipt=run_vitis(case,executable=vitis_executable,timeout_seconds=vitis_timeout)
                else:
                    receipt=run_vivado(case,executable=vivado_executable,timeout_seconds=vivado_timeout)
                outcome["stages"][stage]={"status":"COMPLETED" if receipt["completed_successfully"] else "FAILED",
                                          "receipt_schema":receipt.get("schema"),
                                          "returncode":receipt.get("returncode")}
            except (OSError,ValueError,RuntimeError) as exc:
                outcome["stages"][stage]={"status":"FAILED","error":f"{type(exc).__name__}: {exc}"}
            if outcome["stages"][stage]["status"]!="COMPLETED" and not continue_on_error:
                break
        outcomes.append(outcome)
        if execute and not continue_on_error and any(
            status["status"]=="FAILED" for status in outcome["stages"].values()):
            break
    output={"schema":"fpgai-research/explicit-campaign-run/v1",
            "campaign_sha256":sha256(index_file.read_bytes()).hexdigest(),
            "tools_executed":execute,"selected_stages":list(stages),
            "first_ready_index":first,"requested_limit":limit,
            "require_csim_pass":require_csim_pass,"cases":outcomes,
            "note":"Tool exit does not prove E2/E3/E4; collect attested reports separately"}
    if execute:
        from fpgai_research.experiments.execution_history import record_execution
        record_execution(root,"campaign",output)
        (root/"last_campaign_execution.json").write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    return output
