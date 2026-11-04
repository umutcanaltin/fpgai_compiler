"""Explicit, bounded vendor execution across prepared three-stage Model IR suites.

Dry-run is the default. Execute only on a licensed workstation, and treat
the outcome as tool invocation status, NOT E2/E3/E4 numerical/physical proof.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import re
from fpgai_research.experiments.full_suite import audit_suite
from fpgai_research.experiments.execution import run_campaign

def run_full_suite(directory, *, stages=("vitis",),execute=False,first=0,limit=8,
                   continue_on_error=False,require_csim_pass=False,
                   vitis_executable="vitis_hls",
                   vivado_executable="vivado"):
    if not stages or any(s not in ("vitis","vivado") for s in stages) or len(set(stages))!=len(stages) or stages==("vivado","vitis"):
        raise ValueError("Select unique ordered Vitis/Vivado stages")
    if any(type(x) is not bool for x in (execute,continue_on_error,require_csim_pass)):
        raise ValueError("Execution flags must be booleans")
    if require_csim_pass and "vivado" not in stages:
        raise ValueError("require_csim_pass requires Vivado stage")
    if type(first) is not int or type(limit) is not int or first<0 or not 1<=limit<=256:
        raise ValueError("Select a bounded nonnegative case index and count")
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Suite must be a regular directory")
    attested=audit_suite(root,require_host=False,require_all_axes=False)
    if (not attested.get("all_prepared_models_have_legal_executable_subsets",
                         attested["all_models_source_ready"])
            or attested.get("blocked_models", 0)):
        raise ValueError("Every model needs an audited legal-case subset before native execution")
    manifest=root/"suite_results.json"
    members=json.loads(manifest.read_text())["studies"]
    records=[]
    for item in members:
        name=item["id"]
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,40}",name):
            raise ValueError("Unsafe suite experiment identity")
        outcome=run_campaign(root/name,stages=tuple(stages),execute=execute,
            first=first,limit=limit,continue_on_error=continue_on_error,
            require_csim_pass=require_csim_pass,
            vitis_executable=vitis_executable,vivado_executable=vivado_executable)
        records.append({"id":name,"execution":outcome})
        if execute and not continue_on_error and any(
            status.get("status") in ("FAILED","BLOCKED")
            for case in outcome["cases"] for status in case["stages"].values()):
            break
    result={"schema":"fpgai-research/full-paper-suite-execution/v1",
        "suite_sha256":sha256(manifest.read_bytes()).hexdigest(),
        "stages":list(stages),"tools_executed":execute,
        "first_ready_index":first,"limit_per_model":limit,
        "require_csim_pass":require_csim_pass,"models":records,
        "note":"Completed vendor command is not E2/E3/E4; audit fresh source/RTL-attested reports separately"}
    if execute:
        from fpgai_research.experiments.execution_history import record_execution
        record_execution(root,"suite",result)
        (root/"last_suite_execution.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result
