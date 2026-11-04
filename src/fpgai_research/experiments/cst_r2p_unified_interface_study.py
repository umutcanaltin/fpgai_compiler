"""R2p prospective two-domain C3 interface contract study, source E0/GCC E1.

This is a source-only, create-only controlled regression of the SAME
policy-neutral facade on one static MatMul chain and one bounded
two-reader fanout. Native source compilers are reused. The saved JSON
read-only audit checks provenance/denominators/status consistency, not
independent original GCC numeric execution. No Vitis/Vivado, trained RL,
hardware QoR or arbitrary DAG claim.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
import subprocess

import numpy as np

from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.model_ir.decision_interface import DecisionAction, create_state
from fpgai_research.model_ir.fanout_decision_interface import create_fanout_state
from fpgai_research.model_ir.unified_decision_interface import (
    unified_action_mask, unified_graph_observation, unified_source_mask,
    unified_state_hash,
)

SCHEMA = "fpgai-research/cst-R2p-two-domain-policy-neutral-E1-study/v1"
AUDIT_SCHEMA = "fpgai-research/cst-R2p-two-domain-policy-neutral-read-only-audit/v1"
DOMAINS = ("sequential", "multi_reader_fanout")
FINAL_STATUSES = (
    "E1_GCC_QUALIFIED", "REJECTED", "UNKNOWN_E0",
    "UNKNOWN_E1_NOT_RUN", "FAILED_E1",
)


def _sha64(value) -> bool:
    return isinstance(value,str) and re.fullmatch("[a-f0-9]{64}",value) is not None


def _cases():
    chain = create_state(make_chain(3, (3,5,4,2)), exchange_policy="row_chain")
    fanout = create_fanout_state(sample_fanout(
        2, m=2, k=2, n=3, q=4), layout="disjoint")
    return (
        (DOMAINS[0], chain,
         (
             DecisionAction("operator","mm0","pes",2),
             DecisionAction("operator","mm0","pes",0),
             DecisionAction("exchange","model","exchange_policy","whole_tensor"),
         ),
         np.linspace(-.125,.125,9,dtype=np.float32).reshape(3,3)),
        (DOMAINS[1], fanout,
         (
             DecisionAction("operator","reader0","pes",2),
             DecisionAction("operator","producer","pes",2),
             DecisionAction("memory","model","layout","per_element_optimal"),
         ),
         np.linspace(-.1,.1,4,dtype=np.float32).reshape(2,2)),
    )


def _source_sha() -> str:
    repo = Path(__file__).resolve().parents[3]
    tracked = subprocess.run(
        ["git","-C",str(repo),"status","--porcelain","--untracked-files=no"],
        capture_output=True,text=True,check=True,timeout=10).stdout
    if tracked.strip():
        raise RuntimeError("Original R2p requires clean tracked source revision")
    revision = subprocess.run(
        ["git","-C",str(repo),"rev-parse","HEAD"],
        capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    if re.fullmatch("[0-9a-f]{40}",revision) is None:
        raise ValueError("Original R2p requires exact full source Git SHA")
    return revision


def run(*, output: str | Path) -> dict:
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Original R2p GCC study strictly create-only")
    revision = _source_sha()
    target.mkdir(parents=True)
    cases = []
    for domain, state, actions, input_values in _cases():
        original_hash = unified_state_hash(state)
        observation = unified_graph_observation(state)
        source_e0 = unified_action_mask(state, actions)
        try:
            host_e1 = unified_source_mask(
                state, actions, runtime_input=input_values,
                max_e1_executions=len(actions))
        except Exception as err:
            (target / "FAILED.json").write_text(
                json.dumps({"source_revision":revision,"domain":domain,
                            "status":"FAILED_ORIGINAL_R2p",
                            "type":type(err).__name__,
                            "reason":str(err)[:800]},indent=2,sort_keys=True)+"\n")
            raise
        if (source_e0["domain"] != domain or host_e1["domain"] != domain
                or source_e0["parent_state_sha256"] != original_hash
                or host_e1["parent_state_sha256"] != original_hash
                or source_e0["e0_certified"] != host_e1["e0_certified"]
                or source_e0["candidate_count"] != len(actions)
                or host_e1["candidate_count"] != len(actions)
                or any(a["cpp_sha256"] != b["cpp_sha256"]
                       for a,b in zip(source_e0["rows"],host_e1["rows"]))
                or unified_state_hash(state) != original_hash):
            raise ValueError("R2p E0 and actual GCC E1 original action identities diverge")
        case = {
            "domain":domain,
            "source_revision":revision,
            "original_source_state_sha256": original_hash,
            "original_fixed_input_sha256":sha256(input_values.tobytes()).hexdigest(),
            "original_graph_operator_count":len(state.model.operators),
            "policy_observation_sha256":sha256(json.dumps(
                observation,sort_keys=True,separators=(",",":"),
                allow_nan=False).encode()).hexdigest(),
            "policy_observation":observation,
            "e0_mask":source_e0,
            "e1_mask":host_e1,
        }
        cases.append(case)
        with (target / (domain+".json")).open("x",encoding="utf-8") as out:
            json.dump(case,out,indent=2,sort_keys=True)
            out.write("\n")
    population = {}
    for c in cases:
        for status in c["e1_mask"]["statuses"]:
            population[status] = population.get(status,0)+1
    summary = {
        "schema": SCHEMA,
        "original_source_revision": revision,
        "original_graph_domains":list(DOMAINS),
        "original_graph_count":len(cases),
        "original_action_count":sum(c["e1_mask"]["candidate_count"] for c in cases),
        "status_denominators":population,
        "original_gcc_e1_attempts":sum(c["e1_mask"]["gcc_e1_attempts"] for c in cases),
        "original_gcc_e1_qualified":population.get("E1_GCC_QUALIFIED",0),
        "original_vitis_calls":0, "original_vivado_calls":0,
        "rl_training_calls":0,
        "evidence":"same bounded C3 facade, existing E0 source compilers and executed GCC E1",
        "limitations":[
            "One small sequential chain and one exact two-reader fanout only",
            "Source/GCC host-float finite input, no general-DAG proof",
            "No native fixed-point Vitis, physical FPGA resources or trained RL",
            "Cross-domain normalization is not equivalence of compiler semantics",
        ],
    }
    with (target / "summary.json").open("x",encoding="utf-8") as out:
        json.dump(summary,out,indent=2,sort_keys=True)
        out.write("\n")
    return summary


def audit(study: str | Path) -> dict:
    """Read-only saved JSON provenance and population re-audit; no GCC replay."""
    root = Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R2p original study missing or symlinked")
    files = [root/"summary.json",*(root/(domain+".json") for domain in DOMAINS)]
    if any(f.is_symlink() or not f.is_file() for f in files):
        raise ValueError("R2p original source receipts missing or symlinked")
    summary = json.loads(files[0].read_text())
    if (summary.get("schema") != SCHEMA
            or summary.get("original_graph_domains") != list(DOMAINS)
            or summary.get("original_graph_count") != len(DOMAINS)
            or not isinstance(summary.get("original_source_revision"),str)
            or re.fullmatch("[a-f0-9]{40}",summary["original_source_revision"]) is None
            or summary.get("original_vitis_calls") != 0
            or summary.get("original_vivado_calls") != 0
            or summary.get("rl_training_calls") != 0):
        raise ValueError("R2p original cohort/source revision incorrect")
    cases = [json.loads(f.read_text()) for f in files[1:]]
    original_inputs = [
        np.linspace(-.125,.125,9,dtype=np.float32),
        np.linspace(-.1,.1,4,dtype=np.float32),
    ]
    populations = {}
    attempted = 0
    for (domain, row, inp) in zip(DOMAINS,cases,original_inputs):
        e0 = row.get("e0_mask")
        e1 = row.get("e1_mask")
        observation = row.get("policy_observation")
        if (row.get("domain") != domain
                or row.get("source_revision") != summary["original_source_revision"]
                or row.get("original_fixed_input_sha256")
                     != sha256(inp.tobytes()).hexdigest()
                or not _sha64(row.get("original_source_state_sha256"))
                or row.get("original_graph_operator_count") != (
                    3 if domain==DOMAINS[0] else 5)
                or not isinstance(e0,dict) or not isinstance(e1,dict)
                or not isinstance(observation,dict)
                or observation.get("domain") != domain
                or observation.get("schema")
                    != "fpgai-research/cst-R2p-unified-weight-private-graph/v1"
                or sha256(json.dumps(observation,sort_keys=True,
                    separators=(",",":"),allow_nan=False).encode()).hexdigest()
                    != row.get("policy_observation_sha256")
                or e0.get("domain") != domain or e1.get("domain") != domain
                or e0.get("parent_state_sha256") != row["original_source_state_sha256"]
                or e1.get("parent_state_sha256") != row["original_source_state_sha256"]
                or e0.get("candidate_count") != 3 or e1.get("candidate_count") != 3
                or e0.get("native_vitis_calls") != 0
                or e1.get("native_vitis_calls") != 0
                or e1.get("original_vivado_calls") != 0
                or e1.get("rl_policies_trained") != 0
                or e1.get("physical_feasibility_checked") is not False):
            raise ValueError("R2p original E0/E1 or graph identity invalid")
        if (not isinstance(e0.get("rows"),list)
                or not isinstance(e1.get("rows"),list)
                or len(e0["rows"]) != 3 or len(e1["rows"]) != 3
                or len(e0.get("e0_certified",[])) != 3
                or len(e1.get("e0_certified",[])) != 3
                or list(e0["e0_certified"]) != list(e1["e0_certified"])):
            raise ValueError("R2p original complete action denominator missing")
        actions = (
            (("operator","mm0","pes",2),
             ("operator","mm0","pes",0),
             ("exchange","model","exchange_policy","whole_tensor"))
            if domain==DOMAINS[0] else
            (("operator","reader0","pes",2),
             ("operator","producer","pes",2),
             ("memory","model","layout","per_element_optimal"))
        )
        for i,(initial,measured,expected) in enumerate(
                zip(e0["rows"],e1["rows"],actions)):
            expected_action = dict(zip(("scope","target","field","value"),expected))
            if (initial.get("action") != expected_action
                    or measured.get("action") != expected_action
                    or measured.get("action_index") != i
                    or initial.get("cpp_sha256") != measured.get("cpp_sha256")
                    or measured.get("status") not in FINAL_STATUSES):
                raise ValueError("R2p action treatment or source CPP hash changed")
            status = measured["status"]
            if (measured.get("e1_gcc_qualified") is not
                    (status=="E1_GCC_QUALIFIED")
                    or (status=="E1_GCC_QUALIFIED" and
                        (not measured.get("e0_certified") or
                         not measured.get("source_admissible")
                         or not isinstance(measured.get("gcc_receipt"),dict)))
                    or (status=="REJECTED" and
                        (measured.get("source_admissible")
                         or measured.get("e0_certified")))
                    or (status!="E1_GCC_QUALIFIED"
                        and measured.get("gcc_receipt") is not None)):
                raise ValueError("R2p unqualified action falsely claimed GCC E1")
            if status == "E1_GCC_QUALIFIED":
                receipt=measured["gcc_receipt"]
                if (receipt.get("original_input_sha256")
                        != row["original_fixed_input_sha256"]):
                    raise ValueError("R2p original GCC receipt input SHA mismatch")
                if domain==DOMAINS[0]:
                    if (not _sha64(receipt.get("source_qualification_sha256"))
                            or not _sha64(receipt.get("source_trace_sha256"))):
                        raise ValueError("R2p sequential GCC numeric/trace receipt missing")
                    for field in ("host_max_absolute_error","host_rmse"):
                        error=receipt.get(field)
                        if (type(error) not in (int,float)
                                or not isfinite(error) or error<0):
                            raise ValueError("R2p invalid original GCC numeric error")
                elif (not all(_sha64(receipt.get(key)) for key in (
                        "source_cpp_sha256",
                        "original_stdout_sha256",
                        "original_runtime_source_event_sha256"))
                      or receipt.get("source_cpp_sha256")
                          != measured["cpp_sha256"]
                      or type(receipt.get("original_actual_address_witness_count"))
                          is not int
                      or receipt["original_actual_address_witness_count"] < 1):
                    raise ValueError("R2p fanout original GCC source/trace receipt missing")
            populations[status] = populations.get(status,0)+1
        attempted += e1.get("gcc_e1_attempts",0)
        if e1.get("statuses") != [r["status"] for r in e1["rows"]]:
            raise ValueError("R2p original per-action status summary disagrees")
    if (summary.get("original_action_count") != 6
            or summary.get("status_denominators") != populations
            or summary.get("original_gcc_e1_attempts") != attempted
            or summary.get("original_gcc_e1_qualified")
                != populations.get("E1_GCC_QUALIFIED",0)):
        raise ValueError("R2p source experiment summary population forged")
    return {
        "schema": AUDIT_SCHEMA,
        "original_source_revision": summary["original_source_revision"],
        "observed_graph_domains":list(DOMAINS),
        "observed_original_actions":6,
        "outcome_counts":populations,
        "gcc_e1_attempts":attempted,
        "original_vendor_calls_during_audit":0,
        "rl_training_calls":0,
        "evidence":"read-only JSON input/graph/source status and population audit",
        "limitations":[
            "Not a second GCC source execution or independent arithmetic oracle",
            "Not general graph legalization, native RTL or trained RL evidence",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output")
    p.add_argument("--study")
    p.add_argument("--audit-output")
    args=p.parse_args(argv)
    if bool(args.output)==bool(args.study):
        p.error("Specify exactly one of --output original study or --study audit")
    if args.study:
        result=audit(args.study)
        if args.audit_output:
            dest=Path(args.audit_output)
            if dest.exists() or dest.is_symlink():
                raise FileExistsError("R2p read-only audit output is create-only")
            with dest.open("x",encoding="utf-8") as out:
                json.dump(result,out,indent=2,sort_keys=True)
                out.write("\n")
    else:
        if args.audit_output:
            p.error("--audit-output is only for --study")
        result=run(output=args.output)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
