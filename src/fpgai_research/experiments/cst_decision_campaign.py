"""E0/E1 controlled study of policy-neutral CST architectural decisions.

Tests a complete-design action catalog against an independent *restricted*
domain oracle. The oracle describes only the prespecified 1-op PE/SIMD and
B-row actions and the known compiler target cap; it is NOT a general proof
of arbitrary Phi-derived tensor availability, device feasibility or RL
benefit. Random and greedy policy inputs use exactly the same transition API.
No Gym, torch, trained RL, or original Vitis calls.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess

import numpy as np

from fpgai_research.architecture.resolver import TargetLimits
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.decision_interface import (
    DecisionAction, Accepted, create_state, transition, action_mask,
    graph_observation, qualify_accepted_source)
from fpgai_research.validation.host import validate_host

SCHEMA="fpgai-research/cst-search-interface-research/v1"
CATALOG=(
    (3,(3,5,4,2),2),
    (5,(3,5,4,2),4),
    (4,(4,3,7,2),2),
    (3,(3,5,4,3,2),4),
)


def actions(state):
    first=state.model.operators[0].name
    return tuple([
        *(DecisionAction("operator",first,"pes",n) for n in (0,1,2,4)),
        *(DecisionAction("operator",first,"simd",n) for n in (0,1,2,4)),
        *(DecisionAction("exchange","model","row_block",n) for n in
          (0,1,2,state.model.tensor(state.model.inputs[0]).shape[0])),
        DecisionAction("exchange","model","exchange_policy","whole_tensor"),
    ])


def restricted_oracle(action, *, max_parallel, rows):
    """Independent algebra for the prespecified all-ones initial candidates."""
    if action.scope=="operator" and action.field in ("pes","simd"):
        return type(action.value) is int and 1 < action.value <= max_parallel
    if action.scope=="exchange" and action.field=="row_block":
        return type(action.value) is int and 1 < action.value <= rows
    return (action.scope=="exchange" and action.field=="exchange_policy"
            and action.value=="whole_tensor")


def run(*, output: str | Path, graphs: int = 4,
        host_checks: int = 3, seed: int = 37) -> dict:
    if type(graphs) is not int or not 1<=graphs<=4:
        raise ValueError("graphs must be 1..4")
    if type(host_checks) is not int or not 0<=host_checks<=graphs:
        raise ValueError("host_checks must be 0..graphs")
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError("seed must be a bounded uint32")
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Create-only RL-ready IR research study")
    rng=random.Random(seed)
    cases=[]; policies=[]; qualified_hosts=[]; host_used=0
    for graph_id,(rows,widths,cap) in enumerate(CATALOG[:graphs]):
        model=make_chain(rows,widths)
        state=create_state(model,exchange_policy="row_chain",
                           limits=TargetLimits(max_parallel_units=cap))
        proposals=actions(state)
        mask=action_mask(state,proposals)
        matched=0
        for index,(action,admitted) in enumerate(zip(proposals,mask["legal"])):
            expected=restricted_oracle(action,max_parallel=cap,rows=rows)
            outcome=transition(state,action)
            if admitted != expected or admitted != isinstance(outcome,Accepted):
                raise AssertionError(
                    f"Independent restricted action oracle disagreement: "
                    f"{graph_id}, {index}, {action}, expected={expected}")
            matched+=1
            cases.append({
                "graph":graph_id,"rows":rows,"operators":len(model.operators),
                "target_parallel_cap":cap,"candidate":index,
                "action_scope":action.scope,"action_target":action.target,
                "action_field":action.field,"action_value":repr(action.value),
                "oracle_legal":expected,"mask_accepted":admitted,
                "compiler_accepted":isinstance(outcome,Accepted),
                "source_state_sha256":(
                    outcome.certificate["state_sha256"] if admitted else None),
                "generated_cpp_sha256":(
                    outcome.certificate["cpp_sha256"] if admitted else None),
                "rejection_code":(None if admitted else outcome.code),
                "native_vitis_calls":0,
            })
        valid=[a for a,ok in zip(proposals,mask["legal"]) if ok]
        if not valid:
            raise AssertionError("No legal architectural edits")
        # Two deliberately simple decision policies using exactly the same
        # transition API, with equal one-action budget. Greedy chooses the
        # smallest source scratch / requested parallel knob under this tiny
        # catalog; this is NOT an HLS QoR predictor.
        random_action=rng.choice(valid)
        greedy_action=min(valid,key=lambda a: (
            0 if a.field=="row_block" else 1,
            a.value if type(a.value) is int else 999))
        for policy,chosen in (("uniform_random_legal",random_action),
                              ("source_scratch_heuristic",greedy_action)):
            result=transition(state,chosen)
            if not isinstance(result,Accepted):
                raise AssertionError("Accepted mask proposal rejected by compiler")
            policies.append({
                "graph":graph_id,"policy":policy,"action":asdict(chosen),
                "source_state_sha256":result.certificate["state_sha256"],
                "cpp_sha256":result.certificate["cpp_sha256"],
                "legal":True,"candidate_budget":1,"hls_budget":0,
                "objective_measured":False,
            })
        if host_used<host_checks:
            chosen=transition(state,DecisionAction(
                "operator",model.operators[0].name,"pes",2))
            if not isinstance(chosen,Accepted):
                raise AssertionError("Reference host candidate unexpectedly invalid")
            x=np.linspace(-.125,.125,rows*widths[0],dtype=np.float32).reshape(
                rows,widths[0])
            verified=qualify_accepted_source(
                chosen, runtime_input=x, verify_value_trace=True)
            if (not verified["host_numerical"]["accepted"] or
                    verified["source_value_lifetime"]["status"] !=
                    "SOURCE_VALUE_LIFETIMES_VERIFIED"):
                raise AssertionError("Accepted CST action produced unqualified GCC source")
            qualified_hosts.append({
                "graph":graph_id,
                "action":asdict(DecisionAction("operator",
                    model.operators[0].name,"pes",2)),
                "state_sha256":verified["state_sha256"],
                "qualification_sha256":verified["qualification_sha256"],
                "cpp_sha256":verified["cpp_sha256"],
                "actual_value_trace_sha256":verified[
                    "source_value_lifetime"]["source_trace_sha256"],
                "max_absolute_error":verified[
                    "host_numerical"]["max_absolute_error"],
                "native_vitis_calls":0,
            })
            host_used+=1
        obs=graph_observation(state)
        if (len([n for n in obs["nodes"] if n["type"]=="operator"])
                != len(model.operators)):
            raise AssertionError("Operator observation incomplete")
    revision=subprocess.run(
        ["git","rev-parse","HEAD"],capture_output=True,text=True,timeout=5)
    summary={
        "schema":SCHEMA,"source_revision":revision.stdout.strip(),
        "graphs":graphs,"candidate_actions":len(cases),
        "independent_restricted_oracle_agreements":len(cases),
        "valid_actions":sum(r["compiler_accepted"] for r in cases),
        "rejected_actions":sum(not r["compiler_accepted"] for r in cases),
        "false_legal":0,"false_illegal":0,"host_numerical_passes":host_used,
        "actual_mapped_value_lifetime_passes":len(qualified_hosts),
        "policy_trajectories":len(policies),
        "candidate_budget_per_policy":1,
        "native_vitis_calls":0,"trained_rl_policies":0,
        "evidence":"E0-finite-complete-design-actions-plus-E1-selected-GCC",
        "scientific_limitations":[
            "Independent oracle only covers the prespecified all-ones state and two fields plus B",
            "No learned policy, transfer learning or RL sample efficiency measured",
            "Current row-chain proof uses atomic whole-row stage completions",
            "Not a Phi-derived per-element readiness, fan-out or hardware resource proof",
            "Greedy uses only source-level settings, not actual HLS latency",
        ],
    }
    target.mkdir(parents=True)
    with (target/"cases.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cases[0]))
        writer.writeheader();writer.writerows(cases)
    (target/"policies.json").write_text(
        json.dumps(policies,indent=2,sort_keys=True)+"\n")
    (target/"qualified_source_checks.json").write_text(
        json.dumps(qualified_hosts,indent=2,sort_keys=True)+"\n")
    (target/"summary.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--graphs",type=int,default=4)
    p.add_argument("--host-checks",type=int,default=3)
    p.add_argument("--seed",type=int,default=37)
    a=p.parse_args(argv)
    print(json.dumps(run(output=a.output,graphs=a.graphs,
                         host_checks=a.host_checks,seed=a.seed),indent=2))


if __name__=="__main__":
    main()
