"""Original CPU-only R3/R4: GNN-ready graph and policy-independent CST decisions.

This is an actual compiler/IR experiment, not RL training. The same
complete candidate state/action is fed through the SAME compiler
transition for three proposers and receives identical source-bound
certificates. Equivalent graph renamings retain anonymous observation
features and mask outcomes; full source identity hashes are expected
to change. SOURCE-admissible but uncertified UNKNOWN actions are not
legal under the conservative certificate-backed action mask.

The fixed small graph suite proves no unseen-network transfer,
no RL sample efficiency and no native HLS/physical feasibility.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess

from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.contracts import FunctionalModel
from fpgai_research.model_ir.decision_interface import (
    Accepted, DecisionAction, action_mask, certified_action_mask,
    create_state, graph_observation, state_hash, transition,
)

SCHEMA="fpgai-research/cst-rl-compatible-ir-r3-r4/v1"
CATALOG=((2,3,(3,5,2)),(3,3,(3,5,4,2)),(4,3,(3,5,4,3,2)))
PROPOSERS=("human_supplied","random_policy_supplied","heuristic_supplied")


def _anonymous_rename(model):
    names={t.name:"renamed_"+t.name for t in model.tensors}
    return FunctionalModel(
        tensors=tuple(replace(t,name=names[t.name]) for t in model.tensors),
        operators=tuple(replace(op,name="r_"+op.name,
            inputs=tuple(names[n] for n in op.inputs),
            outputs=tuple(names[n] for n in op.outputs))
            for op in model.operators),
        inputs=tuple(names[n] for n in model.inputs),
        outputs=tuple(names[n] for n in model.outputs),
    )


def _visible_features(observation):
    if observation["observation_kind"]!="public-static-no-weights":
        raise ValueError("A GNN-ready study must exclude raw trained weights")
    if any("data" in node for node in observation["nodes"]):
        raise ValueError("Raw tensor payload leaked into graph observation")
    return {
        "nodes":observation["nodes"],
        "edges":observation["edges"],
        "exchange_policy":observation["exchange_policy"],
        "row_block":observation["row_block"],
        "target_limits":observation["target_limits"],
        "boundary_evidence_status":observation["boundary_evidence_status"],
    }


def run(*,output:str|Path)->dict:
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Original RL-compatible interface study must be create-only")
    rows=[]
    for depth,rows_count,widths in CATALOG:
        model=make_chain(rows_count,widths)
        state=create_state(model,exchange_policy="row_chain")
        renamed=create_state(_anonymous_rename(model),
                             exchange_policy="row_chain")
        first=model.operators[0].name
        renamed_first=renamed.model.operators[0].name
        if state_hash(state)==state_hash(renamed):
            raise AssertionError("Renaming changed canonical source identifiers unexpectedly")
        obs=graph_observation(state)
        renamed_obs=graph_observation(renamed)
        if _visible_features(obs)!=_visible_features(renamed_obs):
            raise AssertionError("Anonymous GNN graph representation changed under renaming")
        if sum(n["type"]=="operator" for n in obs["nodes"])!=depth:
            raise AssertionError("Graph depth missing from variable-size observation")
        actions=(
            DecisionAction("operator",first,"pes",2),
            DecisionAction("operator",first,"pes",0),
            DecisionAction("exchange","model","exchange_policy","whole_tensor"),
        )
        renamed_actions=(
            DecisionAction("operator",renamed_first,"pes",2),
            DecisionAction("operator",renamed_first,"pes",0),
            actions[2],
        )
        mask=certified_action_mask(state,actions)
        rmask=certified_action_mask(renamed,renamed_actions)
        if (mask["status"]!=rmask["status"] or
                mask["status"]!=("BOUNDED_MAPPED_VALUE","REJECTED","UNKNOWN")
                or mask["legal"]!=(True,False,False) or
                mask["native_hls_feasibility_checked"]):
            raise AssertionError("Certificate-backed legality mask not sound for known controls")
        if action_mask(state,actions)["legal"]!=(True,False,True):
            raise AssertionError("Source and certified legality must remain distinguishable")
        policy_results=[transition(state,actions[0]) for _ in PROPOSERS]
        if (not all(isinstance(x,Accepted) for x in policy_results) or
                len({json.dumps(x.certificate,sort_keys=True)
                     for x in policy_results})!=1):
            raise AssertionError("Identical policy proposals must receive identical certificates")
        first_result=policy_results[0]
        if (state_hash(state)==state_hash(first_result.state) or
                first_result.certificate["parent_state_sha256"]!=state_hash(state)
                or first_result.certificate["native_vitis_calls"]!=0):
            raise AssertionError("Accepted action lacks immutable source-bound transition")
        rows.append({
            "graph_depth":depth,
            "operator_count":depth,
            "tensor_nodes":sum(n["type"]=="tensor" for n in obs["nodes"]),
            "graph_edges":len(obs["edges"]),
            "renamed_graph_feature_equivalent":True,
            "renamed_legality_outcomes_equivalent":True,
            "accepted_certificate_equal_for_all_proposers":True,
            "accepted_source_sha256":first_result.certificate["cpp_sha256"],
            "certified_legal_actions":sum(mask["legal"]),
            "certified_rejected_actions":sum(x=="REJECTED" for x in mask["status"]),
            "certified_unknown_actions":sum(x=="UNKNOWN" for x in mask["status"]),
            "source_legal_but_certificate_unknown_actions":sum(
                a and status=="UNKNOWN"
                for a,status in zip(mask["source_admissible"],mask["status"])),
            "native_vitis_calls":0,
            "trained_rl_policies":0,
        })
    revision=subprocess.run(
        ["git","rev-parse","HEAD"],capture_output=True,text=True,timeout=5)
    summary={
        "schema":SCHEMA,
        "source_revision":revision.stdout.strip() if revision.returncode==0 else None,
        "graphs":len(rows),"variable_operator_depths":[r["graph_depth"] for r in rows],
        "renaming_invariance_cases":len(rows),
        "identical_proposer_certificate_cases":len(rows),
        "policy_types":list(PROPOSERS),
        "certified_action_proposals":3*len(rows),
        "certified_legal_actions":sum(r["certified_legal_actions"] for r in rows),
        "source_admissible_but_unknown_actions":sum(
            r["source_legal_but_certificate_unknown_actions"] for r in rows),
        "native_vitis_calls":0,
        "rl_policies_trained":0,
        "rl_sample_efficiency_measured":False,
        "external_peer_parity_established":False,
        "evidence":"E0-certificate-backed-complete-design-action-protocol-only",
        "limitations":[
            "Three small static MatMul chain depths; not general NCHW fanout IR",
            "Only bounded owner/value E0 certificates; no per-cycle or RTL proof",
            "Identical proposer labels intentionally replay the same action, not trained policies",
            "No HLS resources, actual RL reward, convergence or unseen-network generalization",
            "Graph renaming tests do not prove general graph-isomorphism equivariance",
        ],
    }
    dest.mkdir(parents=True)
    with (dest/"cases.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (dest/"summary.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    print(json.dumps(run(output=args.output),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
