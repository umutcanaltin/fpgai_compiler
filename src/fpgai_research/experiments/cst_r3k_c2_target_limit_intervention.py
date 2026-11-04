"""R3k C2: factorial target-limit intervention with identical source actions.

Twelve distinct static MatMul fanout models (three reader counts × four
new tensor shapes), compared under a strict vs relaxed *source-level*
parallel-unit bound. All eight predeclared actions in each pair are
evaluated by the SAME existing compiler source transition and compared
to a second direct replay, with matched model/action identities.

This tests whether bounded source action legality changes under an
explicit feasibility constraint, not HLS physical feasibility or trained
agent sample efficiency. Mask construction costs one compile per action.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from fpgai_research.architecture.resolver import TargetLimits
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r3d_c1c2_finite_contract_ablation import (
    _git_revision, structural_signature,
)
from fpgai_research.experiments.cst_r3j_c1_heldout_semantics import SHAPES,READER_COUNTS
from fpgai_research.model_ir.decision_interface import DecisionAction
from fpgai_research.model_ir.fanout_decision_interface import create_fanout_state
from fpgai_research.model_ir.unified_decision_interface import (
    unified_action_mask,unified_graph_observation,unified_state_hash,unified_transition,
)

SCHEMA="fpgai-research/R3k-C2-target-limit-matched-intervention/v1"
CASES=tuple((reader,*shape) for reader in READER_COUNTS for shape in SHAPES)
LIMITS=(1,4)
ACTIONS_PER_STATE=8
EXPECTED_STRICT=(
    "REJECTED", "REJECTED", "REJECTED", "REJECTED",
    "BOUNDED_CERTIFIED_E0","REJECTED","REJECTED","REJECTED",
)
EXPECTED_RELAXED=(
    "BOUNDED_CERTIFIED_E0","BOUNDED_CERTIFIED_E0",
    "REJECTED","REJECTED","BOUNDED_CERTIFIED_E0",
    "REJECTED","REJECTED","REJECTED",
)


def _actions(reader:int,q:int)->tuple[DecisionAction,...]:
    return (
        DecisionAction("operator","reader0","pes",2),
        DecisionAction("operator",f"reader{reader-1}","pes",2),
        DecisionAction("operator","reader0","pes",0),
        DecisionAction("operator","producer","pes",2),
        DecisionAction("memory","model","layout","per_element_optimal"),
        DecisionAction("memory","model","layout","disjoint"),
        DecisionAction("operator","reader_missing","pes",2),
        DecisionAction("operator","reader0","pes",q+1),
    )


def evaluate(config:tuple[int,...])->dict:
    if config not in CASES:
        raise ValueError("R3k: source model outside predeclared factorial population")
    readers,m,k,n,q=config
    model=sample_fanout(readers,m=m,k=k,n=n,q=q)
    actions=_actions(readers,q)
    if len(actions)!=ACTIONS_PER_STATE or len(set(actions))!=ACTIONS_PER_STATE:
        raise AssertionError("R3k: candidate actions not unique or complete")
    observed=[]
    for cap,expected in zip(LIMITS,(EXPECTED_STRICT,EXPECTED_RELAXED)):
        state=create_fanout_state(
            model,layout="disjoint",limits=TargetLimits(max_parallel_units=cap))
        original_hash=unified_state_hash(state)
        obs=unified_graph_observation(state)
        mask=unified_action_mask(state,actions)
        replay=tuple(unified_transition(state,a) for a in actions)
        if (tuple(mask["statuses"])!=expected
                or tuple(item.status for item in replay)!=expected
                or mask["rows"]!=[item.as_receipt() for item in replay]
                or unified_state_hash(state)!=original_hash
                or tuple(mask["e0_certified"])!=
                   tuple(s=="BOUNDED_CERTIFIED_E0" for s in expected)):
            raise AssertionError("R3k: source constraints or masked compiler verdicts disagree")
        counts=Counter(expected)
        observed.append({
            "source_parallel_limit":cap,
            "original_source_state_sha256":original_hash,
            "graph_public_features_sha256":structural_signature(obs)["digest"],
            "statuses":list(expected),
            "source_cpp_sha256":[item.cpp_sha256 for item in replay],
            "e0_certified":counts["BOUNDED_CERTIFIED_E0"],
            "rejected":counts["REJECTED"],
            "e0_unknown":counts["UNKNOWN_E0"],
            "source_mask_compiler_transitions":len(actions),
            "blind_replay_compiler_transitions":len(actions),
            "original_vendor_calls":0,
        })
    if observed[0]["original_source_state_sha256"]==observed[1]["original_source_state_sha256"]:
        raise AssertionError("R3k: changing source bound did not change source identity")
    expected_flips=[i for i in range(ACTIONS_PER_STATE)
                    if observed[0]["statuses"][i]!=observed[1]["statuses"][i]]
    if expected_flips!=[0,1]:
        raise AssertionError("R3k: only expected reader width actions should flip")
    if observed[0]["graph_public_features_sha256"]==observed[1]["graph_public_features_sha256"]:
        raise AssertionError("R3k: policy observation leaked or hid source limit")
    return {
        "source_model":[*config],
        "source_parameter_count":5,
        "candidate_population_per_limit":ACTIONS_PER_STATE,
        "source_limits":list(LIMITS),
        "identical_candidate_actions":[a.__dict__ for a in actions],
        "target_intervention_flip_indices":expected_flips,
        "target_intervention_flip_count":len(expected_flips),
        "results":observed,
        "hls_or_routed_resource_feasibility_tested":False,
        "trained_agents":0,
    }


def run()->dict:
    rows=[evaluate(config) for config in CASES]
    if len(rows)!=12:
        raise AssertionError("R3k prespecified source population incomplete")
    total=sum(len(r["results"])*ACTIONS_PER_STATE for r in rows)
    return {
        "schema":SCHEMA,
        "matched_static_source_model_pairs":len(rows),
        "source_parallel_limits":list(LIMITS),
        "actions_per_limit":ACTIONS_PER_STATE,
        "total_compiler_masked_candidates":total,
        "total_independently_replayed_candidates":total,
        "certified":[
            sum(r["results"][i]["e0_certified"] for r in rows) for i in range(2)],
        "rejected":[
            sum(r["results"][i]["rejected"] for r in rows) for i in range(2)],
        "unknown_e0":[
            sum(r["results"][i]["e0_unknown"] for r in rows) for i in range(2)],
        "target_limit_status_flips":sum(
            r["target_intervention_flip_count"] for r in rows),
        "rows":rows,
        "gcc_calls":0,"vitis_calls":0,"vivado_calls":0,"trained_agents":0,
        "scope":[
            "12 predeclared MatMul fanout model shapes, two source target limits only",
            "Same compiler/action set under source bound intervention",
            "Not an independent general resource oracle or native FPGA HLS qualification",
            "Every mask entails 8 actual source compiler transitions",
            "No CNN graph or trained RL result",
        ],
    }


def create_only(report:dict,*,output:str|Path,source_revision:str)->None:
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R3k matched source evidence strictly create-only")
    if (report.get("schema")!=SCHEMA
            or re.fullmatch(r"[0-9a-f]{40}",source_revision) is None):
        raise ValueError("R3k: source revision or schema mismatch")
    root.mkdir(parents=True,exist_ok=False)
    (root/"r3k_matched_target_limit_source_masks.json").write_text(
        json.dumps({**report,"source_revision":source_revision},
                   sort_keys=True,indent=2,allow_nan=False)+"\n",
        encoding="utf-8")


def audit(study:str|Path)->dict:
    root=Path(study)
    receipt=root/"r3k_matched_target_limit_source_masks.json"
    if root.is_symlink() or not root.is_dir() or receipt.is_symlink() or not receipt.is_file():
        raise ValueError("R3k original receipt missing/linked")
    r=json.loads(receipt.read_text())
    if (r.get("schema")!=SCHEMA
            or re.fullmatch(r"[0-9a-f]{40}",r.get("source_revision") or "") is None
            or r.get("matched_static_source_model_pairs")!=len(CASES)
            or r.get("source_parallel_limits")!=list(LIMITS)
            or r.get("actions_per_limit")!=ACTIONS_PER_STATE
            or r.get("total_compiler_masked_candidates")!=len(CASES)*2*ACTIONS_PER_STATE
            or r.get("total_independently_replayed_candidates")!=len(CASES)*2*ACTIONS_PER_STATE
            or r.get("target_limit_status_flips")!=2*len(CASES)
            or r.get("certified")!=[len(CASES),len(CASES)*3]
            or r.get("rejected")!=[len(CASES)*7,len(CASES)*5]
            or r.get("unknown_e0")!=[0,0]
            or any(r.get(field)!=0 for field in
                   ("gcc_calls","vitis_calls","vivado_calls","trained_agents"))
            or not isinstance(r.get("rows"),list) or len(r["rows"])!=len(CASES)):
        raise ValueError("R3k original matched source population corrupted")
    for expected_config,row in zip(CASES,r["rows"]):
        outcomes=row.get("results")
        if (row.get("source_model")!=list(expected_config)
                or row.get("candidate_population_per_limit")!=ACTIONS_PER_STATE
                or row.get("target_intervention_flip_indices")!=[0,1]
                or not isinstance(outcomes,list) or len(outcomes)!=2):
            raise ValueError("R3k original paired target intervention missing")
        for index,entry in enumerate(outcomes):
            statuses=(EXPECTED_STRICT,EXPECTED_RELAXED)[index]
            if (entry.get("source_parallel_limit")!=LIMITS[index]
                    or entry.get("statuses")!=list(statuses)
                    or entry.get("source_mask_compiler_transitions")!=ACTIONS_PER_STATE
                    or entry.get("blind_replay_compiler_transitions")!=ACTIONS_PER_STATE
                    or not isinstance(entry.get("source_cpp_sha256"),list)
                    or len(entry["source_cpp_sha256"])!=ACTIONS_PER_STATE
                    or any(s is not None and re.fullmatch(r"[a-f0-9]{64}",s) is None
                           for s in entry["source_cpp_sha256"])):
                raise ValueError("R3k original source action/status receipt altered")
    return {
        "schema":SCHEMA,
        "original_source_revision":r["source_revision"],
        "model_pairs":len(CASES),
        "candidate_population":len(CASES)*2*ACTIONS_PER_STATE,
        "causally_changed_source_action_statuses":2*len(CASES),
        "receipt_sha256":sha256(receipt.read_bytes()).hexdigest(),
        "new_compiler_vendor_calls":0,
    }


def main(argv:list[str]|None=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output")
    mode.add_argument("--study")
    args=parser.parse_args(argv)
    if args.study:
        print(json.dumps(audit(args.study),sort_keys=True,indent=2))
        return
    rev=_git_revision()
    report=run()
    create_only(report,output=args.output,source_revision=rev)
    print(json.dumps({
        "schema":SCHEMA,"source_revision":rev,
        "matched_source_model_pairs":report["matched_static_source_model_pairs"],
        "candidate_population":report["total_compiler_masked_candidates"],
        "target_limit_status_flips":report["target_limit_status_flips"],
        "original_vendor_calls":0,
    },sort_keys=True,indent=2))


if __name__=="__main__":
    main()
