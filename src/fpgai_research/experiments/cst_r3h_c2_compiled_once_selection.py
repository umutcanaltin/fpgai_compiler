"""R3h: measured C2 compiled-once source catalog consumption ablation.

Shows a precise API-level causal improvement: after paying for the full
E0 mask, selecting a certified candidate needs zero additional *source
compiler transitions* when it retains the original typed state. It does
NOT imply that computing a mask is more efficient than trying a single
unmasked action, or that any trained agent improves HLS performance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter_ns

from fpgai_research.experiments.cst_r3d_c1c2_finite_contract_ablation import _git_revision
from fpgai_research.experiments.cst_r3f_c2_variable_graph_coverage import COHORTS, _case
from fpgai_research.model_ir.compiled_action_catalog import (
    build_catalog, select_certified,
)
from fpgai_research.model_ir.unified_decision_interface import (
    unified_action_mask, unified_state_hash,
)

SCHEMA="fpgai-research/R3h-compiled-once-C2-source-selection/v1"


def evaluate(name: str,width: int)->dict:
    if (name,width) not in COHORTS:
        raise ValueError("R3h: graph not in preregistered C2 cohort")
    state,actions=_case(name,width)
    original_hash=unified_state_hash(state)
    start=perf_counter_ns()
    catalog=build_catalog(state,actions)
    catalog_ns=perf_counter_ns()-start
    start=perf_counter_ns()
    mask=unified_action_mask(state,actions)
    replay_ns=perf_counter_ns()-start
    if (catalog.statuses()!=mask["statuses"]
            or catalog.mask()!=mask["e0_certified"]
            or catalog.as_receipt()["rows"]!=mask["rows"]
            or unified_state_hash(state)!=original_hash):
        raise ValueError("R3h: compiled-once catalog differs from original source mask")
    selected=next((i for i,good in enumerate(catalog.mask()) if good),None)
    if selected is None:
        raise ValueError("R3h: no source-certified action in fixed cohort")
    start=perf_counter_ns()
    result=select_certified(catalog,selected,current_state=state)
    selection_ns=perf_counter_ns()-start
    if (result.as_receipt()!=mask["rows"][selected]
            or result.state_sha256!=unified_state_hash(result.next_state)
            or result.next_state is state
            or unified_state_hash(state)!=original_hash):
        raise ValueError("R3h: cached source selection broke identity")
    counts=Counter(catalog.statuses())
    return {
        "cohort":name,"parameter":width,
        "parent_source_state_sha256":original_hash,
        "catalog_compiler_calls":len(actions),
        "legacy_mask_compiler_calls":len(actions),
        "legacy_second_compile_if_selected":1,
        "catalog_additional_compile_if_selected":0,
        "certified_action_count":counts["BOUNDED_CERTIFIED_E0"],
        "unknown_e0_count":counts["UNKNOWN_E0"],
        "rejected_count":counts["REJECTED"],
        "chosen_action_index":selected,
        "chosen_action":mask["rows"][selected]["action"],
        "selected_source_cpp_sha256":result.cpp_sha256,
        "selected_next_state_sha256":result.state_sha256,
        "catalog_wall_ns":catalog_ns,
        "independent_mask_wall_ns":replay_ns,
        "cached_selection_wall_ns":selection_ns,
        "native_vitis_calls":0,"trained_agents":0,
    }


def run()->dict:
    rows=[evaluate(n,w) for n,w in COHORTS]
    if len(rows)!=6:
        raise AssertionError("R3h: C2 source cohort incomplete")
    return {
        "schema":SCHEMA,
        "original_graph_population":len(rows),
        "original_candidate_population":sum(r["catalog_compiler_calls"] for r in rows),
        "original_compiled_source_selection_population":len(rows),
        "catalog_source_compiler_calls":sum(r["catalog_compiler_calls"] for r in rows),
        "independent_legacy_mask_compiler_calls":sum(
            r["legacy_mask_compiler_calls"] for r in rows),
        "catalog_additional_selection_compiler_calls":0,
        "legacy_additional_selection_compiler_calls":len(rows),
        "rows":rows,
        "new_gcc_calls":0,"new_vitis_calls":0,
        "new_vivado_calls":0,"trained_rl_policies":0,
        "limits":[
            "API-level compile-once selection saving occurs AFTER mask construction cost",
            "A single blind unmasked proposal costs one source compiler transition, not eight",
            "No hardware performance, physical resource/clock or trained-agent efficiency result",
            "Timing data are exploratory CPU wall nanoseconds, not randomized statistical comparisons",
            "Six hand-catalogued variable-size source graph cohorts, not arbitrary real-world workloads",
        ],
    }


def create_only(result:dict,*,output:str|Path,source_revision:str)->None:
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R3h: create-only original source report")
    if (result.get("schema")!=SCHEMA
            or re.fullmatch(r"[a-f0-9]{40}",source_revision) is None):
        raise ValueError("R3h: source revision/schema invalid")
    root.mkdir(parents=True,exist_ok=False)
    (root/"r3h_compiled_once_selection.json").write_text(
        json.dumps({**result,"source_revision":source_revision},
                   indent=2,sort_keys=True,allow_nan=False)+"\n",
        encoding="utf-8")


def audit(study:str|Path)->dict:
    root=Path(study)
    file=root/"r3h_compiled_once_selection.json"
    if root.is_symlink() or not root.is_dir() or file.is_symlink() or not file.is_file():
        raise ValueError("R3h: missing/noncanonical original source report")
    j=json.loads(file.read_text())
    rows=j.get("rows")
    if (j.get("schema")!=SCHEMA
            or re.fullmatch(r"[a-f0-9]{40}",j.get("source_revision") or "") is None
            or j.get("original_graph_population")!=len(COHORTS)
            or j.get("original_candidate_population")!=8*len(COHORTS)
            or j.get("original_compiled_source_selection_population")!=len(COHORTS)
            or not isinstance(rows,list) or len(rows)!=len(COHORTS)
            or any(j.get(k)!=0 for k in (
                "catalog_additional_selection_compiler_calls",
                "new_gcc_calls","new_vitis_calls","new_vivado_calls",
                "trained_rl_policies"))):
        raise ValueError("R3h: incomplete or tampered original source study")
    for row,(name,width) in zip(rows,COHORTS):
        if (row.get("cohort")!=name or row.get("parameter")!=width
                or row.get("catalog_compiler_calls")!=8
                or row.get("legacy_mask_compiler_calls")!=8
                or row.get("legacy_second_compile_if_selected")!=1
                or row.get("catalog_additional_compile_if_selected")!=0
                or not isinstance(row.get("chosen_action_index"),int)
                or not 0<=row["chosen_action_index"]<8
                or not re.fullmatch(r"[a-f0-9]{64}",
                                    row.get("selected_source_cpp_sha256") or "")
                or not re.fullmatch(r"[a-f0-9]{64}",
                                    row.get("selected_next_state_sha256") or "")
                or any(type(row.get(f)) is not int or row[f]<0 for f in (
                    "catalog_wall_ns","independent_mask_wall_ns",
                    "cached_selection_wall_ns"))):
            raise ValueError("R3h: cached selection or expense receipt invalid")
    if (j.get("catalog_source_compiler_calls")!=8*len(COHORTS)
            or j.get("independent_legacy_mask_compiler_calls")!=8*len(COHORTS)
            or j.get("legacy_additional_selection_compiler_calls")!=len(COHORTS)):
        raise ValueError("R3h: compiler transition denominator altered")
    return {
        "schema":SCHEMA,
        "source_revision":j["source_revision"],
        "candidate_population":8*len(COHORTS),
        "catalog_compiler_calls":j["catalog_source_compiler_calls"],
        "compiled_once_selection_additional_compiler_calls":0,
        "source_receipt_sha256":sha256(file.read_bytes()).hexdigest(),
        "new_compiler_or_vendor_calls_during_audit":0,
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
    sha=_git_revision()
    result=run()
    create_only(result,output=args.output,source_revision=sha)
    print(json.dumps({
        "schema":SCHEMA,"source_revision":sha,
        "graph_population":result["original_graph_population"],
        "candidate_population":result["original_candidate_population"],
        "catalog_compiler_calls":result["catalog_source_compiler_calls"],
        "catalog_additional_selection_compiler_calls":
            result["catalog_additional_selection_compiler_calls"],
        "new_vendor_calls":0,"trained_agents":0,
    },sort_keys=True,indent=2))


if __name__=="__main__":
    main()
