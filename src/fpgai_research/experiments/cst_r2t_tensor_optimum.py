"""R2t: prospective exact whole-tensor baseline versus original R2q 36 cases.

This is a NEW source-only campaign, not an edit to frozen R2q.
No new GCC/Vitis/Vivado/RL calls. Unknown budget cases are explicit,
and cannot be scored as feasible optimum. Output is create-only.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import json
from pathlib import Path
import re
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import source_layout
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    FANOUTS, SHAPES, POLICY_NAMES, policy_widths,
)
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.tensor_exact_allocation import (
    exact_tensor_source_baseline,
)

SCHEMA = "fpgai-research/cst-R2t-original-36-exact-tensor-comparator-E0/v1"
CASE_COUNT = len(FANOUTS) * len(SHAPES) * len(POLICY_NAMES)
COLUMNS = (
    "case", "fanout_readers", "shape", "policy",
    "disjoint_source_slots", "greedy_whole_tensor_slots",
    "proven_optimal_whole_tensor_slots", "proven_tensor_lower_bound_slots",
    "committed_element_exact_slots", "tensor_optimum_status",
    "search_nodes", "source_event_sha256", "tensor_layout_bases",
)


def source_revision() -> str:
    repo = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    if status.strip():
        raise RuntimeError("R2t original source requires clean tracked Git")
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    if re.fullmatch("[a-f0-9]{40}",result) is None:
        raise ValueError("R2t exact original full source SHA absent")
    return result


def calculate_catalog(*, max_search_nodes: int = 250_000) -> list[dict]:
    if type(max_search_nodes) is not int or max_search_nodes < 1:
        raise ValueError("Finite exact tensor search budget required")
    cases = []
    for r in FANOUTS:
        for dims in SHAPES:
            m,k,n,q=dims
            model=sample_fanout(r,m=m,k=k,n=n,q=q)
            for policy in POLICY_NAMES:
                base=ArchitectureRequest()
                widths=policy_widths(r,policy)
                arch=resolve(model,base,{
                    f"reader{i}":replace(base,pes=width)
                    for i,width in enumerate(widths) if width!=1
                })
                spec, slotmap, exact_units, optimal, events=source_layout(
                    arch, layout="per_element_optimal"
                )
                result=exact_tensor_source_baseline(
                    events,max_search_nodes=max_search_nodes
                )
                disjoint=len(slotmap)
                if (result.lower_bound < exact_units
                        or result.greedy_feasible_slots < result.lower_bound
                        or disjoint < result.greedy_feasible_slots
                        or (result.status=="OPTIMAL_PROVED"
                            and (result.exact_feasible_slots is None
                                 or not exact_units<=result.exact_feasible_slots
                                 <=result.greedy_feasible_slots))):
                    raise AssertionError("R2t inconsistent original source baseline")
                cases.append({
                    "case":f"case_{len(cases):03d}",
                    "fanout_readers":r, "shape":list(dims), "policy":policy,
                    "disjoint_source_slots":disjoint,
                    "greedy_whole_tensor_slots":result.greedy_feasible_slots,
                    "proven_optimal_whole_tensor_slots":result.exact_feasible_slots,
                    "proven_tensor_lower_bound_slots":result.lower_bound,
                    "committed_element_exact_slots":exact_units,
                    "tensor_optimum_status":result.status,
                    "search_nodes":result.search_nodes,
                    "source_event_sha256":result.source_event_sha256
                        if result.source_event_sha256 is not None
                        else optimal.schedule_sha256,
                    "tensor_layout_bases":[list(item) for item in result.bases]
                        if result.bases is not None else None,
                })
    if len(cases)!=CASE_COUNT:
        raise AssertionError("R2t original full 36-case denominator missing")
    return cases


def summarize(cases: list[dict], revision: str, budget: int) -> dict:
    if len(cases)!=CASE_COUNT:
        raise ValueError("R2t 36 cases required for full original summary")
    proved=[c for c in cases if c["tensor_optimum_status"]=="OPTIMAL_PROVED"]
    unknown=[c for c in cases if c["tensor_optimum_status"]=="UNKNOWN_BUDGET"]
    if len(proved)+len(unknown)!=CASE_COUNT:
        raise ValueError("R2t unknown exact optimality status")
    return {
        "schema":SCHEMA,"source_revision":revision,
        "max_search_nodes_per_case":budget,
        "original_source_cases":CASE_COUNT,
        "optimality_proved_cases":len(proved),
        "budget_unknown_cases":len(unknown),
        "tensor_optimum_better_than_greedy_cases":sum(
            c["proven_optimal_whole_tensor_slots"]<
            c["greedy_whole_tensor_slots"] for c in proved
        ),
        "element_exact_better_than_proved_tensor_optimum_cases":sum(
            c["committed_element_exact_slots"]<
            c["proven_optimal_whole_tensor_slots"] for c in proved
        ),
        "optimum_equals_tensor_lower_bound_cases":sum(
            c["proven_optimal_whole_tensor_slots"]==
            c["proven_tensor_lower_bound_slots"] for c in proved
        ),
        "unknown_cases":[c["case"] for c in unknown],
        "original_GCC_calls":0, "original_Vitis_calls":0,
        "original_Vivado_calls":0, "trained_RL_policies":0,
        "limits":[
            "Exact contiguous whole-tensor reuse for fixed first-COMMIT to last-READ intervals",
            "Exponential spatial-order search; budget-unknown cases are not certified optimum",
            "Committed-value source scalar slots, not actual native FPGA BRAM or ports",
            "Separate emitted reader accumulator and counter arrays remain present",
            "No proof of general mapping-dependent memory advantage or solver novelty",
        ],
    }


def run(*, output: str|Path, max_search_nodes:int=250_000)->dict:
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R2t original source optimum study create-only")
    revision=source_revision()
    cases=calculate_catalog(max_search_nodes=max_search_nodes)
    summary=summarize(cases,revision,max_search_nodes)
    root.mkdir(parents=True)
    for name,body in (("cases.json",cases),("summary.json",summary)):
        with (root/name).open("x",encoding="utf-8") as output_file:
            json.dump(body,output_file,indent=2,sort_keys=True)
            output_file.write("\n")
    with (root/"comparisons.csv").open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=COLUMNS)
        writer.writeheader()
        for row in cases:
            writer.writerow({
                **row,"shape":json.dumps(row["shape"]),
                "tensor_layout_bases":json.dumps(row["tensor_layout_bases"]),
            })
    return summary


def audit(study: str|Path)->dict:
    root=Path(study)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("R2t original study folder missing or symlinked")
    paths=[root/x for x in ("cases.json","summary.json","comparisons.csv")]
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError("R2t original source receipts missing or symlinked")
    summary=json.loads(paths[1].read_text())
    if (summary.get("schema")!=SCHEMA
            or summary.get("source_revision")!=source_revision()
            or type(summary.get("max_search_nodes_per_case")) is not int
            or summary["max_search_nodes_per_case"]<1):
        raise ValueError("R2t original Git revision or search budget invalid")
    saved=json.loads(paths[0].read_text())
    current=calculate_catalog(
        max_search_nodes=summary["max_search_nodes_per_case"]
    )
    if saved!=current or summary!=summarize(
            current,summary["source_revision"],summary["max_search_nodes_per_case"]):
        raise ValueError("R2t source-optimum saved results differ from recalculation")
    with paths[2].open(newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=list(COLUMNS):
            raise ValueError("R2t CSV header changed")
        read=list(reader)
    if len(read)!=CASE_COUNT:
        raise ValueError("R2t CSV denominator changed")
    for case,expected in zip(read,current):
        original={**expected,
                  "shape":json.dumps(expected["shape"]),
                  "tensor_layout_bases":json.dumps(expected["tensor_layout_bases"])}
        if any(case[field]!=(
            "True" if original[field] is True else
            "False" if original[field] is False else
            "" if original[field] is None else str(original[field]))
            for field in COLUMNS):
            raise ValueError("R2t CSV source measurement altered")
    return {
        "schema":"fpgai-research/cst-R2t-source-exact-read-only-recompute/v1",
        "source_revision":summary["source_revision"],
        "original_cases":CASE_COUNT,
        "proved_optimal_cases":summary["optimality_proved_cases"],
        "budget_unknown_cases":summary["budget_unknown_cases"],
        "original_GCC_calls_during_audit":0,
        "original_vendor_calls_during_audit":0,
        "note":"Same-source exact E0 recomputation, not independent GCC/RTL/Vivado",
    }


def main(argv: list[str]|None=None)->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output")
    p.add_argument("--study")
    p.add_argument("--audit-output")
    p.add_argument("--max-search-nodes",type=int,default=250_000)
    args=p.parse_args(argv)
    if bool(args.output)==bool(args.study):
        p.error("Choose one --output original source study or --study read-only audit")
    if args.study:
        result=audit(args.study)
        if args.audit_output:
            dest=Path(args.audit_output)
            if dest.exists() or dest.is_symlink():
                raise FileExistsError("R2t original audit create-only")
            with dest.open("x",encoding="utf-8") as out:
                json.dump(result,out,indent=2,sort_keys=True)
                out.write("\n")
    else:
        if args.audit_output:
            p.error("--audit-output only valid with --study")
        result=run(output=args.output,max_search_nodes=args.max_search_nodes)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
