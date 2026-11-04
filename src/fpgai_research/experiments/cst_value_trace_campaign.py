"""Prespecified E1 source-executed mapped tensor-value lifetime campaign.

Unlike the E0 row-atomic checker, this executes actual generated GCC C++ with
per-contribution and per-element read/ready instrumentation. The independent
observer requires all reductions to finish before every element commit, all
reads after production, and last-reader release before physical scratch reuse.
No dataflow RTL or actual Vitis cycle claims.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.row_chain import compile_row_chain
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.experiments.cst_nonlinear_campaign import synthetic_nonlinear_model
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.cpp_value_trace import trace_cpp_values


SCHEMA="fpgai-research/cst-executed-mapped-value-campaign/v1"
CATALOG=(
    # First two are an explicit *paired* causal control: same Functional
    # graph and exchange schedule, only operator Phi spatial mapping changes.
    ("MatMul",3,(3,5,4,2),1,False),
    ("MatMul",3,(3,5,4,2),1,True),
    ("MatMul",5,(3,5,4,3,2),2,True),
    ("Nonlinear",2,(),1,False),
    ("Nonlinear",3,(),2,True),
    ("Nonlinear",5,(),5,True),
)


def run(*, output: str | Path, max_cases: int = 6) -> dict:
    if type(max_cases) is not int or not 1<=max_cases<=len(CATALOG):
        raise ValueError("max_cases must be 1..6")
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Create-only source value-trace experiment already exists")
    results=[]
    for case_id,(kind,rows,widths,block,hetero) in enumerate(CATALOG[:max_cases]):
        graph=(make_chain(rows,widths) if kind=="MatMul"
               else synthetic_nonlinear_model(rows,trans_b=int(hetero)))
        base=ArchitectureRequest()
        overrides=({graph.operators[0 if kind=="MatMul" else 1].name:
                    replace(base,pes=2,simd=2,serialization=2)}
                   if hetero else {})
        arch=resolve(graph,base,overrides)
        compiled=compile_row_chain(arch,row_block=block)
        result=trace_cpp_values(compiled.ir,emission=compiled.emission)
        if result["status"]!="SOURCE_VALUE_LIFETIMES_VERIFIED":
            raise AssertionError("Actual emitted tensor value lifetime not verified")
        results.append({
            "case":case_id,"kind":kind,"rows":rows,
            "operators":len(graph.operators),"row_block":block,
            "heterogeneous_mapping":hetero,
            "functional_sha256":result["functional_sha256"],
            "source_cpp_sha256":result["cpp_sha256"],
            "executed_trace_sha256":result["source_trace_sha256"],
            "mapped_contributions":result["mapped_reduction_iterations_verified"],
            "commits":result["element_commits_verified"],
            "reads":result["downstream_and_input_reads_verified"],
            "last_reader_releases":result["last_reader_releases_verified"],
            "source_verified":True,
            "original_vitis_calls":0,
        })
    paired_phi_control = None
    if len(results)>=2:
        control,variant=results[:2]
        if (control["functional_sha256"]!=variant["functional_sha256"]
                or control["row_block"]!=variant["row_block"]):
            raise AssertionError("Phi causal control has unmatched graph or row schedule")
        if (control["executed_trace_sha256"]==variant["executed_trace_sha256"]
                or control["source_cpp_sha256"]==variant["source_cpp_sha256"]):
            raise AssertionError("Phi mapping change did not change actual source ownership trace")
        if (control["commits"]!=variant["commits"]
                or control["last_reader_releases"]!=variant["last_reader_releases"]):
            raise AssertionError("Phi control altered logical tensor cardinalities")
        paired_phi_control = {
            "same_functional_graph":True,
            "same_row_exchange_schedule":True,
            "different_actual_mapping_and_source_traces":True,
            "same_logical_value_and_release_counts":True,
            "claim":"Phi changes source spatial ownership trace, NOT the current row-atomic legal-action mask",
        }
    revision=subprocess.run(
        ["git","rev-parse","HEAD"],capture_output=True,text=True,timeout=5)
    summary={
        "schema":SCHEMA,
        "source_revision":revision.stdout.strip(),
        "cases_attempted":len(results),
        "source_value_trace_passes":sum(x["source_verified"] for x in results),
        "paired_phi_control":paired_phi_control,
        "total_mapped_reduction_iterations":sum(x["mapped_contributions"] for x in results),
        "total_logical_element_commits":sum(x["commits"] for x in results),
        "total_verified_consumer_releases":sum(x["last_reader_releases"] for x in results),
        "original_vitis_calls":0,
        "evidence":"E1-actual-GCC-traced-Phi-reductions-and-element-lifetimes",
        "limits":[
            "Static sequential single-reader MatMul/Gemm/Relu/Flatten row-chain only",
            "Current cross-stage schedule still materializes whole rows",
            "No general fan-out or per-element early-consumption codegen",
            "Actual GCC event ordering is not Vitis cycle, physical bank or RTL proof",
        ],
    }
    target.mkdir(parents=True)
    with (target/"cases.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    (target/"summary.json").write_text(json.dumps(
        summary,indent=2,sort_keys=True)+"\n")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--max-cases",type=int,default=6)
    args=p.parse_args(argv)
    print(json.dumps(run(output=args.output,max_cases=args.max_cases),indent=2))


if __name__=="__main__":
    main()
