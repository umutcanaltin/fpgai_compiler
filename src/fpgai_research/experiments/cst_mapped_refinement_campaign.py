"""Prespecified source-hashed E0-to-E1 mapped-value refinement study.

Three row-block candidates, one Functional graph. Each run validates the
independent mapped logical event verifier and actually compiles/executes
the emitted C++ value/reduction tracer. No vendor HLS/native RL calls.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.validation.mapped_source_refinement import qualify_mapped_source

def run(*,output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Create-only source-refinement study exists")
    arch=resolve(make_chain(3,(3,5,4,2)),ArchitectureRequest())
    cases=[]
    for b in (1,2,3):
        r=qualify_mapped_source(arch,row_block=b)
        if r["status"]!="EXECUTED_SOURCE_REFINEMENT_VERIFIED":
            raise AssertionError("Required source refinement did not qualify")
        cases.append({
            "row_block":b,
            "source_sha256":r["source_cpp_sha256"],
            "logical_event_sha256":r["logical_phi_event_sha256"],
            "executed_trace_sha256":r["actual_cpp_trace_sha256"],
            "values_certified":r["logical_value_count"],
            "actual_last_reads":r["actual_cpp_last_reader_releases"],
            "status":r["status"],"native_vitis_calls":0,
        })
    if len({r["source_sha256"] for r in cases})!=3:
        raise AssertionError("Distinct B source schedules collapsed unexpectedly")
    summary={
        "schema":"fpgai-research/cst-phi-e0-to-e1-study/v1",
        "source_revision":subprocess.run(["git","rev-parse","HEAD"],
                 text=True,capture_output=True,timeout=5).stdout.strip(),
        "matched_source_cases":len(cases),
        "qualified_source_cases":sum(x["status"]=="EXECUTED_SOURCE_REFINEMENT_VERIFIED"
                                     for x in cases),
        "all_source_last_reads_released":all(x["values_certified"]==x["actual_last_reads"]
                                              for x in cases),
        "original_vitis_calls":0,
        "evidence":"E0-mapped-logical-events-plus-E1-actual-C++-source-trace",
        "limits":["No actual RTL/native HLS CSim",
                  "One sequential static chain only",
                  "No multi-reader source lowering or cycle-level proof"],
    }
    root.mkdir(parents=True)
    with (root/"cases.csv").open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(cases[0]))
        writer.writeheader();writer.writerows(cases)
    (root/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    a=p.parse_args(argv)
    print(json.dumps(run(output=a.output),indent=2))
if __name__=="__main__":
    main()
