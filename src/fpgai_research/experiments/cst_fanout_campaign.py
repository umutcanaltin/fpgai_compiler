"""Prespecified E0 two-reader lifetime controls for CST branch semantics.

This is an abstract IR-level proof obligation, not lowered C++ or timing.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

from fpgai_research.model_ir.contracts import FunctionalModel,Operator,Tensor
from fpgai_research.model_ir.fanout_lifetime import (
    LifetimeEvent,canonical_fanout_events,verify_fanout_events,
)

def branch_graph():
    return FunctionalModel(
        tensors=(Tensor("x",(1,2)),Tensor("w",(2,2),data=(1.,0.,0.,1.)),
                 Tensor("h",(1,2)),Tensor("a",(1,2)),Tensor("b",(1,2)),
                 Tensor("g",(1,2)),Tensor("out",(1,2))),
        operators=(Operator("mm","MatMul",("x","w"),("h",)),
                   Operator("read0","Relu",("h",),("a",)),
                   Operator("read1","Relu",("h",),("b",)),
                   Operator("join","Add",("a","b"),("g",)),
                   Operator("final","Relu",("g",),("out",))),
        inputs=("x",),outputs=("out",),
    )

def run(*, output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Create-only fan-out research study already exists")
    m=branch_graph()
    cases=[]
    for layout in ("shared_slot","disjoint_slot"):
        slots={"h":0,"g":0 if layout=="shared_slot" else 2}
        events=canonical_fanout_events(m,slots)
        positive=verify_fanout_events(m,slots,events)
        if positive["status"]!="VERIFIED":
            raise AssertionError("Valid fan-out consumer releases rejected")
        first_reader=next(i for i,e in enumerate(events)
                          if e.kind=="CONSUME" and e.consumer=="read0")
        premature=verify_fanout_events(m,slots,
            events[:first_reader+1]+(LifetimeEvent("COMMIT","g",0),)+
            events[first_reader+1:])
        expected="OVERWRITE_BEFORE_LAST_READER" if layout=="shared_slot" else "DUPLICATE_COMMIT"
        # With disjoint slots the premature commit is memory-safe but the
        # unmodified subsequent schedule duplicates the commit. This does
        # not certify graph semantic execution order.
        if layout=="shared_slot" and premature["reason"]!=expected:
            raise AssertionError("Early overwrite not recognized")
        cases.append({
            "layout":layout,
            "positive_status":positive["status"],
            "multi_reader_values":positive["multi_reader_values"],
            "early_commit_status":premature["status"],
            "early_commit_reason":premature["reason"],
            "early_overwrite_rejected":premature["reason"]=="OVERWRITE_BEFORE_LAST_READER",
            "actual_cpp_generated":False,
            "native_vitis_calls":0,
        })
    summary={
        "schema":"fpgai-research/cst-fanout-abstract-controls/v1",
        "source_revision":subprocess.run(["git","rev-parse","HEAD"],
                 capture_output=True,text=True,timeout=5).stdout.strip(),
        "cases":len(cases),"valid_controls":sum(x["positive_status"]=="VERIFIED" for x in cases),
        "early_alias_rejections":sum(x["early_overwrite_rejected"] for x in cases),
        "executed_source_cases":0,"native_vitis_calls":0,
        "evidence":"E0-abstract-two-reader-last-use-only",
        "limits":["No branching HLS source generated or verified",
                  "Acknowledgments not all per-iteration Phi read events",
                  "Not a timing/cycle, port or physical memory certificate"],
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
