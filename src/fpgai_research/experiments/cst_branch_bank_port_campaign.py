"""L6b original CPU-only declared abstract port/UNKNOWN source evidence."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.experiments.cst_branch_models import sample_branch
from fpgai_research.model_ir.branch_bank_ports import (
    DeclaredPortModel,CycleAssignment,check_declared_ports,
)
from fpgai_research.model_ir.branch_matmul_contract import branch_matmul_source_events
from fpgai_research.model_ir.contracts import ArchitectureRequest

SHAPES=((1,2,3,1),(2,2,3,2),(2,2,2,1))


def run(*,output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Declared port-model evidence is create-only")
    rows=[]
    for dims in SHAPES:
        model=sample_branch(*dims)
        base=ArchitectureRequest()
        overrides={model.operators[1].name:replace(base,pes=2,simd=2)}
        arch=resolve(model,base,overrides)
        offsets=disjoint_branch_offsets(arch)
        events,_=branch_matmul_source_events(arch)
        model2=DeclaredPortModel(2,1,1)
        trials={
            "missing_schedule":check_declared_ports(
                arch,offsets,port_model=model2),
            "source_serial_hypothesis":check_declared_ports(
                arch,offsets,port_model=model2,
                schedule=tuple(CycleAssignment(i,i) for i in range(len(events)))),
            "oversubscribed_same_tick":check_declared_ports(
                arch,offsets,port_model=DeclaredPortModel(1,1,1),
                schedule=tuple(CycleAssignment(i,0) for i in range(len(events)))),
        }
        expected={"missing_schedule":"UNKNOWN_UNSCHEDULED",
                  "source_serial_hypothesis":"ABSTRACT_PORT_FEASIBLE_E0",
                  "oversubscribed_same_tick":"REJECTED_ABSTRACT_PORT_OVERSUBSCRIPTION"}
        for name,receipt in trials.items():
            if receipt["status"]!=expected[name] or receipt["hardware_qualified"]:
                raise AssertionError("Unknown or abstract bank safety was falsely hardware certified")
            rows.append({"dims":json.dumps(dims),"policy":name,
                         "status":receipt["status"],
                         "abstract_only":receipt["certified_abstract_only"],
                         "hardware_qualified":receipt["hardware_qualified"],
                         "source_events":len(events),
                         "original_vitis_calls":0})
    summary={
        "schema":"fpgai-research/cst-declared-port-unknown-controls-e0/v1",
        "source_revision":subprocess.run(["git","rev-parse","HEAD"],check=True,
            capture_output=True,text=True,timeout=5).stdout.strip(),
        "cases":len(rows),"graph_shapes":len(SHAPES),
        "unknown_unscheduled":sum(r["status"]=="UNKNOWN_UNSCHEDULED" for r in rows),
        "abstract_feasible_only":sum(r["status"]=="ABSTRACT_PORT_FEASIBLE_E0" for r in rows),
        "declared_port_oversubscriptions":sum(r["status"]=="REJECTED_ABSTRACT_PORT_OVERSUBSCRIPTION" for r in rows),
        "hardware_qualified_cases":sum(r["hardware_qualified"] for r in rows),
        "original_vitis_calls":0,
        "evidence":"E0-explicit-source-event-abstract-port-check-NOT-hardware",
        "limitations":[
            "No inferred HLS cycle schedule or physical RAM mapping",
            "Hypothetical modulo stripe banks only; actual HLS source monolithic scratch[]",
            "No physical optimization/throughput or original Vitis execution",
        ],
    }
    root.mkdir(parents=True)
    with (root/"cases.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (root/"summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    print(json.dumps(run(output=parser.parse_args(argv).output),indent=2))


if __name__=="__main__":
    main()
