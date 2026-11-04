"""L7 independent semantic-meaning ablation for graph CST read certificates.

Construct in-bounds bijective source-index errors that retain ALL value
read cardinalities, so a lifetime verifier AND event source sharing the
same erroneous map may both accept. A separate NumPy index-grid oracle
must catch that semantic mismatch. This is a finite synthetic
falsification study, not a peer-compiler or original RTL baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.graph_phi import emit_graph_hls
from fpgai_research.experiments.cst_graph_access_fixtures import access_graph
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.graph_phi_contract import (
    graph_disjoint_offsets,graph_source_events,verify_graph_events,
)
from fpgai_research.model_ir import graph_phi_contract as graph_contract
from fpgai_research.model_ir.graph_access_maps import scalar_output_accesses
from fpgai_research.validation.graph_access_oracle import independent_numpy_access_oracle

CASES=(("broadcast",2,3,1),("broadcast",3,4,1),
       ("concat",2,3,2),("concat",3,4,3))


def correlated_map_mutation(model,events,kind):
    """Permute real producer indices, preserving per-source read counts."""
    mutated=[]
    for event in events:
        if event.kind!="READ":
            mutated.append(event)
            continue
        if kind=="broadcast" and event.consumer=="broadcast_add" and event.tensor=="narrow":
            rows,cols=model.tensor("narrow").shape
            row,column=divmod(event.element,cols)
            mutated.append(replace(event,element=((row+1)%rows)*cols+column))
        elif kind=="concat" and event.consumer=="combine" and event.tensor=="a":
            rows,cols=model.tensor("a").shape
            row,column=divmod(event.element,cols)
            mutated.append(replace(event,element=row*cols+(column+1)%cols))
        else:
            mutated.append(event)
    mutated=tuple(mutated)
    if mutated==events:
        raise AssertionError("Prespecified semantic mutation had no effect")
    return mutated


def run(*,output:str|Path):
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Independent semantic ablation output is create-only")
    records=[]
    for kind,m,n,q in CASES:
        model=access_graph(kind,m=m,n=n,q=q)
        arch=resolve(model,ArchitectureRequest())
        offsets=graph_disjoint_offsets(arch)
        events,_=graph_source_events(arch)
        for variant,candidate in (("valid",events),
                                  ("in_bounds_index_permutation",
                                   correlated_map_mutation(model,events,kind))):
            # Deliberately share the same wrong access-map helper between
            # the event producer and the E0 lifetime checker. This is a
            # CONTROLLED synthetic correlated-implementation experiment,
            # not a production compiler mode or a real source mutation.
            canonical=graph_contract.scalar_output_accesses
            def correlated_wrong_map(model,op,output_index):
                mapped=scalar_output_accesses(model,op,output_index)
                wrong=[]
                for port,name,element in mapped:
                    if kind=="broadcast" and op.name=="broadcast_add" and name=="narrow":
                        rows,cols=model.tensor(name).shape
                        row,col=divmod(element,cols)
                        element=((row+1)%rows)*cols+col
                    if kind=="concat" and op.name=="combine" and name=="a":
                        rows,cols=model.tensor(name).shape
                        row,col=divmod(element,cols)
                        element=row*cols+(col+1)%cols
                    wrong.append((port,name,element))
                return tuple(wrong)
            try:
                if variant!="valid":
                    graph_contract.scalar_output_accesses=correlated_wrong_map
                e0=verify_graph_events(arch,offsets,candidate)
            finally:
                graph_contract.scalar_output_accesses=canonical
            independent=independent_numpy_access_oracle(arch,candidate)
            if e0["status"]!="VERIFIED_E0":
                raise AssertionError("Correlated mapping mutation did not preserve E0 lifetime")
            should_accept=(variant=="valid")
            if (independent["status"]=="VERIFIED_INDEPENDENT_ACCESS_E0")!=should_accept:
                raise AssertionError("Independent reference failed to distinguish semantic mapping")
            records.append({
                "graph":kind,"dims":json.dumps([m,n,q]),
                "variant":variant,
                "source_lifetime_accepted":e0["status"]=="VERIFIED_E0",
                "independent_semantics_accepted":should_accept,
                "expected_semantic_legal":should_accept,
                "source_events":len(candidate),
                "missing_read":json.dumps(independent.get("first_missing"),sort_keys=True),
                "unexpected_read":json.dumps(independent.get("first_unexpected"),sort_keys=True),
                "original_vitis_calls":0,
            })
    summary={
        "schema":"fpgai-research/cst-independent-graph-semantics-falsification-e0/v1",
        "source_revision":subprocess.run(["git","rev-parse","HEAD"],
            check=True,text=True,capture_output=True,timeout=5).stdout.strip(),
        "cases":len(records),"positive_controls":len(CASES),
        "negative_correlated_map_mutants":len(CASES),
        "lifetime_only_false_acceptances":sum(not x["expected_semantic_legal"] and
                                           x["source_lifetime_accepted"] for x in records),
        "full_independent_false_acceptances":sum(not x["expected_semantic_legal"] and
                                           x["independent_semantics_accepted"] for x in records),
        "original_vitis_calls":0,
        "evidence":"E0-independent-NumPy-vs-shared-lifetime-semantic-mutation-study",
        "limitations":[
            "Artificial in-bounds index permutations; not a claim of a bug in upstream tools",
            "NumPy oracle checks logical access semantics, NOT Phi/hardware/source cycles",
            "No original Vitis, native numerical benchmark or external compiler run",
        ],
    }
    root.mkdir(parents=True)
    with (root/"cases.csv").open("w",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    (root/"summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    print(json.dumps(run(output=p.parse_args(argv).output),indent=2))


if __name__=="__main__":
    main()
