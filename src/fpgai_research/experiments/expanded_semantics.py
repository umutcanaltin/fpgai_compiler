"""Deterministic expanded E0/E1 Model IR semantic experiment.

Executes generated C++ under GCC host-float against the NumPy reference at
every operator boundary. Never equate results with ap_fixed or native FPGA.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from fpgai_research.model_ir.contracts import Tensor,Operator,FunctionalModel,ArchitectureRequest
from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import verify_refinement
from fpgai_research.validation.tensor_trace import check_tensor_trace

def _fixture(rng):
    weights=tuple(float(x) for x in rng.uniform(-.45,.45,(6,4)).astype(np.float32).flat)
    bias=tuple(float(x) for x in rng.uniform(-.2,.2,4).astype(np.float32))
    scalar=(float(rng.uniform(.3,1.1)),)
    model=FunctionalModel((Tensor("x",(1,2,3)),Tensor("flat",(1,6)),
        Tensor("weights",(6,4),data=weights),Tensor("matmul",(1,4)),
        Tensor("bias",(4,),data=bias),Tensor("added",(1,4)),
        Tensor("relu",(1,4)),Tensor("scale",(),data=scalar),Tensor("y",(1,4))),(
        Operator("reshape","Reshape",("x",),("flat",),(("shape",(0,-1)),)),
        Operator("mm","MatMul",("flat","weights"),("matmul",)),
        Operator("add","Add",("matmul","bias"),("added",)),
        Operator("relu","Relu",("added",),("relu",)),
        Operator("scale","Mul",("relu","scale"),("y",))),("x",),("y",))
    return model,rng.uniform(-.5,.5,(1,2,3)).astype(np.float32)

def run_expanded_semantics(*,seed=2026,fixtures=8,output=None):
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError("Seed must be a nonnegative 32-bit integer")
    if type(fixtures) is not int or not 1<=fixtures<=32:
        raise ValueError("fixtures must be 1..32")
    rng=np.random.default_rng(seed)
    base=ArchitectureRequest()
    options=(
       ("baseline",base),
       ("spatial_mapping",ArchitectureRequest(pes=2,parallelism=2,simd=2,serialization=2,tile=(1,3,2))),
       ("tail_mapping",ArchitectureRequest(pes=3,parallelism=1,simd=4,serialization=3,tile=(1,4,3))),
    )
    records=[]
    for index in range(fixtures):
        model,sample=_fixture(rng)
        functional_hash=None
        mapping_signatures=set()
        for label,request in options:
            architect=resolve(model,base,{"mm":request})
            low=lower(architect)
            stage=verify_refinement(low)
            if functional_hash is None:
                functional_hash=stage["functional_sha256"]
            if stage["functional_sha256"]!=functional_hash:
                raise AssertionError("Functional Model IR changed under architecture specialization")
            traced=check_tensor_trace(low,sample)
            if not traced["accepted"] or traced["operator_count"]!=5:
                raise AssertionError("Intermediate tensor semantic mismatch")
            mapping_signatures.add(stage["effective_architecture_sha256"])
            records.append({"fixture":index,"variant":label,
                "functional_sha256":functional_hash,
                "architecture_sha256":stage["effective_architecture_sha256"],
                "cpp_sha256":traced["cpp_sha256"],
                "operator_boundaries_verified":traced["operator_count"],
                "scalar_values_verified":traced["total_values"],
                "max_absolute_error":max(t["max_absolute_error"] for t in traced["operators"]),
                "operator_kinds":[t["kind"] for t in traced["operators"]],
                "passed":True})
        if len(mapping_signatures)!=len(options):
            raise AssertionError("Operator-local configurations lack unique architecture mappings")
    report={"schema":"fpgai-research/expanded-model-ir-semantic-evidence/v1",
       "seed":seed,"fixtures":fixtures,"architectures_per_fixture":len(options),
       "cases":len(records),"passed":sum(r["passed"] for r in records),
       "operator_boundaries_checked":sum(r["operator_boundaries_verified"] for r in records),
       "scalar_values_checked":sum(r["scalar_values_verified"] for r in records),
       "max_absolute_error":max(r["max_absolute_error"] for r in records),
       "source":"Functional->Architectural->Lowered Model IR->generated HLS C++",
       "evidence":"E0/E1-executed-GCC-host-float",
       "native_vitis_executed":False,"bit_exact_fixed_point":False,
       "limitations":"Bounded deterministic graphs and float-mode GCC; no native ap_fixed or achieved HLS timing",
       "records":records}
    if output is not None:
        path=Path(output)
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            raise FileExistsError("Refusing to overwrite existing scientific receipt")
        path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report
