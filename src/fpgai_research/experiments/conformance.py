"""Repeatable E0/E1 bounded semantic + generated C++ host-float conformance campaign."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from fpgai_research.model_ir.contracts import ArchitectureRequest,FunctionalModel,Tensor,Operator
from fpgai_research.model_ir.lowering import lower
from fpgai_research.architecture.resolver import resolve
from fpgai_research.architecture.mapping import verify_mapping
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.validation.host import validate_host

def _model(rng,index):
    kind=("MatMul","Gemm","Conv")[index%3]
    if kind in ("MatMul","Gemm"):
        m,k,n=(int(rng.integers(1,5)) for _ in range(3))
        weight_shape=(n,k) if kind=="Gemm" else (k,n)
        weights=tuple(float(v) for v in rng.uniform(-1,1,weight_shape[0]*weight_shape[1]))
        x=rng.uniform(-1,1,(m,k)).astype(np.float32)
        tensors=(Tensor("x",(m,k)),Tensor("w",weight_shape,data=weights),Tensor("y",(m,n)))
        attrs=(("transB",1),) if kind=="Gemm" else ()
        model=FunctionalModel(tensors,(Operator("compute",kind,("x","w"),("y",),attrs),),("x",),("y",))
        shape=(m,n,k)
        tile=tuple(int(rng.integers(1,d+1)) for d in shape)
    else:
        channels=int(rng.integers(1,3));co=int(rng.integers(1,4))
        height=int(rng.integers(3,6));width=int(rng.integers(3,6))
        x=rng.uniform(-.5,.5,(1,channels,height,width)).astype(np.float32)
        weights=tuple(float(v) for v in rng.uniform(-1,1,co*channels*4))
        shape=(1,co,height-1,width-1,channels,2,2)
        tensors=(Tensor("x",(1,channels,height,width)),
                 Tensor("w",(co,channels,2,2),data=weights),
                 Tensor("y",(1,co,height-1,width-1)))
        model=FunctionalModel(tensors,(Operator("compute","Conv",("x","w"),("y",)),),("x",),("y",))
        tile=(1,int(rng.integers(1,co+1)),int(rng.integers(1,height)),
              int(rng.integers(1,width)),channels,2,2)
    return model,x,shape,tile,kind

def run_conformance(*,cases:int=12,seed:int=42,output:str|Path|None=None,
                    max_iterations:int=10000)->dict:
    if type(cases) is not int or not 1<=cases<=200:
        raise ValueError("Cases must be 1..200")
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError("Seed must be unsigned 32-bit")
    if type(max_iterations) is not int or max_iterations<1:
        raise ValueError("Positive iteration bound required")
    rng=np.random.default_rng(seed);rows=[]
    for index in range(cases):
        model,x,shape,tile,kind=_model(rng,index)
        contract=ArchitectureRequest(precision_bits=16,integer_bits=6,
           parallelism=1+index%2,pes=1+index%3,simd=1+index%4,
           serialization=1+index%3,tile=tile,pipeline_ii=1+index%2)
        row={"case":index,"operator_kind":kind,"iteration_shape":list(shape),
             "architecture":{**asdict(contract),"tile":list(contract.tile)},
             "evidence":"E0/E1-and-gcc-host-float-only"}
        try:
            arch=resolve(model,ArchitectureRequest(),{"compute":contract})
            mapping=verify_mapping(arch.points[0],max_iterations=max_iterations)
            source=emit_hls(lower(arch))
            host=validate_host(model,source,x)
            row.update(status="PASS" if host["accepted"] else "NUMERICAL_FAIL",
                       semantic_iterations=mapping["iterations"],cpp_sha256=host["cpp_sha256"],
                       input_sha256=host["input_sha256"],rmse=host["rmse"],
                       max_absolute_error=host["max_absolute_error"],cosine=host["cosine"])
        except Exception as exc:
            row.update(status="ERROR",error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    result={"schema":"fpgai-research/bounded-host-conformance/v1","seed":seed,
            "cases":cases,"passed":sum(r["status"]=="PASS" for r in rows),
            "failed":sum(r["status"]!="PASS" for r in rows),
            "native_tool_execution":False,"fixed_point_bit_exact":False,"results":rows}
    if output is not None:
        dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result
