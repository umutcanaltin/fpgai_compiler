"""Deterministic E1 experiment comparing source-loop work to portable Phi.

This is a bounded structural consistency experiment, NOT an RTL or Vitis
schedule validation. Independent loop traversal lives in the architecture
materialization oracle, separate from the coordinate mapper.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from fpgai_research.architecture.materialization_oracle import verify_source_mapping
from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.contracts import ArchitectureRequest, Tensor, Operator, FunctionalModel
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import functional_fingerprint, verify_refinement
from fpgai_research.validation.cpp_phi_trace import trace_cpp_phi


def _case(rng, kind):
    m = int(rng.integers(1, 4))
    n = int(rng.integers(1, 9))
    k = int(rng.integers(1, 9))
    if kind in ('MatMul','Gemm'):
        tensors=(Tensor('x',(m,k)),Tensor('w',(k,n),data=(.125,)*(k*n)),Tensor('y',(m,n)))
        op=Operator('compute',kind,('x','w'),('y',))
        shape=(m,n,k)
        tile=(int(rng.integers(1,m+1)),int(rng.integers(1,n+1)),int(rng.integers(1,k+1)))
    else:
        channels=int(rng.integers(1,5))
        height=int(rng.integers(2,6))
        width=int(rng.integers(2,6))
        out_channels=int(rng.integers(1,6))
        tensors=(Tensor('x',(m,channels,height,width)),
                 Tensor('w',(out_channels,channels,2,2),data=(.125,)*(out_channels*channels*4)),
                 Tensor('y',(m,out_channels,height-1,width-1)))
        op=Operator('compute','Conv',('x','w'),('y',))
        shape=(m,out_channels,height-1,width-1,channels,2,2)
        tile=(m,int(rng.integers(1,out_channels+1)),
              int(rng.integers(1,height)),int(rng.integers(1,width)),channels,2,2)
    model=FunctionalModel(tensors,(op,),('x',),('y',))
    req=ArchitectureRequest(parallelism=int(rng.integers(1,4)),
        pes=int(rng.integers(1,4)), simd=int(rng.integers(1,7)),
        serialization=int(rng.integers(1,5)),tile=tile,
        pipeline_ii=int(rng.integers(1,4)))
    return model,req,shape


def run_materialization_study(*, seed=2026, cases=96, output=None, host_cpp=False):
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('Seed must be a 32-bit nonnegative integer')
    if type(cases) is not int or not 1<=cases<=1024:
        raise ValueError('Cases must be 1..1024')
    if type(host_cpp) is not bool:
        raise ValueError('host_cpp must be boolean')
    rng=np.random.default_rng(seed)
    records=[]
    for index in range(cases):
        kind=('MatMul','Gemm','Conv')[index%3]
        model,request,shape=_case(rng,kind)
        architect=resolve(model,ArchitectureRequest(),{'compute':request})
        low=lower(architect)
        refinement=verify_refinement(low)
        source=verify_source_mapping(architect.points[0],low.operators[0],max_iterations=100_000)
        record={'index':index,'kind':kind,'iteration_shape':list(shape),
            'architecture':{k:list(v) if isinstance(v,tuple) else v for k,v in vars(request).items()},
            'functional_sha256':functional_fingerprint(model),
            'stage_preservation':refinement['stage_invariants_verified'],
            'source_phi_agreement':source['source_phi_spatial_and_fold_agree'],
            'coverage':source['coverage_and_uniqueness'],
            'iterations':source['logical_iterations']}
        if host_cpp:
            trace=trace_cpp_phi(low,max_iterations=100_000)
            record['host_cpp_trace_pass']=trace['source_phi_agreement']
            record['host_cpp_sha256']=trace['cpp_sha256']
            record['host_cpp_trace_sha256']=trace['trace_sha256']
        records.append(record)
    result={'schema':'fpgai-research/source-phi-materialization-study/v1',
      'seed':seed,'cases':cases,
      'matmul_cases':sum(r['kind']=='MatMul' for r in records),
      'gemm_cases':sum(r['kind']=='Gemm' for r in records),
      'conv_cases':sum(r['kind']=='Conv' for r in records),
      'source_phi_agreement_passed':sum(r['source_phi_agreement'] for r in records),
      'stage_preservation_passed':sum(r['stage_preservation'] for r in records),
      'coverage_passed':sum(r['coverage'] for r in records),
      'evidence':'E0/E1-independent-source-loop-model',
      'vitis_executed':False,'rtl_verified':False,
      'host_cpp_trace_executed':host_cpp,
      'host_cpp_trace_passed':sum(r.get('host_cpp_trace_pass',False) for r in records) if host_cpp else None,
      'limitations':'Bounded independent Python model of HLS source loops; not parsed C++, Vitis execution or physical evidence',
      'records':records}
    if output is not None:
        path=Path(output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result
