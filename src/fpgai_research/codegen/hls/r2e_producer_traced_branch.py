"""Separate R2e emitted C++: runtime producer reduction contribution witnesses.

R2d is immutable. This prospective source variant adds a runtime
CONTRIBUTE record after EACH actual input*weight accumulation in its
H-producing C++ loop, plus a separate producer-completion marker after
the real H scratch store. The independent R2e verifier checks event
completeness AND separate numerical FunctionalModel output.

Source traces have no FPGA cycle, PE concurrency or BRAM meaning.
"""
from __future__ import annotations

from hashlib import sha256

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2d_loop_driven_branch import emit_loop_driven_cpp
from fpgai_research.model_ir.branch_matmul_contract import admit_branch

SCHEMA="fpgai-research/cst-R2e-actual-producer-contributions-loop-source/v1"


def emit_producer_contribution_cpp(arch,*,offsets:dict[str,int],
                                   max_events:int=100_000)->Emission:
    (m,k,n,q),(h,a,b,g),_weights,_ops=admit_branch(arch.functional)
    # Use a separate, already certified runtime LOOP lowering, not the
    # historical R2b straight-line prewritten event source.
    original=emit_loop_driven_cpp(arch,offsets=offsets,
                                  max_events=max_events)
    src=original.cpp
    previous=f"      for(int kk=0;kk<{k};++kk) sum+=input[row*{k}+kk]*w0[kk*{n}+col];"
    replacement="\n".join([
        f"      for(int kk=0;kk<{k};++kk) {{",
        f"        const int x_address=row*{k}+kk;",
        f"        const int weight_address=kk*{n}+col;",
        "        sum+=input[x_address]*w0[weight_address];",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2D)",
        '        std::printf("FPGAI_PROD CONTRIBUTE %d %d %d %d\\n",',
        "                    idx,kk,x_address,weight_address);",
        "#endif",
        "      }",
    ])
    if src.count(previous)!=1:
        raise ValueError("Independent R2e source producer loop is not uniquely recognized")
    src=src.replace(previous,replacement,1)
    old="\n".join([
        f"      scratch[{offsets[h]}+idx]=sum;",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2D)",
        f'      std::printf("FPGAI_R2B COMMIT {h} %d\\n",idx);',
    ])
    new="\n".join([
        f"      scratch[{offsets[h]}+idx]=sum;",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2D)",
        '      std::printf("FPGAI_PROD COMMIT %d\\n",idx);',
        f'      std::printf("FPGAI_R2B COMMIT {h} %d\\n",idx);',
        '      std::printf("FPGAI_PROD VALUE %d %.9g\\n",idx,double(sum));',
    ])
    if src.count(old)!=1:
        raise ValueError("Independent producer commit instrumentation not uniquely located")
    src=src.replace(old,new,1)
    return Emission(cpp=src,manifest={
        **original.manifest,
        "schema":SCHEMA,
        "cpp_sha256":sha256(src.encode()).hexdigest(),
        "base_r2d_source_cpp_sha256":original.manifest["cpp_sha256"],
        "expected_producer_H_outputs":m*n,
        "expected_producer_contributions":m*k*n,
        "expected_intermediate_H_numeric_values":m*n,
        "source_producer_reduction_semantics":"each row*column commits after exactly k runtime kk contributions",
        "original_GCC_calls_during_emission":0,
        "original_vendor_calls_during_emission":0,
        "evidence":"E0-new-producer-traced-runtime-loop-source-await-GCC",
        "limitations":[
            "New R2e instrumented source is not original R2d source or data",
            "Printed contribution adjacent to an operand multiply is not bit-exact proof of arithmetic",
            "Independent NumPy and separate typed producer source oracle required",
            "No FPGA hardware-cycle producer schedule, multi-clock, RAM port or physical validation",
        ],
    })
