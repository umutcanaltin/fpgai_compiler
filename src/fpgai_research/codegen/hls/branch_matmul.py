"""Restricted E1 HLS C++ source for *independently mapped* MatMul fan-out.

Three MatMul operators share one immutable Functional DAG but have distinct
requested/effective Phi contracts. Real C++ MatMul loops use the same shared
emitter as sequential row-chain experiments. This is HOST SOURCE evidence,
not original Vitis CSim/CSynth, DATAFLOW timing or physical BRAM.
"""
from __future__ import annotations

from hashlib import sha256

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.row_interleaved import _emit_row_matmul
from fpgai_research.model_ir.branch_matmul_contract import (
    admit_branch, branch_contract,
)
from fpgai_research.model_ir.contracts import ArchitecturalModel
from fpgai_research.model_ir.lowering import lower

SCHEMA="fpgai-research/cst-multi-matmul-branch-source-e1/v1"


def disjoint_branch_offsets(arch:ArchitecturalModel,*,reuse_after_last_read:bool=False):
    (m,k,n,q),(h,a,b,g),weights,ops=admit_branch(arch.functional)
    # g is not live until both independent readers have completed, and all
    # h contributions have been consumed; g can alias the h base.
    return {h:0,a:m*n,b:m*n+m*q,
            g:0 if reuse_after_last_read else m*n+2*m*q}


def emit_branch_matmul_hls(arch: ArchitecturalModel,*,offsets:dict[str,int],
                          max_events:int=100_000)->Emission:
    proof=branch_contract(arch,offsets,max_events=max_events)
    model=arch.functional
    (m,k,n,q),(h,a,b,g),weights,ops=admit_branch(model)
    lo=lower(arch)
    widths={(point.effective.precision_bits,point.effective.integer_bits)
             for point in arch.points}
    bits,integer=next(iter(widths))
    scratch=proof["source_scratch_slots"]
    lines=[
        "// Source-only E1; mapped MatMul producer and TWO independently mapped readers.",
        "// All memories are sequential scratch; no HLS DATAFLOW or RAM port proof.",
        "#ifdef FPGAI_HOST_FLOAT",
        "#include <cmath>","#include <cstdio>",
        "using data_t=float; // not fixed-point equivalent",
        "#else","#include <ap_fixed.h>",
        f"using data_t=ap_fixed<{bits},{integer},AP_RND,AP_SAT>;",
        "#endif",
    ]
    for idx,weight in enumerate(weights):
        data=model.tensor(weight).data
        literals=", ".join(f"data_t({v:.9g})" for v in data)
        lines.append(f"static const data_t w{idx}[{len(data)}] = {{{literals}}};")
    lines.extend([
        f'extern "C" void fpgai_model(const data_t input[{m*k}], data_t output[{m*q}]) {{',
        f"  data_t scratch[{scratch}];",
    ])
    for name in (h,a,b,g):
        lines.append(f"  data_t* scratch_{name}=scratch+{offsets[name]};")
    for stage,(out_name,inn,wn,inner,out_width,reader) in enumerate((
        (h,"input","w0",k,n,None),
        (a,f"scratch_{h}","w1",n,q,ops[1].name),
        (b,f"scratch_{h}","w2",n,q,ops[2].name),
    )):
        lines.append(f"  // Phi-mapped MatMul stage {stage}: {ops[stage].name}")
        lines.append(f"  for (int mi=0; mi<{m}; ++mi) {{")
        _emit_row_matmul(
            lines,ordinal=stage,lo=lo.operators[stage],
            input_array=inn,weight_array=wn,
            output_array=f"scratch_{out_name}",input_k=inner,
            output_n=out_width,lhs_is_row_buffer=stage>0,
            lhs_is_full_matrix=stage>0,
            output_is_row_buffer=False,
            fanout_commit_tensor=out_name,
            fanout_read_tensor=h if stage>0 else None,
            fanout_read_consumer=reader,
        )
        lines.append("  }")
    lines.extend([
        f"  for(int idx=0;idx<{m*q};++idx) {{",
        f"    data_t va=scratch_{a}[idx];",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_BRANCH)",
        f'    std::printf("FPGAI_BRANCH READ {a} %d {ops[3].name} %d\\n",idx,idx);',
        "#endif",
        f"    data_t vb=scratch_{b}[idx];",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_BRANCH)",
        f'    std::printf("FPGAI_BRANCH READ {b} %d {ops[3].name} %d\\n",idx,idx);',
        "#endif",
        f"    scratch_{g}[idx]=va+vb;",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_FANOUT)",
        f'    std::printf("FPGAI_FANOUT COMMIT {g} %d\\n",idx);',
        "#endif",
        "  }",
        f"  for(int idx=0;idx<{m*q};++idx) {{",
        f"    data_t value=scratch_{g}[idx];",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_BRANCH)",
        f'    std::printf("FPGAI_BRANCH READ {g} %d {ops[4].name} %d\\n",idx,idx);',
        "#endif",
        "    output[idx]=value<data_t(0)?data_t(0):value;",
        "  }",
        "}",
    ])
    cpp="\n".join(lines)+"\n"
    manifest={
        "schema":SCHEMA,
        "functional_sha256":proof["functional_sha256"],
        "architecture_sha256":proof["architecture_sha256"],
        "cpp_sha256":sha256(cpp.encode()).hexdigest(),
        "logical_event_sha256":proof["source_order_event_sha256"],
        "last_use_sha256":proof["last_use_sha256"],
        "phi_iteration_sha256":proof["phi_iteration_sha256"],
        "phi_iterations":proof["phi_iterations"],
        "scratch_offsets":dict(sorted(offsets.items())),
        "scratch_elements":scratch,
        "disjoint_source_slots":m*n+3*m*q,
        "mapped_h_reads":proof["actual_required_h_reads"],
        "input_shape":[m,k],"output_shape":[m,q],
        "precision_bits":bits,"integer_bits":integer,
        "native_vitis_calls":0,
        "evidence":"E1-source-generation-only; independent GCC trace required",
        "limitations":[
            "Restricted three-MatMul, five-op diamond; not general DAG lowering",
            "Independent producer/reader Phi source lowering, host source order only",
            "No fixed-point CSim, bank ports, HLS runtime cycles or physical QoR",
        ],
    }
    return Emission(cpp,manifest)
