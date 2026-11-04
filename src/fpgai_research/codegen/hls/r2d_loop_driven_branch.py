"""New R2d F1 experiment: runtime-decoded Phi reader loops (not event unrolling).

Unlike R2b/R2c's per-event C++ literal-index statements, this source
computes all H READ indices and tensor output positions dynamically in
C++ from a runtime loop counter, PE-group width and per-reader mapping.
Each actual scratch-array load is witnessed using its runtime subscript.

The independent Python Phi-issue reference does NOT generate the source
loop body. This remains serialized HOST C++: no physical PE parallelism,
HLS cycle schedule, synthesized ports or proof of scientific firstness.
The original R2b/R2c emitter and case artifacts are immutable.
"""
from __future__ import annotations

from hashlib import sha256

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.model_ir.branch_matmul_contract import (
    admit_branch, validate_branch_events,
)
from fpgai_research.model_ir.independent_phi_issue_reference import (
    validate_phi_issue_conformance,
)
from fpgai_research.model_ir.interleaved_branch_schedule import interleaved_branch_events
from fpgai_research.model_ir.semantics import architecture_fingerprint,functional_fingerprint

SCHEMA="fpgai-research/cst-R2d-loop-driven-Phi-source-e1/v1"


def emit_loop_driven_cpp(arch,*,offsets:dict[str,int],
                         max_events:int=100_000)->Emission:
    (m,k,n,q),(h,a,b,g),weights,ops=admit_branch(arch.functional)
    producer=arch.points[0].effective
    if (producer.parallelism,producer.pes,producer.simd,producer.serialization)!=(1,1,1,1) \
            or producer.tile or producer.pipeline_ii is not None:
        raise ValueError("R2d producer is scalar-only in this source prototype")
    readers=[arch.points[i].effective for i in (1,2)]
    for cfg in readers:
        if (cfg.parallelism!=1 or cfg.serialization!=1 or cfg.tile
                or cfg.pipeline_ii is not None or not 1<=cfg.pes<=q
                or not 1<=cfg.simd<=n):
            raise ValueError("R2d reader source prototype supports bounded PE/SIMD only")
    if any((p.effective.precision_bits,p.effective.integer_bits)!=(
            producer.precision_bits,producer.integer_bits) for p in arch.points):
        raise ValueError("R2d requires one exact source numeric precision")
    events=interleaved_branch_events(arch,max_events=max_events).events
    independent=validate_phi_issue_conformance(arch,events)
    if independent["status"]!="VERIFIED_E0":
        raise ValueError("R2d independent source issue policy failed")
    validity=validate_branch_events(arch,offsets,events,max_events=max_events)
    if validity["status"]!="VERIFIED_E0":
        raise ValueError("R2d source scratch memory is not certified: "+str(validity))
    total=validity["source_scratch_slots"]
    bits,integer=producer.precision_bits,producer.integer_bits
    c=[
        "// New R2d actual runtime-index loop-driven C++; not literal source-event unrolling.",
        "// No hardware-cycle, PE concurrency, RAM banking or II claim.",
        "#ifdef FPGAI_HOST_FLOAT",
        "#include <cstdio>",
        "using data_t=float;",
        "#else",
        "#include <ap_fixed.h>",
        f"using data_t=ap_fixed<{bits},{integer},AP_RND,AP_SAT>;",
        "#endif",
        "",
        "// Independent execution algorithm: invert logical issue rank into a",
        "// PE group, reduction step and output index at C++ *runtime*.",
        "// The Python spec uses a closed-form forward rank, not this loop.",
        "static void r2d_decode_issue(int ordinal, int width, int n, int q,",
        "                             int *row, int *col, int *reduction) {",
        "  const int per_row=n*q;",
        "  *row=ordinal/per_row;",
        "  int within=ordinal%per_row;",
        "  for(int group=0;group<q;group+=width) {",
        "    const int active=(group+width<=q)?width:q-group;",
        "    const int block=active*n;",
        "    if(within<block) {",
        "      *reduction=within/active;",
        "      *col=group+within%active;",
        "      return;",
        "    }",
        "    within-=block;",
        "  }",
        "  *col=-1; *reduction=-1; // unreachable for admitted shape",
        "}",
    ]
    for i,w in enumerate(weights):
        vals=arch.functional.tensor(w).data
        c.append(
            f"static const data_t w{i}[{len(vals)}] = {{"+
            ", ".join(f"data_t({float(value):.9g})" for value in vals)+
            "};")
    c += [
        f'extern "C" void fpgai_model(const data_t input[{m*k}], data_t output[{m*q}]) {{',
        f"  data_t scratch[{total}]={{}};",
        f"  data_t acc_a[{m*q}]={{}};",
        f"  data_t acc_b[{m*q}]={{}};",
        f"  data_t add_a[{m*q}]={{}};",
        f"  data_t add_b[{m*q}]={{}};",
        f"  int count_a[{m*q}]={{}};",
        f"  int count_b[{m*q}]={{}};",
        f"  for(int row=0;row<{m};++row) {{",
        f"    for(int col=0;col<{n};++col) {{",
        "      const int idx=row*"+str(n)+"+col;",
        "      data_t sum=data_t(0);",
        f"      for(int kk=0;kk<{k};++kk) sum+=input[row*{k}+kk]*w0[kk*{n}+col];",
        f"      scratch[{offsets[h]}+idx]=sum;",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2D)",
        f'      std::printf("FPGAI_R2B COMMIT {h} %d\\n",idx);',
        "#endif",
        "    }",
        "  }",
        f"  for(int ordinal=0;ordinal<{m*n*q};++ordinal) {{",
    ]
    for stage,tensor,acc,counter in (
            (0,a,"acc_a","count_a"),(1,b,"acc_b","count_b")):
        spec=readers[stage]
        source_name=ops[stage+1].name
        prefix="r0" if stage==0 else "r1"
        c += [
            "    {",
            f"      int row=-1,col=-1,reduction=-1;",
            f"      r2d_decode_issue(ordinal,{spec.pes},{n},{q},&row,&col,&reduction);",
            f"      if(row<0 || row>={m} || col<0 || col>={q} || reduction<0 || reduction>={n}) return;",
            f"      const int source_idx=row*{n}+reduction;",
            f"      const int output_idx=row*{q}+col;",
            f"      const int source_addr={offsets[h]}+source_idx;",
            f"      data_t v=scratch[source_addr];",
            f"      {acc}[output_idx]+=v*w{stage+1}[reduction*{q}+col];",
            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2D)",
            f'      std::printf("FPGAI_R2B READ {h} %d {source_name} %d 0 %d %d 0\\n",'
            f'source_idx,output_idx,col%{spec.pes},reduction%{spec.simd});',
            "#endif",
            f"      ++{counter}[output_idx];",
            f"      if({counter}[output_idx]=={n}) {{",
            f"        scratch[{offsets[tensor]}+output_idx]={acc}[output_idx];",
            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2D)",
            f'        std::printf("FPGAI_R2B COMMIT {tensor} %d\\n",output_idx);',
            "#endif",
            "      }",
            "    }",
        ]
    c += [
        "  }",
        f"  for(int idx=0;idx<{m*q};++idx) {{",
        f"    add_a[idx]=scratch[{offsets[a]}+idx];",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2D)",
        f'    std::printf("FPGAI_R2B READ {a} %d {ops[3].name} %d -1 -1 -1 -1\\n",idx,idx);',
        "#endif",
        f"    add_b[idx]=scratch[{offsets[b]}+idx];",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2D)",
        f'    std::printf("FPGAI_R2B READ {b} %d {ops[3].name} %d -1 -1 -1 -1\\n",idx,idx);',
        "#endif",
        f"    scratch[{offsets[g]}+idx]=add_a[idx]+add_b[idx];",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2D)",
        f'    std::printf("FPGAI_R2B COMMIT {g} %d\\n",idx);',
        "#endif",
        "  }",
        f"  for(int idx=0;idx<{m*q};++idx) {{",
        f"    const int source_addr={offsets[g]}+idx;",
        "    data_t v=scratch[source_addr];",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2D)",
        f'    std::printf("FPGAI_R2B READ {g} %d {ops[4].name} %d -1 -1 -1 -1\\n",idx,idx);',
        "#endif",
        "    output[idx]=v>data_t(0)?v:data_t(0);",
        "  }",
        "}",
    ]
    cpp="\n".join(c)+"\n"
    return Emission(cpp=cpp,manifest={
        "schema":SCHEMA,
        "cpp_sha256":sha256(cpp.encode()).hexdigest(),
        "functional_sha256":functional_fingerprint(arch.functional),
        "architecture_sha256":architecture_fingerprint(arch),
        "source_schedule":"runtime-ordinal-decoder-PE-group-interleaved-v1",
        "fixed_source_event_SHA256":validity["source_order_event_sha256"],
        "scratch_offsets":dict(sorted(offsets.items())),
        "scratch_unit_slots":total,
        "source_mutable_array_elements_excluding_count_arrays":total+4*m*q,
        "source_additional_count_array_elements":2*m*q,
        "source_all_declared_mutable_array_elements":total+6*m*q,
        "source_read_events":validity["read_events"],
        "source_commit_events":validity["commit_events"],
        "original_gcc_calls":0,"original_vitis_calls":0,
        "original_vivado_calls":0,
        "evidence":"E0-source-generated-runtime-index-loop-await-GCC-validation",
        "limitations":[
            "Runtime C++ rank decoder executes in serial on host: no FPGA PE concurrency",
            "Independent Python oracle is E0 and measures ordered source READs, not hardware timing",
            "Source arrays include two additional reduction completion counters",
            "H producer computes a full MatMul before readers start; no streaming overlap",
            "Only bounded static five-operator diamond, not arbitrary graph",
        ],
    })
