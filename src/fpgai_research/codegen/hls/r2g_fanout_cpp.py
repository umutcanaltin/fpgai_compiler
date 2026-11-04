"""CST R2g: runtime-loop C++ for 2..4 fanout MatMul source graphs.

Distinct NEW source backend. Dynamic ordinal-to-(row,col,kk) C++ loop
computes each reader's H load subscript, with a runtime contribution
witness and actual intermediate-value witness after each COMMIT.
A separately implemented Python forward-rank oracle verifies the
resulting READ order. This generator does not unroll Python events
into C++ statements or modify the frozen R2b/R2e original cohorts.

Host GCC/float32 is not Vitis HLS, actual PE concurrency, physical
BRAM memory reuse, RTL, bit-exact ap_fixed, or arbitrary DAG support.
"""
from __future__ import annotations

from hashlib import sha256

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.model_ir.fanout_contract import (
    admit_fanout, disjoint_fanout_offsets,
)
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint, functional_fingerprint,
)

SCHEMA="fpgai-research/cst-R2g-runtime-fanout-C++-e1/v1"


def emit_fanout_cpp(arch,*,offsets:dict[str,int]|None=None)->Emission:
    spec=admit_fanout(arch)
    if offsets is None:
        offsets=disjoint_fanout_offsets(spec)
    legal=disjoint_fanout_offsets(spec)
    if offsets!=legal:
        # Avoid claiming a general aliased-source compiler before source
        # liveness of all compiler-created C++ intermediates is verified.
        raise ValueError("R2g source backend admits only certified disjoint layout")
    m,k,n,q=spec.m,spec.k,spec.n,spec.q
    r=len(spec.readers)
    model=arch.functional
    fmt={(p.effective.precision_bits,p.effective.integer_bits)
         for p in arch.points}
    if len(fmt)!=1:
        raise ValueError("R2g mixed numeric precision requires explicit lowering")
    bits,integer=next(iter(fmt))
    total=legal[spec.intermediates[-1]]+m*q
    c=[
        "// CST R2g: dynamic C++ ordinal decoder for 2..4 fanout readers.",
        "// Source events, not clock/BRAM-port/RTL execution semantics.",
        "#include <cstdio>",
        "#ifdef FPGAI_HOST_FLOAT",
        "using data_t=float;",
        "#else",
        "#include <ap_fixed.h>",
        f"using data_t=ap_fixed<{bits},{integer},AP_RND,AP_SAT>;",
        "#endif",
        "static void r2g_decode_issue(int ordinal,int width,int n,int q,",
        "                             int *row,int *col,int *kk) {",
        "  *row=ordinal/(n*q);",
        "  int within=ordinal%(n*q);",
        "  for(int start=0;start<q;start+=width) {",
        "    const int active=start+width<=q?width:q-start;",
        "    const int block=active*n;",
        "    if(within<block) {",
        "      *kk=within/active;",
        "      *col=start+within%active;",
        "      return;",
        "    }",
        "    within-=block;",
        "  }",
        "  *col=-1; *kk=-1;",
        "}",
    ]
    for wi,name in enumerate(spec.weights):
        data=model.tensor(name).data
        c.append(
            f"static const data_t w{wi}[{len(data)}] = {{"+
            ", ".join(f"data_t({float(v):.9g})" for v in data)+"};")
    c +=[
        f'extern "C" void fpgai_model(const data_t input[{m*k}],data_t output[{m*q}]) {{',
        f"  data_t scratch[{total}]={{}};",
        f"  data_t acc[{r}][{m*q}]={{}};",
        f"  int count[{r}][{m*q}]={{}};",
        f"  for(int row=0;row<{m};++row) {{",
        f"    for(int col=0;col<{n};++col) {{",
        f"      const int h_idx=row*{n}+col;",
        "      data_t sum=data_t(0);",
        f"      for(int kk=0;kk<{k};++kk) {{",
        f"        const int x_idx=row*{k}+kk;",
        f"        const int w_idx=kk*{n}+col;",
        "        sum+=input[x_idx]*w0[w_idx];",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2G)",
        f'        std::printf("FPGAI_R2G CONTRIBUTE {spec.h} %d {spec.producer} -1 %d %d %d\\n",',
        "                    h_idx,kk,x_idx,w_idx);",
        "#endif",
        "      }",
        f"      scratch[{legal[spec.h]}+h_idx]=sum;",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2G)",
        f'      std::printf("FPGAI_R2G COMMIT {spec.h} %d\\n",h_idx);',
        f'      std::printf("FPGAI_R2G VALUE {spec.h} %d %.9g\\n",h_idx,double(sum));',
        "#endif",
        "    }",
        "  }",
        f"  for(int ordinal=0;ordinal<{m*n*q};++ordinal) {{",
    ]
    for i,(reader,value) in enumerate(zip(spec.readers,spec.reader_values)):
        e=arch.points[i+1].effective
        c +=[
            "    {",
            "      int row=-1,col=-1,kk=-1;",
            f"      r2g_decode_issue(ordinal,{e.pes},{n},{q},&row,&col,&kk);",
            f"      if(row<0 || row>={m} || col<0 || col>={q} || kk<0 || kk>={n}) return;",
            f"      const int h_idx=row*{n}+kk;",
            f"      const int output_idx=row*{q}+col;",
            f"      const int h_addr={legal[spec.h]}+h_idx;",
            "      data_t h_val=scratch[h_addr];",
            f"      acc[{i}][output_idx]+=h_val*w{i+1}[kk*{q}+col];",
            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2G)",
            f'      std::printf("FPGAI_R2G READ {spec.h} %d {reader} %d 0 %d %d 0\\n",',
            f"                  h_idx,output_idx,col%{e.pes},kk%{e.simd});",
            "#endif",
            f"      ++count[{i}][output_idx];",
            f"      if(count[{i}][output_idx]=={n}) {{",
            f"        scratch[{legal[value]}+output_idx]=acc[{i}][output_idx];",
            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2G)",
            f'        std::printf("FPGAI_R2G COMMIT {value} %d\\n",output_idx);',
            f'        std::printf("FPGAI_R2G VALUE {value} %d %.9g\\n",',
            f"                    output_idx,double(acc[{i}][output_idx]));",
            "#endif",
            "      }",
            "    }",
        ]
    c.append("  }")
    current=spec.reader_values[0]
    for i,(merge,value) in enumerate(zip(spec.add_ops,spec.add_values)):
        next_value=spec.reader_values[i+1]
        c +=[
            f"  for(int idx=0;idx<{m*q};++idx) {{",
            f"    const int left_address={legal[current]}+idx;",
            f"    const int right_address={legal[next_value]}+idx;",
            "    const data_t left=scratch[left_address];",
            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2G)",
            f'    std::printf("FPGAI_R2G READ {current} %d {merge} %d -1 -1 -1 -1\\n",idx,idx);',
            "#endif",
            "    const data_t right=scratch[right_address];",
            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2G)",
            f'    std::printf("FPGAI_R2G READ {next_value} %d {merge} %d -1 -1 -1 -1\\n",idx,idx);',
            "#endif",
            "    const data_t sum=left+right;",
            f"    scratch[{legal[value]}+idx]=sum;",
            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2G)",
            f'    std::printf("FPGAI_R2G COMMIT {value} %d\\n",idx);',
            f'    std::printf("FPGAI_R2G VALUE {value} %d %.9g\\n",idx,double(sum));',
            "#endif",
            "  }",
        ]
        current=value
    c +=[
        f"  for(int idx=0;idx<{m*q};++idx) {{",
        f"    const int source_addr={legal[current]}+idx;",
        "    const data_t v=scratch[source_addr];",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2G)",
        f'    std::printf("FPGAI_R2G READ {current} %d {spec.final} %d -1 -1 -1 -1\\n",idx,idx);',
        "#endif",
        "    output[idx]=v>data_t(0)?v:data_t(0);",
        "  }",
        "}",
    ]
    cpp="\n".join(c)+"\n"
    return Emission(cpp=cpp,manifest={
        "schema":SCHEMA,
        "source_cpp_sha256":sha256(cpp.encode()).hexdigest(),
        "functional_sha256":functional_fingerprint(model),
        "architecture_sha256":architecture_fingerprint(arch),
        "fanout":r,"shape":[m,k,n,q],
        "source_offsets":dict(sorted(legal.items())),
        "source_scratch_array_elements":total,
        "source_accumulator_array_elements":r*m*q,
        "source_counter_array_elements":r*m*q,
        "source_all_declared_mutable_array_elements":total+2*r*m*q,
        "producer_contribution_count":m*k*n,
        "reader_H_read_count":r*m*n*q,
        "original_GCC_calls_during_emission":0,
        "original_native_tool_calls_during_emission":0,
        "runtime_C++_indices_computed":True,
        "evidence":"E0-dynamic-fanout-source-await-original-GCC",
        "limitations":[
            "Source disjoint layout only; not per-element aliased scratch",
            "Source loops are serialized, no true concurrent PE or hardware clock schedule",
            "Runtime values are float32, not bit-exact fixed-point HLS output",
            "No native RTL/BRAM or arbitrary graph support",
        ],
    })
