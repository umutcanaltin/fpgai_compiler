"""L6a bounded graph-wide source-lowered MatMul/Relu/Add DAG.

All generated MatMul loops call the original shared source emitter with
per-operator Phi, not a shadow Python approximate lowering. A full source
trace emits every produced value and each intermediate read. The validation
module independently executes and checks that evidence against E0.
"""
from __future__ import annotations

from hashlib import sha256

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.row_interleaved import _emit_row_matmul
from fpgai_research.model_ir.graph_phi_contract import (
    admit_graph,graph_contract,graph_disjoint_offsets,OUT_SINK,
)
from fpgai_research.model_ir.lowering import lower

SCHEMA="fpgai-research/cst-graph-phi-hls-source-e1/v1"


def emit_graph_hls(arch,*,offsets=None,max_events:int=100_000):
    model,produced=admit_graph(arch)
    offsets=(graph_disjoint_offsets(arch) if offsets is None else offsets)
    certificate=graph_contract(arch,offsets,max_events=max_events)
    from fpgai_research.model_ir.graph_edge_witness import edge_witness_receipt
    edge_witness=edge_witness_receipt(arch,offsets,max_events=max_events)
    lowered=lower(arch)
    fmt={(p.effective.precision_bits,p.effective.integer_bits) for p in arch.points}
    bits,integer=next(iter(fmt))
    symbols={op.outputs[0]:f"tensor_{i}" for i,op in enumerate(model.operators)}
    src=[
        "// L6a bounded graph Phi source; original HLS/RTL not qualified.",
        "#ifdef FPGAI_HOST_FLOAT","#include <cmath>","#include <cstdio>",
        "using data_t=float;",
        "#else","#include <ap_fixed.h>",
        f"using data_t=ap_fixed<{bits},{integer},AP_RND,AP_SAT>;",
        "#endif",
    ]
    for ordinal,op in enumerate(model.operators):
        if op.kind=="MatMul":
            values=model.tensor(op.inputs[1]).data
            src.append(
                f"static const data_t weights_{ordinal}[{len(values)}] = {{"+
                ", ".join(f"data_t({v:.9g})" for v in values)+"};")
    x=model.inputs[0]
    output=model.outputs[0]
    xin=model.tensor(x)
    out_shape=model.tensor(output).shape
    xlen=xin.shape[0]*xin.shape[1]
    outlen=out_shape[0]*out_shape[1]
    src.append(f'extern "C" void fpgai_model(const data_t input[{xlen}], data_t output[{outlen}]) {{')
    src.append(f"  data_t scratch[{certificate['source_array_elements']}];")
    for op in model.operators:
        name=op.outputs[0]
        src.append(f"  data_t* {symbols[name]}=scratch+{offsets[name]};")
    def ptr(name):
        return "input" if name==x else symbols[name]
    for ordinal,op in enumerate(model.operators):
        name=op.outputs[0]
        rows,width=model.tensor(name).shape
        src.append(f"  // {op.name}: {op.kind}; explicit operator Phi if MatMul")
        if op.kind=="MatMul":
            k=model.tensor(op.inputs[0]).shape[1]
            lhs=op.inputs[0]
            src.append(f"  for(int mi=0;mi<{rows};++mi) {{")
            _emit_row_matmul(
                src,ordinal=ordinal,lo=lowered.operators[ordinal],
                input_array=ptr(lhs),weight_array=f"weights_{ordinal}",
                output_array=ptr(name),input_k=k,output_n=width,
                lhs_is_row_buffer=lhs!=x,lhs_is_full_matrix=lhs!=x,
                output_is_row_buffer=False,
                fanout_commit_tensor=name,
                fanout_read_tensor=lhs if lhs in produced else None,
                fanout_read_consumer=op.name if lhs in produced else None,
                tie_fanout_trace_to_load=lhs in produced,
            )
            src.append("  }")
        elif op.kind in ("Relu","Add","Concat"):
            src.append(f"  for(int idx=0;idx<{rows*width};++idx) {{")
            if op.kind=="Concat":
                src.append(f"    const int col=idx%{width};")
                src.append("    data_t value=data_t(0);")
                cursor=0
                for port,name_in in enumerate(op.inputs):
                    source_width=model.tensor(name_in).shape[1]
                    src.append(
                        ("    if" if port==0 else "    else if")+
                        f" (col<{cursor+source_width}) {{")
                    src.append(
                        f"      const int source_index=idx/{width}*{source_width}+(col-{cursor});")
                    src.append(f"      value={ptr(name_in)}[source_index];")
                    if name_in in produced:
                        src.extend([
                            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_BRANCH)",
                            f'      std::printf("FPGAI_BRANCH READ {name_in} %d {op.name} %d\\n",source_index,idx);',
                            "#endif",
                        ])
                    src.append("    }")
                    cursor+=source_width
            else:
                for port,name_in in enumerate(op.inputs):
                    ir,ic=model.tensor(name_in).shape
                    # Explicit per-input 2D broadcast source address. Do not
                    # read idx directly when that input is broadcast.
                    source_expr=f"(idx/{width}*{ic if ir!=1 else 0}+(idx%{width})*{1 if ic!=1 else 0})"
                    src.append(f"    const int source_index_{port}={source_expr};")
                    src.append(f"    data_t value_{port}={ptr(name_in)}[source_index_{port}];")
                    if name_in in produced:
                        src.extend([
                            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_BRANCH)",
                            f'    std::printf("FPGAI_BRANCH READ {name_in} %d {op.name} %d\\n",source_index_{port},idx);',
                            "#endif",
                        ])
                if op.kind=="Relu":
                    src.append("    data_t value=value_0<data_t(0)?data_t(0):value_0;")
                else:
                    src.append("    data_t value=value_0+value_1;")
            src.append(f"    {ptr(name)}[idx]=value;")
            src.extend([
                "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_FANOUT)",
                f'    std::printf("FPGAI_FANOUT COMMIT {name} %d\\n",idx);',
                "#endif",
            ])
            src.append("  }")
    src.extend([
        f"  for(int idx=0;idx<{outlen};++idx) {{",
        f"    data_t value={ptr(output)}[idx];",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_BRANCH)",
        f'    std::printf("FPGAI_BRANCH READ {output} %d {OUT_SINK} %d\\n",idx,idx);',
        "#endif",
        "    output[idx]=value;",
        "  }","}",
    ])
    cpp="\n".join(src)+"\n"
    return Emission(cpp,{
        "schema":SCHEMA,
        "functional_sha256":certificate["functional_sha256"],
        "architecture_sha256":certificate["architecture_sha256"],
        "cpp_sha256":sha256(cpp.encode()).hexdigest(),
        "source_event_sha256":certificate["source_event_sha256"],
        "last_use_sha256":certificate["last_use_sha256"],
        "edge_witness_sha256":edge_witness["source_witness_sha256"],
        "edge_witnesses":edge_witness["edges"],
        "independent_phi_issue_status":certificate["independent_phi_issue_status"],
        "independent_phi_issue_checked_reads":certificate["independent_phi_issue_checked_reads"],
        "independent_phi_issue_checked_operators":certificate["independent_phi_issue_checked_operators"],
        "independent_phi_issue_signature_sha256":certificate["independent_phi_issue_signature_sha256"],
        "operator_phi_sha256":certificate["operator_phi_sha256"],
        "operator_phi_iterations":certificate["operator_phi_iterations"],
        "offsets":dict(sorted(offsets.items())),
        "source_array_elements":certificate["source_array_elements"],
        "event_count":certificate["commits"]+certificate["reads"],
        "operator_count":certificate["operator_count"],
        "edge_count":certificate["edge_count"],
        "native_vitis_calls":0,
        "evidence":"E1-HLS-compatible-source-only; GCC trace separately required",
        "limitations":[
            "Static rank-two MatMul(const RHS), scalar Relu/Add, sequential source",
            "No original Vitis CSim/CSynth, hardware dataflow, bank-port or RTL proof",
        ],
    })
