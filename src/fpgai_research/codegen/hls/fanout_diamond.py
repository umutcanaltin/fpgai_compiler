"""Restricted actual HLS C++ lowering of a two-reader diamond (E1 source).

Precisely supported graph: MatMul(static weight) -> two parallel Relu
operators -> Add -> Relu. All intermediate values are static MxN floats
(M,N>1 permitted) and emitted as source-addressed scratch accesses.
Independent spatial Phi schedules, arbitrary DAGs, mixed precision,
simultaneous HLS dataflow, and bank-port arbitration are NOT implemented.

The E0 alias checker runs on the EXACT sequential source event order, not on
an assumed whole-stage atomic schedule. No vendor/native tools are invoked.
"""
from __future__ import annotations

from hashlib import sha256
import re

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.model_ir.contracts import FunctionalModel, ArchitecturalModel
from fpgai_research.model_ir.fanout_lifetime import (
    LifetimeEvent, verify_fanout_events,
)
from fpgai_research.model_ir.semantics import functional_fingerprint

SCHEMA = "fpgai-research/cst-fanout-diamond-hls-source-e1/v1"


def _admit(model: FunctionalModel):
    if not isinstance(model, FunctionalModel):
        raise TypeError("Typed FunctionalModel required")
    model.validate()
    if len(model.operators) != 5 or len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError("Only five-op, one-input, one-output fan-out diamond is lowered")
    producer, reader0, reader1, join, final = model.operators
    if tuple(op.kind for op in model.operators) != (
            "MatMul", "Relu", "Relu", "Add", "Relu"):
        raise ValueError("Unsupported fan-out operator sequence")
    if any(op.attributes for op in model.operators):
        raise ValueError("Fan-out diamond does not implement operator attributes")
    if len(model.tensors) != 7:
        raise ValueError("Unexpected extra tensors in restricted diamond")
    x = model.inputs[0]
    weight = producer.inputs[1] if len(producer.inputs) == 2 else None
    h, a, b, g, output = (
        producer.outputs[0], reader0.outputs[0], reader1.outputs[0],
        join.outputs[0], final.outputs[0])
    if (not weight or producer.inputs != (x, weight)
            or reader0.inputs != (h,) or reader1.inputs != (h,)
            or join.inputs != (a, b) or final.inputs != (g,)
            or model.outputs != (output,)):
        raise ValueError("Unsupported fan-out DAG wiring/ports")
    dims = model.tensor(x).shape
    w = model.tensor(weight)
    if len(dims) != 2 or len(w.shape) != 2:
        raise ValueError("Only static 2D MatMul input and weight supported")
    m, k = dims
    wk, n = w.shape
    if wk != k or w.data is None:
        raise ValueError("Matrix weight must be static and match inner dimension")
    if any(model.tensor(t).shape != (m, n) for t in (h,a,b,g,output)):
        raise ValueError("All diamond intermediates/output must have matching MxN shape")
    if any(model.tensor(t).data is not None for t in (x,h,a,b,g,output)):
        raise ValueError("Only the matrix weight may be a constant")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", s)
           for s in (h,a,b,g,producer.name,reader0.name,reader1.name,
                     join.name,final.name)):
        raise ValueError("Restricted source trace requires safe C identifiers")
    return (m,k,n), (h,a,b,g), (reader0.name,reader1.name,join.name,final.name), w


def source_order_events(model: FunctionalModel) -> tuple[LifetimeEvent,...]:
    """Explicit per-element order executed by this restricted source emitter."""
    (m,_k,n),(h,a,b,g),(read0,read1,add,last),_ = _admit(model)
    count=m*n
    events=[]
    for idx in range(count):
        events.append(LifetimeEvent("COMMIT",h,idx))
    for idx in range(count):
        events.extend((LifetimeEvent("CONSUME",h,idx,read0),
                       LifetimeEvent("COMMIT",a,idx)))
    for idx in range(count):
        events.extend((LifetimeEvent("CONSUME",h,idx,read1),
                       LifetimeEvent("COMMIT",b,idx)))
    for idx in range(count):
        events.extend((LifetimeEvent("CONSUME",a,idx,add),
                       LifetimeEvent("CONSUME",b,idx,add),
                       LifetimeEvent("COMMIT",g,idx)))
    for idx in range(count):
        events.append(LifetimeEvent("CONSUME",g,idx,last))
    return tuple(events)


def fanout_diamond_offsets(model: FunctionalModel, *, share_released: bool) -> dict[str,int]:
    """Two explicitly named experimental layouts; not an optimality oracle."""
    (m,_k,n),(h,a,b,g),_readers,_weight = _admit(model)
    extent=m*n
    return {
        h:0, a:extent, b:2*extent,
        g:0 if share_released else 3*extent,
    }


def emit_fanout_diamond_hls(
    model: FunctionalModel,
    *,
    offsets: dict[str,int],
    precision_bits: int = 16,
    integer_bits: int = 6,
    max_events: int = 20_000,
    architecture: ArchitecturalModel | None = None,
) -> Emission:
    (m,k,n),(h,a,b,g),(read0,read1,join,last),weight = _admit(model)
    if (type(offsets) is not dict or set(offsets) != {h,a,b,g} or
            any(type(v) is not int or v < 0 for v in offsets.values())):
        raise ValueError("Every intermediate requires one nonnegative scratch base")
    if (type(precision_bits) is not int or type(integer_bits) is not int or
            not 1 <= integer_bits <= precision_bits <= 64):
        raise ValueError("Invalid fixed-point precision")
    extent=m*n
    slots=max(offsets.values())+extent
    events=source_order_events(model)
    mapped_contract = None
    if architecture is not None:
        if not isinstance(architecture, ArchitecturalModel) or architecture.functional != model:
            raise ValueError("Mapped Phi architecture must bind the exact Functional graph")
        from fpgai_research.model_ir.fanout_phi_source import (
            mapped_fanout_contract, mapped_source_order_events,
        )
        mapped_contract=mapped_fanout_contract(architecture,offsets,max_events=max_events)
        events=mapped_source_order_events(architecture,max_events=max_events)
        if (precision_bits,integer_bits) != (
                architecture.points[0].effective.precision_bits,
                architecture.points[0].effective.integer_bits):
            raise ValueError("Emitted fixed-point precision differs from mapped Phi")
        proof={"status":"VERIFIED","event_sha256":mapped_contract["source_event_sha256"],
               "events_checked":len(events)}
    else:
        proof=verify_fanout_events(model,offsets,events,max_events=max_events)
        if proof["status"] != "VERIFIED":
            raise ValueError("SOURCE_ALIAS_UNSAFE: " + str(proof["reason"]) +
                             " " + str(proof.get("counterexample")))
    lit=", ".join(f"data_t({v:.9g})" for v in weight.data)
    src=[
        "// E1 restricted five-op CST diamond: no concurrent DATAFLOW or HLS timing claim.",
        "#ifdef FPGAI_HOST_FLOAT",
        "#include <cmath>", "#include <cstdio>",
        "using data_t = float; // not fixed-point bit exact",
        "#else", "#include <ap_fixed.h>",
        f"using data_t = ap_fixed<{precision_bits}, {integer_bits}, AP_RND, AP_SAT>;",
        "#endif",
        f"static const data_t weights[{k*n}] = {{{lit}}};",
        f'extern "C" void fpgai_model(const data_t input[{m*k}], data_t output[{extent}]) {{',
        f"  data_t scratch[{slots}];",
    ]
    if architecture is None:
        # Keep the historical unmapped emitter byte-identical for archived
        # source receipts and backwards-compatible source trace experiments.
        src.extend([
            f"  for (int mi=0; mi<{m}; ++mi) {{",
            f"    for (int ni=0; ni<{n}; ++ni) {{",
            "      data_t sum = data_t(0);",
            f"      for (int ki=0; ki<{k}; ++ki) sum += input[mi*{k}+ki]*weights[ki*{n}+ni];",
            f"      const int idx = mi*{n}+ni;",
            f"      scratch[{offsets[h]}+idx] = sum;",
            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_FANOUT)",
            f'      std::printf("FPGAI_FANOUT COMMIT {h} %d\\n",idx);',
            "#endif",
            "    }",
            "  }",
        ])
    else:
        from fpgai_research.model_ir.lowering import lower
        from fpgai_research.codegen.hls.row_interleaved import _emit_row_matmul
        resolved=lower(architecture)
        src.extend([
            f"  data_t* scratch_h = scratch + {offsets[h]};",
            f"  for (int mi=0; mi<{m}; ++mi) {{",
        ])
        _emit_row_matmul(
            src,ordinal=0,lo=resolved.operators[0],
            input_array="input",weight_array="weights",
            output_array="scratch_h",input_k=k,output_n=n,
            lhs_is_row_buffer=False,output_is_row_buffer=False,
            fanout_commit_tensor=h,
        )
        src.append("  }")
    for tensor,consumer,target in ((h,read0,a),(h,read1,b)):
        src.extend([
            f"  for (int idx=0; idx<{extent}; ++idx) {{",
            f"    data_t value = scratch[{offsets[tensor]}+idx];",
            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_FANOUT)",
            f'    std::printf("FPGAI_FANOUT CONSUME {tensor} %d {consumer}\\n",idx);',
            "#endif",
            "    if (value < data_t(0)) value = data_t(0);",
            f"    scratch[{offsets[target]}+idx] = value;",
            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_FANOUT)",
            f'    std::printf("FPGAI_FANOUT COMMIT {target} %d\\n",idx);',
            "#endif",
            "  }",
        ])
    src.extend([
        f"  for (int idx=0; idx<{extent}; ++idx) {{",
        f"    data_t va = scratch[{offsets[a]}+idx];",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_FANOUT)",
        f'    std::printf("FPGAI_FANOUT CONSUME {a} %d {join}\\n",idx);',
        "#endif",
        f"    data_t vb = scratch[{offsets[b]}+idx];",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_FANOUT)",
        f'    std::printf("FPGAI_FANOUT CONSUME {b} %d {join}\\n",idx);',
        "#endif",
        f"    scratch[{offsets[g]}+idx] = va + vb;",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_FANOUT)",
        f'    std::printf("FPGAI_FANOUT COMMIT {g} %d\\n",idx);',
        "#endif",
        "  }",
        f"  for (int idx=0; idx<{extent}; ++idx) {{",
        f"    data_t value = scratch[{offsets[g]}+idx];",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_FANOUT)",
        f'    std::printf("FPGAI_FANOUT CONSUME {g} %d {last}\\n",idx);',
        "#endif",
        "    output[idx] = value < data_t(0) ? data_t(0) : value;",
        "  }",
        "}",
    ])
    cpp="\n".join(src)+"\n"
    return Emission(cpp,{
        "schema":SCHEMA,
        "functional_sha256":functional_fingerprint(model),
        "cpp_sha256":sha256(cpp.encode()).hexdigest(),
        "intermediate_offsets":dict(sorted(offsets.items())),
        "scratch_elements":slots,
        "full_disjoint_scratch_elements":4*extent,
        "input_shape":[m,k],"output_shape":[m,n],
        "source_e0_event_sha256":proof["event_sha256"],
        "source_e0_events_checked":len(events),
        "mapped_phi_enabled":architecture is not None,
        "producer_phi_iteration_sha256":(
            mapped_contract["producer_phi_iteration_sha256"]
            if mapped_contract is not None else None),
        "producer_phi_iteration_count":(
            mapped_contract["producer_iteration_count"]
            if mapped_contract is not None else None),
        "architecture_sha256":(
            mapped_contract["architecture_sha256"]
            if mapped_contract is not None else None),
        "precision_bits":precision_bits,
        "integer_bits":integer_bits,
        "supported_graph":"static MatMul -> (Relu,Relu) -> Add -> Relu",
        "evidence":"E1-source-generated-only",
        "native_vitis_executed":False,
        "hardware_memory_proven":False,
        "limitations":[
            "Restricted five-op diamond, no generic branching lowering",
            "Phi-conditioned producer only when explicit architecture supplied; two scalar readers",
            "No simultaneous bank access or native HLS/RTL guarantee",
        ],
    })
