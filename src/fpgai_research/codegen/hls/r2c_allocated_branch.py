"""CST R2c: actual C++ source materialization of certified per-value slots.

A distinct NEW research source backend: original R2b C++/GCC evidence is
untouched. The declared static scratch pool stores each source tensor
element at an exact, potentially aliased address justified by finite
last-reader lifetimes. Every real READ/COMMIT memory operation prints
the same live C++ address variable used by scratch[slot], plus the
historical typed READ/COMMIT witness.

This is serialized host C++, not a hardware bank/port, clock-cycle,
on-chip RAM allocation or universal arbitrary tensor-DAG capability.
"""
from __future__ import annotations

from hashlib import sha256
import json

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.branch_matmul import disjoint_branch_offsets
from fpgai_research.model_ir.branch_matmul_contract import (
    admit_branch, validate_branch_events, _hash,
)
from fpgai_research.model_ir.element_scratch_allocation import (
    allocate_minimum_slots, verify_allocation,
)
from fpgai_research.model_ir.interleaved_branch_schedule import interleaved_branch_events
from fpgai_research.model_ir.semantics import architecture_fingerprint,functional_fingerprint


SCHEMA="fpgai-research/cst-R2c-per-value-allocated-C++-e1/v1"


def emit_r2c_allocated_cpp(arch,*,value_slots:dict[tuple[str,int],int]|None=None,
                           max_events:int=4096)->Emission:
    model=arch.functional
    (m,k,n,q),(h,a,b,g),weights,ops=admit_branch(model)
    if (type(max_events) is not int or
            not 1<=max_events<=4096):
        raise ValueError("R2c source compiler event bound must be 1..4096")
    producer=arch.points[0].effective
    if (producer.parallelism,producer.pes,producer.simd,producer.serialization)!=(
            1,1,1,1) or producer.tile or producer.pipeline_ii is not None:
        raise ValueError("R2c only admits explicitly scalar mapped producer")
    plan=interleaved_branch_events(arch,max_events=max_events)
    from fpgai_research.model_ir.independent_phi_issue_reference import validate_phi_issue_conformance
    independent=validate_phi_issue_conformance(arch,plan.events)
    if independent["status"]!="VERIFIED_E0":
        raise ValueError("Independent mapped Phi issue order refused source: "+str(independent))
    # Pure tensor/read-order legality is established separately, *before*
    # applying source element-level alias addresses.
    witness=validate_branch_events(
        arch,disjoint_branch_offsets(arch),plan.events,max_events=max_events)
    if witness["status"]!="VERIFIED_E0":
        raise ValueError("Uncertified mapped producer/two-reader event sequence")
    canonical=allocate_minimum_slots(plan.events,max_events=max_events)
    assigned=(canonical.slot_map() if value_slots is None else value_slots)
    if type(assigned) is not dict:
        raise ValueError("Exact source value map must be a dict")
    if any(type(slot) is not int or slot<0 for slot in assigned.values()):
        raise ValueError("Nonnegative integer original scratch source slots required")
    slots=max(assigned.values(),default=-1)+1
    check=verify_allocation(plan.events,assigned,unit_slots=slots)
    if check["status"]!="EXACT_SOURCE_SLOT_REPLAY_VERIFIED":
        raise AssertionError("Pre-emission value liveness replay not verified")
    fmt={(point.effective.precision_bits,point.effective.integer_bits)
         for point in arch.points}
    if len(fmt)!=1:
        raise ValueError("R2c does not silently cast mixed numerical precisions")
    bits,integer=next(iter(fmt))
    lines=[
        "// Prospective R2c bounded per-element scratch source, certified E0.",
        "// No HLS DATAFLOW, independent bank/port arbitration or timing claim.",
        "#ifdef FPGAI_HOST_FLOAT",
        "#include <cstdio>",
        "using data_t=float;",
        "#else",
        "#include <ap_fixed.h>",
        f"using data_t=ap_fixed<{bits},{integer},AP_RND,AP_SAT>;",
        "#endif",
    ]
    for i,wn in enumerate(weights):
        weights_data=model.tensor(wn).data
        lines.append(f"static const data_t weight_{i}[{len(weights_data)}] = {{"+
                     ", ".join(f"data_t({v:.9g})" for v in weights_data)+"};")
    lines +=[
        f'extern "C" void fpgai_model(const data_t input[{m*k}], data_t output[{m*q}]) {{',
        f"  data_t scratch[{slots}]={{}};",
        f"  data_t acc_a[{m*q}]={{}};",
        f"  data_t acc_b[{m*q}]={{}};",
        f"  data_t add_a[{m*q}]={{}};",
        f"  data_t add_b[{m*q}]={{}};",
    ]

    def trace(ev,i:int):
        slot_var=f"r2c_slot_{i}"
        idx_var=f"r2c_idx_{i}"
        lines.append("#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2C)")
        lines.append(
            f'  std::printf("FPGAI_SLOT {ev.kind} {ev.tensor} %d %d\\n",'
            f'{idx_var},{slot_var});')
        if ev.kind=="COMMIT":
            lines.append(
                f'  std::printf("FPGAI_R2B COMMIT {ev.tensor} %d\\n",{idx_var});')
        else:
            owner=ev.phi_owner if ev.phi_owner is not None else (-1,)*4
            lines.append(
                f'  std::printf("FPGAI_R2B READ {ev.tensor} %d '
                f'{ev.consumer} {ev.output_element} '+
                " ".join(str(x) for x in owner)+
                r'\n",'+idx_var+");")
        lines.append("#endif")

    for i,ev in enumerate(plan.events):
        tensor,element=ev.tensor,ev.element
        slot=assigned[(tensor,element)]
        lines.append(f"  const int r2c_idx_{i}={element};")
        lines.append(f"  const int r2c_slot_{i}={slot};")
        ref=f"scratch[r2c_slot_{i}]"
        if ev.kind=="COMMIT":
            if tensor==h:
                row,ni=divmod(element,n)
                lines +=[
                    f"  {{ data_t r2c_sum_{i}=data_t(0);",
                    f"    for(int kk=0;kk<{k};++kk) "
                    f"r2c_sum_{i}+=input[{row*k}+kk]*weight_0[kk*{n}+{ni}];",
                    f"    {ref}=r2c_sum_{i};",
                    "  }",
                ]
            elif tensor in (a,b):
                acc="acc_a" if tensor==a else "acc_b"
                lines.append(f"  {ref}={acc}[r2c_idx_{i}];")
            elif tensor==g:
                lines.append(
                    f"  {ref}=add_a[r2c_idx_{i}]+add_b[r2c_idx_{i}];")
            else:
                raise AssertionError("Unrecognized committed tensor in mapped CST")
        elif ev.kind=="READ":
            lines.append(f"  data_t r2c_value_{i}={ref};")
            if tensor==h:
                row,ni=divmod(ev.output_element,q)
                kk=element%n
                if element//n!=row:
                    raise AssertionError("Invalid Phi-conditioned source element")
                if ev.consumer==ops[1].name:
                    source_acc,source_weight="acc_a","weight_1"
                elif ev.consumer==ops[2].name:
                    source_acc,source_weight="acc_b","weight_2"
                else:
                    raise AssertionError("Unsupported mapped reader")
                lines.append(
                    f"  {source_acc}[{ev.output_element}]+="
                    f"r2c_value_{i}*{source_weight}[{kk*q+ni}];")
            elif tensor in (a,b):
                tmp="add_a" if tensor==a else "add_b"
                lines.append(f"  {tmp}[r2c_idx_{i}]=r2c_value_{i};")
            elif tensor==g:
                lines.append(
                    f"  output[r2c_idx_{i}]=r2c_value_{i}>data_t(0)?"
                    f"r2c_value_{i}:data_t(0);")
            else:
                raise AssertionError("Unsupported scratch source READ tensor")
        else:
            raise AssertionError("Unrecognized typed event")
        trace(ev,i)
    lines.append("}")
    cpp="\n".join(lines)+"\n"
    addresses=tuple(sorted((tensor,element,slot)
                            for (tensor,element),slot in assigned.items()))
    allocation_sha=sha256(json.dumps(
        addresses,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return Emission(cpp=cpp,manifest={
        "schema":SCHEMA,
        "functional_sha256":functional_fingerprint(model),
        "architecture_sha256":architecture_fingerprint(arch),
        "cpp_sha256":sha256(cpp.encode()).hexdigest(),
        "source_event_sha256":witness["source_order_event_sha256"],
        "source_lifetime_certificate_sha256":canonical.schedule_sha256,
        "source_allocation_sha256":allocation_sha,
        "source_value_slots":list(addresses),
        "source_scratch_unit_slots":slots,
        "source_minimum_unit_slots":canonical.unit_slots,
        "source_peak_live_intermediates":canonical.peak_simultaneous_live_values,
        "source_allocation_is_optimal_unit_slots":slots==canonical.unit_slots,
        "separate_accumulator_and_join_array_elements":4*m*q,
        "total_declared_mutable_source_array_elements":slots+4*m*q,
        "source_events":len(plan.events),
        "source_read_events":witness["read_events"],
        "source_commit_events":witness["commit_events"],
        "input_shape":[m,k],"output_shape":[m,q],
        "original_gcc_calls_during_emission":0,
        "original_vendor_calls_during_emission":0,
        "evidence":"E0-generated-per-element-source-await-original-GCC-execution",
        "limitations":[
            "Unit scalar value slots; no RAM port/bank/word packing or physical clock semantics",
            "Independent accumulators/add arrays are counted in declared mutable source ARRAY elements; scalar C++ local temporaries are not counted",
            "Source optimality only for fixed total source event order and unit-size scratch values",
            "Source compiler supports only bounded three MatMul diamond; no arbitrary DAG",
        ],
    })
