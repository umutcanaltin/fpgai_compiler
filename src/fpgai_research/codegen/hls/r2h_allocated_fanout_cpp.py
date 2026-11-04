"""R2h: actual certified per-value aliasing in dynamic fanout source C++.

Two source layouts (disjoint and exact unit-slot) use the SAME R2g
runtime ordinal reader algorithm and full intermediate arithmetic.
Every actual scratch LOAD/STORE uses the same C++ address variable
recorded immediately before its corresponding typed READ/COMMIT.

Source address tables are generated from the independently certified
R2f E0 event order and finite-value lifetime allocator. This is NOT
a byte/BRAM/port equivalence result or proof of hardware scheduling.
Historical R2g original source and artifacts remain immutable.
"""
from __future__ import annotations

from hashlib import sha256

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2g_fanout_cpp import emit_fanout_cpp
from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    project_source_value_events,
)
from fpgai_research.model_ir.element_scratch_allocation import (
    allocate_minimum_slots,verify_allocation,
)
from fpgai_research.model_ir.fanout_contract import (
    admit_fanout,disjoint_fanout_offsets,generate_fanout_events,
    verify_fanout_events,verify_fanout_issue,
)
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint,functional_fingerprint,
)

SCHEMA="fpgai-research/cst-R2h-dynamic-allocated-fanout-C++-e1/v1"
LAYOUTS=("disjoint","per_element_optimal")


def source_layout(arch,*,layout:str):
    if layout not in LAYOUTS:
        raise ValueError("R2h requires explicit disjoint or per_element_optimal layout")
    spec=admit_fanout(arch)
    events=generate_fanout_events(arch)
    disjoint=disjoint_fanout_offsets(spec)
    if verify_fanout_events(arch,disjoint,events)["status"]!="VERIFIED_E0":
        raise ValueError("Uncertified original fanout event source")
    if verify_fanout_issue(arch,events)["status"]!="VERIFIED_E0":
        raise ValueError("Uncertified original independent Phi source policy")
    projected=project_source_value_events(events)
    optimal=allocate_minimum_slots(projected)
    if layout=="per_element_optimal":
        assigned=optimal.slot_map()
        capacity=optimal.unit_slots
    else:
        sizes={tensor:spec.m*(spec.n if tensor==spec.h else spec.q)
               for tensor in spec.intermediates}
        assigned={(tensor,i):disjoint[tensor]+i
                  for tensor,size in sizes.items()
                  for i in range(size)}
        capacity=sum(sizes.values())
    verified=verify_allocation(projected,assigned,unit_slots=capacity)
    if verified["status"]!="EXACT_SOURCE_SLOT_REPLAY_VERIFIED":
        raise AssertionError("R2h pre-emission scratch alias safety failed")
    if optimal.unit_slots!=optimal.peak_simultaneous_live_values:
        raise AssertionError("R2h fixed-event optimality bound failed")
    return spec,assigned,capacity,optimal,projected


def emit_allocated_fanout_cpp(arch,*,layout:str)->Emission:
    spec,assigned,capacity,optimal,events=source_layout(arch,layout=layout)
    r=len(spec.readers)
    m,n,q=spec.m,spec.n,spec.q
    legal=disjoint_fanout_offsets(spec)
    generated=emit_fanout_cpp(arch)
    source=generated.cpp

    def one(old,new,*,count=1):
        nonlocal source
        actual=source.count(old)
        if actual!=count:
            raise ValueError(
                "R2h source lowering pattern changed: "+
                repr(old)+" expected "+str(count)+" got "+str(actual))
        source=source.replace(old,new,count)

    tables=[]
    for tensor in spec.intermediates:
        size=m*(n if tensor==spec.h else q)
        values=",".join(str(assigned[(tensor,i)]) for i in range(size))
        tables.append(f"static const int r2h_slot_{tensor}[{size}] = {{{values}}};")
    one('extern "C" void fpgai_model',
        "\n".join(tables)+'\nextern "C" void fpgai_model')
    one(f"  data_t scratch[{generated.manifest['source_scratch_array_elements']}]={{}};",
        f"  data_t scratch[{capacity}]={{}};")
    # The typed event line follows each actual scratch operation,
    # while FPGAI_SLOT binds the exact C++ runtime address.
    def witness(kind,tensor,index,address):
        return (
            f'      std::printf("FPGAI_SLOT {kind} {tensor} %d %d\\n",'
            f'{index},{address});\n'
        )
    # Producer H: write first, then source-address and typed COMMIT traces.
    one(f"      scratch[{legal[spec.h]}+h_idx]=sum;",
        f"      const int h_store_addr=r2h_slot_{spec.h}[h_idx];\n"
        f"      scratch[h_store_addr]=sum;")
    one(f'      std::printf("FPGAI_R2G COMMIT {spec.h} %d\\n",h_idx);',
        witness("COMMIT",spec.h,"h_idx","h_store_addr")+
        f'      std::printf("FPGAI_R2G COMMIT {spec.h} %d\\n",h_idx);')
    # Reader H address is computed inside each independently mapped
    # reader scope; same variable is printed and loaded.
    one(f"      const int h_addr={legal[spec.h]}+h_idx;",
        f"      const int h_addr=r2h_slot_{spec.h}[h_idx];",count=r)
    for i,(reader,tensor) in enumerate(zip(spec.readers,spec.reader_values)):
        original=f'      std::printf("FPGAI_R2G READ {spec.h} %d {reader} %d 0 %d %d 0\\n",'
        one(original,
            witness("READ",spec.h,"h_idx","h_addr")+original)
        one(f"        scratch[{legal[tensor]}+output_idx]=acc[{i}][output_idx];",
            f"        const int reader_store_addr=r2h_slot_{tensor}[output_idx];\n"
            f"        scratch[reader_store_addr]=acc[{i}][output_idx];")
        original=f'        std::printf("FPGAI_R2G COMMIT {tensor} %d\\n",output_idx);'
        one(original,
            witness("COMMIT",tensor,"output_idx","reader_store_addr")+original)
    current=spec.reader_values[0]
    for i,(merge,tensor) in enumerate(zip(spec.add_ops,spec.add_values)):
        right=spec.reader_values[i+1]
        one(f"    const int left_address={legal[current]}+idx;",
            f"    const int left_address=r2h_slot_{current}[idx];")
        one(f"    const int right_address={legal[right]}+idx;",
            f"    const int right_address=r2h_slot_{right}[idx];")
        original=f'    std::printf("FPGAI_R2G READ {current} %d {merge} %d -1 -1 -1 -1\\n",idx,idx);'
        one(original,witness("READ",current,"idx","left_address")+original)
        original=f'    std::printf("FPGAI_R2G READ {right} %d {merge} %d -1 -1 -1 -1\\n",idx,idx);'
        one(original,witness("READ",right,"idx","right_address")+original)
        one(f"    scratch[{legal[tensor]}+idx]=sum;",
            f"    const int add_store_addr=r2h_slot_{tensor}[idx];\n"
            f"    scratch[add_store_addr]=sum;")
        original=f'    std::printf("FPGAI_R2G COMMIT {tensor} %d\\n",idx);'
        one(original,witness("COMMIT",tensor,"idx","add_store_addr")+original)
        current=tensor
    one(f"    const int source_addr={legal[current]}+idx;",
        f"    const int source_addr=r2h_slot_{current}[idx];")
    original=f'    std::printf("FPGAI_R2G READ {current} %d {spec.final} %d -1 -1 -1 -1\\n",idx,idx);'
    one(original,witness("READ",current,"idx","source_addr")+original)

    return Emission(source,{
        **generated.manifest,
        "schema":SCHEMA,
        "base_R2g_source_cpp_sha256":generated.manifest["source_cpp_sha256"],
        "source_cpp_sha256":sha256(source.encode()).hexdigest(),
        "functional_sha256":functional_fingerprint(arch.functional),
        "architecture_sha256":architecture_fingerprint(arch),
        "source_layout":layout,
        "source_slot_map_sha256":sha256(repr(tuple(sorted(assigned.items()))).encode()).hexdigest(),
        "source_slot_map":tuple(sorted(
            (tensor,i,slot) for (tensor,i),slot in assigned.items())),
        "source_scratch_array_elements":capacity,
        "source_accumulator_array_elements":r*m*q,
        "source_counter_array_elements":r*m*q,
        "source_all_declared_mutable_array_elements":capacity+2*r*m*q,
        "source_optimal_unit_slots_for_fixed_order":optimal.unit_slots,
        "source_event_order_sha256":optimal.schedule_sha256,
        "runtime_address_witnesses_for_every_read_and_commit":True,
        "actual_optimized_HLS_BRAM_savings_established":False,
        "original_GCC_calls_during_emission":0,
        "limitations":[
            "Source-memory aliasing is checked only for one fixed serial source event order",
            "C++ scratch use is not RAM-bank, physical BRAM or clock scheduling evidence",
            "Other declared mutable C++ arrays are counted, local scalars excluded",
            "C++ source still bounded to MatMul fanout 2..4 with left Add chain and Relu",
        ],
    })
