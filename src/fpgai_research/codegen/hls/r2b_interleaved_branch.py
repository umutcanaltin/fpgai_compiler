"""Bounded generated C++ for prospective CST R2b Phi-reader interleaving.

The emitted C++ executes real scratch READs/WRITEs in a deterministic
per-contribution sequence; printed events are adjacent to their actual
source operations. This does not implement physical parallel PE hardware,
hardware-cycle scheduling, real RAM-port arbitration or native HLS QoR.
"""
from __future__ import annotations

from hashlib import sha256

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.model_ir.branch_matmul_contract import (
    _hash, admit_branch, validate_branch_events,
)
from fpgai_research.model_ir.interleaved_branch_schedule import (
    interleaved_branch_events,
)
from fpgai_research.model_ir.semantics import (
    architecture_fingerprint, functional_fingerprint,
)

SCHEMA = "fpgai-research/cst-r2b-interleaved-emitted-source-e1/v1"


def emit_r2b_interleaved_cpp(arch, *, offsets: dict[str, int],
                              max_events: int = 100_000) -> Emission:
    """Emit source operations, not merely an E0 trace table.

    Exactly one print is adjacent to each original intermediate scratch
    load/store. A source-executed validator must separately parse,
    numerically verify and certify the actual output.
    """
    model = arch.functional
    (m, k, n, q), (h, a, b, g), weights, ops = admit_branch(model)
    # A separate producer-mapping lowering is required before supporting
    # non-scalar producer Phi; never silently map a non-scalar request.
    producer = arch.points[0].effective
    if (producer.parallelism, producer.pes, producer.simd,
            producer.serialization) != (1, 1, 1, 1) or (
            producer.tile or producer.pipeline_ii is not None):
        raise ValueError("R2b E1 scalar-only producer; no unchecked Phi lowering")
    plan = interleaved_branch_events(arch, max_events=max_events)
    from fpgai_research.model_ir.independent_phi_issue_reference import validate_phi_issue_conformance
    independent = validate_phi_issue_conformance(arch,plan.events)
    if independent["status"]!="VERIFIED_E0":
        raise ValueError("Independent mapped Phi issue order refused source: "+str(independent))
    receipt = validate_branch_events(arch, offsets, plan.events,
                                     max_events=max_events)
    if receipt["status"] != "VERIFIED_E0":
        raise ValueError("R2b emitted scratch unsafe: " + receipt["reason"] +
                         " " + str(receipt.get("counterexample")))
    sizes = {h: m*n, a: m*q, b: m*q, g: m*q}
    slots = receipt["source_scratch_slots"]
    fmt = {(p.effective.precision_bits, p.effective.integer_bits)
           for p in arch.points}
    if len(fmt) != 1:
        raise ValueError("R2b mixed fixed-point formats not source supported")
    bits, integer = next(iter(fmt))
    lines = [
        "// R2b prospective static C++ E1: real source-order PE-grouped reads.",
        "// Not hardware parallelism, HLS DATAFLOW timing, board, bank or port proof.",
        "#ifdef FPGAI_HOST_FLOAT",
        "#include <cstdio>",
        "using data_t=float;",
        "#else",
        "#include <ap_fixed.h>",
        f"using data_t=ap_fixed<{bits},{integer},AP_RND,AP_SAT>;",
        "#endif",
    ]
    for i, wn in enumerate(weights):
        values = model.tensor(wn).data
        lines.append(f"static const data_t weight_{i}[{len(values)}] = {{"+
                     ", ".join(f"data_t({float(v):.9g})" for v in values)+"};")
    lines += [
        f'extern "C" void fpgai_model(const data_t input[{m*k}], data_t output[{m*q}]) {{',
        f"  data_t scratch[{slots}]={{}};",
        f"  data_t acc_a[{m*q}]={{}};",
        f"  data_t acc_b[{m*q}]={{}};",
        f"  data_t add_a[{m*q}]={{}};",
        f"  data_t add_b[{m*q}]={{}};",
    ]
    def trace(event, *, source_index: str | None = None):
        lines.append("#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_R2B)")
        if event.kind == "COMMIT":
            lines.append(
                f'  std::printf("FPGAI_R2B COMMIT {event.tensor} {event.element}\\n");')
        else:
            if source_index is None:
                raise AssertionError("READ witness must use actual runtime source-index variable")
            owner = event.phi_owner if event.phi_owner is not None else (-1,)*4
            lines.append(
                f'  std::printf("FPGAI_R2B READ {event.tensor} %d '
                f'{event.consumer} {event.output_element} ' +
                " ".join(str(x) for x in owner) + r'\n", '+source_index+");")
        lines.append("#endif")

    for i, ev in enumerate(plan.events):
        tensor, element = ev.tensor, ev.element
        phys = offsets[tensor] + element
        if ev.kind == "COMMIT":
            if tensor == h:
                mi, ni = divmod(element, n)
                lines += [
                    f"  {{ data_t value_{i}=data_t(0);",
                    f"    for(int kk=0;kk<{k};++kk) "
                    f"value_{i}+=input[{mi*k}+kk]*weight_0[kk*{n}+{ni}];",
                    f"    scratch[{phys}]=value_{i};",
                    "  }",
                ]
            elif tensor in (a, b):
                source = "acc_a" if tensor == a else "acc_b"
                lines.append(f"  scratch[{phys}]={source}[{element}];")
            elif tensor == g:
                lines.append(
                    f"  scratch[{phys}]=add_a[{element}]+add_b[{element}];")
            else:
                raise AssertionError("Unexpected intermediate COMMIT")
            trace(ev)
        elif ev.kind == "READ":
            if tensor == h:
                mi, ni = divmod(ev.output_element, q)
                kk = element % n
                if element // n != mi:
                    raise AssertionError("Source reader geometry mismatch")
                stage = 0 if ev.consumer == ops[1].name else 1
                if ev.consumer != ops[stage+1].name:
                    raise AssertionError("Unsupported source reader name")
                accum = "acc_a" if stage == 0 else "acc_b"
                lines.append(f"  const int r2b_index_{i}={element};")
                lines.append(
                    f"  data_t r2b_value_{i}=scratch[{offsets[h]}+r2b_index_{i}];")
                lines.append(
                    f"  {accum}[{ev.output_element}]+="
                    f"r2b_value_{i}*weight_{stage+1}[{kk*q+ni}];")
            elif tensor in (a, b):
                dst = "add_a" if tensor == a else "add_b"
                lines.append(f"  const int r2b_index_{i}={element};")
                lines.append(
                    f"  {dst}[{element}]=scratch[{offsets[tensor]}+r2b_index_{i}];")
            elif tensor == g:
                lines.append(f"  const int r2b_index_{i}={element};")
                lines.append(
                    f"  data_t r2b_value_{i}=scratch[{offsets[g]}+r2b_index_{i}];")
                lines.append(
                    f"  output[{element}]=r2b_value_{i}>data_t(0)?"
                    f"r2b_value_{i}:data_t(0);")
            else:
                raise AssertionError("Unexpected intermediate READ")
            trace(ev, source_index=f"r2b_index_{i}")
        else:
            raise AssertionError("Unrecognized source event")
    lines.append("}")
    cpp = "\n".join(lines)+"\n"
    return Emission(cpp=cpp, manifest={
        "schema": SCHEMA,
        "functional_sha256": functional_fingerprint(model),
        "architecture_sha256": architecture_fingerprint(arch),
        "cpp_sha256": sha256(cpp.encode()).hexdigest(),
        "planned_source_event_sha256": receipt["source_order_event_sha256"],
        "planned_logical_value_last_use_sha256": receipt["last_use_sha256"],
        "planned_events": len(plan.events),
        "planned_h_reads": receipt["actual_required_h_reads"],
        "scratch_offsets": dict(sorted(offsets.items())),
        "scratch_slots": slots,
        "schedule": plan.source_schedule,
        "input_shape": [m, k], "output_shape": [m, q],
        "source_evidence": "E0-generated-real-scratch-operations-await-GCC-execution",
        "original_vendor_calls": 0,
        "limitations": [
            "Source operations are serialized, even when a Phi owner denotes PE lanes",
            "The E1 GCC trace must independently verify each actual source load/store",
            "No native ap_fixed CSim, CSynth, RAM-port timing or physical routing",
        ],
    })
