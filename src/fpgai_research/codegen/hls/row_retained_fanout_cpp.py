"""First-paper bounded row-retained H materialization in the canonical R2g emitter.

R2g original is the *sole arithmetic and Phi ordinal decoder*. This
prospective opt-in lowering only changes the loop nest and scratch
addressing: one H row is produced, all its fanout readers use it in
their original Phi order, then its n slots may be reused for the next
row. Other intermediates, reader accumulators, and counters remain
materialized as in R2g. This is NOT general streaming, hardware
pipeline safety, an RTL port certificate, or proven BRAM reduction.

Original R2g/R2h emitters, original studies and their hashes are unchanged.
"""
from __future__ import annotations

from hashlib import sha256
import re

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.r2g_fanout_cpp import emit_fanout_cpp
from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    project_source_value_events,
)
from fpgai_research.model_ir.element_scratch_allocation import verify_allocation
from fpgai_research.model_ir.fanout_contract import (
    admit_fanout, disjoint_fanout_offsets, generate_fanout_events,
    verify_fanout_events, verify_fanout_issue,
)

SCHEMA = "fpgai-research/first-paper-row-retained-fanout-source/v1"


def row_retained_plan(arch) -> dict:
    """Prove the row staging schedule and its full READ/COMMIT slot map at E0.

    Existing R2f checks the logical event order with disjoint offsets.
    The existing independent unit-slot replay then checks actual row
    aliasing. This does not derive a real hardware-cycle schedule.
    """
    spec = admit_fanout(arch)
    m, k, n, q = spec.m, spec.k, spec.n, spec.q
    if m < 2:
        raise ValueError("Row retention experiment requires at least two rows")
    full = generate_fanout_events(arch)
    producer_length = m * n * (k + 1)
    producer = full[:producer_length]
    remainder = full[producer_length:]
    reader_end = next(
        (i for i, event in enumerate(remainder)
         if event.kind == "READ" and event.tensor != spec.h), None
    )
    if reader_end is None:
        raise ValueError("No source fanout consumer/merge boundary")
    reader_phase = remainder[:reader_end]
    merges = remainder[reader_end:]
    if (len(producer) != producer_length
            or any(e.kind not in ("CONTRIBUTE", "COMMIT") or e.tensor != spec.h
                   for e in producer)
            or not reader_phase
            or any(
                not ((e.kind == "READ" and e.tensor == spec.h)
                     or (e.kind == "COMMIT"
                         and e.tensor in spec.reader_values))
                for e in reader_phase
            )):
        raise ValueError("Unexpected canonical R2g source phases")

    rows = []
    for row in range(m):
        lo = row * n * (k + 1)
        hi = (row + 1) * n * (k + 1)
        produced = producer[lo:hi]
        consumed = tuple(
            e for e in reader_phase if (
                (e.kind == "READ" and e.element // n == row)
                or (e.kind == "COMMIT" and e.element // q == row)
            )
        )
        if (len(produced) != n * (k + 1)
                or sum(e.kind == "READ" for e in consumed)
                    != len(spec.readers) * n * q):
            raise ValueError("Row source fanout read multiplicity incomplete")
        rows.extend(produced)
        rows.extend(consumed)
    if tuple(e for e in rows if e.kind == "READ" and e.tensor == spec.h) != tuple(
            e for e in reader_phase if e.kind == "READ"):
        raise ValueError("Row staging changed original Phi reader order")
    if tuple(e for e in rows if e.kind == "COMMIT"
             and e.tensor in spec.reader_values) != tuple(
            e for e in reader_phase if e.kind == "COMMIT"):
        raise ValueError("Row staging changed reader completion sequence")
    source_events = tuple(rows) + merges
    if len(source_events) != len(full):
        raise ValueError("Row source event population changed")
    legal = disjoint_fanout_offsets(spec)
    safety = verify_fanout_events(arch, legal, source_events)
    issue = verify_fanout_issue(arch, source_events)
    if safety["status"] != "VERIFIED_E0" or issue["status"] != "VERIFIED_E0":
        raise ValueError("Row source trace failed existing E0 contract")
    shift = (m - 1) * n
    source_offsets = {tensor: (0 if tensor == spec.h else base - shift)
                      for tensor, base in legal.items()}
    capacities = {tensor: m * (n if tensor == spec.h else q)
                  for tensor in spec.intermediates}
    assigned = {
        (tensor, idx): (idx % n if tensor == spec.h
                        else source_offsets[tensor] + idx)
        for tensor, size in capacities.items() for idx in range(size)
    }
    scratch = sum(capacities.values()) - shift
    replay = verify_allocation(
        project_source_value_events(source_events), assigned,
        unit_slots=scratch,
    )
    if replay["status"] != "EXACT_SOURCE_SLOT_REPLAY_VERIFIED":
        raise ValueError("Row source memory lifetime reuse is not certified")
    return {
        "source_events": source_events,
        "source_event_sha256": safety["source_event_sha256"],
        "source_slot_map": assigned,
        "source_slot_map_sha256": sha256(
            repr(tuple(sorted(assigned.items()))).encode()).hexdigest(),
        "source_offsets": source_offsets,
        "source_scratch_array_elements": scratch,
        "source_full_scratch_array_elements": sum(capacities.values()),
        "source_H_row_buffer_elements": n,
        "source_H_full_elements": m*n,
        "source_saved_scratch_elements": shift,
        "source_reader_H_reads": len(spec.readers) * m*n*q,
        "source_contract": "E0_VALUE_AND_PHI_AND_EXACT_ROW_ALIAS_REPLAY",
        "original_vendor_calls": 0,
    }


def emit_row_retained_fanout_cpp(arch) -> Emission:
    """Derive a new source arm by checked changes to the canonical R2g.

    No second MatMul/Add arithmetic generator or issue-coordinate decoder.
    Source lowering is fail-closed when R2g changes and a pattern no
    longer matches. Every actual scratch load/store gets an adjacent
    host-only source address witness. Trace verification is separate.
    """
    spec = admit_fanout(arch)
    m, n, q = spec.m, spec.n, spec.q
    r = len(spec.readers)
    plan = row_retained_plan(arch)
    baseline = emit_fanout_cpp(arch)
    source = baseline.cpp
    old_offsets = disjoint_fanout_offsets(spec)
    new_offsets = plan["source_offsets"]
    if old_offsets[spec.h] != 0:
        raise ValueError("Unsupported producer offset in canonical fanout")

    def one(old: str, new: str, count: int = 1) -> None:
        nonlocal source
        found = source.count(old)
        if found != count:
            raise ValueError(
                f"Canonical R2g lowering changed: expected {count} "
                f"occurrences of {old!r}, found {found}"
            )
        source = source.replace(old, new)

    old_capacity = baseline.manifest["source_scratch_array_elements"]
    new_capacity = plan["source_scratch_array_elements"]
    one(f"  data_t scratch[{old_capacity}]={{}};",
        f"  data_t scratch[{new_capacity}]={{}};")

    # Fold the full producer pass and reader pass into a row-local
    # source schedule. All source reader ordinals stay in original order.
    one(
        f"  }}\n  for(int ordinal=0;ordinal<{m*n*q};++ordinal) {{",
        f"    for(int ordinal=row*{n*q};"
        f"ordinal<(row+1)*{n*q};++ordinal) {{",
    )
    # This marker appears before every merge and final ReLU loop.
    # Change ONLY the first occurrence (the ordinal-loop boundary).
    close_marker = f"  }}\n  for(int idx=0;idx<{m*q};++idx) {{"
    if source.count(close_marker) != len(spec.add_ops) + 1:
        raise ValueError("Canonical fanout merge-loop boundary changed")
    source = source.replace(
        close_marker,
        f"    }}\n  }}\n  for(int idx=0;idx<{m*q};++idx) {{",
        1,
    )

    one(
        "      scratch[0+h_idx]=sum;",
        f"      const int r4_h_store_addr=h_idx%{n};\n"
        "      scratch[r4_h_store_addr]=sum;",
    )
    one(
        "      const int h_addr=0+h_idx;",
        f"      const int h_addr=h_idx%{n};",
        count=r,
    )

    # All non-H tensors move down after replacing the old m*n H
    # capacity by one row. Substitute original offsets in one pass
    # to avoid colliding with newly introduced numeric offsets.
    pattern = re.compile(r"(?<![A-Za-z_0-9])([0-9]+)\+(output_idx|idx)\b")
    expected_bases = {old_offsets[t]: new_offsets[t]
                      for t in spec.intermediates if t != spec.h}
    touched = set()

    def shift_base(match: re.Match) -> str:
        base = int(match.group(1))
        if base not in expected_bases:
            return match.group(0)
        touched.add(base)
        return f"{expected_bases[base]}+{match.group(2)}"

    source = pattern.sub(shift_base, source)
    if touched != set(expected_bases):
        raise ValueError("R2g tensor offsets not all represented in source")

    def slot_witness(kind: str, tensor: str, index: str, address: str,
                     indent: str) -> str:
        return (f'{indent}std::printf("FPGAI_SLOT {kind} {tensor} %d %d\\n",'
                f'{index},{address});\n')

    # Scratch STORE witnesses and source COMMIT trace must use the same
    # actual address variable. Text is emitted only in trace-enabled host
    # mode and is not a vendor memory observation.
    one(
        f'      std::printf("FPGAI_R2G COMMIT {spec.h} %d\\n",h_idx);',
        slot_witness("COMMIT", spec.h, "h_idx", "r4_h_store_addr", "      ")
        + f'      std::printf("FPGAI_R2G COMMIT {spec.h} %d\\n",h_idx);',
    )
    for reader in spec.readers:
        old = f'      std::printf("FPGAI_R2G READ {spec.h} %d {reader} %d 0 %d %d 0\\n",'
        one(old, slot_witness("READ", spec.h, "h_idx", "h_addr", "      ") + old)
    for i, tensor in enumerate(spec.reader_values):
        base = new_offsets[tensor]
        name = f"r4_reader_store_addr_{i}"
        one(
            f"        scratch[{base}+output_idx]=acc[{i}][output_idx];",
            f"        const int {name}={base}+output_idx;\n"
            f"        scratch[{name}]=acc[{i}][output_idx];",
        )
        old = f'        std::printf("FPGAI_R2G COMMIT {tensor} %d\\n",output_idx);'
        one(old, slot_witness("COMMIT", tensor, "output_idx", name, "        ") + old)
    current = spec.reader_values[0]
    for i, (merge, tensor) in enumerate(zip(spec.add_ops, spec.add_values)):
        right = spec.reader_values[i+1]
        for read_tensor, address in ((current, "left_address"),
                                      (right, "right_address")):
            old = (f'    std::printf("FPGAI_R2G READ {read_tensor} '
                   f'%d {merge} %d -1 -1 -1 -1\\n",idx,idx);')
            one(old, slot_witness("READ", read_tensor, "idx",
                                  address, "    ") + old)
        base = new_offsets[tensor]
        name = f"r4_merge_store_addr_{i}"
        one(
            f"    scratch[{base}+idx]=sum;",
            f"    const int {name}={base}+idx;\n"
            f"    scratch[{name}]=sum;",
        )
        old = f'    std::printf("FPGAI_R2G COMMIT {tensor} %d\\n",idx);'
        one(old, slot_witness("COMMIT", tensor, "idx", name, "    ") + old)
        current = tensor
    old = (f'    std::printf("FPGAI_R2G READ {current} '
           f'%d {spec.final} %d -1 -1 -1 -1\\n",idx,idx);')
    one(old, slot_witness("READ", current, "idx", "source_addr", "    ") + old)
    # All remaining H access in source must use r4_h_store_addr/h_addr,
    # never a full-tensor offset. An accidental second old H index
    # is a fatal source-lowering mismatch.
    if re.search(r"scratch\[0\+h_idx\]|const int h_addr=0\+h_idx", source):
        raise ValueError("Untransformed H full tensor memory access")
    digest = sha256(source.encode()).hexdigest()
    manifest = {
        **baseline.manifest,
        "schema": SCHEMA,
        "base_R2g_source_cpp_sha256": baseline.manifest["source_cpp_sha256"],
        "source_cpp_sha256": digest,
        "materialization": "H_ROW_RETAINED_ALL_READERS_BEFORE_ROW_REUSE",
        "source_offsets": dict(sorted(new_offsets.items())),
        "source_slot_map_sha256": plan["source_slot_map_sha256"],
        "source_slot_map": tuple(sorted(
            (tensor, idx, address)
            for (tensor, idx), address in plan["source_slot_map"].items())),
        "source_event_order_sha256": plan["source_event_sha256"],
        "source_contract": plan["source_contract"],
        "source_scratch_array_elements": new_capacity,
        "source_all_declared_mutable_array_elements":
            new_capacity + 2 * r * m * q,
        "source_H_row_buffer_elements": n,
        "source_H_full_elements": m * n,
        "source_saved_scratch_elements": plan["source_saved_scratch_elements"],
        "runtime_address_witnesses_for_every_read_and_commit": True,
        "original_GCC_calls_during_emission": 0,
        "original_native_tool_calls_during_emission": 0,
        "HLS_E3_qualified": None,
        "physical_E4_qualified": None,
        "limits": (
            "Static row-major MatMul producer and 2..4 mapped MatMul readers",
            "Serial row-source order; no hardware concurrent port/timing proof",
            "Other reader accumulators and counters remain full materialized arrays",
            "No Vitis HLS BRAM, physical memory, or arbitrary graph guarantee",
        ),
    }
    return Emission(source, manifest)
