"""R4k: experimental FULL-H storage under the EXISTING row-source schedule.

This is the missing schedule-matched control for R4a/R4d row retention.
It reuses the row-retained emitter, keeping every arithmetic statement,
all producer/reader loop nests and source Phi event order unchanged.
The ONLY changes are array capacity and scratch address expressions.
Original R2g/R4a/R4d source emitters and original experiments are unchanged.

SOURCE E0 ONLY until independent host E1 verifies actual generated C++.
Never infer HLS scheduling, RAMB mapping, FPGA throughput or novelty.
"""
from __future__ import annotations

from hashlib import sha256
import re

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.row_retained_fanout_cpp import (
    emit_row_retained_fanout_cpp, row_retained_plan,
)
from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    project_source_value_events,
)
from fpgai_research.model_ir.element_scratch_allocation import verify_allocation
from fpgai_research.model_ir.fanout_contract import (
    admit_fanout, disjoint_fanout_offsets,
)

SCHEMA = "fpgai-research/R4k-full-storage-row-source-schedule/v1"


def full_storage_row_schedule_plan(arch) -> dict:
    """Certify a full disjoint slot map for the SAME R4a row schedule."""
    spec = admit_fanout(arch)
    row = row_retained_plan(arch)
    m, n, q = spec.m, spec.n, spec.q
    disjoint = disjoint_fanout_offsets(spec)
    capacities = {
        tensor: m * (n if tensor == spec.h else q)
        for tensor in spec.intermediates
    }
    assigned = {
        (tensor, index): disjoint[tensor] + index
        for tensor, count in capacities.items()
        for index in range(count)
    }
    full_capacity = sum(capacities.values())
    replay = verify_allocation(
        project_source_value_events(row["source_events"]),
        assigned, unit_slots=full_capacity,
    )
    if replay["status"] != "EXACT_SOURCE_SLOT_REPLAY_VERIFIED":
        raise ValueError("R4k full-layout source replay not certified")
    if full_capacity - row["source_scratch_array_elements"] != (m - 1) * n:
        raise ValueError("R4k source-capacity delta changed")
    return {
        "source_events": row["source_events"],
        "source_event_sha256": row["source_event_sha256"],
        "source_slot_map": assigned,
        "source_slot_map_sha256": sha256(
            repr(tuple(sorted(assigned.items()))).encode()).hexdigest(),
        "source_offsets": disjoint,
        "source_scratch_array_elements": full_capacity,
        "source_H_full_elements": m * n,
        "source_saved_scratch_elements": 0,
        "source_contract": "E0_VALUE_AND_PHI_AND_DISJOINT_ROW_SCHEDULE_REPLAY",
        "vendor_calls": 0,
    }


def emit_full_storage_row_schedule_cpp(arch) -> Emission:
    """Return row scheduling with full materialization, fail-closed edits."""
    spec = admit_fanout(arch)
    m, n, q = spec.m, spec.n, spec.q
    row = emit_row_retained_fanout_cpp(arch)
    row_plan = row_retained_plan(arch)
    plan = full_storage_row_schedule_plan(arch)
    source = row.cpp

    def one(old: str, new: str, count: int = 1) -> None:
        nonlocal source
        actual = source.count(old)
        if actual != count:
            raise ValueError(
                f"R4k row source template drift: {old!r}, expected={count} found={actual}"
            )
        source = source.replace(old, new)

    small = row_plan["source_scratch_array_elements"]
    full = plan["source_scratch_array_elements"]
    one(f"  data_t scratch[{small}]={{}};",
        f"  data_t scratch[{full}]={{}};")
    one(f"      const int r4_h_store_addr=h_idx%{n};",
        "      const int r4_h_store_addr=h_idx;")
    one(f"      const int h_addr=h_idx%{n};",
        "      const int h_addr=h_idx;",
        count=len(spec.readers))

    old_offsets = row_plan["source_offsets"]
    new_offsets = plan["source_offsets"]
    conversion = {
        old_offsets[tensor]: new_offsets[tensor]
        for tensor in spec.intermediates if tensor != spec.h
    }
    # Single regex replacement pass avoids numeric collision between old
    # source offsets and new offsets; no arithmetic/loop statement changed.
    pattern = re.compile(r"(?<![A-Za-z_0-9])([0-9]+)\+(output_idx|idx)\b")
    touched = set()

    def convert(match: re.Match) -> str:
        before = int(match.group(1))
        if before not in conversion:
            return match.group(0)
        touched.add(before)
        return f"{conversion[before]}+{match.group(2)}"

    source = pattern.sub(convert, source)
    if touched != set(conversion):
        raise ValueError("R4k failed to rebase all non-H source scratch addresses")
    if f"const int h_addr=h_idx%{n};" in source or (
            f"const int r4_h_store_addr=h_idx%{n};" in source):
        raise ValueError("R4k row alias remains in full-layout control")
    # An identical source schedule is a property of the independently
    # certified source-event trace AND actual emitted loop scaffold.
    marker = (
        f"    for(int ordinal=row*{n*q};"
        f"ordinal<(row+1)*{n*q};++ordinal) {{"
    )
    if source.count(marker) != 1 or row.cpp.count(marker) != 1:
        raise ValueError("R4k changed the row-source issue schedule")
    digest = sha256(source.encode()).hexdigest()
    return Emission(source, {
        **row.manifest,
        "schema": SCHEMA,
        "base_row_retained_cpp_sha256": row.manifest["source_cpp_sha256"],
        "source_cpp_sha256": digest,
        "materialization": "FULL_H_STORAGE_ROW_SOURCE_SCHEDULE_CONTROL",
        "source_offsets": dict(sorted(new_offsets.items())),
        "source_slot_map_sha256": plan["source_slot_map_sha256"],
        "source_slot_map": tuple(sorted(
            (tensor, idx, address)
            for (tensor, idx), address in plan["source_slot_map"].items())),
        "source_event_order_sha256": plan["source_event_sha256"],
        "source_contract": plan["source_contract"],
        "source_scratch_array_elements": full,
        "source_all_declared_mutable_array_elements":
            full + 2 * len(spec.readers) * m * q,
        "source_H_row_buffer_elements": m * n,
        "source_H_full_elements": m * n,
        "source_saved_scratch_elements": 0,
        "original_vendor_calls": 0,
        "HLS_E3_qualified": None,
        "physical_E4_qualified": None,
        "limitations": (
            "Same source Phi issue schedule, not a proved hardware cycle schedule",
            "Host C++ source only until E1 independent runtime testing",
            "No native RTL, physical RAM/BRAM or measured FPGA clock",
        ),
    })
