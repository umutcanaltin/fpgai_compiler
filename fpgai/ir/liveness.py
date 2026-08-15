from __future__ import annotations

"""Tensor producer/consumer and live-range analysis for FPGAI IR graphs."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TensorLiveRange:
    name: str
    producer: str | None
    producer_index: int | None
    consumers: tuple[str, ...]
    consumer_indices: tuple[int, ...]
    first_live_step: int
    last_live_step: int
    graph_input: bool
    graph_output: bool
    constant: bool
    buffer_slot: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "producer_index": self.producer_index,
            "consumers": list(self.consumers),
            "consumer_indices": list(self.consumer_indices),
            "first_live_step": self.first_live_step,
            "last_live_step": self.last_live_step,
            "graph_input": self.graph_input,
            "graph_output": self.graph_output,
            "constant": self.constant,
            "buffer_slot": self.buffer_slot,
        }


def analyze_tensor_liveness(graph: Any) -> dict[str, Any]:
    ops = list(getattr(graph, "ops", []) or [])
    graph_inputs = set(getattr(graph, "inputs", []) or [])
    graph_outputs = set(getattr(graph, "outputs", []) or [])
    constants = set(getattr(graph, "constants", {}) or {})

    producers: dict[str, tuple[str, int]] = {}
    consumers: dict[str, list[tuple[str, int]]] = {}
    for index, op in enumerate(ops):
        for name in getattr(op, "outputs", []) or []:
            producers[str(name)] = (str(op.name), index)
        for name in getattr(op, "inputs", []) or []:
            consumers.setdefault(str(name), []).append((str(op.name), index))

    tensor_names = set(getattr(graph, "tensors", {}) or {}) | set(producers) | set(consumers) | graph_inputs | graph_outputs | constants
    preliminary: list[dict[str, Any]] = []
    for name in sorted(tensor_names):
        producer = producers.get(name)
        uses = consumers.get(name, [])
        producer_index = producer[1] if producer else None
        first = -1 if name in graph_inputs or name in constants else (producer_index if producer_index is not None else 0)
        consumer_indices = tuple(index for _, index in uses)
        if consumer_indices:
            last = max(consumer_indices)
        elif name in graph_outputs:
            last = len(ops)
        else:
            last = first
        if name in graph_outputs:
            last = max(last, len(ops))
        preliminary.append({
            "name": name,
            "producer": producer[0] if producer else None,
            "producer_index": producer_index,
            "consumers": tuple(op_name for op_name, _ in uses),
            "consumer_indices": consumer_indices,
            "first": first,
            "last": last,
            "graph_input": name in graph_inputs,
            "graph_output": name in graph_outputs,
            "constant": name in constants,
        })

    # Greedy interval-coloring for reusable activation buffers. Constants are not
    # assigned activation slots because they belong to parameter/storage planning.
    reusable = sorted((row for row in preliminary if not row["constant"]), key=lambda row: (row["first"], row["last"], row["name"]))
    slots_last: list[int] = []
    slot_by_name: dict[str, int] = {}
    for row in reusable:
        chosen = None
        for slot, last in enumerate(slots_last):
            if last < row["first"]:
                chosen = slot
                break
        if chosen is None:
            chosen = len(slots_last)
            slots_last.append(row["last"])
        else:
            slots_last[chosen] = row["last"]
        slot_by_name[row["name"]] = chosen

    ranges = [TensorLiveRange(
        name=row["name"], producer=row["producer"], producer_index=row["producer_index"], consumers=row["consumers"], consumer_indices=row["consumer_indices"], first_live_step=row["first"], last_live_step=row["last"], graph_input=row["graph_input"], graph_output=row["graph_output"], constant=row["constant"], buffer_slot=None if row["constant"] else slot_by_name[row["name"]]
    ) for row in preliminary]

    branch_tensors = [item.name for item in ranges if len(item.consumers) > 1]
    merge_ops = [str(op.name) for op in ops if len([x for x in (getattr(op, "inputs", []) or []) if x not in constants]) > 1]
    max_live = 0
    live_by_step: dict[str, list[str]] = {}
    for step in range(-1, len(ops) + 1):
        live = [item.name for item in ranges if not item.constant and item.first_live_step <= step <= item.last_live_step]
        live_by_step[str(step)] = live
        max_live = max(max_live, len(live))

    return {
        "schema": "fpgai.tensor-liveness/v1",
        "graph_name": str(getattr(graph, "name", "main")),
        "tensor_count": len(ranges),
        "activation_buffer_slots": len(slots_last),
        "maximum_simultaneously_live_tensors": max_live,
        "branch_tensors": branch_tensors,
        "merge_ops": merge_ops,
        "has_branching": bool(branch_tensors or merge_ops),
        "sequential_current_buffer_compatible": not bool(branch_tensors or merge_ops),
        "tensors": {item.name: item.to_dict() for item in ranges},
        "live_by_step": live_by_step,
        "buffer_policy": "Greedy interval reuse metadata; branch-aware HLS code generation consumes a precision-compatible allocation derived from these live ranges.",
    }


def write_tensor_liveness_report(payload: Mapping[str, Any], reports_dir: str | Path) -> tuple[Path, Path]:
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "tensor_liveness.json"
    md_path = root / "tensor_liveness.md"
    json_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Tensor liveness", "", f"- Tensors: {payload.get('tensor_count')}", f"- Activation buffer slots: {payload.get('activation_buffer_slots')}", f"- Maximum simultaneously live tensors: {payload.get('maximum_simultaneously_live_tensors')}", f"- Branching detected: `{payload.get('has_branching')}`", f"- Sequential current-buffer compatible: `{payload.get('sequential_current_buffer_compatible')}`", "", "Producer/consumer and live-range metadata is consumed by the branch-aware HLS buffer allocator for supported DAG profiles."]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
