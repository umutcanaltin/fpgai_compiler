from __future__ import annotations

"""Liveness-aware activation-buffer allocation for HLS inference graphs."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from fpgai.backends.hls.emit.types_h import _default_precision, _op_precision_from_attrs, _spec_to_ap, _tensor_cpp_type
from fpgai.ir.liveness import analyze_tensor_liveness


@dataclass(frozen=True)
class HLSBufferSlot:
    slot: int
    name: str
    cpp_type: str
    words: int
    tensors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "name": self.name,
            "cpp_type": self.cpp_type,
            "words": self.words,
            "tensors": list(self.tensors),
        }


def _words(graph: Any, tensor_name: str) -> int:
    spec = graph.get_tensor(tensor_name) if hasattr(graph, "get_tensor") else None
    shape = getattr(spec, "shape", None) if spec is not None else None
    if not shape:
        raise ValueError(f"HLSBUF001: missing static shape for tensor {tensor_name!r}")
    total = 1
    for dim in shape:
        value = int(dim)
        if value <= 0:
            raise ValueError(f"HLSBUF002: dynamic shape for tensor {tensor_name!r}")
        total *= value
    return total


def _tensor_cpp_types(graph: Any, raw_cfg: Mapping[str, Any] | None) -> dict[str, str]:
    raw = dict(raw_cfg or {})
    defaults = _default_precision(raw)
    result: dict[str, str] = {}
    default_act = _spec_to_ap(defaults["activation"])
    # Seed all declared tensors so persistent state and non-port tensors retain
    # their explicit integer/floating dtype even when they are not graph inputs.
    for name, spec in (getattr(graph, "tensors", {}) or {}).items():
        result[str(name)] = _tensor_cpp_type(spec, default_act)
    for name in getattr(graph, "inputs", []) or []:
        spec = graph.get_tensor(str(name)) if hasattr(graph, "get_tensor") else None
        result[str(name)] = _tensor_cpp_type(spec, default_act)
    for index, op in enumerate(getattr(graph, "ops", []) or []):
        precision = _op_precision_from_attrs(op, defaults)
        out_type = _spec_to_ap(precision["activation"])
        for name in getattr(op, "outputs", []) or []:
            spec = graph.get_tensor(str(name)) if hasattr(graph, "get_tensor") else None
            result[str(name)] = _tensor_cpp_type(spec, out_type)
    return result


def build_hls_buffer_allocation(
    graph: Any,
    *,
    raw_cfg: Mapping[str, Any] | None = None,
    tensor_liveness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Allocate reusable HLS activation buffers using liveness + C++ type compatibility.

    Buffers are reused only when live ranges do not overlap and the underlying
    resolved C++ scalar type is identical. This preserves layerwise precision
    semantics while allowing real DAG scheduling to consume liveness metadata.
    """

    liveness = dict(tensor_liveness or analyze_tensor_liveness(graph))
    tensors = liveness.get("tensors", {})
    cpp_types = _tensor_cpp_types(graph, raw_cfg)

    rows: list[dict[str, Any]] = []
    for name, entry in tensors.items():
        if not isinstance(entry, Mapping) or bool(entry.get("constant", False)):
            continue
        spec = graph.get_tensor(str(name)) if hasattr(graph, "get_tensor") else None
        semantics = getattr(spec, "semantics", None) if spec is not None else None
        state_obj = getattr(semantics, "state", None) if semantics is not None else None
        memory_obj = getattr(semantics, "memory", None) if semantics is not None else None
        persistent = bool(getattr(state_obj, "persistent_across_invocations", False))
        state_kind = str(getattr(state_obj, "kind", "stateless") or "stateless")
        storage = str(getattr(memory_obj, "storage", "unspecified") or "unspecified")
        rows.append({
            "tensor": str(name),
            "first": int(entry.get("first_live_step", 0)),
            "last": int(entry.get("last_live_step", 0)),
            "cpp_type": cpp_types.get(str(name), _spec_to_ap(_default_precision(dict(raw_cfg or {}))["activation"])),
            "words": _words(graph, str(name)),
            "persistent": persistent,
            "state_kind": state_kind,
            "storage": storage,
        })
    rows.sort(key=lambda row: (row["first"], row["last"], row["tensor"]))

    slots: list[dict[str, Any]] = []
    tensor_to_buffer: dict[str, str] = {}
    tensor_to_slot: dict[str, int] = {}
    for row in rows:
        chosen: int | None = None
        # Persistent runtime-session tensors are physical state: never reuse
        # their storage with ephemeral activation live ranges.
        if not bool(row.get("persistent")):
            for slot_index, slot in enumerate(slots):
                if bool(slot.get("persistent")):
                    continue
                if slot["cpp_type"] == row["cpp_type"] and int(slot["last"]) < int(row["first"]):
                    chosen = slot_index
                    break
        if chosen is None:
            chosen = len(slots)
            slot_name = f"fpgai_state_{row['tensor']}" if bool(row.get("persistent")) else f"fpgai_buffer_{chosen}"
            slots.append({
                "slot": chosen,
                "name": slot_name.replace("/", "_").replace(".", "_"),
                "cpp_type": row["cpp_type"],
                "words": int(row["words"]),
                "last": int(row["last"]),
                "tensors": [row["tensor"]],
                "persistent": bool(row.get("persistent")),
                "state_kind": row.get("state_kind", "stateless"),
                "storage": row.get("storage", "unspecified"),
            })
        else:
            slot = slots[chosen]
            slot["last"] = int(row["last"])
            slot["words"] = max(int(slot["words"]), int(row["words"]))
            slot["tensors"].append(row["tensor"])
        tensor_to_buffer[row["tensor"]] = slots[chosen]["name"]
        tensor_to_slot[row["tensor"]] = chosen

    public_slots = []
    for slot in slots:
        row = HLSBufferSlot(
            slot=int(slot["slot"]),
            name=str(slot["name"]),
            cpp_type=str(slot["cpp_type"]),
            words=int(slot["words"]),
            tensors=tuple(str(x) for x in slot["tensors"]),
        ).to_dict()
        row.update({
            "persistent": bool(slot.get("persistent", False)),
            "state_kind": str(slot.get("state_kind", "stateless")),
            "storage": str(slot.get("storage", "unspecified")),
        })
        public_slots.append(row)

    resource_provenance: dict[str, Any] = {}
    for slot in public_slots:
        resource_provenance[slot["name"]] = {
            "tensors": list(slot["tensors"]),
            "cpp_type": slot["cpp_type"],
            "words": slot["words"],
        }

    return {
        "schema": "fpgai.hls-buffer-allocation/v1",
        "mode": "liveness",
        "graph_name": str(getattr(graph, "name", "main")),
        "slot_count": len(public_slots),
        "slots": public_slots,
        "tensor_to_buffer": tensor_to_buffer,
        "tensor_to_slot": tensor_to_slot,
        "resource_provenance": resource_provenance,
        "policy": "Buffers are greedily reused only across non-overlapping live ranges with identical resolved scalar types.",
    }


def build_legacy_buffer_provenance(graph: Any) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    inputs = list(getattr(graph, "inputs", []) or [])
    if inputs:
        mapping["layer_in"] = {"tensors": [str(inputs[0])]}
    for index, op in enumerate(getattr(graph, "ops", []) or []):
        outputs = list(getattr(op, "outputs", []) or [])
        if outputs:
            mapping[f"layer_{index}_out"] = {"tensors": [str(outputs[0])]}
    return mapping


def write_hls_buffer_allocation_report(payload: Mapping[str, Any], reports_dir: str | Path) -> tuple[Path, Path]:
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "hls_buffer_allocation.json"
    md_path = root / "hls_buffer_allocation.md"
    json_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# HLS buffer allocation",
        "",
        f"- Mode: `{payload.get('mode')}`",
        f"- Buffer slots: {payload.get('slot_count')}",
        "",
        "| Buffer | Type | Words | Tensor history |",
        "|---|---|---:|---|",
    ]
    for slot in payload.get("slots", []) or []:
        lines.append(
            f"| `{slot.get('name')}` | `{slot.get('cpp_type')}` | {slot.get('words')} | "
            + ", ".join(f"`{x}`" for x in slot.get("tensors", []))
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
