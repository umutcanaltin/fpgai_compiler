from __future__ import annotations

"""Network-level execution planning shared by compiler analysis and HLS codegen."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from fpgai.config.access import get_path


NETWORK_EXECUTION_MODES = {"sequential", "dataflow", "phase_shared", "parallel"}


def _norm(value: Any) -> str:
    text = str(value if value is not None else "sequential").strip().lower().replace("-", "_")
    if text in {"auto", "unspecified", "default", ""}:
        return "sequential"
    if text == "streamed":
        return "dataflow"
    return text


def requested_network_execution_mode(raw_cfg: Mapping[str, Any] | None) -> str:
    raw = raw_cfg or {}
    requested = get_path(raw, "architecture.network.execution.mode", None)
    if requested is None:
        requested = get_path(raw, "hls.execution_mode", "sequential")
    mode = _norm(requested)
    if mode not in NETWORK_EXECUTION_MODES:
        raise ValueError(
            "NETEXEC001: architecture.network.execution.mode must be one of "
            f"{sorted(NETWORK_EXECUTION_MODES)}, got {requested!r}"
        )
    return mode


def _shape_signature(descriptor: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    def first(name: str) -> tuple[int, ...]:
        values = getattr(descriptor, name, []) or []
        if not values:
            return ()
        shape = tuple(int(x) for x in values[0])
        if len(shape) > 1 and shape[0] == 1:
            shape = shape[1:]
        return shape
    return first("input_shapes"), first("output_shapes")


def _reuse_groups(descriptors: Sequence[Any]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[tuple[int, Any]]] = {}
    for index, desc in enumerate(descriptors):
        op_type = str(getattr(desc, "op_type", ""))
        if op_type not in {"MatMul", "Dense", "RMSNorm", "LayerNormalization", "SiLU", "Relu", "Add", "Mul"}:
            continue
        in_shape, out_shape = _shape_signature(desc)
        attrs = dict(getattr(desc, "attrs", {}) or {})
        precision_tag = attrs.get("precision_tag")
        key = (op_type, in_shape, out_shape, str(precision_tag or ""))
        buckets.setdefault(key, []).append((index, desc))

    groups: list[dict[str, Any]] = []
    group_id = 0
    for key, entries in sorted(buckets.items(), key=lambda item: str(item[0])):
        if len(entries) < 2:
            continue
        op_type, in_shape, out_shape, precision_tag = key
        groups.append({
            "id": f"reuse_group_{group_id}",
            "op_type": op_type,
            "member_indices": [index for index, _ in entries],
            "members": [str(getattr(desc, "node_name", f"layer_{index}")) for index, desc in entries],
            "input_shape": list(in_shape),
            "output_shape": list(out_shape),
            "precision_tag": precision_tag or None,
            "compatibility": "same_op_shape_precision",
            "physical_status": "planned",
        })
        group_id += 1
    return groups


@dataclass(frozen=True)
class NetworkExecutionPlan:
    requested_mode: str
    resolved_mode: str
    source: str
    dataflow_pragma: bool = False
    reuse_groups: list[dict[str, Any]] = field(default_factory=list)
    training_phase_schedule: dict[str, str] = field(default_factory=dict)
    physical_status: str = "implemented"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fpgai.network-execution-plan/v1",
            "requested_mode": self.requested_mode,
            "resolved_mode": self.resolved_mode,
            "source": self.source,
            "dataflow_pragma": bool(self.dataflow_pragma),
            "reuse_groups": [dict(group) for group in self.reuse_groups],
            "training_phase_schedule": dict(self.training_phase_schedule),
            "physical_status": self.physical_status,
            "notes": list(self.notes),
        }


def build_network_execution_plan(
    raw_cfg: Mapping[str, Any] | None,
    descriptors: Sequence[Any] = (),
    *,
    pipeline_mode: str = "inference",
) -> NetworkExecutionPlan:
    raw = raw_cfg or {}
    source = "architecture.network.execution.mode" if get_path(raw, "architecture.network.execution.mode", None) is not None else "hls.execution_mode"
    requested = requested_network_execution_mode(raw)
    mode = requested
    is_training = str(pipeline_mode).strip().lower() == "training_on_device"
    groups = _reuse_groups(descriptors) if mode == "phase_shared" else []
    notes: list[str] = []
    physical_status = "implemented"
    dataflow_pragma = mode == "dataflow" and not is_training

    if mode == "dataflow" and is_training:
        physical_status = "planning_only"
        notes.append("Training network dataflow is planned explicitly by forward/backward/update phase; no top-level DATAFLOW pragma is emitted yet.")
    if mode == "phase_shared":
        physical_status = "planning_only"
        notes.append("Compatible reuse groups are resolved, but physical shared-engine HLS allocation is not yet emitted.")
    if mode == "parallel":
        physical_status = "planning_only"
        notes.append("Network-level parallel branch replication requires explicit branch scheduling and is not emitted globally yet.")

    training_schedule: dict[str, str] = {}
    if is_training:
        for phase in ("forward", "backward", "update"):
            value = get_path(raw, f"training.schedule.{phase}.mode", None)
            if value is not None:
                training_schedule[phase] = _norm(value)

    return NetworkExecutionPlan(
        requested_mode=requested,
        resolved_mode=mode,
        source=source,
        dataflow_pragma=dataflow_pragma,
        reuse_groups=groups,
        training_phase_schedule=training_schedule,
        physical_status=physical_status,
        notes=notes,
    )
