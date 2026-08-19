from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence

from fpgai.config.access import get_path
from fpgai.ir import Graph
from fpgai.ir.semantics import mark_ir_level

_AUTO = {None, "", "auto", "unspecified", "default"}
_STORAGE = {"bram", "uram", "ddr", "host", "external", "stream", "recompute"}
_BACKENDS = {"hls", "vitis_hls", "vhdl", "rtl", "external", "auto", "unspecified"}
_EXECUTION = {"auto", "unspecified", "sequential", "dataflow", "serialized", "phase_shared", "parallel", "streamed"}


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower().replace("-", "_")


def _is_auto(value: Any) -> bool:
    return _norm(value) in {_norm(x) for x in _AUTO}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _match(rule: Mapping[str, Any], *, op: Any, index: int) -> bool:
    match = _mapping(rule.get("match"))
    if not match:
        return False
    if "index" in match and int(match["index"]) != index:
        return False
    if "name" in match and str(match["name"]) != str(op.name):
        return False
    if "op_type" in match and str(match["op_type"]) != str(op.op_type):
        return False
    provider = None
    ext = _mapping((op.attrs or {}).get("_fpgai_external_operator"))
    comp = _mapping((op.attrs or {}).get("_fpgai_composite_provider"))
    if ext:
        provider = ext.get("package_id")
    elif comp:
        provider = comp.get("provider")
    if "provider" in match and str(match["provider"]) != str(provider or ""):
        return False
    return True


def _merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key == "match":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _global_defaults(raw_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "memory": {
            "weight_storage": get_path(raw_cfg, "memory.weight_storage", "auto"),
            "activation_storage": get_path(raw_cfg, "memory.activation_storage", "auto"),
            "gradient_storage": get_path(raw_cfg, "memory.gradient_storage", "auto"),
            "optimizer_state_storage": get_path(raw_cfg, "memory.optimizer_state_storage", "auto"),
        },
        "implementation": {
            "backend": "auto",
            "preferred": (),
        },
        "execution": {
            "mode": "auto",
        },
        "transport": {
            "protocol": "auto",
        },
        "buffering": {
            "storage": "auto",
        },
    }


def _requested_for_op(raw_cfg: Mapping[str, Any], op: Any, index: int) -> Dict[str, Any]:
    requested = _global_defaults(raw_cfg)
    architecture = _mapping(raw_cfg.get("architecture"))
    requested = _merge(requested, _mapping(architecture.get("network")))
    requested = _merge(requested, _mapping(architecture.get("defaults")))
    rules = architecture.get("layers", ())
    if isinstance(rules, Sequence) and not isinstance(rules, (str, bytes)):
        for rule in rules:
            if isinstance(rule, Mapping) and _match(rule, op=op, index=index):
                requested = _merge(requested, rule)

    # Existing implementation selection stays authoritative and is folded into
    # the same per-node request instead of inventing a parallel selection path.
    impl_root = _mapping(raw_cfg.get("implementations"))
    op_rules = _mapping(impl_root.get("operators"))
    node_rules = _mapping(impl_root.get("nodes"))
    ext = _mapping((op.attrs or {}).get("_fpgai_external_operator"))
    operator_id = str(ext.get("operator_id", ""))
    selection: Dict[str, Any] = {}
    for key in (operator_id, str(op.op_type)):
        if key and isinstance(op_rules.get(key), Mapping):
            selection = _merge(selection, op_rules[key])
    if isinstance(node_rules.get(str(op.name)), Mapping):
        selection = _merge(selection, node_rules[str(op.name)])
    if selection:
        requested["implementation"] = _merge(dict(_mapping(requested.get("implementation"))), selection)
    return requested


def _validate_choice(kind: str, value: Any, allowed: Iterable[str], rejected: list[dict[str, Any]]) -> str | None:
    normalized = _norm(value)
    if _is_auto(normalized):
        return None
    if normalized not in set(allowed):
        rejected.append({"field": kind, "requested": value, "reason": "unsupported_value", "allowed": sorted(set(allowed))})
        return None
    return normalized


def _candidate_backends(op: Any) -> set[str]:
    result = set()
    for candidate in getattr(op.semantics, "implementation_candidates", ()) or ():
        backend = _norm(getattr(candidate, "backend", None))
        if backend:
            result.add(backend)
    return result


def resolve_layer_mechanisms(graph: Graph, raw_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve user-requested architecture choices without silently forcing layers.

    ``auto`` / unspecified choices remain unresolved unless an existing semantic
    contract already selected a value. Explicit choices are attached to the IR
    and rejected deterministically when they conflict with declared candidates.
    The pass is model-agnostic and works for built-in, external-operator and
    externally expanded composite layers.
    """
    report_layers: list[dict[str, Any]] = []
    for index, op in enumerate(graph.ops):
        requested = _requested_for_op(raw_cfg, op, index)
        rejected: list[dict[str, Any]] = []
        resolved: Dict[str, Any] = {}

        memory = _mapping(requested.get("memory"))
        weight_storage = _validate_choice("memory.weight_storage", memory.get("weight_storage"), _STORAGE, rejected)
        activation_storage = _validate_choice("memory.activation_storage", memory.get("activation_storage"), _STORAGE, rejected)
        gradient_storage = _validate_choice("memory.gradient_storage", memory.get("gradient_storage"), _STORAGE, rejected)
        optimizer_storage = _validate_choice("memory.optimizer_state_storage", memory.get("optimizer_state_storage"), _STORAGE, rejected)

        implementation = _mapping(requested.get("implementation"))
        backend = _validate_choice("implementation.backend", implementation.get("backend"), _BACKENDS, rejected)
        candidate_backends = _candidate_backends(op)
        if backend and candidate_backends:
            aliases = {"vitis_hls": "hls", "rtl": "vhdl"}
            compared = aliases.get(backend, backend)
            declared = {aliases.get(x, x) for x in candidate_backends}
            if compared not in declared and "external" not in declared:
                rejected.append({
                    "field": "implementation.backend",
                    "requested": backend,
                    "reason": "no_declared_candidate",
                    "declared_candidates": sorted(candidate_backends),
                })
                backend = None
        if backend:
            op.semantics.selected_backend = backend
            resolved["backend"] = backend

        execution = _mapping(requested.get("execution"))
        requested_execution_mode = execution.get("mode")
        execution_mode = _validate_choice("execution.mode", requested_execution_mode, _EXECUTION, rejected)
        if execution_mode is None and _is_auto(requested_execution_mode):
            # Backend defaults are compiler decisions only when the user left the
            # mechanism on auto. They are recorded explicitly so the backend never
            # silently changes the architecture. Explicit user choices always win.
            selected_backend = _norm(backend or getattr(op.semantics, "selected_backend", None))
            if str(op.op_type) == "MultiHeadAttention" and selected_backend in {"hls", "vitis_hls"}:
                execution_mode = "serialized"
                resolved["execution_mode_source"] = "backend_default:hls.multi_head_attention"
        if execution_mode:
            op.semantics.schedule["execution_mode"] = execution_mode
            op.attrs["execution_mode"] = execution_mode
            resolved["execution_mode"] = execution_mode

        transport = _mapping(requested.get("transport"))
        protocol = _norm(transport.get("protocol"))
        if not _is_auto(protocol):
            for name in op.inputs + op.outputs:
                tensor = graph.get_tensor(str(name))
                if tensor is not None:
                    tensor.semantics.transport.protocol = str(protocol)
            resolved["transport_protocol"] = protocol

        buffering = _mapping(requested.get("buffering"))
        buffer_storage = _validate_choice("buffering.storage", buffering.get("storage"), _STORAGE, rejected)
        if buffer_storage:
            op.semantics.buffering["storage"] = buffer_storage
            resolved["buffer_storage"] = buffer_storage

        # Constants on non-primary inputs are parameter/weight tensors. Never
        # change their storage unless the user actually requested it.
        if weight_storage:
            for name in op.inputs[1:]:
                if str(name) in graph.constants:
                    tensor = graph.get_tensor(str(name))
                    if tensor is not None:
                        tensor.semantics.memory.storage = weight_storage
                        tensor.semantics.memory.residency = "external" if weight_storage in {"ddr", "host", "external"} else "on_chip"
            op.semantics.schedule["weight_storage"] = weight_storage
            resolved["weight_storage"] = weight_storage

        if activation_storage:
            for name in op.outputs:
                tensor = graph.get_tensor(str(name))
                if tensor is not None:
                    tensor.semantics.memory.storage = activation_storage
                    tensor.semantics.memory.residency = "external" if activation_storage in {"ddr", "host", "external"} else "on_chip"
            op.semantics.buffering["activation_storage"] = activation_storage
            resolved["activation_storage"] = activation_storage

        if gradient_storage:
            op.semantics.training["gradient_storage"] = gradient_storage
            resolved["gradient_storage"] = gradient_storage
        if optimizer_storage:
            op.semantics.training["optimizer_state_storage"] = optimizer_storage
            resolved["optimizer_state_storage"] = optimizer_storage

        preferred = implementation.get("preferred", ())
        if isinstance(preferred, str):
            preferred = (preferred,)
        elif not isinstance(preferred, (list, tuple)):
            preferred = ()
        if preferred:
            op.semantics.training.setdefault("implementation_selection", {})
            op.semantics.training["implementation_selection"]["preferred_packages"] = [str(x) for x in preferred]
            resolved["preferred_packages"] = [str(x) for x in preferred]

        provenance = {}
        if isinstance((op.attrs or {}).get("_fpgai_external_operator"), Mapping):
            provenance = {"kind": "ecosystem_operator", **dict(op.attrs["_fpgai_external_operator"])}
        elif isinstance((op.attrs or {}).get("_fpgai_composite_provider"), Mapping):
            provenance = {"kind": "ecosystem_composite", **dict(op.attrs["_fpgai_composite_provider"])}
        else:
            provenance = {"kind": "builtin"}

        entry = {
            "index": index,
            "name": str(op.name),
            "op_type": str(op.op_type),
            "provider": provenance,
            "requested": requested,
            "resolved": resolved,
            "rejected": rejected,
            "status": "rejected" if rejected else "resolved",
        }
        op.semantics.resource_constraints["mechanism_resolution"] = entry
        report_layers.append(entry)

    report = {
        "schema": "fpgai.layer-mechanism-resolution/v1",
        "layer_count": len(report_layers),
        "rejected_layer_count": sum(1 for item in report_layers if item["rejected"]),
        "layers": report_layers,
        "policy": "explicit user choices are preserved when legal; auto/unspecified choices are not silently forced by layer type",
    }
    graph.metadata["layer_mechanism_resolution"] = report
    return report


def materialize_compile_plan_semantics(graph: Graph, compile_plan: Any) -> Dict[str, Any]:
    """Attach the resolved planner architecture to the authoritative FPGAI IR.

    The planner already owns legality/default resolution for PE/SIMD, loop II,
    unroll, partitioning, tiling, buffering, precision and layer memory.  This
    function does not create a second planner; it mirrors the resolved plan into
    per-op IR semantics so downstream reports/MLIR/runtime artifacts can inspect
    the same hardware decisions consumed by code generation.
    """
    layer_plans = list(getattr(compile_plan, "layer_plans", ()) or ())
    by_name = {str(getattr(lp, "node_name", "")): lp for lp in layer_plans}
    materialized = []
    unmatched = []
    for index, op in enumerate(graph.ops):
        lp = by_name.get(str(op.name))
        if lp is None and index < len(layer_plans):
            candidate = layer_plans[index]
            if str(getattr(candidate, "op_type", "")) == str(op.op_type):
                lp = candidate
        if lp is None:
            unmatched.append(str(op.name))
            continue
        arch = getattr(lp, "architecture", None)
        arch_dict = arch.to_dict() if arch is not None and hasattr(arch, "to_dict") else {}
        op.semantics.schedule["architecture"] = arch_dict
        op.semantics.schedule["architecture_signature"] = str(getattr(lp, "architecture_signature", ""))
        if arch_dict:
            pipeline = dict(arch_dict.get("pipeline", {}) or {})
            parallelism = dict(arch_dict.get("parallelism", {}) or {})
            partitioning = dict(arch_dict.get("partitioning", {}) or {})
            tiling = dict(arch_dict.get("tiling", {}) or {})
            op.semantics.schedule["pipeline"] = pipeline
            op.semantics.schedule["parallelism"] = parallelism
            op.semantics.schedule["partitioning"] = partitioning
            op.semantics.schedule["tiling"] = tiling
            op.semantics.buffering.update(dict(arch_dict.get("buffering", {}) or {}))
            # Make hierarchical execution first-class in FPGAI IR.  Layer-level
            # decisions preserve PE/SIMD and buffering; loop-level decisions
            # preserve II, unroll, tiling and partitioning independently.
            op.semantics.execution["layer"] = {
                "pipeline_style": pipeline.get("style"),
                "pipeline_scope": pipeline.get("scope"),
                "pe": parallelism.get("pe", 1),
                "simd": parallelism.get("simd", 1),
                "buffering": dict(arch_dict.get("buffering", {}) or {}),
            }
            op.semantics.execution["loops"] = {
                "pipeline_ii": pipeline.get("ii", 1),
                "pipeline": dict(pipeline.get("loops", {}) or {}),
                "unroll": dict(parallelism.get("unroll", {}) or {}),
                "tiling": dict(tiling.get("sizes", {}) or {}),
                "partitioning": {
                    "factor": partitioning.get("factor", 1),
                    "mode": partitioning.get("mode", "none"),
                    "targets": dict(partitioning.get("targets", {}) or {}),
                },
            }
            memory = dict(arch_dict.get("memory", {}) or {})
            if memory:
                op.semantics.resource_constraints["resolved_memory"] = memory
        op.semantics.resource_constraints["planner_node"] = {
            "node_name": str(getattr(lp, "node_name", op.name)),
            "op_type": str(getattr(lp, "op_type", op.op_type)),
        }
        materialized.append(str(op.name))

    network_execution = dict((getattr(compile_plan, "notes", {}) or {}).get("network_execution", {}) or {})
    graph.semantics.resource_constraints["resolved_architecture"] = {
        "target_board": str(getattr(compile_plan, "target_board", "unknown")),
        "target_part": str(getattr(compile_plan, "target_part", "unknown")),
        "clock_mhz": float(getattr(compile_plan, "clock_mhz", 0.0) or 0.0),
        "architecture_signature": str(getattr(compile_plan, "architecture_signature", "")),
        "network_execution": network_execution,
    }
    graph.semantics.execution["model"] = network_execution
    for op in graph.ops:
        op.semantics.execution.setdefault("model", network_execution)
    mark_ir_level(graph, "architectural", reason="resolved compile-plan architecture materialized")
    report = {
        "schema": "fpgai.ir-plan-materialization/v1",
        "materialized_operator_count": len(materialized),
        "materialized_operators": materialized,
        "unmatched_operators": unmatched,
        "architecture_signature": str(getattr(compile_plan, "architecture_signature", "")),
        "policy": "planner decisions are mirrored into FPGAI IR; codegen remains driven by the existing compile plan",
    }
    graph.metadata["ir_plan_materialization"] = report
    return report
