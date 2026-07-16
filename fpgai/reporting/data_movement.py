from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_RUNTIME_MODE_NUMBERS = {
    "run_inference": 0,
    "import_weights": 1,
    "export_weights": 2,
    "run_training": 2,
    "accumulate_gradients": 3,
    "apply_accumulated_gradients": 4,
    "reset_accumulators": 5,
    "ddr_tiled_training": 7,
    "export_gradients": 8,
    "export_optimizer_state": 9,
}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalise(value: Any, default: str = "") -> str:
    text = str(value if value is not None else default).strip().lower().replace("-", "_")
    aliases = {
        "axis": "axi_stream",
        "stream": "axi_stream",
        "maxi": "m_axi",
        "ddr": "m_axi",
        "axi_dma": "dma",
        "runtime": "ps_runtime",
        "host": "ps_runtime",
        "compile_time": "embedded",
        "const": "embedded",
        "static": "embedded",
    }
    return aliases.get(text, text)




def _bool_value(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled", "tiled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled", "none", "full"}:
            return False
    if value is None:
        return default
    return bool(value)


def _positive_int(value: Any, default: int | None = None) -> int | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _cfg_tiled_enabled(cfg: Mapping[str, Any] | None) -> bool:
    if not isinstance(cfg, Mapping):
        return False
    if _normalise(cfg.get("policy")) == "tiled":
        return True
    tiled = cfg.get("tiled")
    if isinstance(tiled, Mapping):
        if "enabled" in tiled:
            return _bool_value(tiled.get("enabled"), False)
        return any(key in tiled for key in ("tile_size", "size", "words"))
    return _bool_value(tiled, False)


def _cfg_tile_size(cfg: Mapping[str, Any] | None, default: int = 64) -> int:
    if not isinstance(cfg, Mapping):
        return default
    tiled = cfg.get("tiled")
    candidates = [cfg.get("tile_size"), cfg.get("tile_words"), cfg.get("words")]
    if isinstance(tiled, Mapping):
        candidates.extend([tiled.get("tile_size"), tiled.get("size"), tiled.get("words")])
    for candidate in candidates:
        parsed = _positive_int(candidate, None)
        if parsed is not None:
            return parsed
    return default


def _commands(runtime_sequence: Any) -> set[str]:
    sequence = runtime_sequence.get("sequence", []) if isinstance(runtime_sequence, Mapping) else (runtime_sequence or [])
    out: set[str] = set()
    for item in sequence:
        if isinstance(item, Mapping):
            command = item.get("command")
            if command is None and len(item) == 1:
                command = next(iter(item.keys()))
        else:
            command = item
        if command:
            out.add(str(command).strip().lower().replace("-", "_"))
    return out


def _cfg_section(raw_config: Mapping[str, Any] | None, *parts: str) -> dict[str, Any]:
    node: Any = raw_config or {}
    for part in parts:
        if not isinstance(node, Mapping):
            return {}
        node = node.get(part, {})
    return dict(node) if isinstance(node, Mapping) else {}


def _first_cfg(raw_config: Mapping[str, Any] | None, paths: list[tuple[str, ...]]) -> dict[str, Any]:
    for path in paths:
        cfg = _cfg_section(raw_config, *path)
        if cfg:
            return cfg
    return {}


def _movement_cfg_for_role(raw_config: Mapping[str, Any] | None, role: str) -> dict[str, Any]:
    """Return canonical movement config for a tensor role.

    FPGAI currently accepts both the older nested spelling
    data_movement.inputs.import.interface and the newer direct spelling
    data_movement.inputs.interface.  Sprint P keeps both visible here so reports,
    runtime contracts, and generated artifacts resolve the same user decision.
    """
    role_u = role.lower()
    paths = {
        "inputs": [("data_movement", "inputs", "import"), ("data_movement", "inputs"), ("data_movement", "input", "load")],
        "outputs": [("data_movement", "outputs", "export"), ("data_movement", "outputs"), ("data_movement", "output", "store")],
        "weights": [("data_movement", "weights", "import"), ("data_movement", "weights", "load"), ("data_movement", "weights")],
        "labels": [("data_movement", "labels", "import"), ("data_movement", "labels"), ("data_movement", "aux", "load")],
        "gradients": [("data_movement", "gradients", "export"), ("data_movement", "gradients", "store"), ("data_movement", "gradients")],
        "optimizer_state": [("data_movement", "optimizer_state", "export"), ("data_movement", "optimizer_state", "store"), ("data_movement", "optimizer_state")],
        "activations": [("memory", "activations"), ("data_movement", "activations")],
    }.get(role_u, [])
    cfg = _first_cfg(raw_config, paths)
    if any(key in cfg for key in ("interface", "transport", "policy", "storage", "tiled")):
        return cfg
    for nested in ("import", "export", "load", "store"):
        value = cfg.get(nested) if isinstance(cfg, Mapping) else None
        if isinstance(value, Mapping):
            return dict(value)
    return cfg



def _cfg_value(raw_config: Mapping[str, Any] | None, *parts: str, default: Any = None) -> Any:
    node: Any = raw_config or {}
    for part in parts:
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def _storage_for_role(
    raw_config: Mapping[str, Any] | None,
    role: str,
    placement: dict[str, Any] | None,
) -> str:
    """Resolve physical residency separately from transfer configuration."""
    role_u = role.lower()
    placed = str((placement or {}).get("storage") or "").strip().lower().replace("-", "_")
    if placed:
        return placed

    paths: dict[str, list[tuple[str, ...]]] = {
        "weights": [
            ("memory", "storage", "weights"),
            ("memory", "weight_storage"),
            ("training", "storage", "weights"),
        ],
        "activations": [
            ("memory", "storage", "activations"),
            ("memory", "activation_storage"),
            ("training", "storage", "activations"),
        ],
        "gradients": [
            ("training", "storage", "gradients"),
            ("memory", "storage", "gradients"),
            ("memory", "gradient_storage"),
        ],
        "optimizer_state": [
            ("training", "storage", "optimizer_state"),
            ("memory", "storage", "optimizer_state"),
            ("memory", "optimizer_state_storage"),
        ],
    }
    for path in paths.get(role_u, []):
        value = _cfg_value(raw_config, *path, default=None)
        if value is not None and not isinstance(value, Mapping):
            text = str(value).strip().lower().replace("-", "_")
            aliases = {
                "block": "bram",
                "block_ram": "bram",
                "ultra": "uram",
                "ultra_ram": "uram",
                "external": "ddr",
                "external_ddr": "ddr",
                "host": "ps_or_host_memory",
            }
            return aliases.get(text, text)

    if role_u in {"inputs", "outputs", "labels"}:
        return "ps_or_host_memory"
    if role_u == "weights":
        return "bram"
    return "bram"


def _residency_contract_fields(
    *,
    role: str,
    storage: str,
    interface: str,
    policy: str,
    pipeline_mode: str,
    weights_mode: str,
    import_requested: bool,
    export_requested: bool,
    placement: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe residency, movement, mutability, and lifetime as separate axes.

    This is intentionally a compiler contract.  It does not claim that Vivado
    placed a buffer in the requested memory primitive or that board transfers
    have executed successfully.
    """
    role_u = role.lower()
    storage_u = str(storage or "").strip().lower().replace("-", "_")
    interface_u = _normalise(interface)
    policy_u = _normalise(policy, "full") or "full"
    training = pipeline_mode == "training_on_device"
    has_tensor_placement = placement is not None

    if role_u == "weights":
        if weights_mode in {"ddr_tiled", "ddr_tiled_mutable", "tiled", "tiled_mutable"} or storage_u == "ddr":
            primary_residency = "external_ddr"
            source_residency = "external_ddr"
            local_staging = "tile_buffer"
            movement_semantics = "tiled_fetch_and_writeback" if export_requested else "tiled_fetch"
        elif weights_mode in {"bram_import_full", "bram_import_export_full", "uram_import_full", "uram_import_export_full", "import", "import_export"} or (interface_u == "m_axi" and policy_u == "full"):
            primary_residency = "pl_on_chip"
            source_residency = "external_ddr"
            local_staging = "full_tensor_replica"
            movement_semantics = "full_preload_and_export" if export_requested else "full_preload"
        else:
            primary_residency = "pl_on_chip"
            source_residency = "compile_time_constants"
            local_staging = "full_tensor_static"
            movement_semantics = "compile_time_embedding"
        compute_mutated = bool(training)
        runtime_replaceable = bool(import_requested)
        lifetime = "runtime_training_session" if training else "design_or_runtime_session"
    elif role_u in {"inputs", "labels"}:
        primary_residency = "ps_or_host_memory"
        source_residency = "ps_or_host_memory"
        local_staging = "pl_tile_buffer" if policy_u == "tiled" else "pl_stream_or_input_buffer"
        movement_semantics = f"{interface_u or 'internal'}_{policy_u}_import"
        compute_mutated = False
        runtime_replaceable = True
        lifetime = "one_training_record" if role_u == "labels" else "one_model_invocation"
    elif role_u == "outputs":
        primary_residency = "ps_or_host_memory"
        source_residency = "pl_compute_output"
        local_staging = "pl_tile_buffer" if policy_u == "tiled" else "pl_stream_or_output_buffer"
        movement_semantics = f"{interface_u or 'internal'}_{policy_u}_export"
        compute_mutated = True
        runtime_replaceable = False
        lifetime = "one_model_invocation"
    elif role_u == "activations":
        primary_residency = "external_ddr" if storage_u == "ddr" else "pl_on_chip"
        source_residency = "pl_compute_pipeline"
        local_staging = "tile_buffer" if storage_u == "ddr" else "full_or_layer_local_buffer"
        movement_semantics = "internal_activation_spill_reload" if storage_u == "ddr" else "internal_on_chip_reuse"
        compute_mutated = True
        runtime_replaceable = False
        lifetime = "forward_backward_step" if training else "inference_invocation"
    elif role_u == "gradients":
        primary_residency = "external_ddr" if storage_u == "ddr" else "pl_on_chip"
        source_residency = "pl_backward_pipeline"
        local_staging = "tile_or_accumulator_buffer" if storage_u == "ddr" else "gradient_accumulator"
        movement_semantics = f"internal_accumulate_then_{interface_u}_{policy_u}_export" if export_requested else "internal_accumulate"
        compute_mutated = True
        runtime_replaceable = False
        lifetime = "batch_accumulation_window"
    elif role_u == "optimizer_state":
        primary_residency = "external_ddr" if storage_u == "ddr" else "pl_on_chip"
        source_residency = "runtime_initial_state"
        local_staging = "tile_or_state_buffer" if storage_u == "ddr" else "optimizer_state_buffer"
        movement_semantics = f"internal_update_then_{interface_u}_{policy_u}_export" if export_requested else "internal_persistent_state"
        compute_mutated = bool(training)
        runtime_replaceable = False
        lifetime = "runtime_training_session"
    else:
        primary_residency = "unknown"
        source_residency = "unknown"
        local_staging = "unknown"
        movement_semantics = "unknown"
        compute_mutated = False
        runtime_replaceable = False
        lifetime = "unknown"

    return {
        "primary_residency": primary_residency,
        "source_or_destination_residency": source_residency,
        "local_staging": local_staging,
        "movement_semantics": movement_semantics,
        "compute_mutated": bool(compute_mutated),
        "runtime_replaceable": bool(runtime_replaceable),
        "writeback_required": bool(compute_mutated and primary_residency == "external_ddr") or bool(export_requested),
        "lifetime": lifetime,
        "representation_status": "tensor_level_memory_plan" if has_tensor_placement else "aggregate_config_or_runtime_contract",
    }


def _write_residency_md(path: Path, payload: Mapping[str, Any]) -> None:
    rows = payload.get("tensor_classes", []) if isinstance(payload.get("tensor_classes", []), list) else []
    lines = [
        "# Memory residency and movement contract",
        "",
        "Residency, movement, mutability, and lifetime are independent compiler axes.",
        "",
        "| tensor class | primary residency | source/destination | local staging | movement | compute-mutated | lifetime | representation |",
        "|---|---|---|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {role} | `{primary_residency}` | `{source_or_destination_residency}` | `{local_staging}` | `{movement_semantics}` | `{compute_mutated}` | `{lifetime}` | `{representation_status}` |".format(**row)
        )
    limitations = payload.get("limitations", []) if isinstance(payload.get("limitations", []), list) else []
    if limitations:
        lines.extend(["", "## Current implementation boundaries"])
        lines.extend(f"- {item}" for item in limitations)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _edge_dict(edge: Any) -> dict[str, Any]:
    if isinstance(edge, Mapping):
        return dict(edge)
    if hasattr(edge, "to_dict"):
        try:
            value = edge.to_dict()
            return dict(value) if isinstance(value, Mapping) else {}
        except Exception:
            return {}
    return {
        "tensor_name": getattr(edge, "tensor_name", None),
        "direction": getattr(edge, "direction", None),
        "source": getattr(edge, "source", None),
        "destination": getattr(edge, "destination", None),
        "size_bytes": getattr(edge, "size_bytes", 0),
        "transfer_bytes": getattr(edge, "transfer_bytes", 0),
        "notes": getattr(edge, "notes", {}) or {},
    }


def _placement_by_kind(memory_plan: Any) -> dict[str, dict[str, Any]]:
    placements = getattr(memory_plan, "placements", []) or []
    by_kind: dict[str, dict[str, Any]] = {}
    for placement in placements:
        notes = dict(getattr(placement, "notes", {}) or {})
        kind = str(getattr(placement, "kind", notes.get("kind", "tensor"))).strip().lower()
        entry = {
            "tensor_name": str(getattr(placement, "tensor_name", kind)),
            "kind": kind,
            "storage": str(getattr(placement, "region", notes.get("region", "internal"))).strip().lower(),
            "size_bytes": int(getattr(placement, "size_bytes", 0) or 0),
            "double_buffer": bool(getattr(placement, "double_buffer", False)),
            "notes": notes,
        }
        by_kind.setdefault(kind, entry)
    return by_kind


def _edge_by_kind(communication_plan: Any) -> dict[str, dict[str, Any]]:
    edges = getattr(communication_plan, "edges", []) or []
    by_kind: dict[str, dict[str, Any]] = {}
    for edge in edges:
        data = _edge_dict(edge)
        notes = dict(data.get("notes", {}) or {})
        kind = str(notes.get("kind", data.get("tensor_name", "tensor"))).strip().lower()
        interface = _normalise(notes.get("interface") or notes.get("mode"))
        transport = _normalise(notes.get("transport"))
        if interface in {"", "internal"}:
            mode = _normalise(notes.get("mode"))
            if mode in {"axi_stream", "m_axi", "embedded"}:
                interface = mode
        if not transport:
            if interface == "axi_stream":
                transport = "dma"
            elif interface == "m_axi":
                transport = "ps_runtime"
            else:
                transport = "none"
        entry = {
            "tensor_name": str(data.get("tensor_name") or kind),
            "kind": kind,
            "direction": str(data.get("direction") or ""),
            "source": str(data.get("source") or ""),
            "destination": str(data.get("destination") or ""),
            "interface": interface or "internal",
            "transport": transport,
            "policy": _normalise(notes.get("policy"), "full") or "full",
            "mode": _normalise(notes.get("mode")) or interface or "internal",
            "size_bytes": int(data.get("size_bytes") or 0),
            "transfer_bytes": int(data.get("transfer_bytes") or data.get("size_bytes") or 0),
            "implemented_in_hls": bool(data.get("implemented_in_hls", False)),
            "notes": notes,
        }
        by_kind.setdefault(kind, entry)
    return by_kind


def _runtime_flags(commands: set[str], *, role: str, pipeline_mode: str, weights_mode: str) -> tuple[bool, bool, bool]:
    role_u = role.lower()
    is_training = pipeline_mode == "training_on_device"
    mutable_weights = weights_mode in {
        "bram_import_full",
        "bram_import_export_full",
        "uram_import_full",
        "uram_import_export_full",
        "ddr_tiled",
        "ddr_tiled_mutable",
        "import",
        "import_export",
        "tiled",
        "tiled_mutable",
    }
    if role_u in {"inputs", "labels"}:
        return True, False, False
    if role_u == "outputs":
        return False, True, False
    if role_u == "weights":
        return "import_weights" in commands or mutable_weights, "export_weights" in commands, mutable_weights
    if role_u == "gradients":
        return False, "export_gradients" in commands, False
    if role_u == "optimizer_state":
        return False, "export_optimizer_state" in commands, False
    if role_u == "activations":
        return is_training, False, False
    return False, False, False


def _tensor_entry(
    *,
    role: str,
    placement: dict[str, Any] | None,
    edge: dict[str, Any] | None,
    raw_config: Mapping[str, Any] | None,
    commands: set[str],
    pipeline_mode: str,
    weights_mode: str,
) -> dict[str, Any]:
    role_u = role.lower()
    import_requested, export_requested, mutable = _runtime_flags(
        commands, role=role_u, pipeline_mode=pipeline_mode, weights_mode=weights_mode
    )
    cfg = _movement_cfg_for_role(raw_config, role_u)
    storage = _storage_for_role(raw_config, role_u, placement)
    interface = _normalise(cfg.get("interface") or (edge or {}).get("interface"))
    transport = _normalise(cfg.get("transport") or (edge or {}).get("transport"))

    if role_u == "weights" and weights_mode in {"embedded", "bram_static", "uram_static"} and not import_requested and not export_requested:
        interface = "embedded"
        transport = "none"
    elif not interface:
        interface = str((edge or {}).get("interface") or "internal")
    if not transport:
        if interface == "axi_stream":
            transport = "dma"
        elif interface == "m_axi":
            transport = "ps_runtime"
        elif interface in {"embedded", "internal"}:
            transport = "none"
        else:
            transport = str((edge or {}).get("transport") or "none")

    requested = import_requested or export_requested or role_u in {"inputs", "outputs"} or (role_u == "labels" and pipeline_mode == "training_on_device")
    if role_u in {"gradients", "optimizer_state"} and not export_requested:
        interface = "not_requested"
        transport = "not_requested"
    if role_u == "labels" and pipeline_mode != "training_on_device":
        requested = False
        interface = "not_requested"
        transport = "not_requested"

    reason_parts = []
    if role_u == "weights":
        reason_parts.append(f"weights_mode={weights_mode}")
    if commands:
        reason_parts.append("runtime.sequence=" + ",".join(sorted(commands)))
    if not requested:
        reason_parts.append("not requested by pipeline/runtime")

    policy = _normalise(cfg.get("policy", (edge or {}).get("policy", "full")), "full") or "full"
    tiled = _cfg_tiled_enabled(cfg) or policy == "tiled"
    if tiled and policy in {"", "full"}:
        policy = "tiled"
    tile_size = _cfg_tile_size(cfg, 64) if tiled else None

    residency = _residency_contract_fields(
        role=role_u,
        storage=storage,
        interface=interface,
        policy=policy,
        pipeline_mode=pipeline_mode,
        weights_mode=weights_mode,
        import_requested=bool(import_requested),
        export_requested=bool(export_requested),
        placement=placement,
    )

    return {
        "role": role_u,
        "tensor_name": (placement or edge or {}).get("tensor_name", role_u),
        "storage": storage,
        "interface": interface,
        "transport": transport,
        "direction": (edge or {}).get("direction"),
        "policy": policy,
        "tiled": bool(tiled),
        "tile_size": tile_size,
        "size_bytes": int((placement or edge or {}).get("size_bytes") or 0),
        "transfer_bytes": int((edge or {}).get("transfer_bytes") or (placement or {}).get("size_bytes") or 0),
        "mutable": bool(mutable or residency["compute_mutated"] or residency["runtime_replaceable"]),
        "import_requested": bool(import_requested),
        "export_requested": bool(export_requested),
        "requested": bool(requested),
        "affects_board_fit": bool(requested and interface not in {"embedded", "internal", "none", "not_requested"}),
        **residency,
        "runtime_modes": [
            {"command": command, "mode": _RUNTIME_MODE_NUMBERS.get(command)}
            for command in sorted(commands)
            if (role_u == "weights" and command in {"import_weights", "export_weights"})
            or (role_u == "gradients" and command == "export_gradients")
            or (role_u == "optimizer_state" and command == "export_optimizer_state")
            or (role_u in {"inputs", "outputs"} and command in {"run_inference", "run_training", "accumulate_gradients"})
            or (role_u == "labels" and command in {"run_training", "accumulate_gradients"})
        ],
        "source_config": cfg,
        "reason": "; ".join(reason_parts) if reason_parts else "compiler default movement path",
    }


def _write_md(path: Path, title: str, rows: list[dict[str, Any]], notes: list[str]) -> None:
    lines = [f"# {title}", "", "| tensor | storage | interface | transport | import | export | board-fit | reason |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            "| {role} | `{storage}` | `{interface}` | `{transport}` | `{import_requested}` | `{export_requested}` | `{affects_board_fit}` | {reason} |".format(
                **{**row, "reason": str(row.get("reason", "")).replace("|", "/")}
            )
        )
    if notes:
        lines.extend(["", "## Notes"])
        lines.extend([f"- {note}" for note in notes])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def emit_data_movement_reports(
    out_dir: str | Path,
    *,
    raw_config: Mapping[str, Any] | None,
    pipeline_mode: str,
    weights_mode: str,
    memory_plan: Any,
    communication_plan: Any,
    runtime_sequence: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Emit the Sprint-P tensor movement and PS/PL transfer truth reports.

    The reports are generated from the already-resolved memory plan,
    communication plan, and runtime sequence.  They do not claim board execution;
    they only make requested/absent movement paths explicit and machine-checkable.
    """

    root = Path(out_dir)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    pipeline = str(pipeline_mode or "inference").strip().lower()
    wm = _normalise(weights_mode or "embedded")
    commands = _commands(runtime_sequence or {})
    placements = _placement_by_kind(memory_plan)
    edges = _edge_by_kind(communication_plan)

    role_to_kind = {
        "inputs": "input",
        "outputs": "output",
        "weights": "weight",
        "labels": "target",
        "gradients": "gradient",
        "optimizer_state": "optimizer_state",
        "activations": "activation",
    }
    tensor_rows = [
        _tensor_entry(
            role=role,
            placement=placements.get(kind),
            edge=edges.get(kind),
            raw_config=raw_config,
            commands=commands,
            pipeline_mode=pipeline,
            weights_mode=wm,
        )
        for role, kind in role_to_kind.items()
    ]

    data_payload = {
        "schema_version": 1,
        "artifact_kind": "data_movement_plan",
        "status": "validated",
        "truth_boundary": "Compiler movement contract only; real HLS/Vivado/FPGA success requires corresponding truth artifacts.",
        "pipeline_mode": pipeline,
        "weights_mode": wm,
        "runtime_commands": sorted(commands),
        "tensors": tensor_rows,
        "checks": {
            "unrequested_gradients_export_absent": not any(r["role"] == "gradients" and r["export_requested"] for r in tensor_rows) or "export_gradients" in commands,
            "unrequested_optimizer_export_absent": not any(r["role"] == "optimizer_state" and r["export_requested"] for r in tensor_rows) or "export_optimizer_state" in commands,
            "embedded_weights_have_no_runtime_import": not (wm in {"embedded", "bram_static", "uram_static"} and "import_weights" in commands),
        },
    }

    placement_kinds = sorted({str(getattr(item, "kind", "")).strip().lower() for item in (getattr(memory_plan, "placements", []) or []) if getattr(item, "kind", None)})
    training_state_tensor_level = {"gradient", "optimizer_state"}.issubset(set(placement_kinds))
    residency_payload = {
        "schema_version": 1,
        "artifact_kind": "memory_residency_contract",
        "status": "available",
        "pipeline_mode": pipeline,
        "weights_mode": wm,
        "axes": {
            "residency": "where the authoritative tensor/state lives during compute",
            "movement": "how data reaches or leaves that residency",
            "mutability": "whether compute or runtime commands may change the state",
            "lifetime": "the interval over which the state must persist",
        },
        "tensor_classes": tensor_rows,
        "checks": {
            "residency_and_movement_are_separate": all(bool(row.get("primary_residency")) and bool(row.get("movement_semantics")) for row in tensor_rows),
            "ddr_weight_residency_uses_tiled_movement": not any(
                row.get("role") == "weights"
                and row.get("primary_residency") == "external_ddr"
                and row.get("policy") != "tiled"
                for row in tensor_rows
            ),
            "full_preload_is_classified_as_on_chip_compute_residency": not any(
                row.get("role") == "weights"
                and row.get("movement_semantics") in {"full_preload", "full_preload_and_export"}
                and row.get("primary_residency") != "pl_on_chip"
                for row in tensor_rows
            ),
            "training_state_has_tensor_level_memory_plan_entries": bool(training_state_tensor_level),
        },
        "limitations": [
            "HOST/PS placements are logical endpoints; they do not by themselves prove a specific physical PS-DDR allocation.",
            "BRAM/URAM placement is a compiler request until generated HLS bindings and synthesis reports confirm the realized primitive.",
            "External-DDR execution is a paper-safe runtime claim only after generated HLS, Vivado implementation, and physical-board validation pass.",
            *(
                []
                if pipeline != "training_on_device" or training_state_tensor_level
                else [
                    "Gradients and optimizer state are currently represented by aggregate training storage fields rather than tensor-level MemoryPlan placements; the scientific IR sprint must promote them to explicit state objects."
                ]
            ),
        ],
    }

    transfers = []
    for row in tensor_rows:
        if row.get("requested"):
            transfers.append(
                {
                    "tensor": row["role"],
                    "interface": row["interface"],
                    "transport": row["transport"],
                    "direction": row.get("direction"),
                    "size_bytes": row.get("size_bytes"),
                    "transfer_bytes": row.get("transfer_bytes"),
                    "policy": row.get("policy"),
                    "tiled": bool(row.get("tiled")),
                    "tile_size": row.get("tile_size"),
                    "runtime_modes": row.get("runtime_modes", []),
                    "requires_dma": row.get("transport") == "dma" or row.get("interface") == "axi_stream",
                    "requires_m_axi": row.get("interface") == "m_axi",
                    "requires_tlast": row.get("interface") == "axi_stream",
                    "requires_tiled_ddr": row.get("interface") == "m_axi" and bool(row.get("tiled")),
                    "affects_board_fit": row.get("affects_board_fit"),
                }
            )
    transfer_payload = {
        "schema_version": 1,
        "artifact_kind": "ps_pl_transfer_plan",
        "status": "validated",
        "truth_boundary": "PS/PL transfer contract derived from resolved movement decisions; no board execution is claimed here.",
        "pipeline_mode": pipeline,
        "runtime_commands": sorted(commands),
        "transfers": transfers,
        "requirements": {
            "axi_dma": any(t["requires_dma"] for t in transfers),
            "m_axi": any(t["requires_m_axi"] for t in transfers),
            "tlast": any(t["requires_tlast"] for t in transfers),
            "tiled": any(bool(t.get("tiled")) for t in transfers),
            "ddr_tiled": any(bool(t.get("requires_tiled_ddr")) for t in transfers),
            "tiled_transfers": [t["tensor"] for t in transfers if bool(t.get("tiled"))],
            "runtime_buffers": [t["tensor"] for t in transfers if t.get("interface") in {"m_axi", "axi_stream"}],
        },
    }

    data_json = reports_dir / "data_movement_plan.json"
    transfer_json = reports_dir / "ps_pl_transfer_plan.json"
    residency_json = reports_dir / "memory_residency_contract.json"
    residency_md = reports_dir / "memory_residency_contract.md"
    data_json.write_text(json.dumps(data_payload, indent=2, sort_keys=True), encoding="utf-8")
    transfer_json.write_text(json.dumps(transfer_payload, indent=2, sort_keys=True), encoding="utf-8")
    residency_json.write_text(json.dumps(residency_payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_residency_md(residency_md, residency_payload)
    _write_md(
        reports_dir / "data_movement_plan.md",
        "Data movement plan",
        tensor_rows,
        [data_payload["truth_boundary"]],
    )
    _write_md(
        reports_dir / "ps_pl_transfer_plan.md",
        "PS/PL transfer plan",
        tensor_rows,
        [transfer_payload["truth_boundary"]],
    )

    return {
        "data_movement_plan_json": str(data_json),
        "data_movement_plan_md": str(reports_dir / "data_movement_plan.md"),
        "ps_pl_transfer_plan_json": str(transfer_json),
        "ps_pl_transfer_plan_md": str(reports_dir / "ps_pl_transfer_plan.md"),
        "memory_residency_contract_json": str(residency_json),
        "memory_residency_contract_md": str(residency_md),
    }


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            return dict(value) if isinstance(value, Mapping) else {}
    except Exception:
        return {}
    return {}


def _read_all_hls_source(root: Path) -> str:
    hls = root / "hls"
    if not hls.exists():
        return ""
    parts: list[str] = []
    for path in sorted(hls.rglob("*")):
        if path.is_file() and path.suffix in {".cpp", ".h", ".hpp"}:
            try:
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
    return "\n".join(parts)


def _read_top_hls_source(root: Path) -> str:
    """Return the generated top implementation source, not the CSim harness.

    The HLS tree also contains ``tb.cpp`` declarations that may use stream
    variable names even when the synthesized top in ``deeplearn.cpp`` is an
    m_axi design.  Primary interface validation must therefore read the actual
    top implementation file first and avoid using testbench declarations as top
    port evidence.
    """
    candidates = [
        root / "hls" / "src" / "deeplearn.cpp",
        root / "hls" / "src" / "top.cpp",
    ]
    for path in candidates:
        try:
            if path.exists():
                return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
    return ""


def _tensor_by_role(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("tensors", []) if isinstance(payload.get("tensors", []), list) else []:
        if isinstance(row, Mapping):
            out[str(row.get("role", "")).strip().lower()] = dict(row)
    return out


def _runtime_buffer_plan(root: Path) -> tuple[bool, set[str]]:
    """Return whether buffer_plan.json is present plus declared buffer names.

    An empty ``buffers`` list is a valid runtime package state for examples that
    only use primary input/output runtime_io and do not request auxiliary
    labels/weights/gradients/optimizer buffers.  Do not treat an empty valid
    buffer plan as a missing report.
    """
    path = root / "runtime_package" / "buffer_plan.json"
    payload = _load_json_file(path)
    if not path.exists() or not isinstance(payload, Mapping):
        return False, set()
    entries = payload.get("buffers", [])
    if entries is None:
        entries = []
    names: set[str] = set()
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, Mapping) and entry.get("name") is not None:
            names.add(str(entry.get("name")))
    return True, names


def _check(name: str, expected: bool, actual: bool, *, evidence: str) -> dict[str, Any]:
    return {
        "name": name,
        "expected": bool(expected),
        "actual": bool(actual),
        "passed": bool(expected) == bool(actual),
        "evidence": evidence,
    }


def _check_value(name: str, expected: Any, actual: Any, *, evidence: str) -> dict[str, Any]:
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "passed": expected == actual,
        "evidence": evidence,
    }



def _runtime_package_manifest(root: Path) -> dict[str, Any]:
    return _load_json_file(root / "runtime_package" / "package_manifest.json")


def _runtime_io_entry(manifest: Mapping[str, Any], role: str) -> dict[str, Any]:
    runtime_io = manifest.get("runtime_io", {}) if isinstance(manifest.get("runtime_io", {}), Mapping) else {}
    if role == "inputs":
        entry = (runtime_io.get("inputs", {}) or {}).get("import", {}) if isinstance(runtime_io.get("inputs", {}), Mapping) else {}
    elif role == "outputs":
        entry = (runtime_io.get("outputs", {}) or {}).get("export", {}) if isinstance(runtime_io.get("outputs", {}), Mapping) else {}
    else:
        entry = {}
    return dict(entry) if isinstance(entry, Mapping) else {}


def _movement_kind(row: Mapping[str, Any]) -> str:
    interface = _normalise(row.get("interface"))
    policy = _normalise(row.get("policy"), "full") or "full"
    if interface == "m_axi":
        return "m_axi_tiled" if bool(row.get("tiled")) or policy == "tiled" else "m_axi_full"
    if interface == "axi_stream":
        return "axi_stream_tiled" if bool(row.get("tiled")) or policy == "tiled" else "axi_stream_full"
    return interface or "internal"




def _source_macro_int(source: str, macro: str) -> int | None:
    import re

    patterns = [
        rf"static\s+const\s+int\s+{re.escape(macro)}\s*=\s*(\d+)\s*;",
        rf"#define\s+{re.escape(macro)}\s+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, source)
        if match:
            return _positive_int(match.group(1), None)
    return None


def _tile_macro_for(role: str, kind: str) -> str | None:
    candidates = _tile_macro_candidates_for(role, kind)
    return candidates[0] if candidates else None


def _tile_macro_candidates_for(role: str, kind: str) -> list[str]:
    prefix = "INPUT" if role == "inputs" else "OUTPUT"
    if kind == "m_axi_tiled":
        # Training-on-device tops use explicit training tile macros while the
        # original inference m_axi path uses the generic macros.  Keep both
        # accepted so movement validation reflects the generated artifact
        # status instead of hard-coding one backend spelling.
        return [f"FPGAI_{prefix}_TILE_SIZE", f"FPGAI_TRAIN_{prefix}_TILE_SIZE"]
    if kind == "axi_stream_tiled":
        return [f"FPGAI_AXIS_{prefix}_TILE_SIZE", f"FPGAI_TRAIN_AXIS_{prefix}_TILE_SIZE"]
    return []


def _source_macro_int_any(source: str, macros: list[str]) -> tuple[int | None, str | None]:
    for macro in macros:
        value = _source_macro_int(source, macro)
        if value is not None:
            return value, macro
    return None, macros[0] if macros else None


def _source_has_training_hybrid_axi(source: str) -> bool:
    signatures = _top_kernel_signatures(source)
    if not signatures:
        return False
    joined = "\n".join(signatures)
    has_training_streams = (
        "hls::stream" in joined
        and ("& in" in joined or "&in" in joined)
        and ("& out" in joined or "&out" in joined)
        and ("& aux" in joined or "&aux" in joined)
    )
    has_tensor_m_axi = any(
        token in source
        for token in (
            "m_axi port=input_mem",
            "m_axi port=label_mem",
            "m_axi port=output_mem",
            "m_axi port=weights_mem",
            "m_axi port=gradients_mem",
            "m_axi port=optimizer_state_mem",
        )
    )
    return bool(has_training_streams and has_tensor_m_axi)


def _top_kernel_signatures(source: str) -> list[str]:
    """Return extern deeplearn argument lists from generated top declarations.

    The aggregated HLS source also contains helper functions and testbench stream
    variables.  Primary I/O interface checks must look at the top-kernel
    signature only; otherwise helper/testbench tokens can be misread as top
    ports (for example a helper ``hls::stream<axis_t>& in`` in an m_axi design).
    """
    import re

    signatures: list[str] = []
    pattern = re.compile(r'(?:extern\s+"C"\s+)?void\s+deeplearn\s*\((.*?)\)\s*(?:;|\{)', re.S)
    for match in pattern.finditer(source):
        signatures.append(match.group(1))
    return signatures


def _signature_has_stream_port(signature: str, names: tuple[str, ...]) -> bool:
    import re

    for name in names:
        if re.search(rf"hls::stream\s*<\s*axis_t\s*>\s*&\s*{re.escape(name)}\b", signature):
            return True
    return False


def _source_io_actuals(source: str, role: str) -> dict[str, bool]:
    """Return generated-source evidence for primary I/O movement.

    Inference and training top functions intentionally use different stream port
    names today: inference uses ``in_stream``/``out_stream`` while the training
    top uses ``in``/``out`` plus the auxiliary label stream.  Check only the
    top-kernel ``deeplearn`` signature so helper/testbench stream variables do
    not create false positives for m_axi examples.
    """
    signatures = _top_kernel_signatures(source)
    # If no top signature is found, do not fall back to scanning the whole
    # source.  Whole-source scanning creates false positives from helpers or
    # testbench stream variables in m_axi designs.
    input_stream_port = any(
        _signature_has_stream_port(sig, ("in_stream", "in"))
        for sig in signatures
    )
    output_stream_port = any(
        _signature_has_stream_port(sig, ("out_stream", "out"))
        for sig in signatures
    )
    if role == "inputs":
        return {
            "m_axi_port": "m_axi port=input_mem" in source and "input_mem" in source,
            "stream_port": input_stream_port,
            "m_axi_tiled": ("FPGAI_INPUT_TILE_SIZE" in source or "FPGAI_TRAIN_INPUT_TILE_SIZE" in source) and ("m_axi tiled input import" in source or "training tiled input import" in source),
            "axis_tiled": ("FPGAI_AXIS_INPUT_TILE_SIZE" in source or "FPGAI_TRAIN_AXIS_INPUT_TILE_SIZE" in source) and ("in_stream.read()" in source or "read_f32(in)" in source),
        }
    return {
        "m_axi_port": "m_axi port=output_mem" in source and "output_mem" in source,
        "stream_port": output_stream_port,
        "m_axi_tiled": ("FPGAI_OUTPUT_TILE_SIZE" in source or "FPGAI_TRAIN_OUTPUT_TILE_SIZE" in source) and ("m_axi tiled output export" in source or "training tiled output export" in source),
        "axis_tiled": ("FPGAI_AXIS_OUTPUT_TILE_SIZE" in source or "FPGAI_TRAIN_AXIS_OUTPUT_TILE_SIZE" in source) and ("out_stream.write(packet)" in source or "write_f32(out" in source),
        "tlast": "packet.last" in source or "write_f32(out" in source,
    }

def _append_source_io_checks(checks: list[dict[str, Any]], source: str, role: str, row: Mapping[str, Any]) -> None:
    if not row or not bool(row.get("requested")):
        return
    kind = _movement_kind(row)
    actual = _source_io_actuals(source, role)
    prefix = "input" if role == "inputs" else "output"
    if kind.startswith("m_axi"):
        checks.append(_check(f"cpp_{prefix}_m_axi_port_matches_plan", True, actual["m_axi_port"], evidence=f"{prefix}_mem m_axi interface pragma"))
        hybrid_training_axi = _source_has_training_hybrid_axi(source) and actual["m_axi_port"]
        checks.append(_check(
            f"cpp_{prefix}_stream_port_absent_for_m_axi",
            False,
            actual["stream_port"] and not hybrid_training_axi,
            evidence=(
                f"hls::stream<axis_t>& {'in_stream/in' if role == 'inputs' else 'out_stream/out'}"
                + ("; accepted as training hybrid control/runtime stream because tensor payload uses m_axi" if hybrid_training_axi else "")
            ),
        ))
        if kind == "m_axi_tiled":
            macros = _tile_macro_candidates_for(role, kind)
            checks.append(_check(f"cpp_{prefix}_m_axi_tiling_matches_plan", True, actual["m_axi_tiled"], evidence=f"{'/'.join(macros)} and m_axi tiled {prefix}"))
            expected_tile = row.get("tile_size")
            if expected_tile:
                actual_tile, macro = _source_macro_int_any(source, macros)
                checks.append(_check_value(f"cpp_{prefix}_m_axi_tile_size_matches_plan", int(expected_tile), actual_tile, evidence=macro or "tile macro"))
    elif kind.startswith("axi_stream"):
        checks.append(_check(f"cpp_{prefix}_stream_port_matches_plan", True, actual["stream_port"], evidence=f"hls::stream<axis_t>& {'in_stream/in' if role == 'inputs' else 'out_stream/out'}"))
        checks.append(_check(f"cpp_{prefix}_m_axi_port_absent_for_stream", False, actual["m_axi_port"], evidence=f"{prefix}_mem m_axi interface pragma"))
        if kind == "axi_stream_tiled":
            checks.append(_check(f"cpp_{prefix}_axis_tiling_matches_plan", True, actual["axis_tiled"], evidence=f"FPGAI_AXIS_{prefix.upper()}_TILE_SIZE and stream tile loop"))
            expected_tile = row.get("tile_size")
            if expected_tile:
                macro = _tile_macro_for(role, kind)
                checks.append(_check_value(f"cpp_{prefix}_axis_tile_size_matches_plan", int(expected_tile), _source_macro_int(source, macro or ""), evidence=macro or "tile macro"))
        if role == "outputs":
            checks.append(_check("cpp_output_stream_tlast_matches_plan", True, actual["tlast"], evidence="packet.last/write_f32(out)"))


def _append_source_aux_role_checks(checks: list[dict[str, Any]], source: str, role: str, row: Mapping[str, Any]) -> None:
    """Validate non-primary I/O tensor roles against generated HLS tokens.

    These checks intentionally use concrete generated artifact tokens instead of
    report-only intent.  They extend Sprint P's contract beyond input/output:
    requested roles must have their HLS path, and unrequested export/import paths
    must stay absent.
    """
    role_u = role.lower()
    requested = bool(row.get("requested"))
    import_requested = bool(row.get("import_requested"))
    export_requested = bool(row.get("export_requested"))
    interface = _normalise(row.get("interface"))
    kind = _movement_kind(row)

    if role_u == "labels":
        label_m_axi = "label_mem" in source and "m_axi port=label_mem" in source
        label_stream = "axis_label_tile" in source or "AXI-stream tiled label import" in source or "read_f32(aux)" in source
        if not requested:
            checks.append(_check("cpp_labels_path_absent_when_not_requested", False, label_m_axi or label_stream, evidence="label_mem/axis_label_tile/read_f32(aux)"))
            return
        if kind.startswith("m_axi"):
            checks.append(_check("cpp_labels_m_axi_port_matches_plan", True, label_m_axi, evidence="label_mem m_axi interface pragma"))
            hybrid_training_axi = _source_has_training_hybrid_axi(source) and label_m_axi
            checks.append(_check(
                "cpp_labels_stream_path_absent_for_m_axi",
                False,
                label_stream and not hybrid_training_axi,
                evidence="axis_label_tile/read_f32(aux)" + ("; accepted as training hybrid auxiliary stream because label_mem is the tensor payload path" if hybrid_training_axi else ""),
            ))
        elif kind.startswith("axi_stream"):
            checks.append(_check("cpp_labels_stream_path_matches_plan", True, label_stream, evidence="axis_label_tile/read_f32(aux)"))
            checks.append(_check("cpp_labels_m_axi_port_absent_for_stream", False, label_m_axi, evidence="label_mem m_axi interface pragma"))
        return

    if role_u == "weights":
        weight_port = "weights_mem" in source and "m_axi port=weights_mem" in source
        import_mode = "FPGAI_MODE_IMPORT_WEIGHTS" in source or ("mode 1" in source and "import_weights" in source)
        export_mode = "FPGAI_MODE_EXPORT_WEIGHTS" in source or "export_weights" in source
        embedded = interface == "embedded"
        runtime_commands = {
            str(item.get("command", "")).strip().lower().replace("-", "_")
            for item in row.get("runtime_modes", [])
            if isinstance(item, Mapping)
        }
        explicit_import_command = "import_weights" in runtime_commands
        explicit_export_command = "export_weights" in runtime_commands
        if embedded and not import_requested and not export_requested:
            checks.append(_check("cpp_embedded_weights_have_no_runtime_weight_port", False, weight_port or import_mode or export_mode, evidence="weights_mem/import/export weight mode tokens"))
            return
        if import_requested or export_requested or bool(row.get("mutable")):
            checks.append(_check("cpp_weight_m_axi_port_matches_plan", True, weight_port, evidence="weights_mem m_axi interface pragma"))
            # Weight import/export handlers are available modes in the generated
            # training top.  Their presence does not mean a transfer was
            # requested for this run.  Validate required command handlers when
            # runtime_modes/runtime.sequence requests them, but never fail merely
            # because optional handlers are present and unused.
            if explicit_import_command:
                checks.append(_check("cpp_weight_import_mode_matches_plan", True, import_mode, evidence="FPGAI_MODE_IMPORT_WEIGHTS or mode 1 import_weights"))
            if explicit_export_command:
                checks.append(_check("cpp_weight_export_mode_matches_plan", True, export_mode, evidence="FPGAI_MODE_EXPORT_WEIGHTS or export_weights"))
        return

    if role_u == "gradients":
        grad_port = "gradients_mem" in source and "m_axi port=gradients_mem" in source
        grad_mode = "FPGAI_MODE_EXPORT_GRADIENTS" in source or "export_gradients" in source
        grad_tiled = "gradient_export_tile" in source or "gradient export tiled" in source
        if not export_requested:
            checks.append(_check("cpp_gradient_export_path_absent_when_not_requested", False, grad_port or grad_mode, evidence="gradients_mem/export_gradients"))
            return
        checks.append(_check("cpp_gradient_export_mode_matches_plan", True, grad_mode and "gradients_mem" in source, evidence="FPGAI_MODE_EXPORT_GRADIENTS/export_gradients and gradients_mem"))
        if interface == "m_axi":
            checks.append(_check("cpp_gradient_m_axi_port_matches_plan", True, grad_port, evidence="gradients_mem m_axi interface pragma"))
        if kind == "m_axi_tiled":
            checks.append(_check("cpp_gradient_tiled_export_matches_plan", True, grad_tiled, evidence="gradient_export_tile or tiled gradient export comment"))
        return

    if role_u == "optimizer_state":
        opt_port = "optimizer_state_mem" in source and "m_axi port=optimizer_state_mem" in source
        opt_mode = "FPGAI_MODE_EXPORT_OPTIMIZER_STATE" in source or "export_optimizer_state" in source or "mode 9" in source
        if not export_requested:
            checks.append(_check("cpp_optimizer_state_export_path_absent_when_not_requested", False, opt_port or opt_mode, evidence="optimizer_state_mem/export_optimizer_state"))
            return
        checks.append(_check("cpp_optimizer_state_export_mode_matches_plan", True, opt_mode and "optimizer_state_mem" in source, evidence="FPGAI_MODE_EXPORT_OPTIMIZER_STATE/export_optimizer_state and optimizer_state_mem"))
        if interface == "m_axi":
            checks.append(_check("cpp_optimizer_state_m_axi_port_matches_plan", True, opt_port, evidence="optimizer_state_mem m_axi interface pragma"))
        return


def _buffer_name_for_role(role: str) -> str | None:
    return {
        "labels": "labels",
        "weights": "weights",
        "gradients": "gradients_mem",
        "optimizer_state": "optimizer_state_mem",
    }.get(role)


def _append_runtime_aux_role_checks(checks: list[dict[str, Any]], names: set[str], role: str, row: Mapping[str, Any]) -> None:
    buffer_name = _buffer_name_for_role(role)
    if not buffer_name:
        return
    if role == "weights":
        expected = bool(row.get("import_requested") or row.get("export_requested") or row.get("mutable"))
    elif role in {"gradients", "optimizer_state"}:
        expected = bool(row.get("export_requested"))
    else:
        expected = bool(row.get("requested"))
    checks.append(_check(f"runtime_{role}_buffer_matches_plan", expected, buffer_name in names, evidence=f"runtime_package/buffer_plan.json contains {buffer_name}"))


def _append_runtime_io_checks(checks: list[dict[str, Any]], manifest: Mapping[str, Any], role: str, row: Mapping[str, Any]) -> None:
    if not row or not bool(row.get("requested")) or _normalise(row.get("interface")) not in {"m_axi", "axi_stream"}:
        return
    entry = _runtime_io_entry(manifest, role)
    prefix = "input" if role == "inputs" else "output"
    expected_interface = _normalise(row.get("interface"))
    expected_transport = _normalise(row.get("transport"))
    expected_policy = _normalise(row.get("policy"), "full") or "full"
    expected_resolved = (
        f"m_axi_{'import' if role == 'inputs' else 'export'}_{'tiled' if bool(row.get('tiled')) or expected_policy == 'tiled' else 'full'}"
        if expected_interface == "m_axi"
        else f"dma_stream_{'import' if role == 'inputs' else 'export'}_{'tiled' if bool(row.get('tiled')) or expected_policy == 'tiled' else 'full'}"
    )
    checks.append(_check(f"runtime_{prefix}_interface_matches_plan", True, _normalise(entry.get("interface")) == expected_interface, evidence="runtime_package/package_manifest.json runtime_io"))
    checks.append(_check(f"runtime_{prefix}_transport_matches_plan", True, _normalise(entry.get("transport")) == expected_transport, evidence="runtime_package/package_manifest.json runtime_io"))
    checks.append(_check(f"runtime_{prefix}_resolved_movement_matches_plan", True, str(entry.get("resolved")) == expected_resolved, evidence="runtime_package/package_manifest.json runtime_io.resolved"))

def _movement_source_checks(root: Path, data_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = _read_all_hls_source(root)
    top_source = _read_top_hls_source(root)
    tensors = _tensor_by_role(data_payload)
    checks: list[dict[str, Any]] = []

    if source:
        # Primary input/output interface evidence must come from the generated
        # top implementation only.  The full HLS tree includes tb.cpp extern
        # declarations and helper streams that are not synthesized ports.
        io_source = top_source or ""
        _append_source_io_checks(checks, io_source, "inputs", tensors.get("inputs", {}))
        _append_source_io_checks(checks, io_source, "outputs", tensors.get("outputs", {}))
        for role in ("labels", "weights", "gradients", "optimizer_state"):
            _append_source_aux_role_checks(checks, source, role, tensors.get(role, {}))
    else:
        checks.append({"name": "cpp_source_available", "expected": True, "actual": False, "passed": False, "evidence": "hls/**/*.cpp|h|hpp"})
    return checks


def _movement_runtime_checks(root: Path, data_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    buffer_plan_available, names = _runtime_buffer_plan(root)
    manifest = _runtime_package_manifest(root)
    tensors = _tensor_by_role(data_payload)
    checks: list[dict[str, Any]] = []

    if buffer_plan_available:
        for role in ("labels", "weights", "gradients", "optimizer_state"):
            _append_runtime_aux_role_checks(checks, names, role, tensors.get(role, {}))
    else:
        checks.append({"name": "runtime_buffer_plan_available", "expected": True, "actual": False, "passed": False, "evidence": "runtime_package/buffer_plan.json"})
    if manifest:
        _append_runtime_io_checks(checks, manifest, "inputs", tensors.get("inputs", {}))
        _append_runtime_io_checks(checks, manifest, "outputs", tensors.get("outputs", {}))
    else:
        checks.append({"name": "runtime_package_manifest_available", "expected": True, "actual": False, "passed": False, "evidence": "runtime_package/package_manifest.json"})
    return checks


def _movement_vivado_checks(root: Path, transfer_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    vivado = _load_json_file(root / "reports" / "vivado_bd_validation.json")
    req = transfer_payload.get("requirements", {}) if isinstance(transfer_payload.get("requirements", {}), Mapping) else {}
    checks: list[dict[str, Any]] = []
    if not vivado:
        checks.append({"name": "vivado_bd_validation_available", "expected": True, "actual": False, "passed": False, "evidence": "reports/vivado_bd_validation.json"})
        return checks
    if vivado.get("status") == "not_requested":
        checks.append({"name": "vivado_not_requested_skips_interface_check", "expected": True, "actual": True, "passed": True, "evidence": "build.stages.vivado_project=false"})
        return checks
    bd_checks = vivado.get("checks", {}) if isinstance(vivado.get("checks", {}), Mapping) else {}
    checks.append(_check("vivado_axi_dma_requirement_matches_transfer_plan", bool(req.get("axi_dma")), bool(bd_checks.get("axi_dma_required")), evidence="vivado_bd_validation.checks.axi_dma_required"))
    checks.append(_check("vivado_m_axi_requirement_matches_transfer_plan", bool(req.get("m_axi")), bool(bd_checks.get("m_axi_memory_required")), evidence="vivado_bd_validation.checks.m_axi_memory_required"))
    return checks


def _movement_board_fit_checks(root: Path, transfer_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    board_fit = _load_json_file(root / "reports" / "board_fit.json")
    req = transfer_payload.get("requirements", {}) if isinstance(transfer_payload.get("requirements", {}), Mapping) else {}
    checks: list[dict[str, Any]] = []
    if not board_fit:
        checks.append({"name": "board_fit_available", "expected": True, "actual": False, "passed": False, "evidence": "reports/board_fit.json"})
        return checks
    iface = ((board_fit.get("derived_requirements") or {}).get("interface_requirements") or {}) if isinstance(board_fit.get("derived_requirements"), Mapping) else {}
    checks.append({
        "name": "board_fit_counts_unrequested_paths_as_zero_or_absent",
        "expected": True,
        "actual": not bool(req.get("axi_dma")) or int(iface.get("dma_count") or 0) >= 1,
        "passed": not bool(req.get("axi_dma")) or int(iface.get("dma_count") or 0) >= 1,
        "evidence": "board_fit.derived_requirements.interface_requirements.dma_count",
    })
    checks.append({
        "name": "board_fit_counts_m_axi_when_transfer_plan_requires_it",
        "expected": bool(req.get("m_axi")),
        "actual": int(iface.get("m_axi_bundles") or 0) >= 1,
        "passed": (not bool(req.get("m_axi"))) or int(iface.get("m_axi_bundles") or 0) >= 1,
        "evidence": "board_fit.derived_requirements.interface_requirements.m_axi_bundles",
    })
    return checks


def emit_movement_contract_validation(
    out_dir: str | Path,
    *,
    data_movement_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Cross-check Sprint-P movement reports against generated artifacts.

    This is still structural validation. It proves the compiler artifacts agree
    with the movement contract; it does not claim HLS synthesis, Vivado
    implementation, bitstream generation, or board execution.
    """
    root = Path(out_dir)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    data_payload = _load_json_file(reports_dir / "data_movement_plan.json")
    transfer_payload = _load_json_file(reports_dir / "ps_pl_transfer_plan.json")

    sections = {
        "generated_cpp": _movement_source_checks(root, data_payload),
        "runtime_package": _movement_runtime_checks(root, data_payload),
        "vivado_bd": _movement_vivado_checks(root, transfer_payload),
        "board_fit": _movement_board_fit_checks(root, transfer_payload),
    }
    all_checks = [check for checks in sections.values() for check in checks]
    failed_checks = [
        {
            "section": section,
            "name": str(check.get("name")),
            "expected": check.get("expected"),
            "actual": check.get("actual"),
            "evidence": check.get("evidence"),
        }
        for section, checks in sections.items()
        for check in checks
        if not bool(check.get("passed"))
    ]
    status = "passed" if not failed_checks else "failed"
    payload = {
        "schema_version": 2,
        "artifact_kind": "movement_contract_validation",
        "status": status,
        "passed": status == "passed",
        "blocking_failure": bool(failed_checks),
        "validation_scope": "static_cross_artifact_movement_contract",
        "source_reports": {
            "data_movement_plan": str(reports_dir / "data_movement_plan.json"),
            "ps_pl_transfer_plan": str(reports_dir / "ps_pl_transfer_plan.json"),
        },
        "summary": {
            "total_checks": len(all_checks),
            "passed_checks": len(all_checks) - len(failed_checks),
            "failed_checks": len(failed_checks),
            "failed_check_names": [item["name"] for item in failed_checks],
        },
        "failed_checks": failed_checks,
        "checks": sections,
        "truth_boundary": "Static agreement check only; real transfer execution requires HLS/Vivado/bitstream/board runtime truth artifacts.",
    }
    json_path = reports_dir / "movement_contract_validation.json"
    md_path = reports_dir / "movement_contract_validation.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Movement contract validation",
        "",
        f"Status: `{status}`",
        f"Blocking failure: `{str(bool(failed_checks)).lower()}`",
        f"Failed checks: `{len(failed_checks)}`",
        "",
        "## Checks",
    ]
    for section, checks in sections.items():
        lines.extend(["", f"### {section}"])
        for check in checks:
            marker = "PASS" if bool(check.get("passed")) else "FAIL"
            lines.append(f"- **{marker}** {check.get('name')}: expected=`{check.get('expected')}` actual=`{check.get('actual')}` evidence=`{check.get('evidence')}`")
    if failed_checks:
        lines.extend(["", "## Blocking failures"])
        for item in failed_checks:
            lines.append(f"- `{item['section']}.{item['name']}` expected=`{item['expected']}` actual=`{item['actual']}` evidence=`{item['evidence']}`")
    lines.extend(["", "## Truth boundary", str(payload["truth_boundary"])])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "movement_contract_validation_json": str(json_path),
        "movement_contract_validation_md": str(md_path),
        "movement_contract_validation_status": status,
        "movement_contract_validation_failed_checks": len(failed_checks),
        "movement_contract_validation_blocking_failure": bool(failed_checks),
    }


def movement_contract_validation_summary(out_dir: str | Path) -> dict[str, Any]:
    """Return the movement contract result for manifest/status consumers.

    The manifest should not only link the JSON file; it should also expose a
    loud status bit so artifact disagreement is visible without opening every
    report.
    """
    root = Path(out_dir)
    payload = _load_json_file(root / "reports" / "movement_contract_validation.json")
    if not payload:
        return {
            "status": "missing",
            "passed": False,
            "blocking_failure": True,
            "failed_checks": None,
            "report": str(root / "reports" / "movement_contract_validation.json"),
        }
    summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), Mapping) else {}
    failed = int(summary.get("failed_checks") or len(payload.get("failed_checks", []) or []))
    status = str(payload.get("status") or ("passed" if failed == 0 else "failed"))
    return {
        "status": status,
        "passed": status == "passed" and failed == 0,
        "blocking_failure": bool(payload.get("blocking_failure", failed > 0)),
        "failed_checks": failed,
        "failed_check_names": list(summary.get("failed_check_names", [])) if isinstance(summary.get("failed_check_names", []), list) else [],
        "report": str(root / "reports" / "movement_contract_validation.json"),
    }
