"""Runtime I/O, buffer, and execution-plan generation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

def _runtime_io_movement_summary(communication_plan: Any | None) -> dict[str, Any]:
    summary = {
        "inputs": {
            "import": {
                "interface": "axi_stream",
                "transport": "dma",
                "policy": "full",
                "resolved": "dma_stream_import_full",
            }
        },
        "outputs": {
            "export": {
                "interface": "axi_stream",
                "transport": "dma",
                "policy": "full",
                "resolved": "dma_stream_export_full",
            }
        },
    }
    edges = getattr(communication_plan, "edges", []) or []
    for edge in edges:
        notes = getattr(edge, "notes", {}) or {}
        kind = str(notes.get("kind", "")).strip().lower()
        interface = str(notes.get("interface") or "").strip().lower().replace("-", "_")
        transport = str(notes.get("transport") or "").strip().lower().replace("-", "_")
        policy = str(notes.get("policy") or "").strip().lower().replace("-", "_")
        mode = str(notes.get("mode") or "").strip().lower().replace("-", "_")
        if not interface:
            if mode in {"ddr", "m_axi", "maxi"}:
                interface = "m_axi"
            elif mode in {"stream", "streamed", "axis", "axi_stream"}:
                interface = "axi_stream"
        if not transport:
            transport = "ps_runtime" if interface == "m_axi" else ("dma" if interface == "axi_stream" else "none")
        tiled_flag = notes.get("tiled")
        tile_size = notes.get("tile_size")
        if isinstance(tiled_flag, str):
            tiled = tiled_flag.strip().lower() in {"1", "true", "yes", "on", "enabled", "tiled"}
        else:
            tiled = bool(tiled_flag)
        if policy == "tiled":
            tiled = True
        if not policy:
            policy = "tiled" if tiled else "full"
        try:
            tile_size = int(tile_size) if tile_size is not None else None
        except Exception:
            tile_size = None
        if kind in {"input", "inputs", "activation_in"}:
            if interface == "m_axi" and policy == "tiled":
                resolved = "m_axi_import_tiled"
            elif interface == "m_axi" and policy == "full":
                resolved = "m_axi_import_full"
            elif interface == "axi_stream" and policy == "tiled":
                resolved = "dma_stream_import_tiled"
            else:
                resolved = "dma_stream_import_full"
            entry = {"interface": interface or "axi_stream", "transport": transport, "policy": policy, "resolved": resolved, "tiled": bool(tiled)}
            if tile_size is not None and bool(tiled):
                entry["tile_size"] = tile_size
            summary["inputs"] = {"import": entry}
        if kind in {"output", "outputs", "activation_out"}:
            if interface == "m_axi" and policy == "tiled":
                resolved = "m_axi_export_tiled"
            elif interface == "m_axi" and policy == "full":
                resolved = "m_axi_export_full"
            elif interface == "axi_stream" and policy == "tiled":
                resolved = "dma_stream_export_tiled"
            else:
                resolved = "dma_stream_export_full"
            entry = {"interface": interface or "axi_stream", "transport": transport, "policy": policy, "resolved": resolved, "tiled": bool(tiled)}
            if tile_size is not None and bool(tiled):
                entry["tile_size"] = tile_size
            summary["outputs"] = {"export": entry}
    return summary


def _plan_notes(plan) -> dict[str, Any]:
    if plan is None:
        return {}
    if hasattr(plan, "notes") and isinstance(getattr(plan, "notes"), dict):
        return dict(getattr(plan, "notes"))
    if isinstance(plan, dict):
        notes = plan.get("notes", plan)
        return dict(notes) if isinstance(notes, dict) else {}
    return {}


def _runtime_activation_storage_summary(memory_plan: Any | None) -> dict[str, Any]:
    notes = _plan_notes(memory_plan)
    resolved = str(notes.get("resolved_activation_storage") or "bram").strip().lower().replace("-", "_")
    if resolved not in {"bram", "uram"}:
        resolved = "bram"
    return {
        "storage": resolved,
        "resolved": f"activation_{resolved}",
        "local_buffers": True,
    }


def _file_word_count(path: Path | None, *, word_bytes: int = 4) -> int | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    size = path.stat().st_size
    return max(1, (size + word_bytes - 1) // word_bytes)


def _runtime_buffer_entry(
    name: str,
    *,
    role: str,
    direction: str,
    words: int | None = None,
    dtype: str = "float32",
    required_for_modes: list[int] | None = None,
    source: str | None = None,
    logical_shape: list[int] | None = None,
) -> dict[str, Any]:
    resolved_words = max(1, int(words or 1))
    return {
        "name": name,
        "role": role,
        "dtype": dtype,
        "shape": list(logical_shape or [resolved_words]),
        "physical_words": resolved_words,
        "words": resolved_words,
        "bytes": resolved_words * 4,
        "direction": direction,
        "required_for_modes": list(required_for_modes or []),
        "source": source,
    }


def _emit_runtime_buffer_plans(
    root: Path,
    package_dir: Path,
    *,
    runtime_sequence: Mapping[str, Any],
    runtime_weights: Mapping[str, Any],
    pipeline_mode: str | None,
) -> dict[str, Any]:
    """Emit runtime buffer metadata consumed by generated board_runtime/runtime_api.

    The plan is intentionally conservative: it records the PS/PL buffers needed by
    generated runtime commands and uses packaged artifact sizes when exact model
    tensor shape metadata is not yet available. This is real runtime metadata, not
    a board-execution claim; real execution still requires a deployed overlay.
    """
    sequence = list(runtime_sequence.get("sequence", [])) if isinstance(runtime_sequence, Mapping) else []
    commands: list[str] = []
    for item in sequence:
        command = item.get("command") if isinstance(item, Mapping) else str(item)
        if command:
            commands.append(str(command))

    pipeline = str(pipeline_mode or "").lower()
    is_training = "train" in pipeline or any(
        c in {
            "run_training",
            "reset_accumulators",
            "accumulate_gradients",
            "apply_accumulated_gradients",
            "export_gradients",
            "export_optimizer_state",
        }
        for c in commands
    )

    dataset_manifest_path = root / "validation" / "dataset" / "dataset_manifest.json"
    dataset_manifest: dict[str, Any] = {}
    if dataset_manifest_path.exists():
        try:
            loaded = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                dataset_manifest = loaded
        except Exception:
            dataset_manifest = {}
    dataset_sample_count = max(1, int(dataset_manifest.get("sample_count") or 1))
    dataset_input_shape = [int(v) for v in dataset_manifest.get("input_shape_per_sample", []) if int(v) > 0]
    dataset_input_words_per_sample = int(dataset_manifest.get("input_words_per_sample") or 0)

    input_words = (
        _file_word_count(root / "validation" / "dataset" / "inputs.bin")
        or _file_word_count(root / "input.bin")
        or 1
    )
    output_words = _file_word_count(root / "output.bin") or 1
    output_values_per_sample = (
        output_words // dataset_sample_count
        if dataset_sample_count > 1 and output_words % dataset_sample_count == 0
        else output_words
    )
    input_logical_shape = (
        [dataset_sample_count, *dataset_input_shape]
        if dataset_manifest and dataset_input_shape
        else [input_words]
    )
    output_logical_shape = (
        [dataset_sample_count, output_values_per_sample]
        if dataset_sample_count > 1
        else [output_words]
    )
    gradient_words = (
        _file_word_count(root / "gradients_after.bin")
        or _file_word_count(root / "gradients_export.bin")
        or _file_word_count(root / "training_reference" / "grads_ref.bin")
        or _file_word_count(root / "training_reference" / "gradients_after_ref.bin")
        or 1
    )
    optimizer_words = (
        _file_word_count(root / "optimizer_state_after.bin")
        or _file_word_count(root / "training_reference" / "optimizer_state_after_ref.bin")
        or 1
    )

    buffers: list[dict[str, Any]] = [
        _runtime_buffer_entry(
            "input",
            role="model_input",
            direction="ps_to_pl",
            words=input_words,
            required_for_modes=[2, 3] if is_training else [],
            source="inputs/input.bin",
            logical_shape=input_logical_shape,
        ),
        _runtime_buffer_entry(
            "output",
            role="model_output",
            direction="pl_to_ps",
            words=output_words,
            required_for_modes=[2, 3] if is_training else [],
            source="outputs/output.bin",
            logical_shape=output_logical_shape,
        ),
    ]

    if bool(runtime_weights.get("import_required")) or bool(runtime_weights.get("present")):
        buffers.append(
            _runtime_buffer_entry(
                "weights",
                role="weight_import",
                direction="bidirectional" if bool(runtime_weights.get("export_supported")) else "ps_to_pl",
                words=int(runtime_weights.get("total_words") or 1),
                required_for_modes=[1],
                source=str(runtime_weights.get("weights_bin") or "weights/weights.bin"),
            )
        )

    if is_training:
        buffers.append(
            _runtime_buffer_entry(
                "labels",
                role="training_labels",
                direction="ps_to_pl",
                words=output_words,
                required_for_modes=[2, 3],
                source="inputs/labels.bin",
            )
        )

    if "export_gradients" in commands:
        buffers.append(
            _runtime_buffer_entry(
                "gradients_mem",
                role="gradient_export",
                direction="pl_to_ps",
                words=gradient_words,
                required_for_modes=[8],
                source="outputs/gradients_after.bin",
            )
        )

    if "export_optimizer_state" in commands:
        buffers.append(
            _runtime_buffer_entry(
                "optimizer_state_mem",
                role="optimizer_state_export",
                direction="pl_to_ps",
                words=optimizer_words,
                required_for_modes=[9],
                source="outputs/optimizer_state_after.bin",
            )
        )

    by_name = {b["name"]: b for b in buffers}
    buffer_plan = {
        "schema_version": 1,
        "package_kind": "fpgai_runtime_buffer_plan",
        "truth_boundary": "Generated buffer allocation/binding metadata only; real board execution still requires deployed Vivado/bitstream artifacts.",
        "dataset": {
            "enabled": bool(dataset_manifest),
            "sample_count": dataset_sample_count,
            "input_words_per_sample": dataset_input_words_per_sample or None,
            "output_values_per_sample": output_values_per_sample,
        },
        "buffers": list(by_name.values()),
    }

    mode_map = {
        "run_inference": 0,
        "import_weights": 1,
        "export_weights": 2,
        "run_training": 2,
        "accumulate_gradients": 3,
        "apply_accumulated_gradients": 4,
        "reset_accumulators": 5,
        "export_gradients": 8,
        "export_optimizer_state": 9,
    }
    execution_items: list[dict[str, Any]] = []
    for item in sequence:
        command = item.get("command") if isinstance(item, Mapping) else str(item)
        args = dict(item.get("args", {})) if isinstance(item, Mapping) and isinstance(item.get("args", {}), Mapping) else {}
        command = str(command)
        sync_before: list[str] = []
        sync_after: list[str] = []
        capture: str | None = None
        if command == "import_weights":
            sync_before.append("weights")
        elif command == "export_weights":
            sync_after.append("weights")
        elif command == "run_training":
            sync_before.extend(["input", "labels"])
            sync_after.append("output")
        elif command == "accumulate_gradients":
            sync_before.extend(["input", "labels"])
        elif command == "export_gradients":
            sync_after.append("gradients_mem")
            capture = "outputs/gradients_after.bin"
        elif command == "export_optimizer_state":
            sync_after.append("optimizer_state_mem")
            capture = "outputs/optimizer_state_after.bin"
        elif command == "run_inference":
            sync_before.append("input")
            sync_after.append("output")
            if dataset_sample_count > 1:
                args["repeat"] = dataset_sample_count
        execution_items.append(
            {
                "command": command,
                "mode": mode_map.get(command),
                "args": args,
                "sync_before": [name for name in sync_before if name in by_name],
                "sync_after": [name for name in sync_after if name in by_name],
                "capture": capture,
            }
        )

    execution_plan = {
        "schema_version": 1,
        "package_kind": "fpgai_runtime_execution_plan",
        "sequence": execution_items,
    }

    buffer_plan_path = package_dir / "buffer_plan.json"
    execution_plan_path = package_dir / "runtime_execution_plan.json"
    buffer_plan_path.write_text(json.dumps(buffer_plan, indent=2, sort_keys=True), encoding="utf-8")
    execution_plan_path.write_text(json.dumps(execution_plan, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "buffer_plan": buffer_plan,
        "runtime_execution_plan": execution_plan,
        "files": {
            "buffer_plan": {
                "path": "runtime_package/buffer_plan.json",
                "package_path": "buffer_plan.json",
                "present": True,
                "bytes": buffer_plan_path.stat().st_size,
            },
            "runtime_execution_plan": {
                "path": "runtime_package/runtime_execution_plan.json",
                "package_path": "runtime_execution_plan.json",
                "present": True,
                "bytes": execution_plan_path.stat().st_size,
            },
        },
    }
