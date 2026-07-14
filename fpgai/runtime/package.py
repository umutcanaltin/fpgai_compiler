from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from pathlib import Path
from typing import Any, Mapping


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _copy_if_exists(src: Path, dst: Path) -> dict[str, Any] | None:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": src.as_posix(),
        "package_path": dst.as_posix(),
        "bytes": dst.stat().st_size,
    }


def _first_existing(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        hits = sorted(root.glob(pattern))
        for hit in hits:
            if hit.is_file():
                return hit
    return None


def _collect_existing(root: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        out.extend([p for p in sorted(root.glob(pattern)) if p.is_file()])
    return out


def _artifact_status(path: Path | None) -> dict[str, Any]:
    return {
        "present": bool(path is not None and path.exists()),
        "path": path.as_posix() if path is not None else None,
    }



_RUNTIME_WEIGHT_ARRAY_RE = re.compile(
    r"const\s+[A-Za-z_][A-Za-z0-9_:<>]*\s+([WB]\d+)_init\s*\[\s*(\d+)\s*\]\s*=\s*\{(.*?)\};",
    re.DOTALL,
)


def _normalise_weights_mode(weights_mode: str | None) -> str:
    mode = str(weights_mode or "").strip().lower().replace("-", "_")
    aliases = {
        "bram": "bram_static",
        "embedded": "bram_static",
        "embedded_bram": "bram_static",
        "onchip": "bram_static",
        "on_chip": "bram_static",
        "static": "bram_static",
        "const": "bram_static",
        "ddr": "bram_import_full",
        "dma_ddr": "bram_import_full",
        "runtime": "bram_import_full",
        "preload": "bram_import_full",
        "direct": "bram_import_full",
        "uram": "uram_import_full",
    }
    return aliases.get(mode, mode)


def _float_to_packed32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _parse_float_initializer_values(body: str) -> list[float]:
    values: list[float] = []
    for raw in body.replace("\n", " ").split(","):
        token = raw.strip()
        if not token:
            continue
        token = token.replace("f", "")
        values.append(float(token))
    return values


def _runtime_param_source(root: Path) -> Path | None:
    candidates = [
        root / "hls" / "src" / "fpgai_params.cpp",
        root / "hls" / "fpgai_params.cpp",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    hits = sorted(root.glob("hls/**/fpgai_params.cpp"))
    for hit in hits:
        if hit.is_file():
            return hit
    return None


def _parse_runtime_weight_entries(root: Path) -> tuple[Path | None, list[dict[str, Any]], list[int]]:
    src = _runtime_param_source(root)
    if src is None:
        return None, [], []

    text = src.read_text(encoding="utf-8", errors="replace")
    entries: list[dict[str, Any]] = []
    words: list[int] = []
    offset = 0

    for match in _RUNTIME_WEIGHT_ARRAY_RE.finditer(text):
        base_name = match.group(1)
        declared_count = int(match.group(2))
        values = _parse_float_initializer_values(match.group(3))

        if len(values) != declared_count:
            raise ValueError(
                f"Runtime weight initializer {base_name}_init declares {declared_count} values "
                f"but parser found {len(values)} in {src}"
            )

        packed = [_float_to_packed32(v) for v in values]
        words.extend(packed)
        entries.append(
            {
                "name": base_name,
                "kind": "bias" if base_name.startswith("B") else "weight",
                "offset_words": offset,
                "count_words": declared_count,
            }
        )
        offset += declared_count

    return src, entries, words


def _runtime_weight_payload_required(weights_mode: str | None, entries: list[dict[str, Any]]) -> bool:
    mode = _normalise_weights_mode(weights_mode)
    if mode in {
        "bram_import_full",
        "bram_import_export_full",
        "uram_import_full",
        "uram_import_export_full",
        "ddr_tiled",
        "ddr_tiled_mutable",
        "tile_cached",
    }:
        return True
    if mode in {"bram_static", "uram_static", "none", "static", "const"}:
        return False

    return bool(entries)


def _emit_runtime_weight_payload(
    root: Path,
    package_dir: Path,
    *,
    weights_mode: str | None,
) -> dict[str, Any]:
    src, entries, words = _parse_runtime_weight_entries(root)
    required = _runtime_weight_payload_required(weights_mode, entries)

    summary: dict[str, Any] = {
        "required": required,
        "present": False,
        "weights_mode": _normalise_weights_mode(weights_mode),
        "import_required": required,
        "export_supported": _normalise_weights_mode(weights_mode) in {"bram_import_export_full", "uram_import_export_full", "ddr_tiled_mutable"},
        "reload_before_each_compute": False,
        "format": "packed32",
        "source": src.as_posix() if src is not None else None,
        "total_words": 0,
        "word_bytes": 4,
        "weights_bin": None,
        "weight_layout": None,
        "status": "not_required",
    }

    if not required:
        return {"summary": summary, "files": {}}

    if not entries:
        summary["status"] = "missing_generated_runtime_parameters"
        return {"summary": summary, "files": {}}

    weights_dir = package_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    weights_bin = weights_dir / "weights.bin"
    weights_bin.write_bytes(b"".join(struct.pack("<I", word) for word in words))

    layout = {
        "format": "packed32",
        "source": "generated_hls_parameters",
        "parameter_source": src.as_posix() if src is not None else None,
        "entries": entries,
        "total_words": len(words),
        "word_bytes": 4,
    }

    layout_path = weights_dir / "weight_layout.json"
    layout_path.write_text(json.dumps(layout, indent=2, sort_keys=True), encoding="utf-8")

    summary.update(
        {
            "present": True,
            "total_words": len(words),
            "weights_bin": "weights/weights.bin",
            "weight_layout": "weights/weight_layout.json",
            "status": "created",
        }
    )

    return {
        "summary": summary,
        "files": {
            "weights_bin": {
                "source": src.as_posix() if src is not None else None,
                "package_path": "weights/weights.bin",
                "bytes": weights_bin.stat().st_size,
            },
            "weight_layout": {
                "source": layout_path.as_posix(),
                "package_path": "weights/weight_layout.json",
                "bytes": layout_path.stat().st_size,
            },
        },
    }


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



def _emit_board_runtime_backend(package_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Emit a generated board-runtime adapter for PYNQ/KV260-style backends.

    The adapter wraps a real board object, drives generated HLS mode numbers, and
    can allocate/synchronise PYNQ buffers from the generated buffer_plan.json.
    """
    backend_path = package_dir / "board_runtime.py"
    backend_path.write_text(
        "\n".join(
            [
                '"""FPGAI generated board-runtime adapter.\n\nThis module maps runtime commands to generated HLS mode numbers and allocates\nPYNQ/KV260 buffers from buffer_plan.json. It does not fake hardware execution:\nwithout a real Overlay/IP or explicit fake object in tests, board calls fail.\n"""',
                "from __future__ import annotations",
                "",
                "import json",
                "from pathlib import Path",
                "from typing import Any",
                "",
                "FPGAI_MODE_RUN_TRAINING = 2",
                "FPGAI_MODE_ACCUMULATE_GRADIENTS = 3",
                "FPGAI_MODE_APPLY_ACCUMULATED_GRADIENTS = 4",
                "FPGAI_MODE_RESET_ACCUMULATORS = 5",
                "FPGAI_MODE_EXPORT_GRADIENTS = 8",
                "FPGAI_MODE_EXPORT_OPTIMIZER_STATE = 9",
                "",
                "def load_buffer_plan(package_dir: str | Path | None = None) -> dict[str, Any]:",
                "    base = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent",
                "    path = base / 'buffer_plan.json'",
                "    if not path.exists():",
                "        return {'buffers': []}",
                "    return json.loads(path.read_text(encoding='utf-8'))",
                "",
                "def _numpy_dtype(dtype: str) -> Any:",
                "    import numpy as np  # imported lazily so runtime_api imports without numpy when unused",
                "    mapping = {'float32': np.float32, 'float': np.float32, 'uint32': np.uint32, 'int32': np.int32}",
                "    return mapping.get(str(dtype).lower(), np.float32)",
                "",
                "def allocate_buffers_from_plan(buffer_plan: dict[str, Any] | None = None, *, allocate_fn: Any | None = None, package_dir: str | Path | None = None) -> dict[str, Any]:",
                "    plan = buffer_plan or load_buffer_plan(package_dir)",
                "    if allocate_fn is None:",
                "        try:",
                "            from pynq import allocate as allocate_fn  # type: ignore",
                "        except Exception as exc:",
                "            raise RuntimeError('PYNQ allocate() is not available; pass allocate_fn=... in tests or run on the target board.') from exc",
                "    buffers: dict[str, Any] = {}",
                "    for entry in plan.get('buffers', []):",
                "        name = str(entry['name'])",
                "        shape = tuple(int(v) for v in entry.get('shape', [int(entry.get('words', 1))]))",
                "        dtype = _numpy_dtype(str(entry.get('dtype', 'float32')))",
                "        buffers[name] = allocate_fn(shape=shape, dtype=dtype)",
                "    return buffers",
                "",
                "def _sync_to_device(buf: Any) -> None:",
                "    if hasattr(buf, 'sync_to_device'):",
                "        buf.sync_to_device()",
                "    elif hasattr(buf, 'flush'):",
                "        buf.flush()",
                "",
                "def _sync_from_device(buf: Any) -> None:",
                "    if hasattr(buf, 'sync_from_device'):",
                "        buf.sync_from_device()",
                "    elif hasattr(buf, 'invalidate'):",
                "        buf.invalidate()",
                "",
                "def sync_buffers(buffers: dict[str, Any], names: list[str], *, direction: str) -> None:",
                "    for name in names:",
                "        if name not in buffers:",
                "            raise RuntimeError(f'Runtime buffer {name!r} is required by the execution plan but is not allocated/bound.')",
                "        if direction == 'to_device':",
                "            _sync_to_device(buffers[name])",
                "        elif direction == 'from_device':",
                "            _sync_from_device(buffers[name])",
                "        else:",
                "            raise ValueError(f'Unsupported buffer sync direction: {direction}')",
                "",
                "class PynqDmaMmioBackend:",
                "    \"\"\"Concrete PYNQ/KV260-style backend for generated FPGAI runtime packages.\"\"\"",
                "    def __init__(self, *, bitfile: str | Path | None = None, overlay: Any | None = None, ip_name: str | None = None, dma_name: str | None = None, buffers: dict[str, Any] | None = None, mode_offset: int = 0x10, ap_ctrl_offset: int = 0x00, start_mask: int = 0x01, done_mask: int = 0x02, timeout_cycles: int = 1000000):",
                "        self.mode_offset = int(mode_offset)",
                "        self.ap_ctrl_offset = int(ap_ctrl_offset)",
                "        self.start_mask = int(start_mask)",
                "        self.done_mask = int(done_mask)",
                "        self.timeout_cycles = int(timeout_cycles)",
                "        self.buffers = dict(buffers or {})",
                "        if overlay is None:",
                "            if bitfile is None:",
                "                raise ValueError('PynqDmaMmioBackend requires either overlay=... or bitfile=...')",
                "            try:",
                "                from pynq import Overlay  # type: ignore",
                "            except Exception as exc:",
                "                raise RuntimeError('PYNQ is not available; install/use this backend on the target board.') from exc",
                "            overlay = Overlay(str(bitfile))",
                "        self.overlay = overlay",
                "        self.ip = self._resolve_ip(ip_name)",
                "        self.dma = self._resolve_dma(dma_name)",
                "",
                "    def _resolve_ip(self, ip_name: str | None) -> Any:",
                "        names: list[str] = []",
                "        if ip_name:",
                "            names.append(ip_name)",
                "        names.extend(['deeplearn_0', 'deeplearn', 'deeplearn_top_0'])",
                "        for name in names:",
                "            if hasattr(self.overlay, name):",
                "                return getattr(self.overlay, name)",
                "        ip_dict = getattr(self.overlay, 'ip_dict', {}) or {}",
                "        for name in names:",
                "            if name in ip_dict and hasattr(self.overlay, name):",
                "                return getattr(self.overlay, name)",
                "        if hasattr(self.overlay, 'write') and hasattr(self.overlay, 'read'):",
                "            return self.overlay",
                "        raise RuntimeError(f'Could not resolve generated HLS IP. Tried: {names}')",
                "",
                "    def _resolve_dma(self, dma_name: str | None) -> Any | None:",
                "        names: list[str] = []",
                "        if dma_name:",
                "            names.append(dma_name)",
                "        names.extend(['axi_dma_0', 'dma', 'axi_dma'])",
                "        for name in names:",
                "            if hasattr(self.overlay, name):",
                "                return getattr(self.overlay, name)",
                "        return None",
                "",
                "    def _wait_done(self) -> None:",
                "        for _ in range(max(1, self.timeout_cycles)):",
                "            status = int(self.ip.read(self.ap_ctrl_offset))",
                "            if status & self.done_mask:",
                "                return",
                "        raise TimeoutError('Timed out waiting for generated HLS IP to finish.')",
                "",
                "    def call_mode(self, mode: int, **kwargs: Any) -> Any:",
                "        self.ip.write(self.mode_offset, int(mode))",
                "        if kwargs.get('input_buffer') is not None and self.dma is not None:",
                "            self.dma.sendchannel.transfer(kwargs['input_buffer'])",
                "        self.ip.write(self.ap_ctrl_offset, self.start_mask)",
                "        if self.dma is not None and kwargs.get('output_buffer') is not None:",
                "            self.dma.recvchannel.transfer(kwargs['output_buffer'])",
                "            self.dma.sendchannel.wait()",
                "            self.dma.recvchannel.wait()",
                "        else:",
                "            self._wait_done()",
                "        return {'mode': int(mode)}",
                "",
                "    def bind_buffer(self, logical_name: str, buffer: Any) -> None:",
                "        self.buffers[str(logical_name)] = buffer",
                "",
                "    def bind_buffers(self, buffers: dict[str, Any]) -> None:",
                "        for name, buf in dict(buffers).items():",
                "            self.bind_buffer(str(name), buf)",
                "",
                "    def read_buffer(self, logical_name: str) -> bytes:",
                "        if logical_name not in self.buffers:",
                "            raise RuntimeError(f'No board buffer bound for {logical_name!r}.')",
                "        buf = self.buffers[logical_name]",
                "        _sync_from_device(buf)",
                "        if isinstance(buf, (bytes, bytearray, memoryview)):",
                "            return bytes(buf)",
                "        if hasattr(buf, 'tobytes'):",
                "            return bytes(buf.tobytes())",
                "        try:",
                "            return memoryview(buf).tobytes()",
                "        except TypeError as exc:",
                "            raise TypeError(f'Buffer {logical_name!r} cannot be converted to bytes.') from exc",
                "",
                "    def run_training(self, inputs: Any | None = None, labels: Any | None = None, *, steps: int = 1) -> Any:",
                "        result = None",
                "        for _ in range(int(steps)):",
                "            result = self.call_mode(FPGAI_MODE_RUN_TRAINING, inputs=inputs, labels=labels, input_buffer=self.buffers.get('input'), output_buffer=self.buffers.get('output'))",
                "        return result",
                "",
                "    def reset_accumulators(self) -> Any:",
                "        return self.call_mode(FPGAI_MODE_RESET_ACCUMULATORS)",
                "",
                "    def accumulate_gradients(self, inputs: Any | None = None, labels: Any | None = None, *, steps: int = 1) -> Any:",
                "        result = None",
                "        for _ in range(int(steps)):",
                "            result = self.call_mode(FPGAI_MODE_ACCUMULATE_GRADIENTS, inputs=inputs, labels=labels, input_buffer=self.buffers.get('input'))",
                "        return result",
                "",
                "    def apply_accumulated_gradients(self) -> Any:",
                "        return self.call_mode(FPGAI_MODE_APPLY_ACCUMULATED_GRADIENTS)",
                "",
                "    def export_gradients(self) -> bytes:",
                "        self.call_mode(FPGAI_MODE_EXPORT_GRADIENTS)",
                "        return self.read_buffer('gradients_mem')",
                "",
                "    def export_optimizer_state(self) -> bytes:",
                "        self.call_mode(FPGAI_MODE_EXPORT_OPTIMIZER_STATE)",
                "        return self.read_buffer('optimizer_state_mem')",
                "",
                "def create_pynq_backend(*, bitfile: str | Path | None = None, overlay: Any | None = None, ip_name: str | None = None, dma_name: str | None = None, buffers: dict[str, Any] | None = None, mode_offset: int = 0x10, buffer_plan: dict[str, Any] | None = None, allocate_fn: Any | None = None) -> PynqDmaMmioBackend:",
                "    allocated = allocate_buffers_from_plan(buffer_plan, allocate_fn=allocate_fn) if buffer_plan is not None or allocate_fn is not None else {}",
                "    allocated.update(dict(buffers or {}))",
                "    return PynqDmaMmioBackend(bitfile=bitfile, overlay=overlay, ip_name=ip_name, dma_name=dma_name, buffers=allocated, mode_offset=mode_offset)",
                "",
                "class FPGAIBoardRuntime:",
                "    def __init__(self, backend: Any, *, package_dir: str | Path | None = None, buffers: dict[str, Any] | None = None):",
                "        if backend is None:",
                "            raise ValueError('FPGAIBoardRuntime requires a real backend object.')",
                "        self.backend = backend",
                "        self.package_dir = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent",
                "        self.buffers = dict(buffers or getattr(backend, 'buffers', {}) or {})",
                "        if self.buffers and hasattr(self.backend, 'bind_buffers'):",
                "            self.backend.bind_buffers(self.buffers)",
                "",
                "    def bind_buffers(self, buffers: dict[str, Any]) -> None:",
                "        self.buffers.update(dict(buffers))",
                "        if hasattr(self.backend, 'bind_buffers'):",
                "            self.backend.bind_buffers(buffers)",
                "        elif hasattr(self.backend, 'bind_buffer'):",
                "            for name, buf in dict(buffers).items():",
                "                self.backend.bind_buffer(name, buf)",
                "",
                "    def sync_before(self, names: list[str]) -> None:",
                "        sync_buffers(self.buffers, names, direction='to_device')",
                "",
                "    def sync_after(self, names: list[str]) -> None:",
                "        sync_buffers(self.buffers, names, direction='from_device')",
                "",
                "    def _call_mode(self, mode: int, **kwargs: Any) -> Any:",
                "        if hasattr(self.backend, 'call_mode'):",
                "            return self.backend.call_mode(mode, **kwargs)",
                "        if hasattr(self.backend, 'set_mode'):",
                "            self.backend.set_mode(mode)",
                "            if hasattr(self.backend, 'run'):",
                "                return self.backend.run(**kwargs)",
                "        raise RuntimeError('Backend must implement call_mode(mode, **kwargs) or set_mode(mode)+run().')",
                "",
                "    def _read_payload(self, logical_name: str, fallback_method: str | None = None) -> bytes:",
                "        if fallback_method and hasattr(self.backend, fallback_method):",
                "            payload = getattr(self.backend, fallback_method)()",
                "        elif hasattr(self.backend, 'read_buffer'):",
                "            payload = self.backend.read_buffer(logical_name)",
                "        elif hasattr(self.backend, 'read_m_axi'):",
                "            payload = self.backend.read_m_axi(logical_name)",
                "        elif logical_name in self.buffers:",
                "            buf = self.buffers[logical_name]",
                "            _sync_from_device(buf)",
                "            payload = buf.tobytes() if hasattr(buf, 'tobytes') else memoryview(buf).tobytes()",
                "        else:",
                "            raise RuntimeError(f'Backend cannot read exported payload {logical_name!r}.')",
                "        if not isinstance(payload, (bytes, bytearray, memoryview)):",
                "            raise TypeError(f'Backend returned non-bytes payload for {logical_name!r}.')",
                "        return bytes(payload)",
                "",
                "    def _write_capture(self, payload: bytes, default_relpath: str, capture_path: str | Path | None) -> Path:",
                "        target = Path(capture_path) if capture_path is not None else self.package_dir / default_relpath",
                "        target.parent.mkdir(parents=True, exist_ok=True)",
                "        target.write_bytes(payload)",
                "        return target",
                "",
                "    def run_training(self, inputs: Any | None = None, labels: Any | None = None, *, steps: int = 1) -> Any:",
                "        if int(steps) < 1:",
                "            raise ValueError('steps must be >= 1')",
                "        if hasattr(self.backend, 'run_training'):",
                "            return self.backend.run_training(inputs=inputs, labels=labels, steps=int(steps))",
                "        result = None",
                "        for _ in range(int(steps)):",
                "            result = self._call_mode(FPGAI_MODE_RUN_TRAINING, inputs=inputs, labels=labels, input_buffer=self.buffers.get('input'), output_buffer=self.buffers.get('output'))",
                "        return result",
                "",
                "    def reset_accumulators(self) -> Any:",
                "        if hasattr(self.backend, 'reset_accumulators'):",
                "            return self.backend.reset_accumulators()",
                "        return self._call_mode(FPGAI_MODE_RESET_ACCUMULATORS)",
                "",
                "    def accumulate_gradients(self, inputs: Any | None = None, labels: Any | None = None, *, steps: int = 1) -> Any:",
                "        if int(steps) < 1:",
                "            raise ValueError('steps must be >= 1')",
                "        result = None",
                "        for _ in range(int(steps)):",
                "            if hasattr(self.backend, 'accumulate_gradients'):",
                "                result = self.backend.accumulate_gradients(inputs=inputs, labels=labels, steps=1)",
                "            else:",
                "                result = self._call_mode(FPGAI_MODE_ACCUMULATE_GRADIENTS, inputs=inputs, labels=labels, input_buffer=self.buffers.get('input'))",
                "        return result",
                "",
                "    def apply_accumulated_gradients(self) -> Any:",
                "        if hasattr(self.backend, 'apply_accumulated_gradients'):",
                "            return self.backend.apply_accumulated_gradients()",
                "        return self._call_mode(FPGAI_MODE_APPLY_ACCUMULATED_GRADIENTS)",
                "",
                "    def export_gradients(self, *, capture_path: str | Path | None = None) -> bytes:",
                "        if hasattr(self.backend, 'export_gradients'):",
                "            payload = self.backend.export_gradients()",
                "        else:",
                "            self._call_mode(FPGAI_MODE_EXPORT_GRADIENTS)",
                "            payload = self._read_payload('gradients_mem', 'read_gradients')",
                "        payload = bytes(payload)",
                "        self._write_capture(payload, 'outputs/gradients_after.bin', capture_path)",
                "        return payload",
                "",
                "    def export_optimizer_state(self, *, capture_path: str | Path | None = None) -> bytes:",
                "        if hasattr(self.backend, 'export_optimizer_state'):",
                "            payload = self.backend.export_optimizer_state()",
                "        else:",
                "            self._call_mode(FPGAI_MODE_EXPORT_OPTIMIZER_STATE)",
                "            payload = self._read_payload('optimizer_state_mem', 'read_optimizer_state')",
                "        payload = bytes(payload)",
                "        self._write_capture(payload, 'outputs/optimizer_state_after.bin', capture_path)",
                "        return payload",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"source": backend_path.as_posix(), "package_path": "board_runtime.py", "bytes": backend_path.stat().st_size, "present": True}

def _emit_runtime_api(package_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    api_path = package_dir / "runtime_api.py"
    runtime_weights = dict(payload.get("runtime_weights") or {})
    sequence = dict(payload.get("runtime_sequence") or {})
    commands = [item.get("command") for item in sequence.get("sequence", []) if isinstance(item, dict)]
    runtime_source = r'''
"""FPGAI generated runtime API scaffold.

This file is generated from package_manifest.json. It validates runtime commands,
allocates/binds runtime buffers from buffer_plan.json, delegates physical
execution to the generated PYNQ/KV260 board backend, and writes an auditable
runtime execution report for every run_sequence() call.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PACKAGE_DIR / 'package_manifest.json'
RUN_SEQUENCE_PATH = PACKAGE_DIR / 'run_sequence.json'
BUFFER_PLAN_PATH = PACKAGE_DIR / 'buffer_plan.json'
RUNTIME_EXECUTION_PLAN_PATH = PACKAGE_DIR / 'runtime_execution_plan.json'
RUNTIME_EXECUTION_REPORT_JSON = PACKAGE_DIR / 'runtime_execution_report.json'
RUNTIME_EXECUTION_REPORT_MD = PACKAGE_DIR / 'runtime_execution_report.md'

def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))

def load_run_sequence() -> dict[str, Any]:
    if not RUN_SEQUENCE_PATH.exists():
        return {'sequence': []}
    return json.loads(RUN_SEQUENCE_PATH.read_text(encoding='utf-8'))

def load_buffer_plan() -> dict[str, Any]:
    if not BUFFER_PLAN_PATH.exists():
        return {'buffers': []}
    return json.loads(BUFFER_PLAN_PATH.read_text(encoding='utf-8'))

def load_runtime_execution_plan() -> dict[str, Any]:
    if not RUNTIME_EXECUTION_PLAN_PATH.exists():
        return {'sequence': []}
    return json.loads(RUNTIME_EXECUTION_PLAN_PATH.read_text(encoding='utf-8'))

def _load_board_runtime_module() -> Any:
    try:
        import board_runtime  # type: ignore
        return board_runtime
    except Exception:
        import importlib.util
        spec = importlib.util.spec_from_file_location('board_runtime', PACKAGE_DIR / 'board_runtime.py')
        if spec is None or spec.loader is None:
            raise RuntimeError('Could not load generated board_runtime.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

_BOUND_BACKEND: Any | None = None
_BOUND_BUFFERS: dict[str, Any] = {}

def allocate_runtime_buffers(*, allocate_fn: Any | None = None, buffer_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    board_runtime = _load_board_runtime_module()
    return board_runtime.allocate_buffers_from_plan(buffer_plan or load_buffer_plan(), allocate_fn=allocate_fn, package_dir=PACKAGE_DIR)

def bind_allocated_buffers(buffers: dict[str, Any]) -> dict[str, Any]:
    global _BOUND_BUFFERS
    _BOUND_BUFFERS = dict(buffers)
    if _BOUND_BACKEND is not None and hasattr(_BOUND_BACKEND, 'bind_buffers'):
        _BOUND_BACKEND.bind_buffers(_BOUND_BUFFERS)
    return _BOUND_BUFFERS

def bind_backend(backend: Any, *, buffers: dict[str, Any] | None = None) -> Any:
    global _BOUND_BACKEND
    board_runtime = _load_board_runtime_module()
    FPGAIBoardRuntime = board_runtime.FPGAIBoardRuntime
    if isinstance(backend, FPGAIBoardRuntime):
        _BOUND_BACKEND = backend
    else:
        _BOUND_BACKEND = FPGAIBoardRuntime(backend, package_dir=PACKAGE_DIR, buffers=buffers or _BOUND_BUFFERS)
    if buffers is not None:
        bind_allocated_buffers(buffers)
    elif _BOUND_BUFFERS and hasattr(_BOUND_BACKEND, 'bind_buffers'):
        _BOUND_BACKEND.bind_buffers(_BOUND_BUFFERS)
    return _BOUND_BACKEND

def get_backend() -> Any | None:
    return _BOUND_BACKEND

def _unsupported_board_call(name: str) -> None:
    raise RuntimeError(f'{name} requires a board-specific runtime backend; call bind_backend(...) with a real board adapter first.')

def _sync_before(names: list[str]) -> None:
    if not names or not _BOUND_BUFFERS:
        return
    if _BOUND_BACKEND is not None and hasattr(_BOUND_BACKEND, 'sync_before'):
        _BOUND_BACKEND.sync_before(names)
        return
    board_runtime = _load_board_runtime_module()
    board_runtime.sync_buffers(_BOUND_BUFFERS, names, direction='to_device')

def _sync_after(names: list[str]) -> None:
    if not names or not _BOUND_BUFFERS:
        return
    if _BOUND_BACKEND is not None and hasattr(_BOUND_BACKEND, 'sync_after'):
        _BOUND_BACKEND.sync_after(names)
        return
    board_runtime = _load_board_runtime_module()
    board_runtime.sync_buffers(_BOUND_BUFFERS, names, direction='from_device')

def import_weights(weights: bytes | None = None) -> Any:
    manifest = load_manifest()
    required = bool(manifest.get('runtime_weights', {}).get('import_required'))
    if required and weights is None and not manifest.get('runtime_weights', {}).get('present'):
        raise ValueError('import_weights requires a weights payload or packaged weights/weights.bin.')
    if _BOUND_BACKEND is not None:
        if weights is not None and 'weights' in _BOUND_BUFFERS:
            target = _BOUND_BUFFERS['weights']
            if hasattr(target, '__setitem__'):
                try:
                    target[:] = weights
                except Exception:
                    pass
        if hasattr(_BOUND_BACKEND, '_call_mode'):
            return _BOUND_BACKEND._call_mode(1)
    _unsupported_board_call('import_weights')

def run_inference(inputs: Any | None = None, *, repeat: int = 1) -> Any:
    if int(repeat) < 1:
        raise ValueError('repeat must be >= 1')
    if _BOUND_BACKEND is not None and hasattr(_BOUND_BACKEND, 'run_inference'):
        return _BOUND_BACKEND.run_inference(inputs=inputs, repeat=int(repeat))
    _unsupported_board_call('run_inference')

def run_training(inputs: Any | None = None, labels: Any | None = None, *, steps: int = 1) -> Any:
    if int(steps) < 1:
        raise ValueError('steps must be >= 1')
    if _BOUND_BACKEND is not None:
        return _BOUND_BACKEND.run_training(inputs=inputs, labels=labels, steps=int(steps))
    _unsupported_board_call('run_training')

def export_weights() -> bytes:
    manifest = load_manifest()
    if not bool(manifest.get('runtime_weights', {}).get('export_supported')):
        raise RuntimeError('export_weights was not generated/supported for this package.')
    _unsupported_board_call('export_weights')

def capture_gradients(payload: bytes, out_path: str | Path | None = None) -> Path:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError('capture_gradients expects a bytes-like gradient payload.')
    target = Path(out_path) if out_path is not None else PACKAGE_DIR / 'outputs' / 'gradients_after.bin'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bytes(payload))
    return target

def export_gradients(*, board_payload: bytes | None = None, capture_path: str | Path | None = None) -> bytes:
    if board_payload is not None:
        capture_gradients(board_payload, capture_path)
        return bytes(board_payload)
    if _BOUND_BACKEND is not None:
        return _BOUND_BACKEND.export_gradients(capture_path=capture_path)
    _unsupported_board_call('export_gradients')

def capture_optimizer_state(payload: bytes, out_path: str | Path | None = None) -> Path:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError('capture_optimizer_state expects a bytes-like optimizer-state payload.')
    target = Path(out_path) if out_path is not None else PACKAGE_DIR / 'outputs' / 'optimizer_state_after.bin'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bytes(payload))
    return target

def export_optimizer_state(*, board_payload: bytes | None = None, capture_path: str | Path | None = None) -> bytes:
    manifest = load_manifest()
    opt_state = manifest.get('runtime_optimizer_state', {})
    if board_payload is not None:
        capture_optimizer_state(board_payload, capture_path)
        return bytes(board_payload)
    if not bool(opt_state.get('capture_supported_by_api', False)):
        raise RuntimeError('optimizer-state capture was not generated/supported for this package.')
    if _BOUND_BACKEND is not None:
        return _BOUND_BACKEND.export_optimizer_state(capture_path=capture_path)
    _unsupported_board_call('export_optimizer_state')

def reset_accumulators() -> Any:
    if _BOUND_BACKEND is not None:
        return _BOUND_BACKEND.reset_accumulators()
    _unsupported_board_call('reset_accumulators')

def accumulate_gradients(inputs: Any | None = None, labels: Any | None = None, *, steps: int = 1) -> Any:
    if int(steps) < 1:
        raise ValueError('steps must be >= 1')
    if _BOUND_BACKEND is not None:
        return _BOUND_BACKEND.accumulate_gradients(inputs=inputs, labels=labels, steps=int(steps))
    _unsupported_board_call('accumulate_gradients')

def apply_accumulated_gradients() -> Any:
    if _BOUND_BACKEND is not None:
        return _BOUND_BACKEND.apply_accumulated_gradients()
    _unsupported_board_call('apply_accumulated_gradients')

def _command_items() -> list[dict[str, Any]]:
    plan_sequence = load_runtime_execution_plan().get('sequence', [])
    if plan_sequence:
        return [dict(item) for item in plan_sequence]
    items: list[dict[str, Any]] = []
    for raw in load_run_sequence().get('sequence', []):
        command = raw.get('command') if isinstance(raw, dict) else str(raw)
        args = raw.get('args', {}) if isinstance(raw, dict) and isinstance(raw.get('args', {}), dict) else {}
        items.append({'command': str(command), 'mode': None, 'args': args, 'sync_before': [], 'sync_after': [], 'capture': None})
    return items

def _call_command(command: str, args: dict[str, Any], *, capture_path: str | Path | None = None) -> Any:
    if command == 'import_weights':
        return import_weights()
    if command == 'run_inference':
        return run_inference(repeat=int(args.get('repeat', 1)))
    if command == 'run_training':
        return run_training(steps=int(args.get('steps', 1)))
    if command == 'export_weights':
        return export_weights()
    if command == 'export_gradients':
        return export_gradients(capture_path=capture_path)
    if command == 'export_optimizer_state':
        return export_optimizer_state(capture_path=capture_path)
    if command == 'reset_accumulators':
        return reset_accumulators()
    if command == 'accumulate_gradients':
        return accumulate_gradients(steps=int(args.get('steps', 1)))
    if command == 'apply_accumulated_gradients':
        return apply_accumulated_gradients()
    raise ValueError(f'Unsupported runtime command: {command}')

def _backend_metadata() -> dict[str, Any]:
    manifest = load_manifest()
    backend = _BOUND_BACKEND
    wrapped = getattr(backend, 'backend', backend)
    hardware = manifest.get('hardware', {}) if isinstance(manifest.get('hardware', {}), dict) else {}
    return {
        'type': type(wrapped).__name__ if wrapped is not None else None,
        'wrapper_type': type(backend).__name__ if backend is not None else None,
        'board': manifest.get('board'),
        'bitfile': hardware.get('bitstream', {}).get('package_path') if isinstance(hardware.get('bitstream', {}), dict) else None,
        'ip_name': manifest.get('top_name'),
        'bound': backend is not None,
        'bound_buffers': sorted(_BOUND_BUFFERS.keys()),
    }

def _capture_status(buffer_name: str | None, capture: str | None) -> dict[str, Any] | None:
    if not capture:
        return None
    path = Path(capture)
    if not path.is_absolute():
        path = PACKAGE_DIR / path
    return {
        'buffer': buffer_name,
        'path': path.relative_to(PACKAGE_DIR).as_posix() if path.is_relative_to(PACKAGE_DIR) else path.as_posix(),
        'status': 'written' if path.exists() else 'missing',
        'bytes': path.stat().st_size if path.exists() else 0,
    }

def _result_summary(result: Any) -> dict[str, Any]:
    if isinstance(result, (bytes, bytearray, memoryview)):
        return {'type': 'bytes', 'bytes': len(bytes(result))}
    if isinstance(result, dict):
        return {'type': 'dict', 'keys': sorted(str(k) for k in result.keys())}
    if result is None:
        return {'type': 'none'}
    return {'type': type(result).__name__}

def _write_runtime_execution_report(report: dict[str, Any]) -> None:
    RUNTIME_EXECUTION_REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    lines = [
        '# FPGAI Runtime Execution Report',
        '',
        f"Status: {report.get('status')}",
        f"Backend: {report.get('backend', {}).get('type')}",
        f"Board: {report.get('backend', {}).get('board')}",
        f"Bitfile: {report.get('backend', {}).get('bitfile')}",
        f"IP: {report.get('backend', {}).get('ip_name')}",
        '',
        '## Commands',
        '',
    ]
    for item in report.get('sequence', []):
        lines.extend([
            f"{int(item.get('index', 0)) + 1}. {item.get('command')}",
            f"   - Mode: {item.get('mode')}",
            f"   - Status: {item.get('status')}",
            f"   - sync_before: {', '.join(item.get('sync_before', [])) or '-'}",
            f"   - sync_after: {', '.join(item.get('sync_after', [])) or '-'}",
            f"   - Capture: {item.get('capture') or '-'}",
            f"   - Latency ms: {item.get('latency_ms')}",
        ])
        if item.get('error'):
            lines.append(f"   - Error: {item.get('error')}")
        lines.append('')
    if report.get('captures'):
        lines.extend(['## Captures', ''])
        for cap in report.get('captures', []):
            lines.append(f"- {cap.get('buffer')}: {cap.get('path')} ({cap.get('status')}, {cap.get('bytes')} bytes)")
        lines.append('')
    if report.get('errors'):
        lines.extend(['## Errors', ''])
        for error in report.get('errors', []):
            lines.append(f"- {error.get('command')}: {error.get('error')}")
        lines.append('')
    RUNTIME_EXECUTION_REPORT_MD.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')

def run_sequence(*, strict: bool = True, write_report: bool = True, return_report: bool = False) -> list[Any] | dict[str, Any]:
    results: list[Any] = []
    sequence_report: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        'schema_version': 1,
        'package_kind': 'fpgai_runtime_execution_report',
        'status': 'passed',
        'backend': _backend_metadata(),
        'sequence': sequence_report,
        'captures': captures,
        'errors': errors,
    }

    for index, item in enumerate(_command_items()):
        command = str(item.get('command'))
        args = dict(item.get('args', {})) if isinstance(item.get('args', {}), dict) else {}
        sync_before = [str(v) for v in item.get('sync_before', [])]
        sync_after = [str(v) for v in item.get('sync_after', [])]
        capture = item.get('capture')
        capture_path = (PACKAGE_DIR / str(capture)) if capture else None
        entry = {
            'index': index,
            'command': command,
            'mode': item.get('mode'),
            'status': 'running',
            'sync_before': sync_before,
            'sync_after': sync_after,
            'capture': str(capture) if capture else None,
            'captures': [],
            'latency_ms': None,
            'result': None,
            'error': None,
        }
        started = time.perf_counter()
        try:
            _sync_before(sync_before)
            result = _call_command(command, args, capture_path=capture_path)
            _sync_after(sync_after)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            entry['status'] = 'passed'
            entry['latency_ms'] = round(elapsed_ms, 6)
            entry['result'] = _result_summary(result)
            cap = _capture_status(sync_after[0] if sync_after else None, str(capture) if capture else None)
            if cap is not None:
                entry['captures'].append(cap)
                captures.append(cap)
            results.append(result)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            error = {'command': command, 'error': str(exc), 'type': type(exc).__name__, 'traceback': traceback.format_exc(limit=8)}
            entry['status'] = 'failed'
            entry['latency_ms'] = round(elapsed_ms, 6)
            entry['error'] = str(exc)
            errors.append(error)
            report['status'] = 'failed'
            sequence_report.append(entry)
            if write_report:
                _write_runtime_execution_report(report)
            if strict:
                raise
            continue
        sequence_report.append(entry)

    if errors:
        report['status'] = 'failed'
    if write_report:
        _write_runtime_execution_report(report)
    if return_report:
        return report
    return results
'''
    runtime_source += "\n" + f"GENERATED_COMMANDS = {commands!r}\n"
    runtime_source += f"RUNTIME_WEIGHT_PAYLOAD_REQUIRED = {bool(runtime_weights.get('required'))!r}\n"
    runtime_source += f"RUNTIME_WEIGHT_EXPORT_SUPPORTED = {bool(runtime_weights.get('export_supported'))!r}\n"
    api_path.write_text(runtime_source, encoding="utf-8")
    return {"source": api_path.as_posix(), "package_path": "runtime_api.py", "bytes": api_path.stat().st_size, "present": True}

def emit_runtime_package(
    out_dir: str | Path,
    *,
    board: str | None = None,
    pipeline_mode: str | None = None,
    top_name: str | None = None,
    hls_artifacts: Mapping[str, Any] | None = None,
    weights_mode: str | None = None,
    communication_plan: Any | None = None,
    memory_plan: Any | None = None,
    build_stages: Mapping[str, Any] | None = None,
    runtime_sequence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a self-describing runtime package from existing compile artifacts.

    This function does not run Vivado, deploy to a board, or infer that hardware
    artifacts exist. It packages files that are already present and records
    bitstream/XSA/HWH status truthfully.
    """

    root = Path(out_dir).resolve()
    package_dir = root / "runtime_package"

    # Vivado/bitstream stages may refresh an already-created runtime package by
    # calling this function with only ``out_dir``. Preserve the compiler-resolved
    # runtime contract before replacing the package directory, otherwise the
    # second packaging pass would silently erase the execution sequence and
    # related metadata.
    previous_payload: dict[str, Any] = {}
    previous_manifest = package_dir / "package_manifest.json"
    if previous_manifest.exists():
        try:
            loaded = json.loads(previous_manifest.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                previous_payload = dict(loaded)
        except (OSError, json.JSONDecodeError):
            previous_payload = {}

    if board is None:
        board = previous_payload.get("board")
    if pipeline_mode is None:
        pipeline_mode = previous_payload.get("pipeline_mode")
    if top_name is None:
        top_name = previous_payload.get("top_name")
    if weights_mode is None:
        prior_weights = previous_payload.get("runtime_weights", {})
        if isinstance(prior_weights, Mapping):
            weights_mode = prior_weights.get("weights_mode")
    if hls_artifacts is None:
        prior_hls = previous_payload.get("hls_artifacts", {})
        if isinstance(prior_hls, Mapping):
            hls_artifacts = dict(prior_hls)
    if build_stages is None:
        prior_stages = previous_payload.get("build_stages", {})
        if isinstance(prior_stages, Mapping):
            build_stages = dict(prior_stages)
    if runtime_sequence is None:
        prior_sequence = previous_payload.get("runtime_sequence", {})
        if isinstance(prior_sequence, Mapping):
            runtime_sequence = dict(prior_sequence)

    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Any] = {}

    copy_plan = {
        "compile_manifest": (root / "manifest.json", package_dir / "manifest.json"),
        "input_bin": (root / "input.bin", package_dir / "inputs" / "input.bin"),
        "output_bin": (root / "output.bin", package_dir / "outputs" / "output.bin"),
        "gradients_after_bin": (root / "gradients_after.bin", package_dir / "outputs" / "gradients_after.bin"),
        "gradients_export_bin": (root / "gradients_export.bin", package_dir / "outputs" / "gradients_export.bin"),
        "grads_ref_bin": (root / "training_reference" / "grads_ref.bin", package_dir / "reference" / "grads_ref.bin"),
        "gradients_after_ref_bin": (root / "training_reference" / "gradients_after_ref.bin", package_dir / "reference" / "gradients_after_ref.bin"),
        "optimizer_state_after_bin": (root / "optimizer_state_after.bin", package_dir / "outputs" / "optimizer_state_after.bin"),
        "optimizer_state_after_ref_bin": (root / "training_reference" / "optimizer_state_after_ref.bin", package_dir / "reference" / "optimizer_state_after_ref.bin"),
        "hls_artifact_metadata": (
            root / "hls_artifact_metadata.json",
            package_dir / "hls" / "hls_artifact_metadata.json",
        ),
        "hls_schedule_summary": (
            root / "hls_schedule_summary.json",
            package_dir / "hls" / "hls_schedule_summary.json",
        ),
        "hls_ii_comparison": (
            root / "hls_ii_comparison.json",
            package_dir / "hls" / "hls_ii_comparison.json",
        ),
    }

    for name, (src, dst) in copy_plan.items():
        copied = _copy_if_exists(src, dst)
        if copied is not None:
            copied["package_path"] = _safe_rel(Path(copied["package_path"]), package_dir)
            files[name] = copied

    # Capture HLS run logs when present.
    hls_logs = _collect_existing(root, ["hls/logs/*.log", "hls/logs/*.json"])
    copied_logs: list[dict[str, Any]] = []
    for src in hls_logs:
        copied = _copy_if_exists(src, package_dir / "hls" / "logs" / src.name)
        if copied is not None:
            copied["package_path"] = _safe_rel(Path(copied["package_path"]), package_dir)
            copied_logs.append(copied)
    if copied_logs:
        files["hls_logs"] = copied_logs

    # Runtime hardware handoff/status. These are presence checks only.
    bitstream = _first_existing(
        root,
        [
            "vivado_bridge/bitstream/*.bit",
            "vivado_bridge/project/**/*.bit",
            "**/*.bit",
        ],
    )
    hwh = _first_existing(
        root,
        [
            "vivado_bridge/bitstream/*.hwh",
            "vivado_bridge/project/**/*.hwh",
            "**/*.hwh",
        ],
    )
    xsa = _first_existing(
        root,
        [
            "vivado_bridge/bitstream/*.xsa",
            "vivado_bridge/project/**/*.xsa",
            "**/*.xsa",
        ],
    )

    hardware = {
        "bitstream": _artifact_status(bitstream),
        "hwh": _artifact_status(hwh),
        "xsa": _artifact_status(xsa),
        "deployable_overlay_present": bool(bitstream is not None and (hwh is not None or xsa is not None)),
    }

    for name, src in {"bitstream": bitstream, "hwh": hwh, "xsa": xsa}.items():
        if src is None:
            continue
        copied = _copy_if_exists(src, package_dir / "hardware" / src.name)
        if copied is not None:
            copied["package_path"] = _safe_rel(Path(copied["package_path"]), package_dir)
            files[name] = copied

    weight_payload = _emit_runtime_weight_payload(root, package_dir, weights_mode=weights_mode)
    files.update(weight_payload["files"])

    runtime_sequence_payload = dict(runtime_sequence or {})
    if runtime_sequence_payload:
        run_sequence_path = package_dir / "run_sequence.json"
        run_sequence_path.write_text(json.dumps(runtime_sequence_payload, indent=2, sort_keys=True), encoding="utf-8")
        files["runtime_sequence"] = {
            "path": "runtime_package/run_sequence.json",
            "package_path": "run_sequence.json",
            "present": True,
        }

    runtime_buffer_plans = _emit_runtime_buffer_plans(
        root,
        package_dir,
        runtime_sequence=runtime_sequence_payload,
        runtime_weights=weight_payload["summary"],
        pipeline_mode=pipeline_mode,
    )
    files.update(runtime_buffer_plans["files"])

    payload: dict[str, Any] = {
        "schema_version": 1,
        "package_kind": "fpgai_runtime_package",
        "status": "created",
        "package_dir": package_dir.as_posix(),
        "source_out_dir": root.as_posix(),
        "board": board,
        "pipeline_mode": pipeline_mode,
        "top_name": top_name,
        "hls_artifacts": dict(hls_artifacts or {}),
        "build_stages": {str(k): bool(v) for k, v in dict(build_stages or {}).items()},
        "runtime_sequence": runtime_sequence_payload,
        "runtime_buffer_plan": runtime_buffer_plans["buffer_plan"],
        "runtime_execution_plan": runtime_buffer_plans["runtime_execution_plan"],
        "hardware": hardware,
        "runtime_weights": weight_payload["summary"],
        "runtime_io": _runtime_io_movement_summary(communication_plan),
        "runtime_activation_storage": _runtime_activation_storage_summary(memory_plan),
        "runtime_gradient_export": {
            "capture_supported_by_api": True,
            "captured_gradients_present": ("gradients_after_bin" in files or "gradients_export_bin" in files),
            "reference_gradients_present": ("grads_ref_bin" in files or "gradients_after_ref_bin" in files),
            "capture_filename": "gradients_after.bin",
            "reference_filename": "grads_ref.bin",
        },
        "runtime_optimizer_state": {
            "capture_supported_by_api": True,
            "captured_state_present": "optimizer_state_after_bin" in files,
            "reference_state_present": "optimizer_state_after_ref_bin" in files,
            "capture_filename": "optimizer_state_after.bin",
            "reference_filename": "optimizer_state_after_ref.bin",
        },
        "files": files,
        "notes": [
            "Runtime package records and copies existing artifacts only.",
            "It does not run Vivado, deploy to hardware, or infer missing bitstream/XSA/HWH files.",
        ],
    }

    board_runtime = _emit_board_runtime_backend(package_dir, payload)
    files["board_runtime"] = board_runtime
    payload["board_runtime"] = {
        "path": "runtime_package/board_runtime.py",
        "package_path": "board_runtime.py",
        "present": True,
        "hls_modes": {
            "run_training": 2,
            "accumulate_gradients": 3,
            "apply_accumulated_gradients": 4,
            "reset_accumulators": 5,
            "export_gradients": 8,
            "export_optimizer_state": 9,
        },
        "backend_contract": "bind a real board object implementing call_mode()/read_buffer(), use generated PynqDmaMmioBackend/create_pynq_backend for PYNQ/KV260, or provide explicit runtime methods; allocate/bind PYNQ buffers from buffer_plan.json with allocate_runtime_buffers()/bind_allocated_buffers()",
    }

    runtime_api = _emit_runtime_api(package_dir, payload)
    files["runtime_api"] = runtime_api
    payload["runtime_api"] = {
        "path": "runtime_package/runtime_api.py",
        "package_path": "runtime_api.py",
        "present": True,
        "functions": [
            "import_weights",
            "run_inference",
            "run_training",
            "export_weights",
            "export_gradients",
            "capture_gradients",
            "export_optimizer_state",
            "capture_optimizer_state",
            "allocate_runtime_buffers",
            "bind_allocated_buffers",
            "bind_backend",
            "get_backend",
            "reset_accumulators",
            "accumulate_gradients",
            "apply_accumulated_gradients",
            "run_sequence",
        ],
        "truth_boundary": "Generated API can allocate/bind PYNQ-style buffers and bind a real board backend object; physical execution still requires a deployed bitstream and board-specific DMA/MMIO implementation.",
    }

    manifest_path = package_dir / "package_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    readme = package_dir / "README_RUNTIME.md"
    readme.write_text(
        "\n".join(
            [
                "# FPGAI Runtime Package",
                "",
                "This package contains runtime-facing artifacts copied from an FPGAI compile output.",
                "",
                f"- board: `{board}`",
                f"- pipeline_mode: `{pipeline_mode}`",
                f"- top_name: `{top_name}`",
                f"- bitstream present: `{hardware['bitstream']['present']}`",
                f"- hwh present: `{hardware['hwh']['present']}`",
                f"- xsa present: `{hardware['xsa']['present']}`",
                f"- deployable overlay present: `{hardware['deployable_overlay_present']}`",
                f"- runtime weight payload required: `{weight_payload['summary']['required']}`",
                f"- runtime weight payload present: `{weight_payload['summary']['present']}`",
                f"- gradient export capture API: `{payload['runtime_gradient_export']['capture_supported_by_api']}`",
                f"- gradient export captured file present: `{payload['runtime_gradient_export']['captured_gradients_present']}`",
                f"- optimizer-state capture API: `{payload['runtime_optimizer_state']['capture_supported_by_api']}`",
                f"- optimizer-state captured file present: `{payload['runtime_optimizer_state']['captured_state_present']}`",
                f"- selected build stages: `{json.dumps(payload['build_stages'], sort_keys=True)}`",
                f"- runtime sequence: `{json.dumps(runtime_sequence_payload.get('sequence', []), sort_keys=True)}`",
                f"- runtime buffers: `{len(runtime_buffer_plans['buffer_plan'].get('buffers', []))}`",
                "",
                "The package is truthful: missing hardware handoff files are recorded as missing.",
                "Use the Vivado bridge flow to generate bitstream/XSA artifacts before board deployment.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    from fpgai.runtime.package_validation import emit_runtime_package_validation

    validation_summary = emit_runtime_package_validation(root, package_dir)
    files["runtime_package_validation_json"] = {
        "package_path": "runtime_package_validation.json",
        "present": True,
    }
    files["runtime_package_validation_md"] = {
        "package_path": "runtime_package_validation.md",
        "present": True,
    }
    payload["runtime_package_validation"] = validation_summary
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    emit_runtime_package_validation(root, package_dir)

    return {
        "path": "runtime_package/package_manifest.json",
        "package_dir": "runtime_package",
        "status": payload["status"],
        "deployable_overlay_present": hardware["deployable_overlay_present"],
        "bitstream_present": hardware["bitstream"]["present"],
        "hwh_present": hardware["hwh"]["present"],
        "xsa_present": hardware["xsa"]["present"],
        "runtime_weight_payload_required": weight_payload["summary"]["required"],
        "runtime_weight_payload_present": weight_payload["summary"]["present"],
        "runtime_weight_total_words": weight_payload["summary"]["total_words"],
        "runtime_package_validation_status": validation_summary["status"],
        "runtime_package_deployability_ready": validation_summary["deployability_ready"],
        "runtime_package_validation_json": validation_summary["validation_json"],
        "file_count": len(files),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create an FPGAI runtime package from a compile output directory.")
    parser.add_argument("out_dir")
    parser.add_argument("--board")
    parser.add_argument("--pipeline-mode")
    parser.add_argument("--top-name")
    ns = parser.parse_args(argv)

    result = emit_runtime_package(
        ns.out_dir,
        board=ns.board,
        pipeline_mode=ns.pipeline_mode,
        top_name=ns.top_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
