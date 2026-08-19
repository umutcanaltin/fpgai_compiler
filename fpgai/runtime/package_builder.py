"""Assemble a self-describing FPGAI runtime package."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from fpgai.runtime.package_artifacts import (
    _artifact_status,
    _collect_existing,
    _copy_if_exists,
    _first_existing,
    _safe_rel,
)
from fpgai.runtime.board_runtime_generator import _emit_board_runtime_backend
from fpgai.runtime.runtime_api_generator import _emit_runtime_api
from fpgai.runtime.runtime_plans import (
    _emit_runtime_buffer_plans,
    _runtime_activation_storage_summary,
    _runtime_io_movement_summary,
)
from fpgai.runtime.runtime_weights import _emit_runtime_weight_payload

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
    persistent_state_plan: Mapping[str, Any] | None = None,
    graph_runtime_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a self-describing runtime package from existing compile artifacts.

    This function does not run Vivado, deploy to a board, or infer that hardware
    artifacts exist. It packages files that are already present and records
    bitstream/XSA/HWH status accurately.
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
    if persistent_state_plan is None:
        prior_state = previous_payload.get("persistent_state", {})
        if isinstance(prior_state, Mapping):
            persistent_state_plan = dict(prior_state)
    if graph_runtime_contract is None:
        prior_contract = previous_payload.get("graph_runtime_contract", {})
        if isinstance(prior_contract, Mapping):
            graph_runtime_contract = dict(prior_contract)

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
        persistent_state_plan=persistent_state_plan,
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
        "persistent_state": dict(persistent_state_plan or {}),
        "graph_runtime_contract": dict(graph_runtime_contract or {}),
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
            "load_graph_runtime_contract",
            "reset_state",
            "import_state",
            "export_state",
            "read_state",
            "write_state",
            "prepare_prefill",
            "prepare_decode",
            "postprocess_detections",
            "reset_accumulators",
            "accumulate_gradients",
            "apply_accumulated_gradients",
            "run_sequence",
        ],
        "validation_boundary": "Generated API can allocate/bind PYNQ-style buffers and bind a real board backend object; physical execution still requires a deployed bitstream and board-specific DMA/MMIO implementation.",
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
                f"- persistent state tensors: `{int((persistent_state_plan or {}).get('tensor_count', 0))}`",
                f"- graph runtime contract keys: `{sorted((graph_runtime_contract or {}).keys())}`",
                "",
                "The package is accurate: missing hardware handoff files are recorded as missing.",
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
        "persistent_state": dict(persistent_state_plan or {}),
        "graph_runtime_contract": dict(graph_runtime_contract or {}),
        "file_count": len(files),
    }
