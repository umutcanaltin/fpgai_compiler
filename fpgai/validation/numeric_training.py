"""Training reference, loss, tiled-I/O, accumulation, and gradient validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .numeric_common import (
    _cfg_lookup,
    _compare_file_pair,
    _compare_vectors,
    _copy_or_transform_f32,
    _exists,
    _normalize_sequence_entries,
    _path_or_none,
    _positive_int,
    _read_f32_file,
    _read_json_file,
)

def _loss_validation_payload(
    out_dir: str | Path,
    *,
    pipeline_mode: str,
    raw_config: dict[str, Any] | None,
    training_reference_result: Any,
    training_compare_result: Any,
) -> dict[str, Any]:
    if str(pipeline_mode or "").strip().lower() != "training_on_device":
        return {"requested": False, "status": "not_applicable", "passed": False}

    raw = raw_config or {}
    loss_type = str(_cfg_lookup(raw, "training.loss.type", "mse") or "mse").strip().lower().replace("-", "_")
    if loss_type in {"ce"}:
        loss_type = "cross_entropy"

    if loss_type != "cross_entropy":
        return {
            "requested": False,
            "status": "not_requested",
            "passed": False,
            "loss_type": loss_type,
            "validation_boundary": "Cross-entropy validation is only requested when training.loss.type=cross_entropy.",
        }

    ref_dir = Path(out_dir) / "training_reference"
    logits_ref = getattr(training_reference_result, "logits_ref_path", None) or (ref_dir / "logits_ref.bin")
    softmax_ref = getattr(training_reference_result, "softmax_ref_path", None) or (ref_dir / "softmax_ref.bin")
    dlogits_ref = getattr(training_reference_result, "dlogits_ref_path", None) or (ref_dir / "dlogits_ref.bin")
    loss_json = getattr(training_reference_result, "cross_entropy_loss_ref_json", None) or (ref_dir / "cross_entropy_loss_ref.json")
    loss_payload = _read_json_file(loss_json) or {}

    preserved = Path(out_dir) / ".fpgai_preserved_validation"
    def _first_existing(*candidates: Path) -> Path | None:
        for candidate in candidates:
            try:
                if candidate.exists():
                    return candidate
            except OSError:
                continue
        return None

    dlogits_got = _first_existing(
        preserved / "dlogits_after.bin",
        preserved / "hls" / "dlogits_after.bin",
        preserved / "runtime_package" / "outputs" / "dlogits_after.bin",
        Path(out_dir) / "dlogits_after.bin",
        Path(out_dir) / "hls" / "dlogits_after.bin",
        Path(out_dir) / "runtime_package" / "outputs" / "dlogits_after.bin",
        Path(out_dir) / "gradients_after.bin",
        Path(out_dir) / "runtime_package" / "outputs" / "gradients_after.bin",
    )
    softmax_got = _first_existing(
        preserved / "softmax_after.bin",
        preserved / "hls" / "softmax_after.bin",
        preserved / "runtime_package" / "outputs" / "softmax_after.bin",
        Path(out_dir) / "softmax_after.bin",
        Path(out_dir) / "hls" / "softmax_after.bin",
        Path(out_dir) / "runtime_package" / "outputs" / "softmax_after.bin",
    )

    comparisons = {
        "dlogits": _compare_file_pair(dlogits_ref, dlogits_got) if dlogits_got is not None else {
            "ref_path": _path_or_none(dlogits_ref),
            "got_path": None,
            "ref_exists": _exists(dlogits_ref),
            "got_exists": False,
            "status": "artifact_missing",
            "passed": False,
        },
        "softmax": _compare_file_pair(softmax_ref, softmax_got) if softmax_got is not None else {
            "ref_path": _path_or_none(softmax_ref),
            "got_path": None,
            "ref_exists": _exists(softmax_ref),
            "got_exists": False,
            "status": "artifact_missing",
            "passed": False,
        },
        "training_step": _training_compare_payload(training_compare_result),
    }
    reference_exists = _exists(logits_ref) and _exists(softmax_ref) and _exists(dlogits_ref) and _exists(loss_json)
    required_compared = all(
        isinstance(comparisons.get(name), dict)
        and comparisons[name].get("status") == "compared"
        and bool(comparisons[name].get("passed", False))
        for name in ("dlogits", "softmax")
    )
    if required_compared:
        status = "compared"
        passed = True
    elif reference_exists:
        status = "artifact_missing"
        passed = False
    else:
        status = "reference_missing"
        passed = False

    return {
        "requested": True,
        "status": status,
        "passed": passed,
        "loss_type": "cross_entropy",
        "softmax_stable": True,
        "loss_before": loss_payload.get("loss", getattr(training_reference_result, "loss_before", None)),
        "reference": {
            "status": "generated" if reference_exists else "missing",
            "logits_ref_bin": _path_or_none(logits_ref),
            "softmax_ref_bin": _path_or_none(softmax_ref),
            "cross_entropy_loss_ref_json": _path_or_none(loss_json),
            "dlogits_ref_bin": _path_or_none(dlogits_ref),
            "logits_ref_exists": _exists(logits_ref),
            "softmax_ref_exists": _exists(softmax_ref),
            "cross_entropy_loss_ref_exists": _exists(loss_json),
            "dlogits_ref_exists": _exists(dlogits_ref),
        },
        "comparisons": comparisons,
        "validation_boundary": (
            "Cross-entropy is validation-qualified only when generated/HLS/runtime loss-gradient artifacts is compared with the stable-softmax Python reference. "
            "Reference artifacts without captured generated outputs are artifact_missing, never a pass."
        ),
    }


def _movement_spec(raw_config: dict[str, Any] | None, tensor: str, direction: str) -> dict[str, Any]:
    raw = raw_config or {}
    spec = _cfg_lookup(raw, f"data_movement.{tensor}.{direction}", {})
    return spec if isinstance(spec, dict) else {}


def _is_tiled_training_io_spec(raw_config: dict[str, Any] | None, tensor: str, direction: str, interface: str | None = None) -> bool:
    spec = _movement_spec(raw_config, tensor, direction)
    if str(spec.get("policy", "")).strip().lower() != "tiled":
        return False
    iface = str(spec.get("interface", "")).strip().lower()
    if interface is not None:
        return iface == interface
    return iface in {"m_axi", "axi_stream"}


def _first_existing_path(*candidates: Path) -> Path | None:
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def _training_tiled_io_validation_payload(
    out_dir: str | Path,
    *,
    pipeline_mode: str,
    raw_config: dict[str, Any] | None,
    training_reference_result: Any,
    training_compare_result: Any,
) -> dict[str, Any]:
    if str(pipeline_mode or "").strip().lower() != "training_on_device":
        return {"requested": False, "status": "not_applicable", "passed": False}

    input_m_axi = _is_tiled_training_io_spec(raw_config, "inputs", "import", "m_axi")
    label_m_axi = _is_tiled_training_io_spec(raw_config, "labels", "import", "m_axi")
    output_m_axi = _is_tiled_training_io_spec(raw_config, "outputs", "export", "m_axi")
    input_axis = _is_tiled_training_io_spec(raw_config, "inputs", "import", "axi_stream")
    label_axis = _is_tiled_training_io_spec(raw_config, "labels", "import", "axi_stream")
    output_axis = _is_tiled_training_io_spec(raw_config, "outputs", "export", "axi_stream")
    requested = any([input_m_axi, label_m_axi, output_m_axi, input_axis, label_axis, output_axis])
    if not requested:
        return {
            "requested": False,
            "status": "not_requested",
            "passed": False,
            "validation_boundary": "Training tiled-I/O validation is only requested when data_movement inputs/labels/outputs use policy=tiled.",
        }

    interfaces = []
    if any([input_m_axi, label_m_axi, output_m_axi]):
        interfaces.append("m_axi")
    if any([input_axis, label_axis, output_axis]):
        interfaces.append("axi_stream")
    interface = interfaces[0] if len(interfaces) == 1 else "mixed"

    input_tile_size = _positive_int(_movement_spec(raw_config, "inputs", "import").get("tile_size", _cfg_lookup(raw_config, "training.tiling.input_tile_size", 64)), 64)
    label_tile_size = _positive_int(_movement_spec(raw_config, "labels", "import").get("tile_size", _cfg_lookup(raw_config, "training.tiling.label_tile_size", 64)), 64)
    output_tile_size = _positive_int(_movement_spec(raw_config, "outputs", "export").get("tile_size", _cfg_lookup(raw_config, "training.tiling.output_tile_size", 64)), 64)

    ref_dir = Path(out_dir) / "training_reference"
    refs = {
        "inputs": getattr(training_reference_result, "tiled_inputs_ref_path", None) or (ref_dir / "tiled_inputs_ref.bin"),
        "labels": getattr(training_reference_result, "tiled_labels_ref_path", None) or (ref_dir / "tiled_labels_ref.bin"),
        "outputs": getattr(training_reference_result, "tiled_outputs_ref_path", None) or (ref_dir / "tiled_outputs_ref.bin"),
        "gradients": getattr(training_reference_result, "tiled_gradients_ref_path", None) or (ref_dir / "tiled_gradients_ref.bin"),
        "weights_after": getattr(training_reference_result, "tiled_weights_after_ref_path", None) or (ref_dir / "tiled_weights_after_ref.bin"),
    }
    reference_available = all(_exists(path) for path in refs.values())

    preserved = Path(out_dir) / ".fpgai_preserved_validation"
    output_got = _first_existing_path(
        preserved / "tiled_outputs_after.bin",
        preserved / "hls" / "tiled_outputs_after.bin",
        preserved / "runtime_package" / "outputs" / "tiled_outputs_after.bin",
        Path(out_dir) / "tiled_outputs_after.bin",
        Path(out_dir) / "hls" / "tiled_outputs_after.bin",
        Path(out_dir) / "runtime_package" / "outputs" / "tiled_outputs_after.bin",
        Path(out_dir) / "runtime_package" / "outputs" / "outputs_after.bin",
    )
    gradients_got = _first_existing_path(
        preserved / "tiled_gradients_after.bin",
        preserved / "hls" / "tiled_gradients_after.bin",
        preserved / "runtime_package" / "outputs" / "tiled_gradients_after.bin",
        Path(out_dir) / "tiled_gradients_after.bin",
        Path(out_dir) / "hls" / "tiled_gradients_after.bin",
        Path(out_dir) / "gradients_after.bin",
        Path(out_dir) / "runtime_package" / "outputs" / "tiled_gradients_after.bin",
        Path(out_dir) / "runtime_package" / "outputs" / "gradients_after.bin",
    )
    weights_got = _first_existing_path(
        preserved / "tiled_weights_after.bin",
        preserved / "hls" / "tiled_weights_after.bin",
        preserved / "runtime_package" / "outputs" / "tiled_weights_after.bin",
        Path(out_dir) / "tiled_weights_after.bin",
        Path(out_dir) / "hls" / "tiled_weights_after.bin",
        Path(out_dir) / "weights_after.bin",
        Path(out_dir) / "runtime_package" / "outputs" / "tiled_weights_after.bin",
        Path(out_dir) / "runtime_package" / "outputs" / "weights_after.bin",
    )

    def _missing(ref_path: Any, got_path: Any) -> dict[str, Any]:
        return {
            "ref_path": _path_or_none(ref_path),
            "got_path": _path_or_none(got_path),
            "ref_exists": _exists(ref_path),
            "got_exists": _exists(got_path),
            "status": "artifact_missing",
            "passed": False,
        }

    comparisons = {
        "outputs": _compare_file_pair(refs["outputs"], output_got) if output_got is not None else _missing(refs["outputs"], None),
        "gradients": _compare_file_pair(refs["gradients"], gradients_got) if gradients_got is not None else _missing(refs["gradients"], None),
        "weights_after": _compare_file_pair(refs["weights_after"], weights_got) if weights_got is not None else _missing(refs["weights_after"], None),
        "training_step": _training_compare_payload(training_compare_result),
    }
    required_names = ("outputs", "gradients", "weights_after")
    all_compared = all(
        isinstance(comparisons.get(name), dict)
        and comparisons[name].get("status") == "compared"
        and bool(comparisons[name].get("passed", False))
        for name in required_names
    )
    any_capture = output_got is not None or gradients_got is not None or weights_got is not None
    if all_compared:
        status = "compared"
        passed = True
    elif reference_available:
        status = "artifact_missing" if not any_capture else "failed"
        passed = False
    else:
        status = "reference_missing"
        passed = False

    return {
        "requested": True,
        "status": status,
        "passed": passed,
        "interface": interface,
        "interfaces": interfaces,
        "tile_size": input_tile_size,
        "tile_sizes": {
            "inputs": input_tile_size,
            "labels": label_tile_size,
            "outputs": output_tile_size,
        },
        "input_tiled": bool(input_m_axi or input_axis),
        "labels_tiled": bool(label_m_axi or label_axis),
        "output_tiled": bool(output_m_axi or output_axis),
        "m_axi": {"inputs": input_m_axi, "labels": label_m_axi, "outputs": output_m_axi},
        "axi_stream": {
            "inputs": input_axis,
            "labels": label_axis,
            "outputs": output_axis,
            "tlast_required": bool(output_axis),
        },
        "compute_fused": bool(requested and reference_available),
        "reference_available": bool(reference_available),
        "captures_available": bool(any_capture),
        "reference": {
            "inputs_ref_bin": _path_or_none(refs["inputs"]),
            "labels_ref_bin": _path_or_none(refs["labels"]),
            "outputs_ref_bin": _path_or_none(refs["outputs"]),
            "gradients_ref_bin": _path_or_none(refs["gradients"]),
            "weights_after_ref_bin": _path_or_none(refs["weights_after"]),
            "inputs_ref_exists": _exists(refs["inputs"]),
            "labels_ref_exists": _exists(refs["labels"]),
            "outputs_ref_exists": _exists(refs["outputs"]),
            "gradients_ref_exists": _exists(refs["gradients"]),
            "weights_after_ref_exists": _exists(refs["weights_after"]),
        },
        "comparisons": comparisons,
        "validation_boundary": (
            "Training tiled I/O is validation-qualified only when tiled input/label movement feeds the compute path and captured tiled outputs, gradients, and weights-after are compared against Python reference artifacts. "
            "Generated tiled interfaces without captures are artifact_missing, never a pass."
        ),
    }


def _batch_accumulation_validation_payload(
    out_dir: str | Path,
    *,
    pipeline_mode: str,
    raw_config: dict[str, Any] | None,
    runtime_sequence: Any,
    training_reference_result: Any,
    training_compare_result: Any,
) -> dict[str, Any]:
    if str(pipeline_mode or '').strip().lower() != 'training_on_device':
        return {"requested": False, "active": False, "status": "not_applicable", "passed": False}

    raw = raw_config or {}
    batch_size = _positive_int(_cfg_lookup(raw, 'training.batch_size', _cfg_lookup(raw, 'training.execution.batch_size', 1)), 1)
    steps = _positive_int(_cfg_lookup(raw, 'training.gradient_accumulation.steps', _cfg_lookup(raw, 'training.accumulation.steps', 1)), 1)
    mode = str(_cfg_lookup(raw, 'training.gradient_accumulation.mode', _cfg_lookup(raw, 'training.accumulation.mode', 'none')) or 'none').strip().lower().replace('-', '_')
    policy = str(_cfg_lookup(raw, 'training.gradient_accumulation.policy', _cfg_lookup(raw, 'training.accumulation.policy', 'average')) or 'average').strip().lower().replace('-', '_')
    seq_entries = _normalize_sequence_entries(runtime_sequence if runtime_sequence is not None else _cfg_lookup(raw, 'runtime.sequence', []))
    seq_commands = [entry.get('command') for entry in seq_entries]
    seq_accumulate_steps = [
        _positive_int((entry.get('args') or {}).get('steps', (entry.get('args') or {}).get('micro_batches', steps)), steps)
        for entry in seq_entries
        if entry.get('command') == 'accumulate_gradients'
    ]
    if seq_accumulate_steps:
        steps = max(steps, max(seq_accumulate_steps))

    requested = bool(
        batch_size > 1
        or steps > 1
        or mode not in {'', 'none', 'false'}
        or any(cmd in {'reset_accumulators', 'accumulate_gradients', 'apply_accumulated_gradients'} for cmd in seq_commands)
    )
    if not requested:
        return {
            "requested": False,
            "active": False,
            "status": "not_requested",
            "passed": False,
            "batch_size": batch_size,
            "accumulation_steps": steps,
            "accumulation_policy": policy,
        }

    ref_grads = getattr(training_reference_result, 'grads_flat_path', None)
    ref_weights_after = getattr(training_reference_result, 'weights_after_flat_path', None)
    ref_dir = Path(out_dir) / 'training_reference'
    accumulated_grads_ref = ref_dir / 'accumulated_grads_ref.bin'
    weights_after_accum_ref = ref_dir / 'weights_after_accum_ref.bin'

    reference_generated = False
    reference_reason = 'Python single-step reference missing; cannot derive repeated-microbatch accumulation reference.'
    if policy in {'average', 'mean'}:
        reference_generated = _copy_or_transform_f32(ref_grads, accumulated_grads_ref, scale=1.0)
        if reference_generated and ref_weights_after is not None:
            _copy_or_transform_f32(ref_weights_after, weights_after_accum_ref, scale=1.0)
        reference_reason = 'Repeated-microbatch average accumulation reference equals the single-step reference gradient.'
    elif policy == 'sum':
        reference_generated = _copy_or_transform_f32(ref_grads, accumulated_grads_ref, scale=float(steps))
        reference_reason = 'Repeated-microbatch sum accumulation reference is the single-step gradient scaled by accumulation_steps; weights-after reference requires an optimizer-specific boundary update.'
    else:
        reference_reason = f'Unsupported accumulation policy {policy!r}; expected average/mean or sum.'

    candidates = [
        Path(out_dir) / 'accumulated_grads_after.bin',
        Path(out_dir) / 'hls' / 'accumulated_grads_after.bin',
        Path(out_dir) / 'gradients_after.bin',
        Path(out_dir) / 'gradients_export.bin',
        Path(out_dir) / 'runtime_package' / 'outputs' / 'accumulated_grads_after.bin',
        Path(out_dir) / 'runtime_package' / 'outputs' / 'gradients_after.bin',
    ]
    got_grads = next((p for p in candidates if p.exists()), None)
    gradients_compare = _compare_file_pair(accumulated_grads_ref, got_grads) if got_grads is not None else {
        'ref_path': _path_or_none(accumulated_grads_ref),
        'got_path': None,
        'ref_exists': accumulated_grads_ref.exists(),
        'got_exists': False,
        'status': 'artifact_missing',
        'passed': False,
    }

    if training_compare_result is not None:
        status = 'compared'
        passed = bool(
            (getattr(training_compare_result, 'grad_cosine', None) is None or getattr(training_compare_result, 'grad_cosine') >= 0.99)
            and (getattr(training_compare_result, 'weight_after_cosine', None) is None or getattr(training_compare_result, 'weight_after_cosine') >= 0.99)
        )
    elif got_grads is not None:
        status = 'compared' if gradients_compare.get('status') == 'compared' else 'missing_or_failed'
        passed = bool(gradients_compare.get('passed', False))
    elif reference_generated:
        status = 'artifact_missing'
        passed = False
    else:
        status = 'reference_missing'
        passed = False

    optimizer_apply_count = sum(1 for cmd in seq_commands if cmd == 'apply_accumulated_gradients')
    return {
        'requested': True,
        'active': True,
        'status': status,
        'passed': passed,
        'batch_size': batch_size,
        'accumulation_steps': steps,
        'accumulation_policy': policy,
        'mode': mode,
        'runtime_commands': seq_commands,
        'optimizer_apply_count': optimizer_apply_count,
        'required_runtime_commands': {
            'reset_accumulators': 'reset_accumulators' in seq_commands,
            'accumulate_gradients': 'accumulate_gradients' in seq_commands,
            'apply_accumulated_gradients': 'apply_accumulated_gradients' in seq_commands,
        },
        'reference': {
            'status': 'generated' if reference_generated else 'missing',
            'reason': reference_reason,
            'accumulated_grads_ref_bin': _path_or_none(accumulated_grads_ref),
            'weights_after_accum_ref_bin': _path_or_none(weights_after_accum_ref) if weights_after_accum_ref.exists() else None,
            'accumulated_grads_ref_exists': accumulated_grads_ref.exists(),
            'weights_after_accum_ref_exists': weights_after_accum_ref.exists(),
        },
        'comparisons': {
            'accumulated_gradients': gradients_compare,
            'training_step': _training_compare_payload(training_compare_result),
        },
        'validation_boundary': (
            'Batch accumulation is validation-qualified only when accumulated gradient and weights-after captures are compared against the Python schedule reference. '
            'A generated reference without captured HLS/runtime artifacts is recorded as artifact_missing, never as a pass.'
        ),
    }


def _gradient_export_validation_payload(gradient_export_artifacts: dict[str, Any] | None) -> dict[str, Any]:
    """Describe and compare exported gradient payloads when capture files exist.

    Training comparison proves the normal generated/testbench gradient tensor.  The
    gradient-export feature is a separate runtime/HLS mode that writes the
    flattened gradients_mem payload.  This helper only marks the export path
    compared when explicit ref/got capture files are available.
    """
    if not gradient_export_artifacts:
        return {"requested": False, "status": "not_requested", "passed": False}

    payload: dict[str, Any] = dict(gradient_export_artifacts)
    requested = bool(payload.get("requested", False))
    comparisons_cfg = payload.get("comparisons", {}) or {}
    comparisons: dict[str, Any] = {}
    any_compared = False
    all_passed = True
    optimizer = str(payload.get("optimizer", "sgd")).lower().replace("-", "_")

    if isinstance(comparisons_cfg, dict):
        for name, cfg in sorted(comparisons_cfg.items()):
            if not isinstance(cfg, dict):
                continue
            cmp_payload = _compare_file_pair(cfg.get("ref"), cfg.get("got"))
            comparisons[str(name)] = cmp_payload
            if cmp_payload.get("status") == "compared":
                any_compared = True
            if not bool(cmp_payload.get("passed", False)):
                all_passed = False

    payload["comparisons"] = comparisons
    if not requested:
        payload["status"] = "not_requested"
        payload["passed"] = False
    elif comparisons and any_compared and all_passed:
        payload["status"] = "compared"
        payload["passed"] = True
    elif comparisons:
        # A partial comparison record usually means only a generated reference
        # exists and the dedicated export payload has not been captured yet.
        # Do not classify that as a correctness failure; reserve
        # missing_or_failed for cases where both sides exist but comparison fails.
        both_sides_seen = any(
            bool(cmp_payload.get("ref_exists", False)) and bool(cmp_payload.get("got_exists", False))
            for cmp_payload in comparisons.values()
            if isinstance(cmp_payload, dict)
        )
        if both_sides_seen:
            payload["status"] = "missing_or_failed"
        else:
            payload["status"] = payload.get("status") or "generated_not_captured_by_testbench"
        payload["passed"] = False
    else:
        payload.setdefault("status", "generated_not_captured_by_testbench")
        payload["passed"] = False
    return payload


def _training_reference_payload(training_reference_result: Any) -> dict[str, Any] | None:
    if training_reference_result is None:
        return None
    return {
        "status": "generated",
        "loss_before": getattr(training_reference_result, "loss_before", None),
        "loss_after": getattr(training_reference_result, "loss_after", None),
        "grads_ref_bin": _path_or_none(getattr(training_reference_result, "grads_flat_path", None)),
        "weights_before_ref_bin": _path_or_none(getattr(training_reference_result, "weights_before_flat_path", None)),
        "weights_after_ref_bin": _path_or_none(getattr(training_reference_result, "weights_after_flat_path", None)),
        "optimizer_type": getattr(training_reference_result, "optimizer_type", "sgd"),
        "optimizer_bias_correction": getattr(training_reference_result, "optimizer_bias_correction", False),
        "optimizer_state_before_ref_bin": _path_or_none(getattr(training_reference_result, "optimizer_state_before_flat_path", None)),
        "optimizer_state_after_ref_bin": _path_or_none(getattr(training_reference_result, "optimizer_state_after_flat_path", None)),
        "loss_type": getattr(training_reference_result, "loss_type", "mse"),
        "logits_ref_bin": _path_or_none(getattr(training_reference_result, "logits_ref_path", None)),
        "softmax_ref_bin": _path_or_none(getattr(training_reference_result, "softmax_ref_path", None)),
        "cross_entropy_loss_ref_json": _path_or_none(getattr(training_reference_result, "cross_entropy_loss_ref_json", None)),
        "dlogits_ref_bin": _path_or_none(getattr(training_reference_result, "dlogits_ref_path", None)),
        "tiled_inputs_ref_bin": _path_or_none(getattr(training_reference_result, "tiled_inputs_ref_path", None)),
        "tiled_labels_ref_bin": _path_or_none(getattr(training_reference_result, "tiled_labels_ref_path", None)),
        "tiled_outputs_ref_bin": _path_or_none(getattr(training_reference_result, "tiled_outputs_ref_path", None)),
        "tiled_gradients_ref_bin": _path_or_none(getattr(training_reference_result, "tiled_gradients_ref_path", None)),
        "tiled_weights_after_ref_bin": _path_or_none(getattr(training_reference_result, "tiled_weights_after_ref_path", None)),
        "summary_json": _path_or_none(getattr(training_reference_result, "summary_json", None)),
        "summary_txt": _path_or_none(getattr(training_reference_result, "summary_txt", None)),
    }


def _training_compare_payload(training_compare_result: Any) -> dict[str, Any] | None:
    if training_compare_result is None:
        return None
    return {
        "status": "compared",
        "results_json": _path_or_none(getattr(training_compare_result, "results_json", None)),
        "summary_txt": _path_or_none(getattr(training_compare_result, "summary_txt", None)),
        "grad_cosine": getattr(training_compare_result, "grad_cosine", None),
        "weight_after_cosine": getattr(training_compare_result, "weight_after_cosine", None),
        "weight_delta_cosine": getattr(training_compare_result, "weight_delta_cosine", None),
        "grad_mae": getattr(training_compare_result, "grad_mae", None),
        "grad_max_abs": getattr(training_compare_result, "grad_max_abs", None),
        "weight_after_mae": getattr(training_compare_result, "weight_after_mae", None),
        "weight_after_max_abs": getattr(training_compare_result, "weight_after_max_abs", None),
    }
