"""Numeric validation report helpers.

This module does not pretend that a design has been numerically validated.
It records exactly which reference/testbench/HLS artifacts exist and whether
there is enough evidence to claim numeric correctness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import struct
import math

from fpgai.analysis.hls_estimate_compare import parse_hls_csynth_report


def _path_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(Path(value))
    except TypeError:
        return str(value)


def _exists(value: Any) -> bool:
    if value is None:
        return False
    try:
        return Path(value).exists()
    except TypeError:
        return False


def _read_f32_file(path: Any) -> list[float] | None:
    if path is None:
        return None
    try:
        data = Path(path).read_bytes()
    except Exception:
        return None
    if len(data) % 4 != 0:
        return None
    if not data:
        return []
    try:
        return list(struct.unpack('<' + 'f' * (len(data) // 4), data))
    except Exception:
        return None


def _compare_vectors(ref: list[float], got: list[float]) -> dict[str, Any]:
    n = min(len(ref), len(got))
    if len(got) == 0 and len(ref) > 0:
        status = 'empty_generated_output'
    elif len(ref) != len(got):
        status = 'shape_mismatch'
    else:
        status = 'compared'
    if n == 0:
        return {
            'status': status,
            'num_ref': len(ref),
            'num_got': len(got),
            'num_compared': 0,
            'mse': None,
            'mae': None,
            'max_abs_error': None,
            'cosine_similarity': None,
        }
    diffs = [float(got[i] - ref[i]) for i in range(n)]
    abs_diffs = [abs(x) for x in diffs]
    mse = sum(x * x for x in diffs) / n
    mae = sum(abs_diffs) / n
    max_abs = max(abs_diffs)
    dot = sum(float(ref[i]) * float(got[i]) for i in range(n))
    nr = math.sqrt(sum(float(ref[i]) * float(ref[i]) for i in range(n)))
    ng = math.sqrt(sum(float(got[i]) * float(got[i]) for i in range(n)))
    cosine = None if nr == 0.0 or ng == 0.0 else dot / (nr * ng)
    return {
        'status': status,
        'num_ref': len(ref),
        'num_got': len(got),
        'num_compared': n,
        'mse': mse,
        'mae': mae,
        'max_abs_error': max_abs,
        'cosine_similarity': cosine,
    }


def _compare_file_pair(
    ref_path: Any,
    got_path: Any,
    *,
    max_abs_error_limit: float = 1e-3,
    mean_abs_error_limit: float | None = None,
    rmse_limit: float | None = None,
    min_cosine_similarity: float | None = None,
) -> dict[str, Any]:
    ref = _read_f32_file(ref_path)
    got = _read_f32_file(got_path)
    payload = {
        'ref_path': _path_or_none(ref_path),
        'got_path': _path_or_none(got_path),
        'ref_exists': _exists(ref_path),
        'got_exists': _exists(got_path),
        'limits': {
            'max_abs_error_limit': max_abs_error_limit,
            'mean_abs_error_limit': mean_abs_error_limit,
            'rmse_limit': rmse_limit,
            'min_cosine_similarity': min_cosine_similarity,
        },
    }
    if ref is None or got is None:
        payload.update({'status': 'missing_or_unreadable', 'passed': False})
        return payload
    metrics = _compare_vectors(ref, got)
    max_abs = metrics.get('max_abs_error')
    mae = metrics.get('mae')
    mse = metrics.get('mse')
    rmse = math.sqrt(float(mse)) if mse is not None and float(mse) >= 0.0 else None
    cosine = metrics.get('cosine_similarity')

    checks = []
    checks.append({
        'name': 'max_abs_error',
        'value': max_abs,
        'limit': max_abs_error_limit,
        'passed': max_abs is not None and max_abs <= max_abs_error_limit,
    })
    if mean_abs_error_limit is not None:
        checks.append({
            'name': 'mae',
            'value': mae,
            'limit': mean_abs_error_limit,
            'passed': mae is not None and mae <= mean_abs_error_limit,
        })
    if rmse_limit is not None:
        checks.append({
            'name': 'rmse',
            'value': rmse,
            'limit': rmse_limit,
            'passed': rmse is not None and rmse <= rmse_limit,
        })
    if min_cosine_similarity is not None:
        checks.append({
            'name': 'cosine_similarity',
            'value': cosine,
            'limit': min_cosine_similarity,
            'passed': cosine is not None and cosine >= min_cosine_similarity,
        })

    passed = metrics['status'] == 'compared' and bool(checks) and all(bool(c.get('passed')) for c in checks)
    payload.update(metrics)
    payload['rmse'] = rmse
    payload['checks'] = checks
    payload['passed'] = bool(passed)
    return payload




def _precision_activation_lsb(raw_config: dict[str, Any] | None) -> float:
    raw = raw_config or {}

    def lookup(path: str, default: Any = None) -> Any:
        node: Any = raw
        for part in path.split('.'):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    activation = lookup('numerics.defaults.activation', {}) or lookup('precision.defaults.activation', {}) or {}
    if not isinstance(activation, dict):
        return 2.0 ** -10
    try:
        total_bits = int(activation.get('total_bits', 16))
        int_bits = int(activation.get('int_bits', 6))
    except Exception:
        return 2.0 ** -10
    frac_bits = max(0, total_bits - int_bits)
    return float(2.0 ** (-frac_bits))


def _precision_aware_inference_limits(raw_config: dict[str, Any] | None) -> dict[str, float]:
    raw = raw_config or {}
    lsb = _precision_activation_lsb(raw)

    def lookup(path: str, default: Any = None) -> Any:
        node: Any = raw
        for part in path.split('.'):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    precision_aware = bool(lookup('benchmark.compare.precision_aware', True))
    base_max = float(lookup('benchmark.compare.max_abs_error', 0.08))
    base_mean = float(lookup('benchmark.compare.mean_abs_error', 0.03))
    base_rmse = float(lookup('benchmark.compare.rmse', 0.04))
    base_cos = float(lookup('benchmark.compare.min_cosine_similarity', 0.95))
    if not precision_aware:
        return {
            'max_abs_error_limit': base_max,
            'mean_abs_error_limit': base_mean,
            'rmse_limit': base_rmse,
            'min_cosine_similarity': base_cos,
        }
    relaxed_cos = 0.93 if lsb >= 0.03125 else 0.95
    return {
        'max_abs_error_limit': float(max(base_max, 4.0 * lsb)),
        'mean_abs_error_limit': float(max(base_mean, 1.5 * lsb)),
        'rmse_limit': float(max(base_rmse, 2.0 * lsb)),
        'min_cosine_similarity': float(min(base_cos, relaxed_cos)),
    }


def _normalize_sequence_entries(runtime_sequence: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if isinstance(runtime_sequence, dict):
        raw_seq = runtime_sequence.get("sequence", [])
    else:
        raw_seq = runtime_sequence or []
    if not isinstance(raw_seq, list):
        return entries
    for item in raw_seq:
        if isinstance(item, str):
            entries.append({"command": item.strip().lower().replace("-", "_"), "args": {}})
        elif isinstance(item, dict):
            if "command" in item:
                entries.append({
                    "command": str(item.get("command", "")).strip().lower().replace("-", "_"),
                    "args": dict(item.get("args", {}) or {}) if isinstance(item.get("args", {}), dict) else {},
                })
            elif len(item) == 1:
                key, value = next(iter(item.items()))
                entries.append({
                    "command": str(key).strip().lower().replace("-", "_"),
                    "args": dict(value or {}) if isinstance(value, dict) else {},
                })
    return entries


def _cfg_lookup(raw_config: dict[str, Any] | None, path: str, default: Any = None) -> Any:
    node: Any = raw_config or {}
    for part in path.split('.'):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return int(default)


def _write_f32_file(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack('<' + 'f' * len(values), *[float(v) for v in values]) if values else b'')


def _copy_or_transform_f32(src: Any, dst: Path, *, scale: float = 1.0) -> bool:
    values = _read_f32_file(src)
    if values is None:
        return False
    _write_f32_file(dst, [float(v) * float(scale) for v in values])
    return True


def _read_json_file(path: Any) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


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
            "truth_boundary": "Cross-entropy validation is only requested when training.loss.type=cross_entropy.",
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
        "truth_boundary": (
            "Cross-entropy is paper-safe only when generated/HLS/runtime loss-gradient evidence is compared with the stable-softmax Python reference. "
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
            "truth_boundary": "Training tiled-I/O validation is only requested when data_movement inputs/labels/outputs use policy=tiled.",
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
        "truth_boundary": (
            "Training tiled I/O is paper-safe only when tiled input/label movement feeds the compute path and captured tiled outputs, gradients, and weights-after are compared against Python reference artifacts. "
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
        'truth_boundary': (
            'Batch accumulation is paper-safe only when accumulated gradient and weights-after captures are compared against the Python schedule reference. '
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


def _optimizer_state_lsb(raw_config: dict[str, Any] | None) -> float:
    """Resolve the optimizer-state fixed-point LSB from the numeric contract."""
    raw = raw_config or {}
    spec = _cfg_lookup(raw, "numerics.training.optimizer_state", None)
    if not isinstance(spec, dict):
        spec = _cfg_lookup(raw, "numerics.defaults.accum", None)
    if not isinstance(spec, dict):
        spec = _cfg_lookup(raw, "precision.defaults.accum", None)
    if not isinstance(spec, dict):
        return float(2.0 ** -16)
    try:
        total_bits = int(spec.get("total_bits", spec.get("bits", 24)))
        int_bits = int(spec.get("int_bits", spec.get("integer_bits", 8)))
    except Exception:
        return float(2.0 ** -16)
    return float(2.0 ** (-max(0, total_bits - int_bits)))


def _optimizer_state_reference_metadata(ref_path: Any) -> dict[str, Any]:
    """Load optimizer update count and canonical parameter layout beside a reference artifact."""
    try:
        ref = Path(ref_path)
    except Exception:
        return {}
    root = ref.parent
    payload: dict[str, Any] = {}
    candidate_roots = [root, root.parent]
    for candidate_root in candidate_roots:
        summary_path = candidate_root / "training_hardware_domain_reference.json"
        try:
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                payload["optimizer_updates"] = int(summary.get("optimizer_updates", 0) or 0)
                payload["optimizer_type"] = str(summary.get("optimizer_type", "sgd"))
                break
        except Exception:
            pass
    for candidate_root in candidate_roots:
        layout_path = candidate_root / "per_sample_trace" / "parameter_layer_map.json"
        try:
            if layout_path.exists():
                layout_payload = json.loads(layout_path.read_text(encoding="utf-8"))
                entries = layout_payload.get("entries", [])
                if isinstance(entries, list):
                    payload["parameter_layout"] = [entry for entry in entries if isinstance(entry, dict)]
                    break
        except Exception:
            pass
    return payload


def _optimizer_state_segments(
    *,
    optimizer: str,
    parameter_layout: list[dict[str, Any]],
    total_words: int,
) -> list[dict[str, Any]]:
    """Build canonical packed optimizer-state segments from the existing parameter layer map."""
    parameter_words = sum(int(entry.get("count", 0) or 0) for entry in parameter_layout)
    if parameter_words <= 0:
        return []
    segments: list[dict[str, Any]] = []
    groups = (("velocity", 0),) if optimizer == "momentum" else (("m", 0), ("v", parameter_words))
    for group, base in groups:
        for entry in parameter_layout:
            count = int(entry.get("count", 0) or 0)
            offset = base + int(entry.get("offset", 0) or 0)
            layer = str(entry.get("layer", "parameter"))
            role = str(entry.get("role", "state"))
            role_prefix = {"weight": "W", "bias": "B", "gamma": "gamma", "beta": "beta"}.get(role, role)
            segments.append({
                "name": f"{group}_{role_prefix}_{layer}",
                "offset": offset,
                "count": count,
                "layer": layer,
                "role": role,
            })
    if optimizer == "adam" and total_words == (2 * parameter_words + 1):
        segments.append({"name": "step", "offset": 2 * parameter_words, "count": 1, "layer": None, "role": "optimizer_step"})
    return segments


def _compare_optimizer_state_pair(
    ref_path: Any,
    got_path: Any,
    *,
    lsb: float,
    optimizer: str = "sgd",
) -> dict[str, Any]:
    """Compare canonical optimizer state with strict and propagated fixed-point classifications."""
    ref = _read_f32_file(ref_path)
    got = _read_f32_file(got_path)
    metadata = _optimizer_state_reference_metadata(ref_path)
    optimizer_updates = int(metadata.get("optimizer_updates", 0) or 0)
    allowed_lsb = 1 if optimizer_updates <= 1 else 2
    payload: dict[str, Any] = {
        "ref_path": _path_or_none(ref_path),
        "got_path": _path_or_none(got_path),
        "ref_exists": _exists(ref_path),
        "got_exists": _exists(got_path),
        "optimizer_updates": optimizer_updates or None,
        "numeric_tolerance": {
            "kind": "optimizer_state_lsb",
            "lsb": float(lsb),
            "single_update_allowed_lsb": 1,
            "multi_update_allowed_lsb": 2,
            "allowed_lsb": allowed_lsb,
        },
    }
    if ref is None or got is None:
        payload.update({"status": "artifact_missing", "passed": False})
        return payload
    metrics = _compare_vectors(ref, got)
    n = min(len(ref), len(got))
    diffs = [float(got[i] - ref[i]) for i in range(n)]
    abs_diffs = [abs(value) for value in diffs]
    ref_norm = math.sqrt(sum(float(value) * float(value) for value in ref))
    diff_norm = math.sqrt(sum(value * value for value in diffs))
    exact_words = sum(1 for value in diffs if value == 0.0)
    eps = 1.0e-15
    within_one_lsb = sum(1 for value in abs_diffs if value <= float(lsb) + eps)
    within_two_lsb = sum(1 for value in abs_diffs if value <= (2.0 * float(lsb)) + eps)
    above_two_lsb = sum(1 for value in abs_diffs if value > (2.0 * float(lsb)) + eps)
    same_shape = len(ref) == len(got)
    all_within_one = bool(same_shape and within_one_lsb == len(ref))
    all_within_two = bool(same_shape and within_two_lsb == len(ref))
    passed = all_within_one if allowed_lsb == 1 else all_within_two
    classification = (
        "bit_exact" if same_shape and exact_words == len(ref)
        else "quantization_aligned" if all_within_one
        else "propagated_quantization_aligned" if optimizer_updates > 1 and all_within_two
        else "failed"
    )
    maximum_lsb_distance = max((int(round(value / float(lsb))) for value in abs_diffs), default=0) if lsb > 0 else None

    segment_payloads: list[dict[str, Any]] = []
    for segment in _optimizer_state_segments(
        optimizer=optimizer,
        parameter_layout=metadata.get("parameter_layout", []),
        total_words=len(ref),
    ):
        start = int(segment["offset"])
        end = start + int(segment["count"])
        segment_diffs = abs_diffs[start:end]
        segment_payloads.append({
            **segment,
            "exact_words": sum(1 for value in segment_diffs if value == 0.0),
            "within_one_lsb": sum(1 for value in segment_diffs if value <= float(lsb) + eps),
            "within_two_lsb": sum(1 for value in segment_diffs if value <= (2.0 * float(lsb)) + eps),
            "above_two_lsb": sum(1 for value in segment_diffs if value > (2.0 * float(lsb)) + eps),
            "max_abs_error": max(segment_diffs, default=0.0),
            "maximum_lsb_distance": max((int(round(value / float(lsb))) for value in segment_diffs), default=0) if lsb > 0 else None,
        })

    payload.update(metrics)
    payload.update({
        "reference_words": len(ref),
        "hls_words": len(got),
        "reference_nonzero": sum(1 for value in ref if value != 0.0),
        "hls_nonzero": sum(1 for value in got if value != 0.0),
        "reference_norm": ref_norm,
        "hls_norm": math.sqrt(sum(float(value) * float(value) for value in got)),
        "relative_l2": (diff_norm / ref_norm) if ref_norm > 0.0 else diff_norm,
        "exact_words": exact_words,
        "within_one_lsb": within_one_lsb,
        "within_two_lsb": within_two_lsb,
        "above_two_lsb": above_two_lsb,
        "all_words_within_one_lsb": all_within_one,
        "all_words_within_two_lsb": all_within_two,
        "maximum_lsb_distance": maximum_lsb_distance,
        "classification": classification,
        "segments": segment_payloads,
        "status": "compared" if same_shape else "shape_mismatch",
        "passed": passed,
    })
    return payload

def _optimizer_state_validation_payload(optimizer_state_artifacts: dict[str, Any] | None, *, raw_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Describe and compare persistent optimizer-state tensors when artifacts exist.

    Momentum and Adam correctness is not only weights-after correctness: their
    persistent state must also be checked.  This helper records requested state
    tensors and compares explicit ref/got float32 files when the testbench or
    runtime path provides them.  Missing files are reported as missing evidence,
    never as a pass.
    """
    if not optimizer_state_artifacts:
        return {"requested": False, "status": "not_requested"}

    payload: dict[str, Any] = dict(optimizer_state_artifacts)
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
            cmp_payload = _compare_optimizer_state_pair(
                cfg.get("ref"), cfg.get("got"), lsb=_optimizer_state_lsb(raw_config), optimizer=optimizer
            )
            comparisons[str(name)] = cmp_payload
            if cmp_payload.get("status") == "compared":
                any_compared = True
            if not bool(cmp_payload.get("passed", False)):
                all_passed = False

    payload["comparisons"] = comparisons
    if optimizer not in {"momentum", "adam"}:
        payload["status"] = "not_applicable"
        payload["implementation_status"] = "not_applicable"
        payload["passed"] = False
    elif not requested:
        payload["status"] = "not_validated"
        payload["implementation_status"] = "not_validated"
        payload["passed"] = False
    elif comparisons and any_compared and all_passed:
        legacy_comparison_only = "layout" not in payload and not any(
            key in payload for key in ("layout_version", "reference_domain", "claim_scope")
        )
        payload["status"] = "compared" if legacy_comparison_only else "implemented"
        payload["implementation_status"] = "implemented"
        payload["passed"] = True
    elif comparisons:
        both_sides_seen = any(
            bool(cmp_payload.get("ref_exists", False)) and bool(cmp_payload.get("got_exists", False))
            for cmp_payload in comparisons.values()
            if isinstance(cmp_payload, dict)
        )
        any_ref_missing_got = any(
            bool(cmp_payload.get("ref_exists", False)) and not bool(cmp_payload.get("got_exists", False))
            for cmp_payload in comparisons.values()
            if isinstance(cmp_payload, dict)
        )
        original_status = str(payload.get("status") or "")
        if both_sides_seen:
            payload["status"] = "failed"
            payload["implementation_status"] = "failed"
            payload["evidence_status"] = "failed"
        elif any_ref_missing_got:
            # Keep the generated-export capability status for legacy contract
            # reports, but expose the stricter numeric proof status separately.
            # Non-export optimizer-state validation remains artifact_missing.
            payload["evidence_status"] = "artifact_missing"
            if original_status == "generated_export_capture_supported":
                payload["status"] = original_status
            else:
                payload["status"] = "artifact_missing"
                payload["implementation_status"] = "not_validated"
        else:
            payload["status"] = "not_validated"
            payload["implementation_status"] = "not_validated"
            payload.setdefault("evidence_status", "artifact_missing")
        payload["passed"] = False
    else:
        payload["status"] = "not_validated" if requested else "not_applicable"
        payload["implementation_status"] = payload["status"]
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



def _resolve_data_path(out_dir: str | Path, value: Any) -> Path | None:
    if value in (None, "", [], {}):
        return None
    try:
        path = Path(str(value))
    except Exception:
        return None
    if path.is_absolute():
        return path
    candidates = [Path(out_dir) / path, Path.cwd() / path]
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return candidates[0]


def _read_float_array(path: Any) -> list[float] | None:
    if path is None:
        return None
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        try:
            import numpy as np  # type: ignore
            return [float(x) for x in np.asarray(np.load(p), dtype=float).reshape(-1).tolist()]
        except Exception:
            return None
    if suffix in {".json", ".jsn"}:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(data, dict):
            for key in ("values", "targets", "labels", "y", "data"):
                if key in data:
                    data = data[key]
                    break
        if isinstance(data, list):
            try:
                return [float(x) for x in data]
            except Exception:
                return None
        return None
    if suffix in {".txt", ".csv", ".tsv"}:
        try:
            raw = p.read_text(encoding="utf-8").replace(",", " ").split()
            return [float(x) for x in raw]
        except Exception:
            return None
    return _read_f32_file(path)


def _read_int_array(path: Any) -> list[int] | None:
    floats = _read_float_array(path)
    if floats is not None:
        try:
            return [int(round(float(x))) for x in floats]
        except Exception:
            return None
    return None


def _reshape_samples(values: list[float], sample_count: int) -> list[list[float]] | None:
    if sample_count <= 0:
        return None
    if len(values) % sample_count != 0:
        return None
    width = len(values) // sample_count
    if width <= 0:
        return None
    return [values[i * width:(i + 1) * width] for i in range(sample_count)]


def _argmax(row: list[float]) -> int:
    if not row:
        return -1
    best = 0
    best_v = float(row[0])
    for idx, value in enumerate(row[1:], start=1):
        if float(value) > best_v:
            best = idx
            best_v = float(value)
    return best


def _topk(row: list[float], k: int) -> set[int]:
    return {idx for idx, _ in sorted(enumerate(row), key=lambda item: float(item[1]), reverse=True)[:max(1, k)]}


def _safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _r2_score(targets: list[float], preds: list[float]) -> float | None:
    n = min(len(targets), len(preds))
    if n == 0:
        return None
    y = [float(targets[i]) for i in range(n)]
    p = [float(preds[i]) for i in range(n)]
    mean_y = sum(y) / n
    ss_tot = sum((v - mean_y) ** 2 for v in y)
    if ss_tot == 0.0:
        return None
    ss_res = sum((p[i] - y[i]) ** 2 for i in range(n))
    return 1.0 - ss_res / ss_tot


def _regression_metrics(targets: list[float], preds: list[float]) -> dict[str, Any]:
    n = min(len(targets), len(preds))
    if n == 0:
        return {"sample_count": 0, "mae": None, "rmse": None, "max_abs_error": None, "r2": None}
    diffs = [float(preds[i]) - float(targets[i]) for i in range(n)]
    abs_diffs = [abs(x) for x in diffs]
    mse = sum(x * x for x in diffs) / n
    return {
        "sample_count": n,
        "mae": sum(abs_diffs) / n,
        "rmse": math.sqrt(mse),
        "max_abs_error": max(abs_diffs),
        "r2": _r2_score(targets[:n], preds[:n]),
    }


def _classification_decision(payload: dict[str, Any], thresholds: Mapping[str, Any]) -> tuple[str, str]:
    accuracy_drop = payload.get("accuracy_drop_pct")
    agreement = payload.get("prediction_agreement_vs_reference")
    max_drop = float(thresholds.get("max_accuracy_drop_pct", 3.0) or 3.0)
    low_drop = float(thresholds.get("recommended_accuracy_drop_pct", 1.0) or 1.0)
    aggressive_drop = float(thresholds.get("aggressive_accuracy_drop_pct", 10.0) or 10.0)
    min_agree = float(thresholds.get("min_prediction_agreement", 0.95) or 0.95)
    aggressive_agree = float(thresholds.get("aggressive_min_prediction_agreement", 0.75) or 0.75)

    if accuracy_drop is not None:
        drop = abs(float(accuracy_drop))
        if drop <= low_drop and (agreement is None or float(agreement) >= 0.99):
            return "recommended_quality", f"accuracy drop {drop:.3g}% is within the recommended threshold"
        if drop <= max_drop and (agreement is None or float(agreement) >= min_agree):
            return "acceptable_tradeoff", f"accuracy drop {drop:.3g}% is within the configured decision threshold"
        if drop <= aggressive_drop and (agreement is None or float(agreement) >= aggressive_agree):
            return "aggressive_compression", f"accuracy drop {drop:.3g}% may be acceptable for resource-constrained deployments"
        return "not_recommended_for_quality", f"accuracy drop {drop:.3g}% exceeds the configured decision threshold"

    if agreement is not None:
        agree = float(agreement)
        if agree >= 0.99:
            return "recommended_quality", f"prediction agreement {agree:.3g} is near-identical to the reference"
        if agree >= min_agree:
            return "acceptable_tradeoff", f"prediction agreement {agree:.3g} is within the configured threshold"
        if agree >= aggressive_agree:
            return "aggressive_compression", f"prediction agreement {agree:.3g} may be acceptable for aggressive compression"
        return "not_recommended_for_quality", f"prediction agreement {agree:.3g} is below the configured threshold"
    return "pending_dataset_labels", "classification labels were not provided and reference prediction agreement could not be computed"


def _regression_decision(payload: dict[str, Any], thresholds: Mapping[str, Any]) -> tuple[str, str]:
    mae_inc = payload.get("mae_increase")
    rmse_inc = payload.get("rmse_increase")
    ref_mae = payload.get("reference_output_mae")
    max_mae_inc = float(thresholds.get("max_mae_increase", 0.01) or 0.01)
    max_rmse_inc = float(thresholds.get("max_rmse_increase", 0.02) or 0.02)
    if mae_inc is not None or rmse_inc is not None:
        mi = abs(float(mae_inc or 0.0))
        ri = abs(float(rmse_inc or 0.0))
        if mi <= max_mae_inc * 0.25 and ri <= max_rmse_inc * 0.25:
            return "recommended_quality", "regression error increase is very small versus the reference"
        if mi <= max_mae_inc and ri <= max_rmse_inc:
            return "acceptable_tradeoff", "regression error increase is within the configured decision threshold"
        if mi <= max_mae_inc * 4.0 and ri <= max_rmse_inc * 4.0:
            return "aggressive_compression", "regression error increase is above the preferred threshold but may be acceptable for resource-constrained deployments"
        return "not_recommended_for_quality", "regression error increase exceeds the configured decision threshold"
    if ref_mae is not None:
        mae = abs(float(ref_mae))
        if mae <= max_mae_inc * 0.25:
            return "recommended_quality", "output MAE versus reference is very small"
        if mae <= max_mae_inc:
            return "acceptable_tradeoff", "output MAE versus reference is within the configured threshold"
        if mae <= max_mae_inc * 4.0:
            return "aggressive_compression", "output MAE versus reference is above the preferred threshold but may be acceptable"
        return "not_recommended_for_quality", "output MAE versus reference exceeds the configured threshold"
    return "pending_targets", "regression targets were not provided and output-error metrics were unavailable"




def _dataset_execution_validation(out_dir: str | Path, *, expected_sample_count: int | None = None, output_values_per_sample: int | None = None) -> dict[str, Any]:
    path = Path(out_dir) / "reports" / "hls_dataset_execution.json"
    base = {
        "artifact_path": str(path),
        "artifact_exists": path.exists(),
        "status": "not_requested",
        "passed": False,
        "checks": [],
    }
    if expected_sample_count is None or expected_sample_count <= 1:
        return base
    if not path.exists():
        # Compatibility path for reference-only unit tests and non-CSim callers.
        # When CSim emits a record, it is validated strictly below.
        return base
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        base.update({"status": "unreadable", "reason": str(exc)})
        return base
    requested = int(record.get("sample_count_requested") or 0)
    executed = int(record.get("sample_count_executed") or 0)
    invocations = int(record.get("inference_invocation_count") or 0)
    generated_words = int(record.get("generated_output_words") or 0)
    per_sample = int(record.get("output_values_per_sample") or output_values_per_sample or 0)
    expected_words = int(expected_sample_count) * int(per_sample)
    checks = [
        {"name": "requested_sample_count", "expected": expected_sample_count, "actual": requested, "passed": requested == expected_sample_count},
        {"name": "executed_sample_count", "expected": expected_sample_count, "actual": executed, "passed": executed == expected_sample_count},
        {"name": "inference_invocation_count", "expected": expected_sample_count, "actual": invocations, "passed": invocations == expected_sample_count},
        {"name": "generated_output_words", "expected": expected_words, "actual": generated_words, "passed": expected_words > 0 and generated_words == expected_words},
    ]
    passed = all(bool(item["passed"]) for item in checks)
    base.update({
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks": checks,
        "record": record,
    })
    return base


def _classification_diagnostics(labels: list[int], predictions: list[int], class_count: int) -> dict[str, Any]:
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    support = [0 for _ in range(class_count)]
    correct = [0 for _ in range(class_count)]
    for target, predicted in zip(labels, predictions):
        if 0 <= target < class_count:
            support[target] += 1
            if 0 <= predicted < class_count:
                matrix[target][predicted] += 1
            if predicted == target:
                correct[target] += 1
    per_class = []
    for class_id in range(class_count):
        per_class.append({
            "class_id": class_id,
            "support": support[class_id],
            "correct": correct[class_id],
            "accuracy": (correct[class_id] / support[class_id]) if support[class_id] else None,
        })
    return {"confusion_matrix": matrix, "per_class_accuracy": per_class}
def _task_quality_payload(
    out_dir: str | Path,
    *,
    raw_config: dict[str, Any] | None,
    output_compare: dict[str, Any] | None,
    inference_reference_artifacts: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = raw_config or {}
    validation = _cfg_lookup(raw, "validation", {})
    if not isinstance(validation, dict):
        validation = {}
    task = str(validation.get("task") or validation.get("type") or "auto").strip().lower().replace("-", "_")
    thresholds = validation.get("decision_thresholds", {}) if isinstance(validation.get("decision_thresholds", {}), dict) else {}
    ref_path = (inference_reference_artifacts or {}).get("outputs_ref")
    hw_path = (inference_reference_artifacts or {}).get("outputs_hw")
    ref_values = _read_float_array(ref_path)
    hw_values = _read_float_array(hw_path)
    base: dict[str, Any] = {
        "schema_version": 1,
        "status": "not_run",
        "task": task,
        "decision_status": "pending_numeric_artifacts",
        "decision_reason": "reference and generated output artifacts were not available",
        "reference_outputs_path": _path_or_none(ref_path),
        "generated_outputs_path": _path_or_none(hw_path),
        "reference_outputs_exist": _exists(ref_path),
        "generated_outputs_exist": _exists(hw_path),
        "dataset_source": ((validation.get("dataset") or {}).get("source") if isinstance(validation.get("dataset"), dict) else validation.get("dataset")) or validation.get("dataset_name") or validation.get("source") or "not_provided",
    }
    if ref_values is None or hw_values is None:
        return base
    if task == "auto":
        # Without labels or targets, default to output-agreement reporting. If labels
        # exist, classification takes priority; if targets exist, regression does.
        if validation.get("labels") or validation.get("label_path"):
            task = "classification"
        elif validation.get("targets") or validation.get("target_path"):
            task = "regression"
        else:
            task = "classification" if len(ref_values) > 1 else "regression"
    base["task"] = task

    if task in {"classification", "multiclass", "categorical"}:
        labels_path = _resolve_data_path(out_dir, validation.get("labels") or validation.get("label_path") or validation.get("y") or (inference_reference_artifacts or {}).get("labels_path"))
        labels = _read_int_array(labels_path) if labels_path is not None else None
        if labels:
            sample_count = len(labels)
            ref_rows = _reshape_samples(ref_values, sample_count)
            hw_rows = _reshape_samples(hw_values, sample_count)
        else:
            sample_count = 1
            ref_rows = [ref_values]
            hw_rows = [hw_values]
        if ref_rows is None or hw_rows is None or len(ref_rows) != len(hw_rows):
            base.update({
                "status": "shape_mismatch",
                "decision_status": "not_recommended_for_quality",
                "decision_reason": "classification output size could not be aligned with labels/samples",
                "labels_path": _path_or_none(labels_path),
                "labels_status": "shape_mismatch" if labels_path is not None else "not_provided",
            })
            return base
        ref_top1 = [_argmax(row) for row in ref_rows]
        hw_top1 = [_argmax(row) for row in hw_rows]
        agreement_count = sum(1 for a, b in zip(ref_top1, hw_top1) if a == b)
        agreement = agreement_count / len(ref_top1) if ref_top1 else None
        class_changes = len(ref_top1) - agreement_count
        conf_deltas = [float(max(hw_rows[i]) - max(ref_rows[i])) for i in range(len(ref_rows)) if ref_rows[i] and hw_rows[i]]
        execution_validation = _dataset_execution_validation(
            out_dir,
            expected_sample_count=len(ref_top1),
            output_values_per_sample=(len(ref_rows[0]) if ref_rows else None),
        )
        payload: dict[str, Any] = {
            **base,
            "status": "compared",
            "execution_validation": execution_validation,
            "labels_path": _path_or_none(labels_path),
            "labels_status": "provided" if labels else "not_provided",
            "sample_count": len(ref_top1),
            "class_count": len(ref_rows[0]) if ref_rows else 0,
            "reference_top1": ref_top1[:16],
            "generated_top1": hw_top1[:16],
            "reference_top1_first": ref_top1[0] if ref_top1 else None,
            "generated_top1_first": hw_top1[0] if hw_top1 else None,
            "prediction_agreement_vs_reference": agreement,
            "class_change_count": class_changes,
            "confidence_delta_mean": _safe_mean(conf_deltas),
            "confidence_delta_max": max([abs(x) for x in conf_deltas]) if conf_deltas else None,
            "target_top1_accuracy": None,
            "generated_top1_accuracy": None,
            "reference_top1_accuracy": None,
            "top1_accuracy_drop_pct": None,
            "generated_top5_accuracy": None,
            "reference_top5_accuracy": None,
            "top5_accuracy_drop_pct": None,
        }
        if labels:
            ref_correct = sum(1 for pred, y in zip(ref_top1, labels) if pred == y)
            hw_correct = sum(1 for pred, y in zip(hw_top1, labels) if pred == y)
            ref_acc = ref_correct / len(labels)
            hw_acc = hw_correct / len(labels)
            k = min(5, len(ref_rows[0]) if ref_rows else 1)
            ref_top5_correct = sum(1 for row, y in zip(ref_rows, labels) if y in _topk(row, k))
            hw_top5_correct = sum(1 for row, y in zip(hw_rows, labels) if y in _topk(row, k))
            ref_diagnostics = _classification_diagnostics(labels, ref_top1, len(ref_rows[0]) if ref_rows else 0)
            hw_diagnostics = _classification_diagnostics(labels, hw_top1, len(hw_rows[0]) if hw_rows else 0)
            diagnostics_dir = Path(out_dir) / "reports"
            confusion_path = diagnostics_dir / "classification_confusion_matrix.json"
            per_class_path = diagnostics_dir / "classification_per_class_accuracy.json"
            confusion_path.write_text(json.dumps({
                "schema_version": 1,
                "artifact_kind": "classification_confusion_matrix",
                "labels": list(range(len(ref_rows[0]) if ref_rows else 0)),
                "reference": ref_diagnostics["confusion_matrix"],
                "generated": hw_diagnostics["confusion_matrix"],
            }, indent=2), encoding="utf-8")
            per_class_path.write_text(json.dumps({
                "schema_version": 1,
                "artifact_kind": "classification_per_class_accuracy",
                "reference": ref_diagnostics["per_class_accuracy"],
                "generated": hw_diagnostics["per_class_accuracy"],
            }, indent=2), encoding="utf-8")
            payload.update({
                "target_top1_accuracy": hw_acc,
                "generated_top1_accuracy": hw_acc,
                "reference_top1_accuracy": ref_acc,
                "top1_accuracy_drop_pct": (hw_acc - ref_acc) * 100.0,
                "generated_top5_accuracy": hw_top5_correct / len(labels),
                "reference_top5_accuracy": ref_top5_correct / len(labels),
                "top5_accuracy_drop_pct": ((hw_top5_correct - ref_top5_correct) / len(labels)) * 100.0,
                "confusion_matrix_path": str(confusion_path),
                "per_class_accuracy_path": str(per_class_path),
                "reference_per_class_accuracy": ref_diagnostics["per_class_accuracy"],
                "generated_per_class_accuracy": hw_diagnostics["per_class_accuracy"],
            })
        if execution_validation.get("status") not in {"not_requested", "passed"}:
            payload["status"] = "execution_record_invalid"
            payload["decision_status"] = "not_recommended_for_quality"
            payload["decision_reason"] = "dataset HLS execution record is missing or inconsistent with the evaluated sample count"
            return payload
        decision, reason = _classification_decision({
            "accuracy_drop_pct": payload.get("top1_accuracy_drop_pct"),
            "prediction_agreement_vs_reference": agreement,
        }, thresholds)
        payload["decision_status"] = decision
        payload["decision_reason"] = reason
        return payload

    if task in {"regression", "numeric_regression"}:
        targets_path = _resolve_data_path(out_dir, validation.get("targets") or validation.get("target_path") or validation.get("y") or (inference_reference_artifacts or {}).get("targets_path"))
        targets = _read_float_array(targets_path) if targets_path is not None else None
        payload = {
            **base,
            "status": "compared",
            "targets_path": _path_or_none(targets_path),
            "targets_status": "provided" if targets is not None else "not_provided",
            "sample_count": min(len(ref_values), len(hw_values)),
            "reference_output_mae": output_compare.get("mae") if output_compare else None,
            "reference_output_rmse": math.sqrt(float(output_compare.get("mse"))) if output_compare and output_compare.get("mse") is not None and float(output_compare.get("mse")) >= 0 else None,
            "reference_output_max_abs_error": output_compare.get("max_abs_error") if output_compare else None,
            "target_mae_reference": None,
            "target_mae_generated": None,
            "target_rmse_reference": None,
            "target_rmse_generated": None,
            "target_r2_reference": None,
            "target_r2_generated": None,
            "mae_increase": None,
            "rmse_increase": None,
        }
        if targets is not None:
            ref_m = _regression_metrics(targets, ref_values)
            hw_m = _regression_metrics(targets, hw_values)
            payload.update({
                "sample_count": min(ref_m.get("sample_count") or 0, hw_m.get("sample_count") or 0),
                "target_mae_reference": ref_m.get("mae"),
                "target_mae_generated": hw_m.get("mae"),
                "target_rmse_reference": ref_m.get("rmse"),
                "target_rmse_generated": hw_m.get("rmse"),
                "target_r2_reference": ref_m.get("r2"),
                "target_r2_generated": hw_m.get("r2"),
                "mae_increase": None if ref_m.get("mae") is None or hw_m.get("mae") is None else float(hw_m["mae"]) - float(ref_m["mae"]),
                "rmse_increase": None if ref_m.get("rmse") is None or hw_m.get("rmse") is None else float(hw_m["rmse"]) - float(ref_m["rmse"]),
            })
        decision, reason = _regression_decision(payload, thresholds)
        payload["decision_status"] = decision
        payload["decision_reason"] = reason
        return payload

    base.update({
        "status": "unsupported_task",
        "decision_status": "pending_task_metadata",
        "decision_reason": f"validation.task={task!r} is not supported by task-quality reporting",
    })
    return base



def _parameter_role_lsb(raw_config: dict[str, Any] | None, role: str) -> float:
    raw = raw_config or {}
    role_key = "weight" if role == "weight" else "bias" if role == "bias" else "weight"
    spec = _cfg_lookup(raw, f"numerics.defaults.{role_key}", None)
    if not isinstance(spec, dict):
        spec = _cfg_lookup(raw, f"precision.defaults.{role_key}", None)
    if not isinstance(spec, dict):
        defaults = {"weight": (20, 8), "bias": (32, 16)}
        total_bits, int_bits = defaults.get(role_key, (20, 8))
    else:
        total_bits = int(spec.get("total_bits", spec.get("bits", 20 if role_key == "weight" else 32)))
        int_bits = int(spec.get("int_bits", spec.get("integer_bits", 8 if role_key == "weight" else 16)))
    return float(2.0 ** (-max(0, total_bits - int_bits)))


def _parameter_update_validation_payload(parameter_artifacts: dict[str, Any] | None, *, raw_config: dict[str, Any] | None = None) -> dict[str, Any]:
    if not parameter_artifacts:
        return {"requested": False, "status": "not_requested", "implementation_status": "not_requested", "passed": False}
    payload = dict(parameter_artifacts)
    ref_path = payload.get("ref")
    got_path = payload.get("got")
    ref = _read_f32_file(ref_path)
    got = _read_f32_file(got_path)
    payload.update({"ref_path": _path_or_none(ref_path), "got_path": _path_or_none(got_path), "ref_exists": _exists(ref_path), "got_exists": _exists(got_path)})
    if ref is None or got is None:
        payload.update({"status": "artifact_missing", "implementation_status": "not_validated", "passed": False})
        return payload
    metadata = _optimizer_state_reference_metadata(ref_path)
    updates = int(metadata.get("optimizer_updates", 0) or 0)
    layout = metadata.get("parameter_layout", [])
    same_shape = len(ref) == len(got)
    segments = []
    all_passed = same_shape
    exact_total = within1_total = within2_total = above2_total = 0
    max_distance = 0
    eps = 1e-15
    for entry in layout:
        start = int(entry.get("offset", 0) or 0); count = int(entry.get("count", 0) or 0); end = start + count
        role = str(entry.get("role", "weight")); lsb = _parameter_role_lsb(raw_config, role)
        diffs = [abs(float(got[i] - ref[i])) for i in range(start, min(end, len(ref), len(got)))]
        exact = sum(v == 0.0 for v in diffs); within1 = sum(v <= lsb + eps for v in diffs); within2 = sum(v <= 2*lsb + eps for v in diffs); above2 = sum(v > 2*lsb + eps for v in diffs)
        allowed = 1 if updates <= 1 else 2
        passed = len(diffs) == count and (within1 == count if allowed == 1 else within2 == count)
        all_passed = all_passed and passed
        distance = max((int(round(v/lsb)) for v in diffs), default=0) if lsb > 0 else 0
        max_distance = max(max_distance, distance)
        exact_total += exact; within1_total += within1; within2_total += within2; above2_total += above2
        segments.append({"name": f"{'W' if role == 'weight' else 'B' if role == 'bias' else role}_{entry.get('layer','parameter')}", "layer": entry.get("layer"), "role": role, "offset": start, "count": count, "lsb": lsb, "exact_words": exact, "within_one_lsb": within1, "within_two_lsb": within2, "above_two_lsb": above2, "maximum_lsb_distance": distance, "max_abs_error": max(diffs, default=0.0), "passed": passed})
    metrics = _compare_vectors(ref, got)
    classification = "bit_exact" if same_shape and exact_total == len(ref) else "quantization_aligned" if same_shape and within1_total == len(ref) else "propagated_quantization_aligned" if updates > 1 and same_shape and within2_total == len(ref) else "failed"
    payload.update(metrics)
    payload.update({"optimizer_updates": updates or None, "reference_words": len(ref), "hls_words": len(got), "exact_words": exact_total, "within_one_lsb": within1_total, "within_two_lsb": within2_total, "above_two_lsb": above2_total, "all_words_within_one_lsb": bool(same_shape and within1_total == len(ref)), "all_words_within_two_lsb": bool(same_shape and within2_total == len(ref)), "maximum_lsb_distance": max_distance, "classification": classification, "segments": segments, "status": "implemented" if all_passed else "failed", "implementation_status": "implemented" if all_passed else "failed", "passed": all_passed})
    return payload


def _training_update_behavior_payload(parameter_validation: dict[str, Any], optimizer_validation: dict[str, Any], *, raw_config: dict[str, Any] | None = None) -> dict[str, Any]:
    parameter_segments = parameter_validation.get("segments", []) or []
    optimizer_comparisons = optimizer_validation.get("comparisons", {}) or {}
    optimizer_after = optimizer_comparisons.get("packed_optimizer_state_after", {}) if isinstance(optimizer_comparisons, dict) else {}
    optimizer_segments = optimizer_after.get("segments", []) or []
    first_parameter = next((segment for segment in parameter_segments if int(segment.get("exact_words", 0)) < int(segment.get("count", 0))), None)
    first_optimizer = next((segment for segment in optimizer_segments if int(segment.get("exact_words", 0)) < int(segment.get("count", 0))), None)
    updates = int(parameter_validation.get("optimizer_updates") or optimizer_after.get("optimizer_updates") or 0)
    per_update: list[dict[str, Any]] = []
    ref_path = Path(str(parameter_validation.get("ref_path"))) if parameter_validation.get("ref_path") else None
    got_path = Path(str(parameter_validation.get("got_path"))) if parameter_validation.get("got_path") else None
    ref_trace = ref_path.parent / "per_update_trace" if ref_path is not None else None
    got_trace = got_path.parent / "per_update_trace" if got_path is not None else None
    optimizer = str(optimizer_validation.get("optimizer", "sgd"))
    if ref_trace is not None and got_trace is not None and ref_trace.exists() and got_trace.exists():
        for update_index in range(1, updates + 1):
            weight_ref = ref_trace / f"update_{update_index:04d}_weights_ref.bin"
            weight_got = got_trace / f"update_{update_index:04d}_weights.bin"
            state_ref = ref_trace / f"update_{update_index:04d}_optimizer_state_ref.bin"
            state_got = got_trace / f"update_{update_index:04d}_optimizer_state.bin"
            weight_cmp = _parameter_update_validation_payload({"requested": True, "ref": weight_ref, "got": weight_got}, raw_config=raw_config)
            state_cmp = _compare_optimizer_state_pair(state_ref, state_got, lsb=_optimizer_state_lsb(raw_config), optimizer=optimizer) if state_ref.exists() or state_got.exists() else {"status": "not_applicable", "passed": optimizer == "sgd", "classification": "bit_exact" if optimizer == "sgd" else "not_applicable"}
            per_update.append({"update": update_index, "weights": weight_cmp, "optimizer_state": state_cmp, "passed": bool(weight_cmp.get("passed")) and bool(state_cmp.get("passed", optimizer == "sgd"))})

    first_divergent_update = None
    first_divergent_layer = None
    first_divergent_tensor = None
    for row in per_update:
        weight_cmp = row.get("weights", {}) or {}
        state_cmp = row.get("optimizer_state", {}) or {}
        weight_non_exact = weight_cmp.get("classification") not in (None, "bit_exact")
        state_non_exact = state_cmp.get("classification") not in (None, "bit_exact", "not_applicable")
        if weight_non_exact or state_non_exact:
            first_divergent_update = row.get("update")
            source = weight_cmp if weight_non_exact else state_cmp
            segments = source.get("segments", []) or []
            boundary = next((segment for segment in segments if int(segment.get("exact_words", 0)) < int(segment.get("count", 0))), None)
            if isinstance(boundary, dict):
                first_divergent_layer = boundary.get("layer")
                first_divergent_tensor = boundary.get("name")
            else:
                first_divergent_tensor = "weights" if weight_non_exact else "optimizer_state"
            break

    if first_divergent_update is None and isinstance(first_parameter, dict):
        first_divergent_update = 1
        first_divergent_layer = first_parameter.get("layer")
        first_divergent_tensor = first_parameter.get("name")
    elif first_divergent_update is None and isinstance(first_optimizer, dict):
        first_divergent_update = 2 if updates and updates > 1 else 1
        first_divergent_layer = first_optimizer.get("layer")
        first_divergent_tensor = first_optimizer.get("name")
    if first_divergent_layer is None and isinstance(first_divergent_tensor, str) and "_" in first_divergent_tensor:
        first_divergent_layer = first_divergent_tensor.split("_", 1)[1]

    complete_trace = bool(updates > 0 and len(per_update) == updates)
    final_passed = bool(parameter_validation.get("passed") and optimizer_validation.get("passed"))
    status = "implemented" if final_passed and (complete_trace or not per_update) else "partial"
    return {
        "artifact_kind": "fpgai_training_update_behavior_trace",
        "schema_version": 3,
        "status": status,
        "optimizer_updates": updates or None,
        "per_update_trace_available": complete_trace,
        "per_update": per_update,
        "first_divergent_update": first_divergent_update,
        "first_divergent_layer": first_divergent_layer,
        "first_divergent_tensor": first_divergent_tensor,
        "first_parameter_boundary": first_parameter,
        "first_optimizer_state_boundary": first_optimizer,
        "propagation_path": [value for value in [first_parameter.get("name") if isinstance(first_parameter, dict) else None, first_optimizer.get("name") if isinstance(first_optimizer, dict) else None] if value],
        "final_classification": optimizer_after.get("classification") or parameter_validation.get("classification"),
        "note": "The first divergence is the first non-bit-exact per-update boundary, even when the comparison remains within the accepted fixed-point tolerance."
    }

def _optimizer_csynth_metrics(report_path: str | Path | None) -> dict[str, Any]:
    if report_path is None:
        return {}
    path = Path(report_path)
    if not path.is_file():
        return {}
    metrics = dict(parse_hls_csynth_report(path))
    text = path.read_text(encoding="utf-8", errors="ignore")
    xml_candidates = [path.with_suffix(".xml"), path.parent / "csynth.xml"]
    for candidate in xml_candidates:
        if candidate.is_file():
            text += "\n" + candidate.read_text(encoding="utf-8", errors="ignore")
    def first(patterns: list[str]) -> float | None:
        import re
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    return float(str(match.group(1)).replace(",", ""))
                except Exception:
                    pass
        return None
    metrics.update({
        "actual_uram": first([r"<URAM>([0-9,.]+)</URAM>", r"\bURAM\b[^0-9\n]*([0-9,]+)"]),
        "latency_min_cycles": first([r"<Best-caseLatency>([0-9,.]+)</Best-caseLatency>", r"Latency[^\n]*min[^0-9\n]*([0-9,]+)"]),
        "latency_max_cycles": first([r"<Worst-caseLatency>([0-9,.]+)</Worst-caseLatency>", r"Latency[^\n]*max[^0-9\n]*([0-9,]+)"]),
        "interval_min_cycles": first([r"<Interval-min>([0-9,.]+)</Interval-min>", r"Interval[^\n]*min[^0-9\n]*([0-9,]+)"]),
        "interval_max_cycles": first([r"<Interval-max>([0-9,.]+)</Interval-max>", r"Interval[^\n]*max[^0-9\n]*([0-9,]+)"]),
        "estimated_clock_period_ns": first([r"<EstimatedClockPeriod>([0-9,.]+)</EstimatedClockPeriod>", r"Estimated[^\n]*Clock[^0-9\n]*([0-9.]+)"]),
    })
    return metrics



def _training_command_latency_payload(
    raw_config: dict[str, Any] | None,
    *,
    source_path: Path | None,
    hls_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Describe command-specific latency boundaries without inventing HLS results.

    Vitis HLS reports one aggregate latency range for the multi-command top.
    FPGAI therefore records that range separately and derives only source-level
    transfer lower bounds for export commands. These lower bounds are not
    presented as scheduled or implemented latency.
    """
    import re

    raw = raw_config or {}
    text = ""
    if source_path is not None and source_path.is_file():
        text = source_path.read_text(encoding="utf-8", errors="ignore")

    weight_words = 0
    for _type_name, _name, count in re.findall(
        r"static\s+(wgt_t|bias_t)\s+([WB]_[A-Za-z0-9_]+)\[(\d+)\]", text
    ):
        weight_words += int(count)
    optimizer_words = None
    match = re.search(r"FPGAI_OPTIMIZER_STATE_EXPORT_WORDS\s*=\s*(\d+)", text)
    if match:
        optimizer_words = int(match.group(1))

    materialization = str(_cfg_lookup(raw, "training.gradients.materialization", "full") or "full")
    tile_size = int(_cfg_lookup(raw, "training.gradients.tile_size", 256) or 256)
    lifetime = str(_cfg_lookup(raw, "training.memory_lifetime.policy", "separate") or "separate")

    aggregate = {
        "latency_min_cycles": hls_metrics.get("latency_min_cycles"),
        "latency_max_cycles": hls_metrics.get("latency_max_cycles"),
        "interval_min_cycles": hls_metrics.get("interval_min_cycles"),
        "interval_max_cycles": hls_metrics.get("interval_max_cycles"),
        "estimated_clock_period_ns": hls_metrics.get("estimated_clock_period_ns"),
        "scope": "multi_command_top_aggregate",
    }
    commands = {
        "run_training": {
            "status": "aggregate_hls_range_only",
            "hls_top_range": aggregate,
            "note": "The current C-synthesis report does not separate runtime command branches.",
        },
        "export_weights": {
            "status": "source_transfer_lower_bound",
            "output_words": weight_words or None,
            "minimum_stream_cycles": weight_words or None,
        },
        "export_gradients": {
            "status": "source_transfer_lower_bound",
            "output_words": weight_words or None,
            "minimum_stream_cycles": weight_words or None,
            "materialization": materialization,
            "tile_size": tile_size if materialization == "tiled" else None,
            "memory_lifetime_policy": lifetime,
        },
        "export_optimizer_state": {
            "status": "source_transfer_lower_bound" if optimizer_words else "not_available",
            "output_words": optimizer_words,
            "minimum_transfer_cycles": optimizer_words,
        },
    }
    return {
        "artifact_kind": "fpgai_training_command_latency",
        "schema_version": 1,
        "status": "implemented" if hls_metrics else "not_validated",
        "aggregate_hls_top": aggregate,
        "commands": commands,
        "claim_scope": "aggregate_hls_range_plus_source_transfer_lower_bounds",
        "note": "Source transfer counts are lower bounds, not command-specific scheduled latency.",
    }


def _training_resource_owner_payload(
    raw_config: dict[str, Any] | None,
    *,
    source_path: Path | None,
    hls_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Describe generated training-memory owners and the YAML knobs that own them.

    Array block counts are conservative source-bound lower-bound estimates. The
    canonical device totals remain the parsed C-synthesis metrics.
    """
    raw = raw_config or {}
    import math
    import re

    type_bits = {
        "wgt_t": int(_cfg_lookup(raw, "numerics.weights.total_bits", 16) or 16),
        "bias_t": int(_cfg_lookup(raw, "numerics.bias.total_bits", 16) or 16),
        "act_t": int(_cfg_lookup(raw, "numerics.activations.total_bits", 16) or 16),
        "grad_wgt_t": int(_cfg_lookup(raw, "numerics.gradients.total_bits", 16) or 16),
        "grad_bias_t": int(_cfg_lookup(raw, "numerics.gradients.total_bits", 16) or 16),
        "opt_t": int(_cfg_lookup(raw, "numerics.optimizer_state.total_bits", 32) or 32),
        "float": 32,
    }
    owners: list[dict[str, Any]] = []
    if source_path is not None and source_path.is_file():
        text = source_path.read_text(encoding="utf-8", errors="ignore")
        macro_counts = {
            name: int(value)
            for name, value in re.findall(r"#define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(\d+)", text)
        }
        decl = re.compile(
            r"(?:static\s+)?(?P<type>wgt_t|bias_t|act_t|grad_wgt_t|grad_bias_t|opt_t|float)\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(?P<count>\d+|[A-Za-z_][A-Za-z0-9_]*)\s*\]\s*;"
        )
        for match in decl.finditer(text):
            type_name = match.group("type")
            name = match.group("name")
            count_token = match.group("count")
            count = int(count_token) if count_token.isdigit() else macro_counts.get(count_token)
            if count is None:
                continue
            bind = re.search(
                rf"#pragma\s+HLS\s+BIND_STORAGE\s+variable={re.escape(name)}\s+type=ram_[12]p\s+impl=(bram|uram)",
                text,
            )
            storage = bind.group(1) if bind else None
            bits = count * type_bits.get(type_name, 32)
            if storage == "uram":
                lower_bound_blocks = int(math.ceil(bits / 294912.0))
            elif storage == "bram":
                lower_bound_blocks = int(math.ceil(bits / 18432.0))
            else:
                lower_bound_blocks = None
            upper = name.upper()
            if upper.startswith("FPGAI_ADAM_") or upper.startswith("FPGAI_MOMENTUM_"):
                role = "optimizer_state"
                knob = "training.storage.optimizer_state"
            elif "GRADIENT_EXPORT_TILE" in upper:
                role = "gradient_export_scratch"
                knob = "training.gradients.tile_size"
            elif upper.startswith("DW_") or upper.startswith("DB_") or "DW_TILE" in upper or "GRAD" in upper:
                role = "gradient"
                knob = "training.storage.parameter_gradient"
            elif upper.startswith("W_") or upper.startswith("B_"):
                role = "parameter"
                knob = "memory.weight_storage"
            elif "TILE" in upper or "BUFFER" in upper or type_name == "act_t":
                role = "activation_or_scratch"
                knob = "data_movement.*.tiled.tile_size"
            else:
                role = "other"
                knob = None
            owners.append({
                "name": name,
                "role": role,
                "element_type": type_name,
                "element_bits": type_bits.get(type_name, 32),
                "count": count,
                "total_bits": bits,
                "source_binding": storage,
                "source_bound_lower_bound_blocks": lower_bound_blocks,
                "owning_yaml_knob": knob,
            })
    owners.sort(key=lambda row: int(row.get("total_bits") or 0), reverse=True)
    knob_trace = [
        {"path": "optimization.parallel.pe", "effective": _cfg_lookup(raw, "optimization.parallel.pe", None), "effect": "output/channel parallelism"},
        {"path": "optimization.parallel.simd", "effective": _cfg_lookup(raw, "optimization.parallel.simd", None), "effect": "input/channel parallelism"},
        {"path": "optimization.parallel.unroll_factor", "effective": _cfg_lookup(raw, "optimization.parallel.unroll_factor", None), "effect": "loop replication"},
        {"path": "optimization.parallel.partition_factor", "effective": _cfg_lookup(raw, "optimization.parallel.partition_factor", None), "effect": "array banking and memory replication"},
        {"path": "optimization.parallel.array_partition_mode", "effective": _cfg_lookup(raw, "optimization.parallel.array_partition_mode", None), "effect": "banking layout"},
        {"path": "memory.weight_storage", "effective": _cfg_lookup(raw, "memory.weight_storage", None), "effect": "parameter memory implementation"},
        {"path": "training.gradients.computation", "effective": _cfg_lookup(raw, "training.gradients.computation", "full_buffer"), "effect": "parameter-gradient compute lifetime: full buffer, tiled accumulation, or fused update"},
        {"path": "training.storage.parameter_gradient", "effective": _cfg_lookup(raw, "training.storage.parameter_gradient", _cfg_lookup(raw, "training.storage.gradient", _cfg_lookup(raw, "memory.gradient_storage", None))), "effect": "parameter-gradient memory implementation"},
        {"path": "training.storage.optimizer_state", "effective": _cfg_lookup(raw, "training.storage.optimizer_state", _cfg_lookup(raw, "memory.optimizer_state_storage", None)), "effect": "optimizer-state memory implementation"},
        {"path": "training.gradients.materialization", "effective": _cfg_lookup(raw, "training.gradients.materialization", "full"), "effect": "full, tiled, or streamed gradient export scratch structure"},
        {"path": "training.gradients.tile_size", "effective": _cfg_lookup(raw, "training.gradients.tile_size", 256), "effect": "bounded gradient export tile size"},
        {"path": "training.memory_lifetime.policy", "effective": _cfg_lookup(raw, "training.memory_lifetime.policy", "separate"), "effect": "per-layer or phase-shared physical export tile ownership"},
        {"path": "training.optimizer.implementation.arithmetic", "effective": _cfg_lookup(raw, "training.optimizer.implementation.arithmetic", "direct"), "effect": "optimizer arithmetic ownership"},
        {"path": "training.optimizer.implementation.update_parallelism", "effective": _cfg_lookup(raw, "training.optimizer.implementation.update_parallelism", 1), "effect": "optimizer update replication"},
        {"path": "data_movement.gradients.export.tiled.tile_size", "effective": _cfg_lookup(raw, "data_movement.gradients.export.tiled.tile_size", _cfg_lookup(raw, "data_movement.gradients.export.tile_size", None)), "effect": "gradient export scratch tile"},
        {"path": "data_movement.inputs.import.tiled.tile_size", "effective": _cfg_lookup(raw, "data_movement.inputs.import.tiled.tile_size", _cfg_lookup(raw, "data_movement.inputs.import.tile_size", None)), "effect": "input activation scratch tile"},
    ]
    actual_bram = hls_metrics.get("actual_bram18")
    actual_lut = hls_metrics.get("actual_lut")
    recommendations = []
    if isinstance(actual_bram, (int, float)) and actual_bram > 0:
        recommendations.extend([
            {"priority": 1, "resource": "bram_18k", "knob": "optimization.parallel.partition_factor", "action": "keep at 1 or reduce banking", "reason": "partitioning can replicate memory banks"},
            {"priority": 2, "resource": "bram_18k", "knob": "training.storage.optimizer_state", "action": "prefer uram when supported", "reason": "persistent optimizer state is a dominant owner"},
            {"priority": 3, "resource": "bram_18k", "knob": "training.storage.parameter_gradient", "action": "select uram when capacity permits", "reason": "moves complete dW/dB owners from BRAM to URAM"},
            {"priority": 4, "resource": "bram_18k", "knob": "training.gradients.materialization", "action": "select tiled or streamed", "reason": "removes full OUT_grad export scratch arrays"},
            {"priority": 5, "resource": "bram_18k", "knob": "training.memory_lifetime.policy", "action": "select phase_shared with tiled materialization", "reason": "reuses one physical export tile across layers"},
        ])
    recommendations.extend([
        {"priority": 7, "resource": "lut", "knob": "optimization.parallel.pe", "action": "reduce", "reason": "reduces replicated datapaths"},
        {"priority": 5, "resource": "lut", "knob": "optimization.parallel.simd", "action": "reduce", "reason": "reduces replicated lane logic"},
        {"priority": 6, "resource": "lut", "knob": "training.optimizer.implementation.update_parallelism", "action": "set to 1", "reason": "serializes optimizer update lanes"},
    ])
    return {
        "artifact_kind": "fpgai_training_resource_ownership",
        "schema_version": 1,
        "status": "implemented" if source_path is not None else "not_available",
        "source": str(source_path) if source_path is not None else None,
        "canonical_hls_totals": hls_metrics,
        "owner_count": len(owners),
        "owners": owners,
        "top_owners": owners[:20],
        "hardware_knob_trace": knob_trace,
        "recommended_knob_actions": recommendations,
        "claim_scope": "generated_source_ownership_with_hls_device_totals",
        "note": "Per-array block values are source-bound lower-bound estimates; canonical totals come from C synthesis.",
    }

def _optimizer_resource_strategy_payload(
    raw_config: dict[str, Any] | None,
    *,
    hls_ran: bool,
    hls_ok: bool | None,
    hls_csynth_report: str | Path | None = None,
) -> dict[str, Any]:
    raw = raw_config or {}
    arithmetic = str(_cfg_lookup(raw, "training.optimizer.implementation.arithmetic", "direct") or "direct").lower().replace("-", "_")
    parallelism = int(_cfg_lookup(raw, "training.optimizer.implementation.update_parallelism", 1) or 1)
    storage = str(_cfg_lookup(raw, "memory.optimizer_state_storage", _cfg_lookup(raw, "training.storage.optimizer_state", "none")) or "none").lower()
    shared_requested = arithmetic == "shared"
    report_path = Path(hls_csynth_report) if hls_csynth_report else None
    report_present = bool(report_path and report_path.is_file())
    metrics = _optimizer_csynth_metrics(report_path) if report_present else {}
    synthesis_status = "available" if hls_ran and hls_ok is True and report_present else ("failed" if hls_ran and hls_ok is False else "not_validated")
    state_impl = "uram" if storage == "uram" else ("bram" if storage == "bram" else None)
    source_path = None
    state_array_bindings: list[dict[str, Any]] = []
    if report_path is not None:
        for parent in [report_path.parent, *report_path.parents]:
            candidate = parent / "src" / "deeplearn.cpp"
            if candidate.is_file():
                source_path = candidate
                break
    if source_path is not None:
        import re
        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"static\s+opt_t\s+(FPGAI_ADAM_[MV]_[WB]_[A-Za-z0-9_]+)\s*\[\s*(\d+)\s*\]\s*;", source_text):
            name = match.group(1)
            size = int(match.group(2))
            bind = re.search(rf"#pragma\s+HLS\s+BIND_STORAGE\s+variable={re.escape(name)}\s+type=ram_2p\s+impl=(bram|uram)", source_text)
            state_array_bindings.append({"name": name, "count": size, "requested_storage": storage, "source_binding": bind.group(1) if bind else None, "binding_status": "materialized" if bind else "missing"})
    synthesized_storage_status = "not_validated"
    if synthesis_status == "available":
        actual_uram = metrics.get("actual_uram")
        if storage == "uram":
            synthesized_storage_status = "observed" if actual_uram not in (None, 0, 0.0) else "not_observed"
        elif storage == "bram":
            synthesized_storage_status = "observed" if metrics.get("actual_bram18") not in (None, 0, 0.0) else "not_observed"
    return {
        "artifact_kind": "fpgai_optimizer_resource_strategy",
        "schema_version": 2,
        "arithmetic": arithmetic,
        "update_parallelism": parallelism,
        "optimizer_state_storage": storage,
        "mechanism": "single_non_inlined_adam_correction_owner" if shared_requested else "direct_per_update_loop_expression",
        "source_generation_status": "implemented",
        "hls_synthesis_status": synthesis_status,
        "hls_ran": bool(hls_ran),
        "hls_ok": hls_ok,
        "csynth_report": str(report_path) if report_path is not None else None,
        "csynth_report_present": report_present,
        "hls_metrics": metrics,
        "state_array_bindings": state_array_bindings,
        "state_binding_source": str(source_path) if source_path is not None else None,
        "training_resource_ownership": _training_resource_owner_payload(raw, source_path=source_path, hls_metrics=metrics),
        "training_command_latency": _training_command_latency_payload(raw, source_path=source_path, hls_metrics=metrics),
        "requested_state_impl": state_impl,
        "synthesized_storage_status": synthesized_storage_status,
        "baseline_comparison_status": "not_run",
        "claim_scope": "generated_and_synthesized_strategy" if synthesis_status == "available" else "generated_strategy",
    }

def emit_numeric_validation_report(
    out_dir: str | Path,
    *,
    pipeline_mode: str,
    source_generated: bool,
    hls_ran: bool = False,
    hls_ok: bool | None = None,
    hls_csynth_report: str | Path | None = None,
    training_reference_result: Any = None,
    training_compare_result: Any = None,
    inference_reference_artifacts: dict[str, Any] | None = None,
    gradient_export_artifacts: dict[str, Any] | None = None,
    optimizer_state_artifacts: dict[str, Any] | None = None,
    parameter_update_artifacts: dict[str, Any] | None = None,
    raw_config: dict[str, Any] | None = None,
    runtime_sequence: Any = None,
) -> dict[str, Path]:
    """Write numeric validation reports and return artifact paths.

    The report is intentionally conservative:
    - inference is marked ``not_run`` unless explicit inference validation
      artifacts are provided;
    - training is marked ``passed`` only when the training comparison result
      exists and exposes a summary/results file;
    - missing HLS/testbench artifacts are recorded as missing evidence, not as
      success.
    """

    out = Path(out_dir)
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    pipeline_mode = str(pipeline_mode or "inference")
    training_reference = _training_reference_payload(training_reference_result)
    training_compare = _training_compare_payload(training_compare_result)

    gradient_export_validation = _gradient_export_validation_payload(gradient_export_artifacts)
    optimizer_state_validation = _optimizer_state_validation_payload(optimizer_state_artifacts, raw_config=raw_config)
    parameter_update_validation = _parameter_update_validation_payload(parameter_update_artifacts, raw_config=raw_config)
    update_behavior_trace = _training_update_behavior_payload(parameter_update_validation, optimizer_state_validation, raw_config=raw_config)
    optimizer_resource_strategy = _optimizer_resource_strategy_payload(raw_config, hls_ran=hls_ran, hls_ok=hls_ok, hls_csynth_report=hls_csynth_report)
    batch_accumulation_validation = _batch_accumulation_validation_payload(
        out,
        pipeline_mode=pipeline_mode,
        raw_config=raw_config,
        runtime_sequence=runtime_sequence,
        training_reference_result=training_reference_result,
        training_compare_result=training_compare_result,
    )
    loss_validation = _loss_validation_payload(
        out,
        pipeline_mode=pipeline_mode,
        raw_config=raw_config,
        training_reference_result=training_reference_result,
        training_compare_result=training_compare_result,
    )
    training_tiled_io_validation = _training_tiled_io_validation_payload(
        out,
        pipeline_mode=pipeline_mode,
        raw_config=raw_config,
        training_reference_result=training_reference_result,
        training_compare_result=training_compare_result,
    )

    task_quality: dict[str, Any] = {"status": "not_applicable", "task": "not_applicable", "decision_status": "not_applicable"}
    if pipeline_mode == "training_on_device":
        if training_compare is not None:
            status = "passed"
            reason = "training reference and generated/testbench comparison artifacts are available"
        elif training_reference is not None:
            status = "reference_only"
            reason = "Python training reference exists, but generated/testbench comparison artifacts are missing"
        else:
            status = "not_run"
            reason = "no training numeric reference or generated/testbench comparison artifacts were found"
    else:
        inference_reference_artifacts = inference_reference_artifacts or {}
        output_compare = None
        if inference_reference_artifacts.get("outputs_hw") and inference_reference_artifacts.get("outputs_ref"):
            limits = _precision_aware_inference_limits(raw_config)
            output_compare = _compare_file_pair(
                inference_reference_artifacts.get("outputs_ref"),
                inference_reference_artifacts.get("outputs_hw"),
                max_abs_error_limit=float(limits["max_abs_error_limit"]),
                mean_abs_error_limit=float(limits["mean_abs_error_limit"]),
                rmse_limit=float(limits["rmse_limit"]),
                min_cosine_similarity=float(limits["min_cosine_similarity"]),
            )
        task_quality = _task_quality_payload(
            out,
            raw_config=raw_config,
            output_compare=output_compare,
            inference_reference_artifacts=inference_reference_artifacts,
        )
        if output_compare is None:
            status = "not_run"
            reason = "inference numeric comparison artifacts are not available for this compile path"
        elif output_compare.get("passed") is True:
            status = "passed"
            reason = "inference output comparison artifacts were compared successfully"
        else:
            compare_status = str(output_compare.get("status") or "failed_numeric_validation")
            if compare_status == "compared":
                status = "failed_tolerance"
                failed_checks = [
                    str(check.get("name"))
                    for check in output_compare.get("checks", [])
                    if isinstance(check, dict) and check.get("passed") is False
                ]
                if failed_checks:
                    reason = "inference output comparison completed but failed tolerance check(s): " + ", ".join(failed_checks)
                else:
                    reason = "inference output comparison completed but failed configured precision-aware tolerance checks"
            elif compare_status in {"shape_mismatch", "missing_or_unreadable", "empty_generated_output"}:
                status = "execution_artifact_invalid" if compare_status == "empty_generated_output" else compare_status
                reason = (
                    "generated inference output artifact is empty"
                    if compare_status == "empty_generated_output"
                    else f"inference output comparison could not be accepted: {compare_status}"
                )
            else:
                status = "failed_numeric_validation"
                reason = f"inference output comparison failed with status={compare_status}"

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "numeric_validation",
        "pipeline_mode": pipeline_mode,
        "status": status,
        "passed": status == "passed",
        "reason": reason,
        "source_generated": bool(source_generated),
        "hls_ran": bool(hls_ran),
        "hls_ok": hls_ok,
        "inference": {
            "status": status if pipeline_mode != "training_on_device" else "not_applicable",
            "inputs_bin": _path_or_none(inference_reference_artifacts.get("inputs_bin")) if inference_reference_artifacts else None,
            "outputs_hw": _path_or_none(inference_reference_artifacts.get("outputs_hw")) if inference_reference_artifacts else None,
            "outputs_ref": _path_or_none(inference_reference_artifacts.get("outputs_ref")) if inference_reference_artifacts else None,
            "outputs_hw_exists": _exists(inference_reference_artifacts.get("outputs_hw")) if inference_reference_artifacts else False,
            "outputs_ref_exists": _exists(inference_reference_artifacts.get("outputs_ref")) if inference_reference_artifacts else False,
            "output_compare": output_compare if pipeline_mode != "training_on_device" else None,
            "task_quality": task_quality if pipeline_mode != "training_on_device" else {"status": "not_applicable", "task": "not_applicable", "decision_status": "not_applicable"},
        },
        "training": {
            "status": status if pipeline_mode == "training_on_device" else "not_applicable",
            "reference": training_reference,
            "comparison": training_compare,
            "checks": [] if training_compare is None else [
                {"name": "gradients", "metric": "cosine_similarity", "value": training_compare.get("grad_cosine"), "passed": training_compare.get("grad_cosine") is None or training_compare.get("grad_cosine") >= 0.99},
                {"name": "weights_after", "metric": "cosine_similarity", "value": training_compare.get("weight_after_cosine"), "passed": training_compare.get("weight_after_cosine") is None or training_compare.get("weight_after_cosine") >= 0.99},
                {"name": "weight_delta", "metric": "cosine_similarity", "value": training_compare.get("weight_delta_cosine"), "passed": training_compare.get("weight_delta_cosine") is None or training_compare.get("weight_delta_cosine") >= 0.99},
            ],
        },
        "gradient_export": gradient_export_validation,
        "optimizer_state_validation": optimizer_state_validation,
        "parameter_update_validation": parameter_update_validation,
        "training_update_behavior_trace": update_behavior_trace,
        "optimizer_resource_strategy": optimizer_resource_strategy,
        "batch_accumulation": batch_accumulation_validation,
        "loss_validation": loss_validation,
        "training_tiled_io": training_tiled_io_validation,
        "paper_claim_allowed": {
            "numeric_correctness": status == "passed",
        },
    }

    json_path = reports / "numeric_validation.json"
    md_path = reports / "numeric_validation.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    lines = [
        "# Numeric validation",
        "",
        f"- Pipeline mode: `{pipeline_mode}`",
        f"- Status: `{status}`",
        f"- Passed: `{str(status == 'passed').lower()}`",
        f"- Reason: {reason}",
        f"- Source generated: `{str(bool(source_generated)).lower()}`",
        f"- HLS ran: `{str(bool(hls_ran)).lower()}`",
    ]
    if pipeline_mode == "training_on_device":
        lines += [
            "",
            "## Training evidence",
            f"- Python reference: `{ 'yes' if training_reference is not None else 'no' }`",
            f"- Generated/testbench comparison: `{ 'yes' if training_compare is not None else 'no' }`",
            f"- Gradient export validation: `{gradient_export_validation.get('status', 'not_requested')}`",
            f"- Optimizer-state validation: `{optimizer_state_validation.get('status', 'not_requested')}`",
            f"- Parameter-update validation: `{parameter_update_validation.get('status', 'not_requested')}`",
            f"- Update behavior trace: `{update_behavior_trace.get('status', 'partial')}`",
            f"- Batch accumulation validation: `{batch_accumulation_validation.get('status', 'not_requested')}`",
            f"- Loss validation: `{loss_validation.get('status', 'not_requested')}`",
            f"- Training tiled-I/O validation: `{training_tiled_io_validation.get('status', 'not_requested')}`",
        ]
    else:
        lines += [
            "",
            "## Inference evidence",
            "- Final output comparison: `not available`" if output_compare is None else f"- Final output comparison: `{output_compare.get('status', 'available')}`",
            f"- Task quality: `{task_quality.get('decision_status', 'not_applicable')}`",
            f"- Task quality reason: {task_quality.get('decision_reason', 'not applicable')}",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    optimizer_json_path = reports / "optimizer_state_validation.json"
    optimizer_md_path = reports / "optimizer_state_validation.md"
    optimizer_json_path.write_text(json.dumps(optimizer_state_validation, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    optimizer_lines = [
        "# Optimizer-state validation",
        "",
        f"- Optimizer: `{optimizer_state_validation.get('optimizer', 'not_applicable')}`",
        f"- Status: `{optimizer_state_validation.get('status', 'not_validated')}`",
        f"- Implementation status: `{optimizer_state_validation.get('implementation_status', optimizer_state_validation.get('status', 'not_validated'))}`",
        f"- Passed: `{str(bool(optimizer_state_validation.get('passed', False))).lower()}`",
        f"- Layout: `{optimizer_state_validation.get('layout', 'canonical_parameter_order')}`",
    ]
    for name, comparison in (optimizer_state_validation.get("comparisons", {}) or {}).items():
        if not isinstance(comparison, dict):
            continue
        optimizer_lines += [
            "",
            f"## {name}",
            f"- Reference words: `{comparison.get('reference_words')}`",
            f"- HLS words: `{comparison.get('hls_words')}`",
            f"- MAE: `{comparison.get('mae')}`",
            f"- Maximum absolute error: `{comparison.get('max_abs_error')}`",
            f"- Relative L2: `{comparison.get('relative_l2')}`",
            f"- Cosine similarity: `{comparison.get('cosine_similarity')}`",
            f"- Exact words: `{comparison.get('exact_words')}`",
            f"- Within one LSB: `{comparison.get('within_one_lsb')}`",
            f"- Within two LSBs: `{comparison.get('within_two_lsb')}`",
            f"- Above two LSBs: `{comparison.get('above_two_lsb')}`",
            f"- Maximum LSB distance: `{comparison.get('maximum_lsb_distance')}`",
            f"- Classification: `{comparison.get('classification')}`",
            f"- All words within one LSB: `{str(bool(comparison.get('all_words_within_one_lsb', False))).lower()}`",
            f"- All words within two LSBs: `{str(bool(comparison.get('all_words_within_two_lsb', False))).lower()}`",
        ]
    optimizer_md_path.write_text("\n".join(optimizer_lines) + "\n", encoding="utf-8")

    parameter_json_path = reports / "parameter_update_validation.json"
    parameter_md_path = reports / "parameter_update_validation.md"
    parameter_json_path.write_text(json.dumps(parameter_update_validation, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    parameter_lines = ["# Parameter-update validation", "", f"- Status: `{parameter_update_validation.get('status', 'not_validated')}`", f"- Classification: `{parameter_update_validation.get('classification')}`", f"- Passed: `{str(bool(parameter_update_validation.get('passed', False))).lower()}`", f"- Maximum LSB distance: `{parameter_update_validation.get('maximum_lsb_distance')}`"]
    for segment in parameter_update_validation.get("segments", []) or []:
        parameter_lines += ["", f"## {segment.get('name')}", f"- Exact words: `{segment.get('exact_words')}/{segment.get('count')}`", f"- Within one LSB: `{segment.get('within_one_lsb')}`", f"- Within two LSBs: `{segment.get('within_two_lsb')}`", f"- Above two LSBs: `{segment.get('above_two_lsb')}`", f"- Maximum LSB distance: `{segment.get('maximum_lsb_distance')}`"]
    parameter_md_path.write_text("\n".join(parameter_lines) + "\n", encoding="utf-8")
    behavior_json_path = reports / "training_update_behavior_trace.json"
    behavior_md_path = reports / "training_update_behavior_trace.md"
    behavior_json_path.write_text(json.dumps(update_behavior_trace, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    behavior_md_path.write_text("# Training update behavior trace\n\n" + f"- Status: `{update_behavior_trace.get('status')}`\n- Optimizer updates: `{update_behavior_trace.get('optimizer_updates')}`\n- Final classification: `{update_behavior_trace.get('final_classification')}`\n- First divergent update: `{update_behavior_trace.get('first_divergent_update')}`\n- First divergent layer: `{update_behavior_trace.get('first_divergent_layer')}`\n- First divergent tensor: `{update_behavior_trace.get('first_divergent_tensor')}`\n- Propagation path: `{' -> '.join(update_behavior_trace.get('propagation_path', [])) or 'none'}`\n", encoding="utf-8")

    behavior_csv_path = reports / "training_update_behavior_trace.csv"
    behavior_csv_lines = ["update,tensor_kind,name,classification,passed,maximum_lsb_distance"]
    for row in update_behavior_trace.get("per_update", []) or []:
        for tensor_kind in ("weights", "optimizer_state"):
            comparison = row.get(tensor_kind, {}) or {}
            segments = comparison.get("segments", []) or []
            if segments:
                for segment in segments:
                    behavior_csv_lines.append(f"{row.get('update')},{tensor_kind},{segment.get('name')},{comparison.get('classification')},{str(bool(comparison.get('passed'))).lower()},{segment.get('maximum_lsb_distance')}")
            else:
                behavior_csv_lines.append(f"{row.get('update')},{tensor_kind},global,{comparison.get('classification')},{str(bool(comparison.get('passed'))).lower()},{comparison.get('maximum_lsb_distance')}")
    behavior_csv_path.write_text("\n".join(behavior_csv_lines) + "\n", encoding="utf-8")
    resource_json_path = reports / "optimizer_resource_strategy.json"
    resource_md_path = reports / "optimizer_resource_strategy.md"
    ownership_json_path = reports / "training_resource_ownership.json"
    ownership_md_path = reports / "training_resource_ownership.md"
    command_latency_json_path = reports / "training_command_latency.json"
    command_latency_md_path = reports / "training_command_latency.md"
    resource_json_path.write_text(json.dumps(optimizer_resource_strategy, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    resource_md_path.write_text("# Optimizer resource strategy\n\n" + f"- Arithmetic: `{optimizer_resource_strategy.get('arithmetic')}`\n- Update parallelism: `{optimizer_resource_strategy.get('update_parallelism')}`\n- State storage: `{optimizer_resource_strategy.get('optimizer_state_storage')}`\n- Mechanism: `{optimizer_resource_strategy.get('mechanism')}`\n- HLS synthesis status: `{optimizer_resource_strategy.get('hls_synthesis_status')}`\n- C synthesis report present: `{str(bool(optimizer_resource_strategy.get('csynth_report_present'))).lower()}`\n- C synthesis report: `{optimizer_resource_strategy.get('csynth_report')}`\n- HLS metrics: `{optimizer_resource_strategy.get('hls_metrics')}`\n- Baseline comparison: `{optimizer_resource_strategy.get('baseline_comparison_status')}`\n", encoding="utf-8")
    ownership = optimizer_resource_strategy.get("training_resource_ownership", {})
    ownership_json_path.write_text(json.dumps(ownership, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    owner_lines = ["# Training resource ownership", "", f"- Status: `{ownership.get('status')}`", f"- Owner count: `{ownership.get('owner_count')}`", f"- Claim scope: `{ownership.get('claim_scope')}`", "", "## Largest generated owners", ""]
    for row in ownership.get("top_owners", [])[:12]:
        owner_lines.append(f"- `{row.get('name')}`: role=`{row.get('role')}`, bits=`{row.get('total_bits')}`, binding=`{row.get('source_binding')}`, knob=`{row.get('owning_yaml_knob')}`")
    owner_lines.extend(["", "## Knob actions", ""])
    for row in ownership.get("recommended_knob_actions", []):
        owner_lines.append(f"- Priority {row.get('priority')}: `{row.get('knob')}` — {row.get('action')} ({row.get('reason')})")
    ownership_md_path.write_text("\n".join(owner_lines) + "\n", encoding="utf-8")
    command_latency = optimizer_resource_strategy.get("training_command_latency", {})
    command_latency_json_path.write_text(json.dumps(command_latency, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    latency_lines = [
        "# Training command latency",
        "",
        f"- Status: `{command_latency.get('status')}`",
        f"- Claim scope: `{command_latency.get('claim_scope')}`",
        f"- Aggregate HLS top range: `{command_latency.get('aggregate_hls_top')}`",
        "",
        "## Commands",
        "",
    ]
    for command, row in command_latency.get("commands", {}).items():
        latency_lines.append(f"- `{command}`: status=`{row.get('status')}`, details=`{row}`")
    command_latency_md_path.write_text("\n".join(latency_lines) + "\n", encoding="utf-8")

    return {
        "numeric_validation_json": json_path,
        "numeric_validation_md": md_path,
        "optimizer_state_validation_json": optimizer_json_path,
        "optimizer_state_validation_md": optimizer_md_path,
        "parameter_update_validation_json": parameter_json_path,
        "parameter_update_validation_md": parameter_md_path,
        "training_update_behavior_trace_json": behavior_json_path,
        "training_update_behavior_trace_md": behavior_md_path,
        "training_update_behavior_trace_csv": behavior_csv_path,
        "optimizer_resource_strategy_json": resource_json_path,
        "optimizer_resource_strategy_md": resource_md_path,
        "training_resource_ownership_json": ownership_json_path,
        "training_resource_ownership_md": ownership_md_path,
        "training_command_latency_json": command_latency_json_path,
        "training_command_latency_md": command_latency_md_path,
    }
