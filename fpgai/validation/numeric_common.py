"""Shared numeric-validation file, vector, configuration, and precision helpers."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

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
