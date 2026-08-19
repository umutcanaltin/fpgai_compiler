"""Generate the public Python runtime API included in a package."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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

def load_graph_runtime_contract() -> dict[str, Any]:
    contract = load_manifest().get('graph_runtime_contract', {})
    return dict(contract) if isinstance(contract, dict) else {}

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

def export_weights(*, capture_path: str | Path | None = None) -> bytes:
    manifest = load_manifest()
    if not bool(manifest.get('runtime_weights', {}).get('export_supported')):
        raise RuntimeError('export_weights was not generated/supported for this package.')
    if _BOUND_BACKEND is not None:
        return _BOUND_BACKEND.export_weights(capture_path=capture_path)
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

def _autoregressive_contract() -> dict[str, Any]:
    contract = load_graph_runtime_contract().get('autoregressive_session', {})
    return contract if isinstance(contract, dict) else {}

def prepare_prefill(*, reset_state_first: bool | None = None) -> dict[str, Any]:
    contract = _autoregressive_contract()
    if not contract:
        raise RuntimeError('prepare_prefill is unavailable because this package has no autoregressive runtime contract.')
    do_reset = bool(contract.get('reset_state_on_prefill', True)) if reset_state_first is None else bool(reset_state_first)
    if do_reset:
        reset_state()
    if _BOUND_BACKEND is not None and hasattr(_BOUND_BACKEND, 'set_runtime_mode'):
        _BOUND_BACKEND.set_runtime_mode('prefill')
    return {'mode': 'prefill', 'reset_state': do_reset, 'contract': contract}

def prepare_decode() -> dict[str, Any]:
    contract = _autoregressive_contract()
    if not contract:
        raise RuntimeError('prepare_decode is unavailable because this package has no autoregressive runtime contract.')
    if _BOUND_BACKEND is not None and hasattr(_BOUND_BACKEND, 'set_runtime_mode'):
        _BOUND_BACKEND.set_runtime_mode('decode')
    return {'mode': 'decode', 'reset_state': False, 'contract': contract}

def postprocess_detections(outputs: Any, **kwargs: Any) -> Any:
    contract = load_graph_runtime_contract().get('detection_output', {})
    if not isinstance(contract, dict) or not contract:
        raise RuntimeError('postprocess_detections is unavailable because this package has no detection output contract.')
    partition = str(contract.get('postprocess_partition', 'ps_or_host'))
    if partition in {'none', 'pl'}:
        raise RuntimeError(f'postprocess_detections is not a host/PS stage for partition={partition}.')
    if _BOUND_BACKEND is not None and hasattr(_BOUND_BACKEND, 'postprocess_detections'):
        return _BOUND_BACKEND.postprocess_detections(outputs, contract=contract, **kwargs)
    _unsupported_board_call('postprocess_detections')

def _persistent_state_contract() -> dict[str, Any]:
    state = load_manifest().get('persistent_state', {})
    return state if isinstance(state, dict) else {}

def reset_state(name: str | None = None) -> Any:
    state = _persistent_state_contract()
    if int(state.get('tensor_count', 0)) <= 0:
        raise RuntimeError('reset_state is unavailable because this package has no persistent state tensors.')
    if _BOUND_BACKEND is not None and hasattr(_BOUND_BACKEND, 'reset_state'):
        return _BOUND_BACKEND.reset_state(name=name)
    _unsupported_board_call('reset_state')

def import_state(payload: Any, *, name: str | None = None) -> Any:
    state = _persistent_state_contract()
    if int(state.get('tensor_count', 0)) <= 0:
        raise RuntimeError('import_state is unavailable because this package has no persistent state tensors.')
    if _BOUND_BACKEND is not None and hasattr(_BOUND_BACKEND, 'import_state'):
        return _BOUND_BACKEND.import_state(payload, name=name)
    _unsupported_board_call('import_state')

def export_state(*, name: str | None = None) -> Any:
    state = _persistent_state_contract()
    if int(state.get('tensor_count', 0)) <= 0:
        raise RuntimeError('export_state is unavailable because this package has no persistent state tensors.')
    if _BOUND_BACKEND is not None and hasattr(_BOUND_BACKEND, 'export_state'):
        return _BOUND_BACKEND.export_state(name=name)
    _unsupported_board_call('export_state')

def read_state(*, name: str | None = None) -> Any:
    state = _persistent_state_contract()
    if int(state.get('tensor_count', 0)) <= 0:
        raise RuntimeError('read_state is unavailable because this package has no persistent state tensors.')
    if _BOUND_BACKEND is not None and hasattr(_BOUND_BACKEND, 'read_state'):
        return _BOUND_BACKEND.read_state(name=name)
    _unsupported_board_call('read_state')

def write_state(payload: Any, *, name: str | None = None) -> Any:
    state = _persistent_state_contract()
    if int(state.get('tensor_count', 0)) <= 0:
        raise RuntimeError('write_state is unavailable because this package has no persistent state tensors.')
    if _BOUND_BACKEND is not None and hasattr(_BOUND_BACKEND, 'write_state'):
        return _BOUND_BACKEND.write_state(payload, name=name)
    _unsupported_board_call('write_state')

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
    if command == 'reset_state':
        return reset_state(name=args.get('name'))
    if command == 'export_state':
        return export_state(name=args.get('name'))
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
