"""One real, immutable Vitis HLS evaluation of a complete typed FPGAI RL state.

No surrogate HLS metrics, no fabricated hardware feasibility, no trained
policy. Preparation/audit are read-only with respect to licensed tools.
Only --execute-native runs original Vitis (once, in a fresh directory).
Independent audit reconstructs model.cpp, tb.cpp, Tcl identity, requested
numeric cohort and vendor evidence using the existing research HLS checks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys

import numpy as np

from fpgai_research.architecture.resolver import TargetLimits
from fpgai_research.experiments.results import summarize_case
from fpgai_research.frontend.onnx import load_onnx
from fpgai_research.toolchains.vitis_hls.package import (
    make_testbench, write_vitis_package,
)
from fpgai_research.toolchains.vitis_hls.runner import run_vitis
from ..environments.hardware_space import (
    HardwareEdit, HardwareState, RESOURCE_NAMES, compile_state, initial_state,
    native_evaluation_plan, transition,
)
from ..runtime_contract import verify_local_compiler_package
from ..timing import audited_hls_timing

SCHEMA = "fpgai-research/rl-hardware-native-case/v1"
AUDIT_SCHEMA = "fpgai-research/rl-hardware-native-audit/v1"


def _hash(data: bytes) -> str:
    return sha256(data).hexdigest()


def _regular_file(path: Path, *, max_bytes: int = 128 * 1024 * 1024) -> Path:
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size > max_bytes):
        raise ValueError(f"Missing or unsafe source/evidence file: {path}")
    return path


def _json(path: Path) -> dict:
    value = json.loads(_regular_file(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected provenance JSON object: {path}")
    return value


def _samples(state: HardwareState, samples) -> np.ndarray:
    if (not isinstance(samples, np.ndarray) or samples.dtype != np.float32
            or samples.ndim < 2 or not samples.flags.c_contiguous
            or not 1 <= samples.shape[0] <= 256 or not np.isfinite(samples).all()):
        raise ValueError("Expected 1..256 contiguous finite original float32 model samples")
    if len(state.model.inputs) != 1 or len(state.model.outputs) != 1:
        raise ValueError("Original Vitis model CSim only supports one input/output")
    expected = tuple(state.model.tensor(state.model.inputs[0]).shape)
    if samples.shape[1:] != expected:
        raise ValueError(f"Original numerical samples must be [N,{expected}], got {samples.shape}")
    return samples


def _sample_sha(samples: np.ndarray) -> str:
    return _hash(json.dumps({
        "dtype": samples.dtype.str, "shape": list(samples.shape),
        "data_sha256": _hash(samples.tobytes()),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _uncertainty(state: HardwareState) -> float:
    # Vitis HLS 2023.2 commonly rounds original target-clock XML to
    # 0.01ns. Keep 0.001ns *source* actions; require representable
    # hundredth-ns clocks for native-qualified source-pinned comparisons
    # until a vendor-specific, independently audited finer-clock protocol.
    if round(state.clock_ns, 2) != state.clock_ns:
        raise ValueError(
            "Native HLS requires a 0.01ns-canonical requested period "
            "because the original Vitis XML rounds target clocks; source "
            "RL clock actions remain continuous at 0.001ns resolution"
        )
    uncertainty = round(state.clock_ns * .27, 2)
    if not 0 < uncertainty < state.clock_ns:
        raise ValueError("Explicit native HLS clock uncertainty invalid")
    return uncertainty


def _canonical_json(value: dict) -> dict:
    """Construct JSON-native evidence before writing or comparing receipts.

    ArchitectureRequest.tile is a tuple in immutable compiler state, but
    json.dump serializes it as a list. Comparing a loaded JSON receipt to
    raw dataclasses.asdict output falsely rejects a genuine saved case.
    Canonicalize both source planning and persisted evidence identically;
    never remove a field, ignore a mismatch, or relax source-hash checks.
    """
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def planned_case(state: HardwareState, samples: np.ndarray) -> dict:
    samples = _samples(state, samples)
    source_plan = native_evaluation_plan(state)
    return _canonical_json({
        **source_plan,
        "schema": SCHEMA,
        "status": "SOURCE_READY_NOT_NATIVE_EXECUTED",
        "sample_shape": list(samples.shape),
        "sample_sha256": _sample_sha(samples),
        "explicit_clock_uncertainty_ns": _uncertainty(state),
        "original_native_evidence_expected": "E3-HLS-synthesis-report",
        "native_vitis_calls": 0,
        "reward": None,
    })


def _expected_testbench(state: HardwareState, samples: np.ndarray) -> str:
    req = state.requests[0]
    return make_testbench(
        state.model, samples, req.precision_bits, req.integer_bits,
        tolerance=state.max_absolute_error,
    )


def audit_native_case(case: str | Path, state: HardwareState, samples: np.ndarray) -> dict:
    """Independent read-only replay; no Vitis/Vivado invocation."""
    case = Path(case).expanduser()
    if case.is_symlink() or not case.is_dir():
        raise ValueError("Missing or symlinked immutable native HLS case")
    expected = planned_case(state, samples)
    if _json(case / "hardware_action.json") != expected:
        raise ValueError("Saved full hardware action/state/samples differ from original task")
    source = compile_state(state).emission
    manifest = _json(case / "manifest.json")
    if (_hash(_regular_file(case / "model.cpp").read_bytes()) != source.manifest["cpp_sha256"]
            or _regular_file(case / "model.cpp").read_text() != source.cpp
            or manifest.get("cpp_sha256") != source.manifest["cpp_sha256"]
            or manifest.get("functional_sha256") != state.functional_sha256
            or manifest.get("part") != state.part
            or manifest.get("clock_ns") != state.clock_ns
            or manifest.get("precision_bits") != state.requests[0].precision_bits
            or manifest.get("integer_bits") != state.requests[0].integer_bits
            or manifest.get("numerical_tolerance") != state.max_absolute_error
            or manifest.get("explicit_clock_uncertainty_ns") != expected["explicit_clock_uncertainty_ns"]):
        raise ValueError("Original model/clock/numeric HLS manifest source attestation mismatch")
    tb = _expected_testbench(state, samples).encode("utf-8")
    if (_hash(_regular_file(case / "tb.cpp").read_bytes()) != _hash(tb)
            or manifest.get("testbench_sha256") != _hash(tb)):
        raise ValueError("Original CSim testbench does not reproduce fixed task samples")
    # Confirm original generated Tcl/manifest and native runner receipts,
    # not a metric from an arbitrary user-authored result JSON.
    _regular_file(case / "run_hls.tcl")
    native_path = case / "vitis_invocation.json"
    native = _json(native_path) if native_path.is_file() else {}
    summary = summarize_case(str(case))
    valid_native = (
        native.get("tool") == "vitis_hls"
        and native.get("completed_successfully") is True
        and native.get("returncode") == 0
        and native.get("inputs_stable_during_execution") is True
        and summary.get("evidence") == "E3-HLS-synthesis-report"
        and summary.get("csim_passed") is True
        and summary.get("source_sha256") == source.manifest["cpp_sha256"]
        and summary.get("target_part") == state.part
        and summary.get("clock_ns") == state.clock_ns
        and summary.get("numerical_tolerance") == state.max_absolute_error
    )
    gates = []
    if not valid_native:
        gates.append("ORIGINAL_NATIVE_E3_CSIM_SOURCE_NOT_QUALIFIED")
    numeric = summary.get("csim_receipt") or {}
    error = numeric.get("max_absolute_error")
    if (type(error) not in (int, float) or not isfinite(error)
            or error > state.max_absolute_error):
        gates.append("REFERENCE_NUMERICAL_ERROR_MISSING_OR_EXCEEDED")
    metrics = summary.get("metrics") or {}
    cycles = metrics.get("latency_worst")
    if (type(cycles) is int or isinstance(cycles, str) and cycles.isdecimal()):
        cycles = int(cycles)
    else:
        cycles = None
    if cycles is None or cycles <= 0:
        gates.append("HLS_WORST_CASE_CYCLES_MISSING")
    consumed = {}
    for field, cap in zip(RESOURCE_NAMES, state.resource_caps):
        amount = metrics.get(field)
        if type(amount) is int or isinstance(amount, str) and amount.isdecimal():
            amount = int(amount)
        else:
            amount = None
        consumed[field] = amount
        if amount is None or amount > cap:
            gates.append("HLS_RESOURCE_MISSING_OR_EXCEEDED_" + field.upper())
    timing = None
    if valid_native:
        try:
            timing = audited_hls_timing(case, native, summary, state.clock_ns)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            gates.append("ORIGINAL_HLS_TIMING_UNVERIFIED_" + type(exc).__name__)
    if timing is None or timing["hls_timing_feasible"] is not True:
        gates.append("ORIGINAL_HLS_SCHEDULING_NOT_FEASIBLE")
    qualified = not gates
    return {
        "schema": AUDIT_SCHEMA,
        "state_sha256": state.state_sha256,
        "sample_sha256": expected["sample_sha256"],
        "functional_sha256": state.functional_sha256,
        "source_cpp_sha256": source.manifest["cpp_sha256"],
        "case": str(case.absolute()),
        "evidence": summary.get("evidence"),
        "native_invocation_present": bool(native),
        "original_vitis_completed_successfully": native.get("completed_successfully"),
        "csim_passed": summary.get("csim_passed"),
        "csim_max_absolute_error": error,
        "resource_caps": dict(zip(RESOURCE_NAMES, state.resource_caps)),
        "original_hls_resource_consumed": consumed,
        "hls_worst_case_cycles": cycles if qualified else None,
        "raw_hls_worst_case_cycles": cycles,
        "requested_clock_ns": state.clock_ns,
        "hls_timing": timing,
        "hls_estimated_latency_ns": cycles * state.clock_ns if qualified else None,
        "qualified": qualified,
        "gate_failures": gates,
        "hardware_physical_fmax_mhz": None,
        "board_qualified": False,
        "reward": None,
        "native_calls_by_audit": 0,
        "note": "HLS-estimated latency only. No FPGA place/route, board inference or learned RL policy.",
    }


def execute_native_case(case: str | Path, state: HardwareState, samples: np.ndarray,
                        *, vitis: str = "vitis_hls",
                        timeout_seconds: int = 7200) -> dict:
    """Explicit one-shot original Vitis. Failed directories remain immutable."""
    source_plan = planned_case(state, samples)
    case = Path(case).expanduser()
    if case.exists() or case.is_symlink():
        raise FileExistsError("Native case already exists, including failures; never overwrite")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 172800:
        raise ValueError("Native Vitis timeout must be 1..172800 seconds")
    # Critical: detect stale installed compiler BEFORE allocating the case
    # and before any licensed original HLS execution.
    verify_local_compiler_package()
    from shutil import which
    if which(vitis) is None:
        raise FileNotFoundError(f"Original Vitis HLS executable not available: {vitis}")
    artifact = compile_state(state)
    source_plan["cpp_sha256"] = artifact.emission.manifest["cpp_sha256"]
    parent = case.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise ValueError("Native case parent must not be a symlink")
    write_vitis_package(
        str(case), state.model, artifact.emission, samples,
        part=state.part, clock_ns=state.clock_ns,
        tolerance=state.max_absolute_error,
        clock_uncertainty_ns=source_plan["explicit_clock_uncertainty_ns"],
    )
    with (case / "hardware_action.json").open("x", encoding="utf-8") as out:
        json.dump(source_plan, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    # Preserve an attempted source-pinned native case even if tool raises.
    run_vitis(case, executable=vitis, timeout_seconds=timeout_seconds)
    return audit_native_case(case, state, samples)


def _load_edit_list(path: str | None) -> list[HardwareEdit]:
    if path is None:
        return []
    records = json.loads(_regular_file(Path(path).expanduser(), max_bytes=1_000_000).read_text())
    if not isinstance(records, list) or len(records) > 64:
        raise ValueError("Architecture action file must contain <=64 typed edit objects")
    edits = []
    for entry in records:
        if (not isinstance(entry, dict) or
                set(entry) - {"scope", "field", "value", "operator_index"} or
                not {"scope", "field", "value"} <= set(entry)):
            raise ValueError("Unknown or missing typed architecture action keys")
        edits.append(HardwareEdit(
            scope=entry["scope"], field=entry["field"], value=entry["value"],
            operator_index=entry.get("operator_index"),
        ))
    return edits


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--samples", required=True, help="Existing float32 .npy with [N,*input_shape]")
    parser.add_argument("--part", required=True)
    parser.add_argument("--clock-ns", required=True, type=float)
    parser.add_argument("--resource-caps", required=True, type=int, nargs=5, metavar="N")
    parser.add_argument("--max-absolute-error", required=True, type=float)
    parser.add_argument("--max-parallel-units", type=int, default=8)
    parser.add_argument("--edits-json", help="Optional source-legal sequential HardwareEdit array")
    parser.add_argument("--output", help="Fresh case for native execution; existing case for --audit")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--execute-native", action="store_true")
    selector.add_argument("--audit", action="store_true", help="Read-only independent original-case audit")
    parser.add_argument("--vitis", default="vitis_hls")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args(argv)
    try:
        if (args.execute_native or args.audit) and not args.output:
            raise ValueError("--output is required for --execute-native or --audit")
        model = load_onnx(args.onnx)
        samples_path = _regular_file(Path(args.samples).expanduser())
        samples = np.load(samples_path, allow_pickle=False)
        state = initial_state(
            model, part=args.part, clock_ns=args.clock_ns,
            resource_caps=tuple(args.resource_caps),
            max_absolute_error=args.max_absolute_error,
            limits=TargetLimits(max_parallel_units=args.max_parallel_units),
        )
        for action in _load_edit_list(args.edits_json):
            result = transition(state, action)
            if result.status != "SOURCE_ACCEPTED":
                raise ValueError("Action rejected by original compiler: " + str(result.reason))
            state = result.next
        if args.audit:
            output = audit_native_case(args.output, state, samples)
        elif args.execute_native:
            output = execute_native_case(
                args.output, state, samples, vitis=args.vitis,
                timeout_seconds=args.timeout,
            )
        else:
            output = planned_case(state, samples)
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
        return 0 if not args.audit and not args.execute_native or output["qualified"] else 3
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"RL native HLS stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
