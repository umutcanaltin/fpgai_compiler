"""Explicit experimental PYNQ-Z2 board DMA + bitstream builder.

This is not the earlier OOC flow. Generates a *new*, isolated project from
source-attested FPGAI HLS C++, reruns vendor CSim/CSynth, exports an AXI IP,
constructs a processor/DMA block design and (if authorized) attempts a
full-board .bit/.hwh build. Hardware inference needs a separate on-device run.

Only PYNQ-Z2 is implemented; KV260's MPSoC/carrier/Linux platform cannot be
silently substituted. Neither external board part nor system timing is guessed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from fpgai_research.toolchains.board.bridge import (
    bridge_cpp, bridge_testbench_cpp, bridge_hls_tcl, validate_tensor_contract,
)
from fpgai_research.toolchains.board.project import (
    PYNQ_Z2_PART, PYNQ_DMA_CLOCK_NS, pynq_z2_board_tcl,
)
from fpgai_research.toolchains.onboarding import resolve_vendor_tool, vendor_environment


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or unsafe original case document: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path.name}")
    return value


def _verify_original_case(case: Path, onnx: Path) -> dict:
    from fpgai_research.experiments.results import summarize_case
    from fpgai_research.toolchains.weights_audit import audit_weight_constants

    manifest = _read_json(case / "manifest.json")
    source = case / "model.cpp"
    if source.is_symlink() or not source.is_file():
        raise ValueError("Missing or symlinked original model.cpp")
    if manifest.get("cpp_sha256") != _hash(source):
        raise ValueError("Original source differs from its source-attested manifest")
    evidence = summarize_case(str(case))
    if (evidence.get("evidence") not in
            ("E3-HLS-synthesis-report", "E4-Vivado-implementation")
            or evidence.get("csim_passed") is not True):
        raise ValueError("Original case must have source-attested E3+ and passing numerical CSim")
    audited = audit_weight_constants(onnx, case)
    if audited.get("constant_literals_match_imported_onnx_values") is not True:
        raise ValueError("Original ONNX parameters failed exact source constant audit")
    validate_tensor_contract(manifest)
    return manifest


def _stop_group(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def _run_vendor(executable: str, args: list[str], cwd: Path,
                environment: dict[str, str] | None, timeout: int,
                label: str) -> dict:
    if label not in ("vitis_board", "vivado_board"):
        raise ValueError("Unknown board vendor stage")
    if args != (["-f", "run_dma_hls.tcl"] if label == "vitis_board" else
                ["-mode", "batch", "-source", "run_board.tcl", "-notrace"]):
        raise ValueError("Only generated, explicitly defined board Tcl can execute")
    if not 1 <= timeout <= 172800:
        raise ValueError("Board native timeout outside permitted range")
    logs = cwd / "logs"
    logs.mkdir(exist_ok=True)
    out_path, err_path = logs / "stdout.log", logs / "stderr.log"
    env = os.environ.copy()
    if environment:
        env.update(environment)
    env["LC_ALL"] = "C"
    started = datetime.now(timezone.utc).isoformat()
    source_names = (("model.cpp", "fpgai_dma.cpp", "fpgai_dma_tb.cpp", "run_dma_hls.tcl")
                    if label == "vitis_board" else ("run_board.tcl",))
    snapshots = {name: _hash(cwd / name) for name in source_names}
    start = time.monotonic()
    with out_path.open("wb") as out, err_path.open("wb") as err:
        proc = subprocess.Popen(
            [executable, *args], cwd=cwd, env=env, stdout=out, stderr=err,
            start_new_session=True,
        )
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_group(proc)
        except BaseException:
            _stop_group(proc)
            raise
    after = {
        name: _hash(cwd / name) if (cwd / name).is_file() else None
        for name in source_names
    }
    source_unchanged = snapshots == after
    version = None
    try:
        proc_version = subprocess.run(
            [executable, "-version"], cwd=cwd, env=env,
            text=True, capture_output=True, timeout=30, check=False,
        )
        if proc_version.returncode == 0:
            version = (proc_version.stdout or proc_version.stderr)[:2048]
    except (OSError, subprocess.TimeoutExpired):
        pass
    receipt = {
        "schema": "fpgai-research/board-native-invocation/v1",
        "tool_version_output": version,
        "input_hashes_before": snapshots,
        "input_hashes_after": after,
        "source_inputs_unchanged": source_unchanged,
        "tool": label,
        "executable": str(Path(executable).resolve()),
        "argv": [executable, *args],
        "started_utc": started,
        "duration_seconds": round(time.monotonic() - start, 2),
        "returncode": None if timed_out else proc.returncode,
        "timed_out": timed_out,
        "stdout_sha256": _hash(out_path),
        "stderr_sha256": _hash(err_path),
        "script_sha256": _hash(cwd / args[1] if label == "vitis_board"
                              else cwd / "run_board.tcl"),
        "completed_successfully": (
            not timed_out and proc.returncode == 0 and source_unchanged
        ),
    }
    (cwd / "invocation.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
    )
    return receipt


def _has_marker(path: Path, marker: str) -> bool:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 64*1024*1024:
        return False
    return marker.encode() in path.read_bytes()


def _write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


def run(args: argparse.Namespace) -> int:
    if args.board != "pynq-z2":
        raise ValueError(
            "KV260 is not yet a supported bitstream target: it needs a distinct "
            "Kria K26 MPSoC board preset, Linux runtime and verified PS/PL design"
        )
    if args.clock_ns != PYNQ_DMA_CLOCK_NS:
        raise ValueError("First PYNQ-Z2 deployment profile uses an explicit 100 MHz 10 ns clock")
    if not args.board_part:
        raise ValueError("Specify --board-part from your installed Vivado PYNQ-Z2 board files")
    script = pynq_z2_board_tcl(
        board_part=args.board_part, part=PYNQ_Z2_PART, clock_ns=args.clock_ns,
    )
    source = Path(args.case).expanduser().resolve(strict=True)
    onnx = Path(args.onnx).expanduser().resolve(strict=True)
    if source.is_symlink() or not source.is_dir() or not onnx.is_file():
        raise ValueError("Original case/ONNX source must be existing regular files")
    manifest = _verify_original_case(source, onnx)
    nin, nout, width, integer = validate_tensor_contract(manifest)
    plan = {
        "schema": "fpgai-research/board-dma-build/v1",
        "board": args.board,
        "board_part": args.board_part,
        "part": PYNQ_Z2_PART,
        "clock_ns": args.clock_ns,
        "input_elements": nin,
        "output_elements": nout,
        "input_shape": manifest["input_shape"],
        "output_shape": manifest["output_shape"],
        "payload_bits": width,
        "integer_bits": integer,
        "transport_bits_per_element": 32,
        "transfer": "one fixed-size frame per HLS ap_ctrl_hs invocation, AXI DMA simple mode",
        "weights": "compile-time ONNX constants preserved in original model.cpp",
        "output": str(Path(args.output).expanduser().absolute()),
        "stage": args.stage,
        "original_case": str(source),
        "hardware_programmed": False,
        "board_inference_verified": False,
    }
    if args.plan or not args.execute_native:
        print(json.dumps({**plan, "action": "PLAN_ONLY_NO_FILES_CREATED"}, indent=2))
        return 0
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to overwrite board experiment: " + str(output))
    if args.hls_timeout < 1 or args.vivado_timeout < 1:
        raise ValueError("Vendor timeout must be positive")
    vitis = resolve_vendor_tool("vitis_hls", args.vitis)
    vivado = resolve_vendor_tool("vivado", args.vivado) if args.stage == "bitstream" else None
    env = vendor_environment([tool for tool in (vitis, vivado) if tool is not None])
    output.mkdir(parents=True, exist_ok=False)
    hls_dir = output / "dma_hls"
    hls_dir.mkdir()
    shutil.copyfile(source / "model.cpp", hls_dir / "model.cpp")
    (hls_dir / "fpgai_dma.cpp").write_text(bridge_cpp(manifest))
    (hls_dir / "fpgai_dma_tb.cpp").write_text(bridge_testbench_cpp(manifest))
    (hls_dir / "run_dma_hls.tcl").write_text(
        bridge_hls_tcl(part=PYNQ_Z2_PART, clock_ns=args.clock_ns)
    )
    _write_json(output / "board_plan.json", {
        **plan,
        "onnx_sha256": _hash(onnx),
        "model_cpp_sha256": _hash(hls_dir / "model.cpp"),
        "bridge_cpp_sha256": _hash(hls_dir / "fpgai_dma.cpp"),
        "original_manifest_sha256": _hash(source / "manifest.json"),
        "result": "SOURCE_READY_VENDOR_UNVERIFIED",
    })
    receipt = _run_vendor(
        vitis.executable, ["-f", "run_dma_hls.tcl"], hls_dir,
        env, args.hls_timeout, "vitis_board",
    )
    ip_component = hls_dir / "fpgai_dma_hls_project" / "solution1" / "impl" / "ip" / "component.xml"
    if (not receipt["completed_successfully"]
            or not ip_component.is_file()
            or not _has_marker(hls_dir / "logs" / "stdout.log", "FPGAI_BOARD_DMA_CSIM_PASS")):
        report = {**plan, "state": "BOARD_HLS_NOT_QUALIFIED",
                  "hls_exit_code": receipt["returncode"],
                  "csim_marker_found": _has_marker(
                      hls_dir / "logs" / "stdout.log", "FPGAI_BOARD_DMA_CSIM_PASS"),
                  "exported_ip_component_found": ip_component.is_file()}
        _write_json(output / "board_build_summary.json", report)
        print(json.dumps(report, indent=2))
        return 3
    if args.stage == "hls":
        report = {**plan, "state": "BOARD_IP_EXPORTED_NOT_BITSTREAM",
                  "hls_exit_code": 0,
                  "exported_ip_component_sha256": _hash(ip_component)}
        _write_json(output / "board_build_summary.json", report)
        print(json.dumps(report, indent=2))
        return 0

    (output / "run_board.tcl").write_text(script)
    bdrun = _run_vendor(
        vivado.executable,
        ["-mode", "batch", "-source", "run_board.tcl", "-notrace"],
        output, env, args.vivado_timeout, "vivado_board",
    )
    artifacts = output / "board_artifacts"
    bit = artifacts / "fpgai_dma.bit"
    hwh = artifacts / "fpgai_dma.hwh"
    from fpgai_research.toolchains.vivado.report import (
        _drc_summary, _worst_negative_slack, _complete_route,
    )
    drc = _drc_summary(artifacts / "drc.rpt")
    route = artifacts / "route_status.rpt"
    route_complete = False
    if route.is_file() and not route.is_symlink():
        try:
            route_complete = _complete_route(route.read_text()) is True
        except ValueError:
            pass
    timing = None
    timing_path = artifacts / "timing_summary.rpt"
    if timing_path.is_file() and not timing_path.is_symlink():
        try:
            timing = _worst_negative_slack(timing_path.read_text())
        except ValueError:
            pass
    outputs_exist = (bit.is_file() and hwh.is_file()
                     and not bit.is_symlink() and not hwh.is_symlink()
                     and bit.stat().st_size > 0 and hwh.stat().st_size > 0)
    # Even advisory-only full board images require an explicit user opt-in.
    # A clean board implementation is never inferred from an OOC model receipt.
    from fpgai_research.toolchains.vivado.advisory_observation import (
        ALLOWED_ADVISORY_RULES,
    )
    core_ok = bool(
        bdrun["completed_successfully"] and outputs_exist and route_complete
        and _has_marker(output / "logs" / "stdout.log", "FPGAI_BOARD_ARTIFACTS_EXPORTED")
        and timing is not None and timing >= 0
        and drc.get("summary_verified") is True
    )
    rules = drc.get("rules") or {}
    advisory_ok = (
        core_ok
        and drc.get("clean") is False
        and drc.get("critical_warnings") == 0
        and drc.get("unconstrained_io") is False
        and bool(rules)
        and set(rules).issubset(ALLOWED_ADVISORY_RULES)
        and type(drc.get("violations")) is int
        and sum(int(v.get("violations", 0)) for v in rules.values())
            == drc["violations"]
        and all(v.get("severity") == "Warning" and
                type(v.get("violations")) is int and v["violations"] > 0
                for v in rules.values())
    )
    strict_clean = core_ok and drc.get("clean") is True
    warnings_accepted = advisory_ok and args.accept_board_advisories
    qualified = strict_clean or warnings_accepted
    state = ("BOARD_BITSTREAM_DRC_CLEAN_UNTESTED_ON_DEVICE" if strict_clean
             else "BOARD_BITSTREAM_WITH_ADVISORIES_NOT_DEVICE_TESTED"
             if warnings_accepted else "BOARD_NOT_QUALIFIED")
    report = {
        **plan,
        "state": state,
        "vivado_exit_code": bdrun["returncode"],
        "timing_slack_ns": timing,
        "routing_verified_complete": route_complete,
        "board_report_sha256": {
            name: _hash(artifacts / name)
            for name in ("route_status.rpt", "timing_summary.rpt",
                         "drc.rpt", "utilization.rpt")
            if (artifacts / name).is_file()
            and not (artifacts / name).is_symlink()
        },
        "drc_clean": drc.get("clean"),
        "drc_violations": drc.get("violations"),
        "drc_critical_warnings": drc.get("critical_warnings"),
        "drc_unconstrained_io": drc.get("unconstrained_io"),
        "drc_rule_counts": drc.get("rules", {}),
        "warning_only_eligible": bool(advisory_ok),
        "user_accepted_board_advisories": args.accept_board_advisories,
        "strict_board_drc_clean_qualified": bool(strict_clean),
        "bitstream_and_hwh_present": outputs_exist,
        "bitstream_sha256": _hash(bit) if outputs_exist else None,
        "hwh_sha256": _hash(hwh) if outputs_exist else None,
        "bitstream_generated": outputs_exist,
        "on_device_inference_verified": False,
        "note": "Board bitstream is not a verified on-device inference result; run board host validation",
    }
    _write_json(output / "board_build_summary.json", report)
    if qualified:
        # A standalone PYNQ-side runtime does not require the compiler Python
        # package, ONNX or any GitHub connectivity on the board.
        shutil.copyfile(
            Path(__file__).resolve().parent / "pynq_runtime.py",
            artifacts / "fpgai_pynq_runtime.py",
        )
        shutil.copyfile(
            output / "board_build_summary.json",
            artifacts / "board_build_summary.json",
        )
    print(json.dumps(report, indent=2))
    return 0 if qualified else 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="Exact original ONNX for source audit")
    parser.add_argument("--case", required=True, help="Existing source-attested HLS case")
    parser.add_argument("--output", required=True, help="New, create-only board build directory")
    parser.add_argument("--board", choices=("pynq-z2", "kv260"), required=True)
    parser.add_argument("--board-part", required=True,
                        help="Exact board_part installed in Vivado; no guessed board presets")
    parser.add_argument("--clock-ns", type=float, default=PYNQ_DMA_CLOCK_NS)
    parser.add_argument("--stage", choices=("hls", "bitstream"), default="bitstream")
    parser.add_argument("--vitis", help="Original Vitis HLS executable or product folder")
    parser.add_argument("--vivado", help="Original Vivado executable or product folder")
    parser.add_argument("--hls-timeout", type=int, default=7200)
    parser.add_argument("--vivado-timeout", type=int, default=14400)
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--accept-board-advisories", action="store_true",
                        help="Explicit opt-in for full board implementation with only "
                             "source-listed warning-level DSP/RTSTAT advisories; "
                             "not strict DRC-clean or hardware inference")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"FPGAI board build blocked: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
