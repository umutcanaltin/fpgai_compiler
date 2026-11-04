"""One-command, local ONNX -> HLS -> Vivado OOC onboarding workflow.

Reuses the canonical compiler preparation, native runners and evidence auditor.
No network access, GitHub Actions, training, device programming or board image
generation. Native tools are never executed without --execute-native.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence

BOARD_PARTS = {"kv260": "xck26-sfvc784-2LV-c"}
STAGES = ("source", "hls", "ooc")
SUMMARY_SCHEMA = "fpgai-research/local-onboarding/v1"


@dataclass(frozen=True)
class VendorTool:
    executable: str
    settings_script: str | None


def resolve_vendor_tool(name: str, supplied: str | None) -> VendorTool:
    """Find an exact AMD executable from a binary, product root or PATH.

    Do not guess which release to use when multiple versions are installed.
    Inspecting paths never sources or executes the toolchain.
    """
    if name not in ("vitis_hls", "vivado"):
        raise ValueError("Unsupported tool")
    if not supplied:
        binary = shutil.which(name)
        if binary is None:
            raise FileNotFoundError(
                f"{name} not on PATH; pass --{'vitis' if name == 'vitis_hls' else 'vivado'} "
                "with an absolute AMD product installation or executable path"
            )
        return VendorTool(str(Path(binary).resolve()), None)

    raw = Path(supplied).expanduser()
    if raw.is_symlink() and not raw.exists():
        raise FileNotFoundError(f"Broken toolchain path: {raw}")
    if not raw.exists():
        raise FileNotFoundError(f"Toolchain path does not exist: {raw}")
    raw = raw.resolve()
    if raw.is_file() and raw.name == "settings64.sh":
        home = raw.parent
        candidates = [home / "bin" / name]
        settings = raw
    elif raw.is_file():
        if raw.name not in (name, name + ".exe"):
            raise ValueError(f"Expected {name} executable, got {raw.name}")
        candidates = [raw]
        settings = raw.parent.parent / "settings64.sh"
    elif raw.is_dir():
        product = "Vitis" if name == "vitis_hls" else "Vivado"
        candidates = [raw / "bin" / name, raw / name]
        candidates.extend(sorted((raw / product).glob("*/bin/" + name))
                          if (raw / product).is_dir() else [])
        settings = None
    else:
        raise ValueError(f"Unsupported toolchain path: {raw}")

    found = sorted({p.resolve() for p in candidates if p.is_file() and os.access(p, os.X_OK)})
    if not found:
        raise FileNotFoundError(f"Cannot locate executable {name} under {raw}")
    if len(found) != 1:
        raise ValueError(f"Several installations of {name} found under {raw}; select a version directory")
    binary = found[0]
    if settings is None:
        settings = binary.parent.parent / "settings64.sh"
    return VendorTool(str(binary), str(settings.resolve()) if settings.is_file() else None)


def vendor_environment(tools: Sequence[VendorTool]) -> dict[str, str] | None:
    """Load trusted AMD settings scripts once and return a subprocess env.

    The scripts are explicitly supplied/located on the user's workstation.
    Never serialize the resulting environment: it can include credentials.
    Source in a child Bash process, not in the Python process or an rc file.
    """
    scripts = list(dict.fromkeys(x.settings_script for x in tools if x.settings_script))
    if not scripts:
        return None
    bash = shutil.which("bash")
    if bash is None:
        raise FileNotFoundError("Bash is required for AMD settings64.sh initialization")
    # Positional arguments avoid shell interpolation of user-supplied paths.
    command = 'set -e\nfor script in "$@"; do source "$script" >/dev/null; done\nenv -0'
    completed = subprocess.run(
        [bash, "-c", command, "fpgai-vendor-setup", *scripts],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=90,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "AMD settings64.sh initialization failed; inspect the installed "
            "vendor scripts and licensing environment on this workstation"
        )
    if len(completed.stdout) > 4_000_000:
        raise ValueError("Vendor environment exceeds the supported size")
    result: dict[str, str] = {}
    for entry in completed.stdout.split(b"\0"):
        if not entry:
            continue
        name, sep, value = entry.partition(b"=")
        if not sep:
            raise ValueError("Invalid environment entry from AMD setup")
        key = os.fsdecode(name)
        if not key or "=" in key:
            raise ValueError("Invalid environment key from AMD setup")
        result[key] = os.fsdecode(value)
    if not result.get("PATH"):
        raise ValueError("AMD setup did not provide PATH")
    return result


def _validate_model(path: Path) -> tuple[int, ...]:
    from fpgai_research.frontend.onnx import load_onnx

    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    model = load_onnx(str(path))
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError(
            "This onboarding workflow currently supports one static runtime "
            "input tensor and one output tensor"
        )
    shape = model.tensor(model.inputs[0]).shape
    if not shape or any(type(d) is not int or d < 1 for d in shape):
        raise ValueError("An ONNX model with positive static input dimensions is required")
    return tuple(shape)


def _select_part(args: argparse.Namespace) -> str:
    from fpgai_research.toolchains.vivado.project import make_vivado_tcl

    part = args.part or BOARD_PARTS.get(args.board)
    if part is None:
        if args.stage == "source":
            return BOARD_PARTS["kv260"]  # Source generation only; no hardware claim.
        raise ValueError("Select --board kv260 or provide the exact --part for native workflows")
    # Use the existing canonical part/clock validation, never interpolate raw Tcl.
    make_vivado_tcl(part=part, clock_ns=args.clock_ns)
    return part


def _create_synthetic_input(filename: Path, shape: tuple[int, ...], seed: int) -> None:
    import numpy as np

    values = np.random.default_rng(seed).uniform(-0.25, 0.25, size=shape).astype(np.float32)
    with filename.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)


def _case_summary(case: Path, *, stage: str, state: str, sample_kind: str,
                  message: str, part: str) -> dict:
    from fpgai_research.experiments.results import summarize_case

    evidence = summarize_case(str(case))
    vivado = evidence.get("vivado_metrics") or {}
    timing = vivado.get("timing") or {}
    drc = vivado.get("drc") or {}
    return {
        "schema": SUMMARY_SCHEMA,
        "requested_stage": stage,
        "state": state,
        "message": message,
        "case_directory": str(case),
        "part": part,
        "sample_kind": sample_kind,
        "evidence_level": evidence["evidence"],
        "csim_passed": evidence.get("csim_passed"),
        "hls_metrics": evidence.get("metrics"),
        "timing_met": timing.get("timing_met"),
        "worst_negative_slack_ns": timing.get("wns_ns"),
        "drc_clean": drc.get("clean"),
        "warnings": evidence.get("warnings", []),
        "routed_ooc": evidence["evidence"] == "E4-Vivado-implementation",
        "board_bitstream_generated": False,
        "device_programmed": False,
    }


def _write_summary(case: Path, payload: dict) -> None:
    # The case is create-only; this single status file is the current
    # invocation's user summary, not a replacement for original receipts.
    (case / "user_run_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _stage_roms(case: Path) -> None:
    """Copy original source-attested Vitis ROM files for Vivado readmem calls."""
    from fpgai_research.toolchains.vivado.rtl_inputs import (
        compare_staged_roms, inventory_vitis_rtl,
    )

    inventory = inventory_vitis_rtl(case)
    if not inventory["rtl_sha256"]:
        raise ValueError("Vitis did not produce supported original Verilog RTL")
    source = case / "fpgai_hls_project" / "solution1" / "syn" / "verilog"
    destination = case / "vivado"
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("Invalid prepared Vivado directory")
    for name in sorted(inventory["rom_sha256"]):
        target = destination / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to replace staged ROM: {target}")
        shutil.copyfile(source / name, target)
    check = compare_staged_roms(destination, inventory["rom_sha256"])
    if not check["matches_original_roms"]:
        raise ValueError("Copied Vivado ROM files do not match source-attested HLS ROMs")


def _plan(args: argparse.Namespace, part: str, shape: tuple[int, ...]) -> dict:
    details = {
        "schema": SUMMARY_SCHEMA,
        "model": str(Path(args.onnx).expanduser().resolve()),
        "input_shape": list(shape),
        "output": str(Path(args.output).expanduser().absolute()),
        "part": part,
        "clock_ns": args.clock_ns,
        "tolerance": args.tolerance,
        "stage": args.stage,
        "uses_synthetic_input": args.input is None,
        "allow_verified_readmem_post_absence": bool(
            getattr(args, "allow_verified_readmem_post_absence", False)),
        "vendor_tools_executed": False,
        "board_bitstream_generated": False,
        "steps": ["ONNX -> Model IR -> HLS C++ -> generated numerical testbench"],
    }
    if args.stage in ("hls", "ooc"):
        details["steps"].append("Vitis HLS C simulation + synthesis; verify E2/E3")
    if args.stage == "ooc":
        details["steps"].append("Vivado out-of-context place/route + timing/DRC; verify E4")
    details["not_covered"] = (
        "Board bitstream, AXI/PS/DMA integration, device programming and "
        "inference on a physical FPGA require separate board-specific work"
    )
    return details


def run(args: argparse.Namespace) -> int:
    from fpgai_research.toolchains.workflow import main as canonical_prepare
    from fpgai_research.toolchains.vitis_hls.runner import run_vitis
    from fpgai_research.toolchains.vivado.runner import run_vivado
    from fpgai_research.experiments.results import summarize_case

    if not 1 <= args.hls_timeout <= 172800 or not 1 <= args.vivado_timeout <= 172800:
        raise ValueError("Vendor tool timeout must be between 1 and 172800 seconds")
    if not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("Sample seed must be between 0 and 2**32 - 1")
    if args.labels and not args.input:
        raise ValueError("--labels requires a real --input sample file")
    if args.execute_native and args.stage == "source":
        raise ValueError("--execute-native is only valid for hls/ooc stages")
    if (getattr(args, "allow_verified_readmem_post_absence", False)
            and args.stage != "ooc"):
        raise ValueError("--allow-verified-readmem-post-absence requires --stage ooc")

    shape = _validate_model(Path(args.onnx).expanduser())
    part = _select_part(args)
    plan = _plan(args, part, shape)
    if args.plan or (args.stage != "source" and not args.execute_native):
        print(json.dumps({**plan, "action": "PLAN_ONLY_NO_FILES_CREATED"}, indent=2))
        return 0

    case = Path(args.output).expanduser().absolute()
    if case.exists() or case.is_symlink():
        raise FileExistsError(
            f"Output already exists: {case}. Select a NEW output directory "
            "to protect original experiment receipts"
        )
    for filename in (args.architecture, args.input, args.labels):
        if filename and not Path(filename).expanduser().is_file():
            raise FileNotFoundError(f"Input not found: {filename}")
    tools: list[VendorTool] = []
    if args.execute_native:
        tools.append(resolve_vendor_tool("vitis_hls", args.vitis))
        if args.stage == "ooc":
            tools.append(resolve_vendor_tool("vivado", args.vivado))
    # Do not source any vendor setup scripts until this opt-in native action.
    environment = vendor_environment(tools) if tools else None
    case.parent.mkdir(parents=True, exist_ok=True)
    sample_kind = "provided_float32_npy" if args.input else "synthetic_smoke_sample"
    sample_file = Path(args.input).expanduser() if args.input else None
    with tempfile.TemporaryDirectory(prefix=".fpgai-onboarding-", dir=case.parent) as tmp:
        if sample_file is None:
            sample_file = Path(tmp) / "sample.npy"
            _create_synthetic_input(sample_file, shape, args.seed)
        prepare_argv = [
            "prepare", "--onnx", str(Path(args.onnx).expanduser()),
            "--input", str(sample_file), "--case", str(case), "--part", part,
            "--clock-ns", str(args.clock_ns), "--tolerance", str(args.tolerance),
        ]
        if args.architecture:
            prepare_argv.extend(("--architecture", args.architecture))
        if args.labels:
            prepare_argv.extend(("--labels", args.labels))
        print("[FPGAI] Preparing ONNX, Model IR, generated HLS source and testbench", flush=True)
        result = canonical_prepare(prepare_argv)
        if result != 0:
            raise RuntimeError(f"Canonical compiler preparation failed: {result}")
        if args.input is None:
            # Copy only locally generated public-style test data; user models
            # and provided datasets are never copied to an additional location.
            shutil.copyfile(sample_file, case / "synthetic_sample.npy")

    if args.stage == "source":
        summary = _case_summary(
            case, stage=args.stage, state="SOURCE_READY", sample_kind=sample_kind,
            message="Generated HLS C++; no vendor program was invoked", part=part,
        )
        _write_summary(case, summary)
        print(json.dumps(summary, indent=2))
        return 0

    print("[FPGAI] Running licensed local Vitis HLS", flush=True)
    vitis = run_vitis(
        case, executable=tools[0].executable,
        timeout_seconds=args.hls_timeout, environment=environment,
    )
    evidence = summarize_case(str(case))
    if (not vitis.get("completed_successfully")
            or evidence["evidence"] != "E3-HLS-synthesis-report"
            or evidence.get("csim_passed") is not True):
        summary = _case_summary(
            case, stage=args.stage, state="HLS_NOT_QUALIFIED",
            sample_kind=sample_kind,
            message="Vitis invocation, numerical C simulation or HLS evidence failed; "
                    "Vivado was not started. Inspect the original logs and receipt.",
            part=part,
        )
        _write_summary(case, summary)
        print(json.dumps(summary, indent=2))
        return 2

    if args.stage == "hls":
        summary = _case_summary(
            case, stage=args.stage, state="HLS_QUALIFIED",
            sample_kind=sample_kind,
            message="Attested Vitis C simulation and synthesis passed; "
                    "this does not establish routed timing or board execution.",
            part=part,
        )
        _write_summary(case, summary)
        print(json.dumps(summary, indent=2))
        return 0

    _stage_roms(case)
    print("[FPGAI] Running local Vivado out-of-context implementation", flush=True)
    vivado = run_vivado(
        case, executable=tools[1].executable,
        timeout_seconds=args.vivado_timeout, environment=environment,
        allow_verified_readmem_post_absence=bool(
            getattr(args, "allow_verified_readmem_post_absence", False)),
    )
    final = summarize_case(str(case))
    routed = final["evidence"] == "E4-Vivado-implementation"
    physical = final.get("vivado_metrics") or {}
    timing_met = (physical.get("timing") or {}).get("timing_met") is True
    drc_clean = (physical.get("drc") or {}).get("clean") is True
    qualified = bool(vivado.get("completed_successfully") and routed
                     and timing_met and drc_clean and final.get("csim_passed") is True)
    summary = _case_summary(
        case, stage=args.stage,
        state="OOC_QUALIFIED" if qualified else "OOC_NOT_QUALIFIED",
        sample_kind=sample_kind,
        message=("Attested routed, timing-clean, DRC-clean model-only implementation; "
                 "no board bitstream generated" if qualified else
                 "Vivado execution or evidence, timing or DRC qualification did not pass; "
                 "inspect reports, logs and warnings"),
        part=part,
    )
    witness = vivado.get("verified_readmem_post_absence_witness") or {}
    staged_after = vivado.get("staged_rom_input_integrity_after") or {}
    summary["vivado_execution_diagnostics"] = {
        "vendor_exit_code": vivado.get("returncode"),
        "vendor_receipt_completed_successfully": vivado.get("completed_successfully"),
        "source_inputs_unchanged": vivado.get("inputs_stable_during_execution"),
        "original_hls_roms_unchanged": vivado.get("rom_inputs_stable_during_execution"),
        "staged_roms_unchanged_after_run":
            vivado.get("staged_rom_inputs_stable_during_execution"),
        "missing_staged_roms_after_run": staged_after.get("missing_original_roms", []),
        "modified_staged_roms_after_run": staged_after.get("modified_original_roms", []),
        "unattributed_staged_roms_after_run": staged_after.get("unattributed_rom_files", []),
        "verified_vendor_read_after_staged_rom_absence":
            witness.get("qualified_postrun_absence"),
        "postrun_absence_witness_reason": witness.get("reason"),
        "postrun_absence_witness_used":
            vivado.get("allow_verified_readmem_post_absence"),
    }
    _write_summary(case, summary)
    print(json.dumps(summary, indent=2))
    return 0 if qualified else 3


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local ONNX-to-HLS/Vivado out-of-context pipeline; no manual vendor GUI"
    )
    parser.add_argument("--onnx", required=True, help="Static ONNX neural-network model")
    parser.add_argument("--output", required=True, help="NEW local case directory")
    parser.add_argument("--input", help="Optional representative float32 .npy sample(s)")
    parser.add_argument("--labels", help="Optional class labels .npy (requires --input)")
    parser.add_argument("--architecture", help="Optional operator mapping JSON")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--board", choices=sorted(BOARD_PARTS),
                        help="Verified FPGA part preset; not a board bitstream")
    target.add_argument("--part", help="Exact target FPGA part identifier")
    parser.add_argument("--clock-ns", type=float, default=10.0)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=197,
                        help="Deterministic synthetic smoke input seed")
    parser.add_argument("--stage", choices=STAGES, default="ooc",
                        help="source (no vendors), hls, or routed ooc (default)")
    parser.add_argument("--vitis", help="Vitis product folder, settings64.sh or vitis_hls")
    parser.add_argument("--vivado", help="Vivado product folder, settings64.sh or vivado")
    parser.add_argument("--hls-timeout", type=int, default=7200)
    parser.add_argument("--vivado-timeout", type=int, default=14400)
    parser.add_argument("--execute-native", action="store_true",
                        help="Explicitly authorize local AMD tool execution")
    parser.add_argument("--allow-verified-readmem-post-absence",
                        action="store_true",
                        help="For a NEW OOC run only, opt into the strict source-and-log-hash "
                             "verified Vivado post-run ROM read witness; never retroactive")
    parser.add_argument("--plan", action="store_true",
                        help="Read-only plan; never execute vendors or write a case")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"FPGAI onboarding error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
