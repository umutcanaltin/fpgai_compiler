"""Read-only inventory of a prepared HLS case versus true FPGA deployment.

This deliberately does NOT infer that a .bit/.xclbin is valid or compatible
with the selected board. Native HLS and OOC Vivado artifacts are insufficient
to certify any on-device inference flow.

Usage:
  python scripts/check_fpga_deployment.py --case build/native_demo/hls_case --target kv260
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGETS = ("kv260", "kr260", "pynq", "alveo", "generic")


def inspect_case(path: str | Path, target: str) -> dict:
    if target not in TARGETS:
        raise ValueError("Target must be kv260, kr260, pynq, alveo, or generic")
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("HLS case must be a directory")

    required = {
        "generated_hls_cpp": (root/"model.cpp").is_file(),
        "numeric_testbench": (root/"tb.cpp").is_file(),
        "vitis_hls_script": (root/"run_hls.tcl").is_file(),
        "source_manifest": (root/"manifest.json").is_file(),
        "vitis_synthesis_verilog": (root/"fpgai_hls_project"/"solution1"/"syn"/"verilog").is_dir(),
        "vivado_ooc_script": (root/"vivado"/"run_vivado.tcl").is_file(),
        "vivado_ooc_timing_report": (root/"vivado"/"reports"/"timing_summary.rpt").is_file(),
        "vivado_ooc_utilization_report": (root/"vivado"/"reports"/"utilization.rpt").is_file(),
        "vivado_ooc_drc_report": (root/"vivado"/"reports"/"drc.rpt").is_file(),
    }

    if required["source_manifest"]:
        doc = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
        if not isinstance(doc,dict):
            raise ValueError("HLS manifest must be an object")
        # Only public target data, never full model constants or generated HDL.
        part = doc.get("part")
        clock_ns = doc.get("clock_ns")
    else:
        part = clock_ns = None

    missing = [key for key, exists in required.items() if not exists]
    next_steps = [
        "Inspect original CSim and synthesis reports before asserting numerical or HLS qualification.",
        "Package appropriate HLS IP or kernel and verify the actual control/data interfaces.",
        "Integrate board-specific clocks, resets, PS/PL or PCIe interconnect, memory/transport and constraints.",
        "Complete independently qualified full-design implementation and generate the correct device image.",
        "Write and verify compatible runtime/host transfers, then compare measured board outputs with reference outputs.",
    ]
    if target in ("kv260", "kr260"):
        next_steps.append(
            "Prepare board-matching Kria platform/image, device-tree/driver and PS↔PL data transport."
        )
    if target=="pynq":
        next_steps.append(
            "Build a complete supported PYNQ overlay (.bit plus matching .hwh) and device-specific software."
        )
    if target=="alveo":
        next_steps.append(
            "Package a compatible Vitis kernel and build a platform-matched .xclbin and XRT host application."
        )

    return {
        "schema":"fpgai-research/fpga-deployment-inventory/v1",
        "target":target,
        "case":str(root),
        "source_manifest_part":part,
        "source_manifest_clock_ns":clock_ns,
        "artifacts":required,
        "missing_artifacts":missing,
        "out_of_context_evidence_is_not_board_qualification":True,
        "complete_board_runtime_verified":False,
        "board_bitstream_generated_by_this_script":False,
        "vendor_commands_executed":False,
        "hardware_programmed":False,
        "next_steps":next_steps,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case",required=True)
    ap.add_argument("--target",choices=TARGETS,default="generic")
    args = ap.parse_args(argv)
    data=inspect_case(args.case,args.target)
    print(json.dumps(data,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
