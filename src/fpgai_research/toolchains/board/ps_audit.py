"""Read-only PYNQ-Z2 PS/DDR preset provenance and native-warning correlation.

Compares an installed third-party board preset with the public Xilinx/PYNQ
Pynq-Z2 hardware source as a reference. Neither is magically validated for
a user's actual PCB or running Linux FSBL. The command NEVER programs an FPGA,
changes PS parameters, modifies Vivado source, or qualifies a bitstream.

Reference: https://github.com/Xilinx/PYNQ/blob/master/boards/Pynq-Z2/petalinux_bsp/hardware_project/pynqz2.tcl
PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_0..3: 0.040, 0.058, -0.009, -0.033 ns.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
from xml.etree import ElementTree as ET


REFERENCE = (0.040, 0.058, -0.009, -0.033)
REFERENCE_SOURCE = (
    "https://github.com/Xilinx/PYNQ/blob/master/boards/Pynq-Z2/"
    "petalinux_bsp/hardware_project/pynqz2.tcl"
)
BOARD_PART = "tul.com.tw:pynq-z2:part0:1.0"
SILICON = "xc7z020clg400-1"
_DDR_LOG_RE = re.compile(
    r"^\s*CRITICAL WARNING:\s*\[(PSU-\d+)\].*?"
    r"PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_([0-3])"
    r"\s+has negative value\s+(-?[0-9]+(?:\.[0-9]+)?)\b",
    re.IGNORECASE,
)


def _file(path: Path, *, limit: int = 64 * 1024 * 1024) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"Missing, oversized or symlinked evidence file: {path}")
    return path


def _json(path: Path) -> dict:
    document = json.loads(_file(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected JSON source evidence object: {path}")
    return document


def _sha(path: Path) -> str:
    return sha256(_file(path).read_bytes()).hexdigest()


def _preset_delays(path: Path) -> tuple[float, ...]:
    root = ET.parse(_file(path, limit=4 * 1024 * 1024)).getroot()
    delays: dict[int, float] = {}
    for item in root.iter("user_parameter"):
        name = item.attrib.get("name", "")
        match = re.fullmatch(
            r"CONFIG\.PCW_UIPARAM_DDR_DQS_TO_CLK_DELAY_([0-3])", name
        )
        if not match:
            continue
        lane = int(match.group(1))
        if lane in delays:
            raise ValueError(f"Duplicate DDR DQS lane {lane} in board preset")
        value = float(item.attrib["value"])
        if not math.isfinite(value) or abs(value) > 10:
            raise ValueError(f"Invalid DDR delay from board preset, lane {lane}")
        delays[lane] = value
    if set(delays) != set(range(4)):
        raise ValueError("Board preset lacks a complete four-lane DDR DQS delay profile")
    return tuple(delays[i] for i in range(4))


def _native_ddr_warnings(directory: Path) -> tuple[dict[int, float], dict[str, int]]:
    from collections import Counter

    observations: dict[int, float] = {}
    codes = Counter()
    for name in ("stdout.log", "stderr.log"):
        log = _file(directory / "logs" / name, limit=128 * 1024 * 1024)
        with log.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if not re.match(r"^\s*CRITICAL WARNING\s*:", line, re.IGNORECASE):
                    continue
                code = re.search(r"\[([^\]]+)\]", line)
                if code:
                    codes[code.group(1)] += 1
                match = _DDR_LOG_RE.search(line)
                if match is None:
                    continue
                lane = int(match.group(2))
                value = float(match.group(3))
                if lane in observations and not math.isclose(
                    observations[lane], value, abs_tol=0.00001
                ):
                    raise ValueError(f"Conflicting native DDR delay warning for lane {lane}")
                observations[lane] = value
    return observations, dict(sorted(codes.items()))


def audit_board_ddr(case_directory: str | Path, board_repository: str | Path) -> dict:
    case = Path(case_directory).expanduser()
    repo = Path(board_repository).expanduser()
    if case.is_symlink() or not case.is_dir():
        raise ValueError("Source case is missing or symlinked")
    if repo.is_symlink() or not repo.is_dir():
        raise ValueError("User-owned board repository is missing or symlinked")

    report = _json(case / "board_build_summary.json")
    if (
        report.get("schema") != "fpgai-research/board-dma-build/v1"
        or report.get("board") != "pynq-z2"
        or report.get("board_part") != BOARD_PART
        or report.get("part") != SILICON
        or report.get("stage") != "bitstream"
        or report.get("state") != "BOARD_NOT_QUALIFIED"
        or report.get("vivado_exit_code") != 0
        or report.get("routing_verified_complete") is not True
        or report.get("bitstream_generated") is not True
        or not isinstance(report.get("timing_slack_ns"), (int, float))
        or not math.isfinite(report["timing_slack_ns"])
        or report["timing_slack_ns"] < 0
    ):
        raise ValueError("Board receipt is not the expected routed, unqualified PYNQ-Z2 bitstream")

    preset = repo / "pynq-z2" / "A.0" / "preset.xml"
    values = _preset_delays(preset)
    actual, native_codes = _native_ddr_warnings(case)
    diagnostics = report.get("vivado_log_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("Missing native critical-warning diagnostics")
    announced = diagnostics.get("critical_warning_codes")
    if native_codes != announced:
        raise ValueError("Native Vivado critical warnings do not match stored case receipt")
    if len(actual) != sum(value < 0 for value in values):
        raise ValueError("Unexpected missing or surplus negative-DDR native warning lanes")
    for lane, observed in actual.items():
        if not values[lane] < 0 or not math.isclose(
            values[lane], observed, abs_tol=0.00001
        ):
            raise ValueError(f"Vivado reported DDR delay differs from installed preset on lane {lane}")
    if (
        diagnostics.get("logs_inspected") is not True
        or diagnostics.get("critical_warning_count") != sum(
            native_codes.values()
        )
    ):
        raise ValueError("Stored native warning classification is incomplete")

    artifacts = case / "board_artifacts"
    digests = report.get("board_report_sha256")
    if not isinstance(digests, dict) or not digests:
        raise ValueError("Missing source-attested full-board implementation report hashes")
    for file_name, expected in digests.items():
        if (
            file_name not in (
                "route_status.rpt", "timing_summary.rpt", "drc.rpt", "utilization.rpt"
            )
            or _sha(artifacts / file_name) != expected
        ):
            raise ValueError(f"Vivado report digest mismatch: {file_name}")
    if (
        _sha(artifacts / "fpgai_dma.bit") != report.get("bitstream_sha256")
        or _sha(artifacts / "fpgai_dma.hwh") != report.get("hwh_sha256")
    ):
        raise ValueError("Bitstream or hardware-handoff digest mismatch")

    lanes = [
        {
            "lane": lane,
            "installed_board_preset_ns": values[lane],
            "public_pynq_reference_ns": REFERENCE[lane],
            "matches_reference": math.isclose(
                values[lane], REFERENCE[lane], abs_tol=0.00001
            ),
            "negative_delay_warning_seen": lane in actual,
        }
        for lane in range(4)
    ]
    return {
        "schema": "fpgai-research/pynq-ps-ddr-readonly-audit/v1",
        "status": "EVIDENCE_CORRELATED_BOARD_REMAINS_UNQUALIFIED",
        "board": "pynq-z2",
        "board_part": BOARD_PART,
        "case": str(case.absolute()),
        "installed_preset_file": str(preset.absolute()),
        "installed_preset_sha256": _sha(preset),
        "public_reference_url": REFERENCE_SOURCE,
        "reference_is_hardware_qualification": False,
        "lanes": lanes,
        "differing_reference_lanes": [
            lane["lane"] for lane in lanes if not lane["matches_reference"]
        ],
        "native_critical_warning_codes": native_codes,
        "native_warnings_match_original_preset": True,
        "case_artifact_hashes_verified": True,
        "vendor_exit_code": report.get("vivado_exit_code"),
        "routed_timing_slack_ns": report.get("timing_slack_ns"),
        "original_board_state": report.get("state"),
        "board_qualified_by_this_audit": False,
        "board_programmed_by_this_audit": False,
        "note": (
            "Read-only source comparison: values from a public reference are "
            "not a replacement for known-good boot image and physical DDR/DMA "
            "verification. This audit changes no PS parameters or DRC severities."
        ),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True, help="Existing full-board Vivado build")
    p.add_argument("--board-repo", required=True,
                   help="Existing third-party PYNQ-Z2 board definitions folder")
    args = p.parse_args(argv)
    try:
        result = audit_board_ddr(args.case, args.board_repo)
    except (OSError, ValueError, RuntimeError, ET.ParseError) as exc:
        print(
            f"FPGAI read-only board DDR audit failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
