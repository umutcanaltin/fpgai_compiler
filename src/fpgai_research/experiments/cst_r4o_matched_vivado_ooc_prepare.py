"""R4o: create-only matched Vivado OOC RTL+ROM physical study preparation.

Source-only preparation; DOES NOT EXECUTE VIVADO. Frozen R4l generated
RTL/ROM assets are re-attested via R4n then COPIED to new isolated cases.
Both arms use identical part/clock/uncertainty/steps and no board pins.
Vivado report files produced later are *not* themselves clean DRC E4 proof.
Original R4a/R4d/R4k/R4l evidence must never be touched.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

from fpgai_research.experiments import cst_r4n_schedule_matched_rtl_memory as r4n
from fpgai_research.experiments import cst_r4k_matched_row_schedule_storage as r4k
from fpgai_research.toolchains.vivado.rtl_inputs import inventory_vitis_rtl

SCHEMA = "fpgai-research/R4o-source-pinned-matched-Vivado-OOC-prepare/v1"
ARMS = ("full_storage", "row_storage")
PART = "xck26-sfvc784-2LV-c"
CLOCK_NS = 12.0
UNCERTAINTY_NS = 1.5
TOP = "fpgai_model"
_MAX_TOTAL_INPUT_BYTES = 256 * 1024 * 1024
_FILE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*\Z", re.ASCII)


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("R4o original/staged source missing or symlinked: " + str(path))
    return sha256(path.read_bytes()).hexdigest()


def _tcl(*, rtl_names: list[str]) -> str:
    # Exact pre-validated simple filenames; each path is Tcl-brace quoted.
    if not rtl_names or any(not _FILE_NAME.fullmatch(n) or
                            not n.endswith((".v", ".sv"))
                            for n in rtl_names):
        raise ValueError("R4o invalid Verilog filenames")
    file_lines = "\n".join(
        f"read_verilog -sv {{rtl/{name}}}"
        if name.endswith(".sv") else f"read_verilog {{rtl/{name}}}"
        for name in rtl_names)
    return (
        "# R4o standalone FPGAI OOC Vivado 2023.2; original Vitis HDL copied.\n"
        "# This is not board-level/IO pin constrained implementation.\n"
        "set_param general.maxThreads 4\n"
        "file mkdir reports\n"
        "file mkdir outputs\n"
        f"{file_lines}\n"
        f"synth_design -top {TOP} -part {PART} -mode out_of_context\n"
        f"create_clock -name ap_clk -period {CLOCK_NS:.2f} "
        "[get_ports ap_clk]\n"
        f"set_clock_uncertainty {UNCERTAINTY_NS:.2f} [get_clocks ap_clk]\n"
        "opt_design\n"
        "place_design\n"
        "route_design\n"
        "report_route_status -file reports/route_status.rpt\n"
        "report_timing_summary -delay_type max -file reports/timing_summary.rpt\n"
        "report_utilization -file reports/utilization.rpt\n"
        "report_drc -file reports/drc.rpt\n"
        "write_checkpoint -force outputs/routed.dcp\n"
        "puts {R4O_VIVADO_OOC_ROUTING_COMMANDS_COMPLETE_REPORTS_REQUIRE_AUDIT}\n"
    )


def inspect(*, r4k_study: str | Path, original_r4d: str | Path,
            original_r4a: str | Path) -> dict:
    """Preflight native E2/E3/RTL+ROM SHA, never execute or stage vendor tools."""
    r4k_study, original_r4d, original_r4a = map(
        Path, (r4k_study, original_r4d, original_r4a))
    for root in (r4k_study, original_r4d, original_r4a):
        if root.is_symlink() or not root.is_dir():
            raise ValueError("R4o original source study root missing or symlinked")
    r4n_report = r4n.inspect(
        r4k_study=r4k_study, original_r4d=original_r4d,
        original_r4a=original_r4a)
    if (r4n_report.get("original_E2_E3_qualified_pair") is not True
            or r4n_report.get("R4l_original_RTL_source_arms_attested") != 2
            or r4n_report.get("R4l_original_arms") != 2
            or [a.get("arm") for a in r4n_report["arms"]] != list(ARMS)
            or r4n_report.get("pair", {}).get("both_RTL_SHA_attested") is not True):
        raise ValueError("R4o matched R4l E2/E3 RTL+ROM source audit incomplete")
    # Physical study readiness DOES NOT depend on a lexical memory parser's
    # ability to interpret every memory site.
    packages = {}
    for arm in r4n_report["arms"]:
        name = arm["arm"]
        pkg = r4k_study / name / "vitis_source"
        receipt = arm.get("rtl_source_receipt") or {}
        original = inventory_vitis_rtl(pkg)
        rtl = receipt.get("rtl_file_sha256")
        rom = receipt.get("rom_file_sha256")
        if (not isinstance(rtl, dict) or not rtl
                or not isinstance(rom, dict)
                or not rom
                or original["rtl_sha256"] != rtl
                or original["rom_sha256"] != rom):
            raise ValueError("R4o original qualified RTL/ROM inputs missing or drifted")
        orig_root = pkg / "fpgai_hls_project" / "solution1" / "syn" / "verilog"
        all_files = {**rtl, **rom}
        if len(all_files) != len(rtl) + len(rom):
            raise ValueError("R4o overlapping Verilog/ROM input names")
        length = 0
        for filename, digest in sorted(all_files.items()):
            if not _FILE_NAME.fullmatch(filename):
                raise ValueError("R4o unsafe original HDL/ROM filename")
            file = orig_root / filename
            if file.is_symlink() or not file.is_file():
                raise ValueError("R4o symlinked/absent original HDL/ROM")
            length += file.stat().st_size
            if length > _MAX_TOTAL_INPUT_BYTES or _sha(file) != digest:
                raise ValueError("R4o original HDL/ROM input budget or SHA mismatch")
        packages[name] = {
            "original_package_path": str(pkg),
            "rtl_sha256": rtl,
            "rom_sha256": rom,
            "original_rtl_file_count": len(rtl),
            "original_rom_file_count": len(rom),
            "HLS_estimated_BRAM18K": arm["HLS_estimated_BRAM18K"],
            "source_cpp_sha256": arm["source_cpp_sha256"],
        }
    return {
        "schema": SCHEMA,
        "status": "ORIGINAL_R4L_RTL_ROM_SOURCE_PINNED_READY_FOR_OOC_STAGING",
        "original_R4k_source_revision": r4n_report["R4k_original_source_revision"],
        "original_R4d_source_revision": r4n_report["original_R4d_source_revision"],
        "original_R4a_source_revision": r4n_report["original_R4a_source_revision"],
        "target_part": PART,
        "requested_clock_ns": CLOCK_NS,
        "source_clock_uncertainty_ns": UNCERTAINTY_NS,
        "top": TOP,
        "arms": packages,
        "both_R4l_E2_E3_qualified": True,
        "physical_E4": None,
        "OOC_board_qualified": False,
        "new_vivado_calls": 0,
    }


def prepare(*, r4k_study: str | Path, original_r4d: str | Path,
            original_r4a: str | Path, output: str | Path) -> dict:
    base = inspect(r4k_study=r4k_study, original_r4d=original_r4d,
                   original_r4a=original_r4a)
    out = Path(output)
    original_roots = tuple(Path(p).resolve() for p in
                           (r4k_study, original_r4d, original_r4a))
    resolved = out.resolve()
    if any(resolved == root or root in resolved.parents
           for root in original_roots):
        raise ValueError("R4o physical staging must be outside original study")
    if out.exists() or out.is_symlink():
        raise FileExistsError("R4o physical OOC output is create-only")
    out.mkdir(parents=True, exist_ok=False)
    try:
        for name in ARMS:
            originals = base["arms"][name]
            case = out / name
            case.mkdir()
            stage = case / "rtl"
            stage.mkdir()
            source = (Path(originals["original_package_path"]) /
                      "fpgai_hls_project" / "solution1" / "syn" / "verilog")
            for filename, digest in sorted({
                **originals["rtl_sha256"], **originals["rom_sha256"],
            }.items()):
                dest = stage / filename
                shutil.copyfile(source / filename, dest, follow_symlinks=False)
                if _sha(dest) != digest:
                    raise ValueError("R4o copied RTL/ROM SHA mismatch")
            script = _tcl(rtl_names=sorted(originals["rtl_sha256"]))
            (case / "run_vivado.tcl").write_text(script)
        report = {
            **base, "status": "SOURCE_ONLY_VIVADO_OOC_PREPARED_NOT_EXECUTED",
            "new_vivado_calls": 0,
            "physical_E4": None,
        }
        (out / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        hashes = {
            str(path.relative_to(out)): _sha(path)
            for path in sorted(out.rglob("*")) if path.is_file()
        }
        (out / "artifact_sha256.json").write_text(
            json.dumps(hashes, indent=2, sort_keys=True) + "\n")
        return report
    except Exception as err:
        (out / "FAILED.json").write_text(json.dumps({
            "type": type(err).__name__,
            "message": str(err)[:3000],
            "original_R4l_files_modified": False,
            "new_vivado_calls": 0,
        }, indent=2, sort_keys=True) + "\n")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r4k-study", required=True)
    parser.add_argument("--original-r4d", required=True)
    parser.add_argument("--original-r4a", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = (
        prepare(
            r4k_study=args.r4k_study,
            original_r4d=args.original_r4d, original_r4a=args.original_r4a,
            output=args.output)
        if args.output else inspect(
            r4k_study=args.r4k_study,
            original_r4d=args.original_r4d, original_r4a=args.original_r4a)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
