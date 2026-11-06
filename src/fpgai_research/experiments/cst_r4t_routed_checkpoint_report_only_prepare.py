"""R4t: source-pinned CREATE-ONLY routed-checkpoint REPORT-ONLY input staging.

Does NOT invoke Vivado/Vitis or any implementation operation. It uses
original R4s/R4r/R4q/R4p attested source and reporting evidence and copies
the existing original R4p routed.dcp to new, separate report-only input
folders. The DCP itself was NOT SHA-attested in original R4p's vendor
invocation receipt; R4t hashes it POST-HOC. Never overstate provenance.

Output Tcl only opens the COPIED checkpoint and emits detailed max-delay
timing and DRC reports. No synthesis, opt, placement, routing, write DCP,
bitstream or original input writes. Execution needs a separate opt-in gate.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from fpgai_research.experiments import cst_r4s_original_timing_drc_context as r4s

SCHEMA = "fpgai-research/R4t-isolated-routed-checkpoint-report-only-prepare/v1"
ARMS = r4s.ARMS
SOURCE_REVISION = r4s.r4r.SOURCE_REVISION
MAX_DCP_BYTES = 1 * 1024 * 1024 * 1024
MAX_TIMING_PATHS = 20


def _sha(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("R4t original checkpoint or copied input missing/symlinked")
    size = path.stat().st_size
    if not (0 < size <= MAX_DCP_BYTES):
        raise ValueError("R4t routed checkpoint size unsupported")
    h = sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    if path.stat().st_size != size:
        raise ValueError("R4t checkpoint size changed during read")
    return h.hexdigest(), size


def _script() -> str:
    return (
        "# R4t isolated report-only Vivado Tcl; review before any native execution.\n"
        "# R4p implementation must NOT be re-run. No source/clock edits.\n"
        "set_param general.maxThreads 4\n"
        "set r4t_case [file normalize [file dirname [info script]]]\n"
        "set r4t_dcp [file join $r4t_case input routed.dcp]\n"
        "set r4t_reports [file join $r4t_case reports]\n"
        "file mkdir $r4t_reports\n"
        "open_checkpoint $r4t_dcp\n"
        f"report_timing -delay_type max -path_type full -input_pins "
        f"-max_paths {MAX_TIMING_PATHS} -nworst 10 "
        "-file [file join $r4t_reports max_delay_paths.rpt]\n"
        "report_timing_summary -delay_type max "
        "-file [file join $r4t_reports timing_summary.rpt]\n"
        "report_drc -file [file join $r4t_reports drc.rpt]\n"
        "puts {R4T_REPORT_ONLY_CHECKPOINT_READ_COMPLETE_NOT_E4}\n"
        "close_design\n"
    )


def inspect(*, study: str | Path, r4o_study: str | Path,
            r4k_study: str | Path, original_r4d: str | Path,
            original_r4a: str | Path) -> dict:
    orig = r4s.inspect(
        study=study, r4o_study=r4o_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a)
    if (orig.get("schema") != r4s.SCHEMA
            or orig.get("status") !=
                "SIGNED_ORIGINAL_R4P_TIMING_DRC_TEXT_FORENSICS_NOT_E4"
            or orig.get("original_source_revision") != SOURCE_REVISION
            or orig.get("source_attested_arms") != 2
            or orig.get("physical_E4") is not None
            or orig.get("original_vendor_artifacts_modified") is not False
            or orig.get("no_new_vivado_or_vitis_calls") is not True
            or [a.get("arm") for a in orig.get("arms", [])] != list(ARMS)):
        raise ValueError("R4t original R4p-through-R4s evidence unqualified")
    root = Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R4t original R4p study absent/symlinked")
    entries = {}
    for arm in orig["arms"]:
        name = arm["arm"]
        original_case = root / name / "vivado"
        outputs = original_case / "outputs"
        if (original_case.is_symlink() or outputs.is_symlink()
                or not outputs.is_dir()):
            raise ValueError("R4t original Vivado checkpoint path unsafe")
        dcp = outputs / "routed.dcp"
        digest, size = _sha(dcp)
        entries[name] = {
            "original_checkpoint_path": str(dcp),
            "posthoc_original_checkpoint_sha256": digest,
            "checkpoint_bytes": size,
            "original_signed_timing_summary_sha256":
                arm["original_signed_timing_summary_sha256"],
            "original_signed_drc_report_sha256":
                arm["original_signed_drc_report_sha256"],
            "original_routed_wns_ns": arm["original_routed_wns_ns"],
            "original_timing_path_detail_status":
                arm["timing_path_source_details"]["status"],
            "original_dcp_sha_contemporaneous_in_R4p_vendor_receipt": False,
        }
    return {
        "schema": SCHEMA,
        "status": "ORIGINAL_R4P_ROUTED_DCP_POSTHOC_PINNED_REPORT_ONLY_READY",
        "original_source_revision": SOURCE_REVISION,
        "original_report_sha_and_vendor_receipts_verified": True,
        "original_checkpoint_provenance_strength": (
            "Existing routed.dcp located in original R4p native output and "
            "SHA-256 pinned POST-HOC; no original invocation-time DCP digest."),
        "arms": entries,
        "planned_new_report_only_vivado_calls": 2,
        "new_native_vivado_calls": 0,
        "original_vendor_artifacts_modified": False,
        "strict_physical_E4": None,
        "board_qualified": False,
    }


def prepare(*, study: str | Path, r4o_study: str | Path,
            r4k_study: str | Path, original_r4d: str | Path,
            original_r4a: str | Path, output: str | Path) -> dict:
    meta = inspect(
        study=study, r4o_study=r4o_study, r4k_study=r4k_study,
        original_r4d=original_r4d, original_r4a=original_r4a)
    dest = Path(output)
    resolved = dest.resolve()
    roots = [Path(p).resolve() for p in (
        study, r4o_study, r4k_study, original_r4d, original_r4a)]
    if any(resolved == root or root in resolved.parents
           or resolved in root.parents for root in roots):
        raise ValueError("R4t separate report-only output overlaps originals")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("R4t report-only output is create-only")
    dest.mkdir(parents=True, exist_ok=False)
    try:
        for name in ARMS:
            case = dest / name
            case.mkdir()
            inputs = case / "input"
            inputs.mkdir()
            original = meta["arms"][name]
            source = Path(original["original_checkpoint_path"])
            checkpoint = inputs / "routed.dcp"
            shutil.copyfile(source, checkpoint, follow_symlinks=False)
            if (_sha(checkpoint) != (
                    original["posthoc_original_checkpoint_sha256"],
                    original["checkpoint_bytes"])
                    or _sha(source) != (
                        original["posthoc_original_checkpoint_sha256"],
                        original["checkpoint_bytes"])):
                raise ValueError("R4t original/copied checkpoint SHA or size changed")
            script = _script()
            (case / "report_only.tcl").write_text(script)
        result = {
            **meta,
            "status": "R4T_ISOLATED_CHECKPOINT_REPORT_ONLY_SOURCE_PREPARED",
            "planned_new_report_only_vivado_calls": 2,
            "new_native_vivado_calls": 0,
            "strict_physical_E4": None,
        }
        (dest / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        manifest = {
            str(p.relative_to(dest)): _sha(p)[0] if p.name == "routed.dcp"
            else sha256(p.read_bytes()).hexdigest()
            for p in sorted(dest.rglob("*")) if p.is_file()
        }
        (dest / "artifact_sha256.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return result
    except BaseException as exc:
        (dest / "FAILED.json").write_text(json.dumps({
            "exception_type": type(exc).__name__,
            "reason": str(exc)[:2000],
            "original_artifacts_modified": False,
            "native_vivado_calls": 0,
        }, sort_keys=True, indent=2) + "\n")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--r4o-study", required=True)
    parser.add_argument("--r4k-study", required=True)
    parser.add_argument("--original-r4d", required=True)
    parser.add_argument("--original-r4a", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    values = dict(study=args.study, r4o_study=args.r4o_study,
                  r4k_study=args.r4k_study,
                  original_r4d=args.original_r4d,
                  original_r4a=args.original_r4a)
    print(json.dumps(
        prepare(**values, output=args.output)
        if args.output else inspect(**values),
        indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
