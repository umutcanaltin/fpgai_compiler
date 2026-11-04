"""Inspect an existing FPGAI native run without running vendors or changing evidence.

In particular, distinguish Vivado exit=0 from a valid integrity-attested E4
result. A post-run ROM read witness displayed here is diagnostic ONLY:
this command never changes or retroactively promotes an original receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def inspect_case(directory: str | Path) -> dict:
    from fpgai_research.experiments.results import summarize_case
    from fpgai_research.toolchains.vivado.report import parse_vivado_reports
    from fpgai_research.toolchains.vivado.rtl_inputs import inventory_vitis_rtl
    from fpgai_research.toolchains.vivado.rom_read_attestation import (
        verify_post_run_readmem_absence,
    )

    case = Path(directory)
    if case.is_symlink() or not case.is_dir():
        raise ValueError("Expected an existing non-symlinked prepared case directory")
    summary = summarize_case(str(case))
    receipt_path = case / "vivado" / "vivado_invocation.json"
    if receipt_path.is_symlink():
        raise ValueError("Symlinked vendor invocation receipt rejected")
    result = {
        "schema": "fpgai-research/read-only-case-diagnosis/v1",
        "case": str(case.resolve()),
        "original_evidence": summary["evidence"],
        "original_csim_passed": summary.get("csim_passed"),
        "original_warnings": summary.get("warnings", []),
        "vendor_invoked_by_diagnosis": False,
        "original_artifacts_modified": False,
        "original_evidence_promoted": False,
        "status": "NO_VIVADO_INVOCATION",
    }
    if not receipt_path.is_file():
        return result
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema") != "fpgai-research/vendor-invocation/v1":
        raise ValueError("Unexpected Vivado receipt schema")
    if receipt.get("tool") != "vivado":
        raise ValueError("The receipt does not describe a Vivado invocation")
    pre = receipt.get("staged_rom_input_integrity_before") or {}
    post = receipt.get("staged_rom_input_integrity_after") or {}
    if not isinstance(pre, dict) or not isinstance(post, dict):
        raise ValueError("Malformed staged-ROM integrity records")
    sha_pre = receipt.get("input_hashes_before_execution") or {}
    sha_post = receipt.get("input_hashes_after_execution") or {}
    if not isinstance(sha_pre, dict) or not isinstance(sha_post, dict):
        raise ValueError("Malformed input hash snapshot")
    result.update({
        "status": ("ATTESTED_VENDOR_RECEIPT" if receipt.get("completed_successfully") is True
                   else "VENDOR_RECEIPT_REJECTED"),
        "vendor_exit_code": receipt.get("returncode"),
        "vendor_timed_out": receipt.get("timed_out"),
        "vendor_receipt_completed_successfully": receipt.get("completed_successfully"),
        "source_and_tcl_input_hashes_unchanged": sha_pre == sha_post,
        "original_hls_roms_unchanged": receipt.get("rom_inputs_stable_during_execution"),
        "staged_roms_intact_before_vivado": pre.get("matches_original_roms"),
        "staged_roms_intact_after_vivado": post.get("matches_original_roms"),
        "missing_staged_roms_after": post.get("missing_original_roms", []),
        "modified_staged_roms_after": post.get("modified_original_roms", []),
        "unattributed_staged_roms_after": post.get("unattributed_rom_files", []),
        "invalid_staged_roms_after": post.get("invalid_rom_files", []),
        "prospective_absence_witness_enabled_in_original_run":
            receipt.get("allow_verified_readmem_post_absence"),
        "original_receipt_absence_witness":
            (receipt.get("verified_readmem_post_absence_witness") or {}).get("reason"),
        "original_fresh_report_names": sorted((receipt.get("fresh_report_sha256") or {}).keys()),
    })
    if (pre.get("matches_original_roms") is True and
            post.get("matches_original_roms") is False and
            isinstance(receipt.get("rom_input_sha256"), dict)):
        # Read the original HLS Verilog only to recover literal readmem names;
        # source files, receipts and vendor logs are never edited.
        inventory = inventory_vitis_rtl(case)
        witness = verify_post_run_readmem_absence(
            expected=receipt["rom_input_sha256"],
            literal_refs=inventory["literal_readmem_references"],
            before=pre, after=post,
            stdout_file=case / "vivado" / "logs" / "vivado.stdout.log",
            expected_stdout_sha256=receipt.get("stdout_sha256"),
        )
        result["read_only_postrun_rom_observation"] = {
            "witness_reason": witness["reason"],
            "original_vendor_log_reports_successful_reads":
                witness["reported_successful_roms"],
            "source_hash_and_original_log_support_absence":
                witness["qualified_postrun_absence"],
            "retroactive_e4_promotion_allowed": False,
        }
    try:
        physical = parse_vivado_reports(case / "vivado")
    except (FileNotFoundError, ValueError) as exc:
        result["physical_report_observation"] = {
            "parsed": False, "reason": f"{type(exc).__name__}: {exc}",
            "attested_e4_claim": False,
        }
    else:
        result["physical_report_observation"] = {
            "parsed": True,
            "timing_met": physical["timing"].get("timing_met"),
            "worst_negative_slack_ns": physical["timing"].get("wns_ns"),
            "drc_verified": physical["drc"].get("summary_verified"),
            "drc_clean": physical["drc"].get("clean"),
            "drc_violations": physical["drc"].get("violations"),
            "attested_e4_claim": False,
            "note": "Read-only unpromoted report observation; original evidence retained",
        }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="Existing native case directory")
    args = parser.parse_args(argv)
    try:
        report = inspect_case(args.case)
    except (OSError, ValueError) as exc:
        print(f"FPGAI diagnose error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
