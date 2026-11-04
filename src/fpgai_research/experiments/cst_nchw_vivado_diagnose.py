"""Read-only diagnostics for original failed Vivado OOC invocations.

Do not regenerate source, modify a source-pinned study, delete a failed
receipt, or invoke any FPGA tool. Error excerpts explain tool failures;
they never qualify E4 evidence and are printed only from hash-matched logs.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from fpgai_research.experiments.cst_nchw_physical import audit_routed_study
from fpgai_research.toolchains.vivado.report import parse_vivado_reports

_ERROR_RE = re.compile(
    r"\b(?:ERROR|FATAL|CRITICAL WARNING)\s*[:\[]|"
    r"\b(?:invalid command name|couldn't read file|"
    r"no such file or directory|failed to|cannot open|can't read)\b",
    re.IGNORECASE,
)
_MAX_EXCERPT_BYTES = 1024 * 1024
_ROM_ACTIVITY_RE = re.compile(
    r"\.(?:dat|mem)\b|readmem[hb]|ROM_AUTO|\bfilemgmt\b|"
    r"memory initialization|\\bimport(?:ed|ing)?\\b", re.IGNORECASE,
)
_MAX_PROJECT_ENTRIES = 50000
_MAX_ROM_BYTES = 64 * 1024 * 1024


def _ensure_plain(path: Path, *, required: bool = False) -> bool:
    if path.is_symlink():
        raise ValueError("Symlinked Vivado evidence is unsupported: " + path.name)
    if not path.exists():
        if required:
            raise FileNotFoundError("Missing Vivado evidence: " + str(path))
        return False
    if not path.is_file():
        raise ValueError("Non-file Vivado evidence: " + str(path))
    return True


def _digest_and_tail(path: Path) -> tuple[str, bytes, int]:
    """Hash and capture a bounded tail from the SAME original read stream."""
    result = sha256()
    tail = b""
    length = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
            length += len(chunk)
            tail = (tail + chunk)[-_MAX_EXCERPT_BYTES:]
    return result.hexdigest(), tail, length


def _inspect_log(path: Path, *, root: Path, expected_sha: object) -> dict:
    item = {
        "path": str(path.relative_to(root)),
        "available": _ensure_plain(path),
        "sha256": None,
        "matches_original_receipt": False,
        "error_lines": [],
        "rom_activity_lines": [],
        "last_lines": [],
    }
    if not item["available"]:
        return item
    item["sha256"], tail, original_bytes = _digest_and_tail(path)
    if not isinstance(expected_sha, str) or item["sha256"] != expected_sha:
        # Never treat subsequently edited logs as original vendor output.
        return item
    item["matches_original_receipt"] = True
    lines = tail.decode("utf-8", errors="replace").splitlines()
    if original_bytes > _MAX_EXCERPT_BYTES and lines:
        lines = lines[1:]  # Discard potentially truncated first line.
    item["error_lines"] = [line[:500] for line in lines
                           if _ERROR_RE.search(line)][-16:]
    item["rom_activity_lines"] = [
        line[:500] for line in lines if _ROM_ACTIVITY_RE.search(line)
    ][-48:]
    item["last_lines"] = [line[:500] for line in lines[-12:]]
    return item


def _project_rom_locations(vendor: Path, expected: object) -> dict:
    """Inventory ROM copies under Vivado's project, read-only and bounded.

    A matching copy elsewhere is a diagnostic observation, NOT proof
    Vivado read it, nor a substitute for the required working ROM.
    """
    project = vendor / "research_implementation"
    info = {
        "project_present": _ensure_plain_directory(project),
        "scanned_entries": 0,
        "scan_truncated": False,
        "project_rom_candidates": [],
    }
    if not info["project_present"]:
        return info
    expected_hashes = expected if isinstance(expected, dict) else {}
    for parent, directories, filenames in os.walk(project, followlinks=False):
        depth = len(Path(parent).relative_to(project).parts)
        # Refuse to follow symlinked directories, including project children.
        directories[:] = [
            d for d in directories
            if not (Path(parent) / d).is_symlink() and depth < 12
        ]
        for name in filenames:
            info["scanned_entries"] += 1
            if info["scanned_entries"] > _MAX_PROJECT_ENTRIES:
                info["scan_truncated"] = True
                return info
            if Path(name).suffix.lower() not in (".dat", ".mem"):
                continue
            path = Path(parent) / name
            item = {"path": str(path.relative_to(vendor)),
                    "sha256": None, "matches_original_sha256": False}
            if path.is_symlink() or not path.is_file():
                item["rejected"] = "symlink_or_nonfile"
            elif path.stat().st_size > _MAX_ROM_BYTES:
                item["rejected"] = "unbounded_file"
            else:
                digest, _, _ = _digest_and_tail(path)
                item["sha256"] = digest
                item["matches_original_sha256"] = (
                    expected_hashes.get(name) == digest)
            info["project_rom_candidates"].append(item)
            if len(info["project_rom_candidates"]) >= 96:
                info["scan_truncated"] = True
                return info
    return info


def _ensure_plain_directory(path: Path) -> bool:
    if path.is_symlink():
        raise ValueError("Symlinked Vivado project directory unsupported")
    return path.is_dir()


def _inspect_original_drc_details(path: Path, *, root: Path,
                                  expected_sha: object,
                                  selected_rules: object) -> dict:
    """Read rule descriptions from an original SHA-256-verified DRC report.

    No DRC policy overrides, no waived warnings, no changes to input bytes.
    A bounded excerpt is diagnostic context, never qualification evidence.
    """
    result={"path":str(path.relative_to(root)),"available":_ensure_plain(path),
            "matches_original_receipt":False,"sha256":None,"rule_examples":{}}
    if not result["available"]:
        return result
    if path.stat().st_size>16*1024*1024:
        result["reason"]="DRC_REPORT_TOO_LARGE_FOR_BOUNDED_DETAIL"
        return result
    digest, tail, _=_digest_and_tail(path)
    result["sha256"]=digest
    if not isinstance(expected_sha,str) or digest!=expected_sha:
        result["reason"]="DRC_REPORT_HASH_MISMATCH"
        return result
    result["matches_original_receipt"]=True
    text=tail.decode("utf-8",errors="replace")
    # Standard Vivado DRC rule details: DPIP-2#1 Warning, etc.
    lines=text.splitlines()
    selected=(selected_rules if isinstance(selected_rules,dict) else {})
    for rule in sorted(selected):
        if not re.fullmatch(r"[A-Z][A-Z0-9_-]*",rule):
            continue
        pattern=re.compile(r"^\s*"+re.escape(rule)+r"#\d+\s+"
                           r"(?:Warning|Advisory|Critical Warning|Error|Info)\s*$",
                           re.I)
        examples=[]
        for idx,line in enumerate(lines):
            if not pattern.match(line):
                continue
            # The first two subsequent non-empty lines normally contain
            # Vivado's rule description and affected object path.
            excerpt=[line.strip()[:220]]
            for nextline in lines[idx+1:idx+9]:
                if nextline.strip():
                    excerpt.append(nextline.strip()[:450])
                    if len(excerpt)>=4:
                        break
            examples.append(excerpt)
            if len(examples)>=2:
                break
        result["rule_examples"][rule]=examples
    return result


def diagnose_failed_case(study: str | Path, *, case_index: int = 0) -> dict:
    """Read the original receipt and matching logs. Make zero vendor calls."""
    if type(case_index) is not int or not 0 <= case_index < 12:
        raise ValueError("Original NCHW physical case index must be 0..11")
    root = Path(study)
    audit = audit_routed_study(root)
    name = f"case_{case_index:03d}"
    case = root / name
    vendor = case / "vivado"
    logs = vendor / "logs"
    for directory in (case, vendor, logs):
        if directory.is_symlink():
            raise ValueError("Symlinked original Vivado directory is unsupported")
    receipt_path = vendor / "vivado_invocation.json"
    result = {
        "schema": "fpgai-research/cst-NCHW-Vivado-failure-diagnostic/v1",
        "case": name,
        "audit_status": audit["cases"][case_index]["status"],
        "new_vendor_calls": 0,
        "new_files_written": 0,
        "original_receipt_present": _ensure_plain(receipt_path),
        "staged_rom_integrity": audit["cases"][case_index].get("staged_rom_integrity"),
        "project_rom_locations": _project_rom_locations(
            vendor, audit["cases"][case_index].get("original_rom_sha256")),
        "returncode": None,
        "timed_out": None,
        "completed_successfully": None,
        "logs": {},
        "physical_report_preview":None,
        "original_drc_rule_details":None,
        "interpretation": "Diagnostic only; no E4 qualification",
    }
    if not result["original_receipt_present"]:
        return result
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict) or receipt.get("tool") != "vivado":
        raise ValueError("Invalid original Vivado receipt")
    result["returncode"] = receipt.get("returncode")
    result["timed_out"] = receipt.get("timed_out")
    result["completed_successfully"] = receipt.get("completed_successfully")
    for label in ("stdout", "stderr"):
        logfile = logs / f"vivado.{label}.log"
        result["logs"][label] = _inspect_log(
            logfile, root=root, expected_sha=receipt.get(f"{label}_sha256"))
    try:
        reports = parse_vivado_reports(vendor)
    except (ValueError,FileNotFoundError) as exc:
        result["physical_report_preview"]={
            "status":"UNQUALIFIED_REPORTS_UNPARSEABLE",
            "report_error":str(exc)[:350],
        }
    else:
        originals=receipt.get("fresh_report_sha256")
        required=("route_status.rpt","timing_summary.rpt",
                  "utilization.rpt","drc.rpt")
        digests=reports["raw_report_sha256"]
        attested=(isinstance(originals,dict)
                  and all(name in digests and originals.get(name)==digests[name]
                          for name in required))
        # Never emit unverified timing/BRAM as if they were original
        # vendor measurements. An attested preview still does NOT
        # qualify E4 when cwd ROM provenance was rejected.
        if attested:
            result["original_drc_rule_details"]=_inspect_original_drc_details(
                vendor/"reports"/"drc.rpt",root=root,
                expected_sha=originals.get("drc.rpt"),
                selected_rules=reports["drc"].get("rules"))
        result["physical_report_preview"]=({
            "status":"UNQUALIFIED_BUT_ORIGINAL_REPORT_HASHES_MATCH",
            "original_report_sha256":digests,
            "routed_nets_complete":reports["routed_nets_complete"],
            "timing":reports["timing"],
            "utilization":reports["utilization"],
            "drc":reports["drc"],
            "e4_qualified":False,
        } if attested else {
            "status":"UNQUALIFIED_REPORT_HASH_MISMATCH",
            "e4_qualified":False,
        })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--case", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(diagnose_failed_case(args.study, case_index=args.case),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
