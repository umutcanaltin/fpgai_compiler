"""Source-attributed, read-only Vivado DRC forensics for an R2j matched pair.

Recheck ORIGINAL drc.rpt files against their immutable R2j export hashes;
extract DSP cell identities and RTSTAT no-load counts. No Vivado calls,
DRC waivers, report rewrites, or E4 promotions are permitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

SCHEMA = "fpgai-research/cst-R2j-source-attributed-DRC-forensics/v1"
_RULE = re.compile(
    r"(?m)^\s*([A-Z][A-Z0-9_-]+)#([1-9][0-9]*)\s+"
    r"(Critical Warning|Warning|Error)\s*$"
)
_DSP = re.compile(r"(?m)^\s*DSP\s+(\S+)\s+(?:input|output|multiplier)\b")
_RTSTAT_COUNT = re.compile(r"\b(\d+)\s+net\(s\) have no routable loads\b")
_REASON = {
    "DPIP-2": "DSP input pipelining guidance",
    "DPOP-3": "DSP PREG output pipelining guidance",
    "DPOP-4": "DSP MREG multiplier pipelining guidance",
    "RTSTAT-10": "Nets with no routable loads",
}


def parse_drc_report(report_text: str) -> list[dict]:
    """Extract numbered violations, not headings or summary rule references."""
    if not isinstance(report_text, str) or not report_text.strip():
        raise ValueError("Empty raw Vivado DRC report")
    matches = list(_RULE.finditer(report_text))
    if not matches:
        raise ValueError("Original DRC report lacks numbered rule evidence")
    violations = []
    observed_indices = set()
    for index, found in enumerate(matches):
        rule, ordinal, severity = found.groups()
        key = (rule, int(ordinal))
        if key in observed_indices:
            raise ValueError("Duplicate numbered Vivado DRC violation")
        observed_indices.add(key)
        body = report_text[found.end():matches[index+1].start()
                           if index + 1 < len(matches) else len(report_text)]
        dsp = _DSP.search(body)
        net_count = _RTSTAT_COUNT.search(body)
        if rule in ("DPIP-2", "DPOP-3", "DPOP-4") and dsp is None:
            raise ValueError("DSP DRC violation is missing its original cell identity")
        if rule == "RTSTAT-10" and net_count is None:
            raise ValueError("RTSTAT-10 is missing the reported unroutable net count")
        violations.append({
            "rule": rule,
            "ordinal": int(ordinal),
            "severity": severity,
            "mechanism": _REASON.get(rule, "UNCLASSIFIED_RULE"),
            "dsp_cell": dsp.group(1) if dsp else None,
            "no_routable_load_net_count": int(net_count.group(1))
            if net_count else None,
            "description_sha256": sha256(body.encode("utf-8")).hexdigest(),
        })
    return violations


def _read_original_report(path: Path, *, expected_sha256: str) -> str:
    if (not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None):
        raise ValueError("Original routed DRC report lacks an attested digest")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16*1024*1024:
        raise ValueError("Original DRC file is missing, unsafe, or oversized")
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Original DRC report bytes differ from routed vendor receipt")
    return data.decode("utf-8")


def forensic_report(*, physical_study: str | Path,
                    export_study: str | Path, pair: int = 20) -> dict:
    from fpgai_research.experiments.cst_r2j_physical_diagnostics import diagnostic

    baseline = diagnostic(export_study, pair=pair)
    if not baseline["both_source_attributed_routed_observations_valid"]:
        raise ValueError("Selected pair has no matched original routed observations")
    root = Path(physical_study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Original physical study path missing or symlinked")
    audit_path = Path(export_study) / "original_physical_audit.json"
    if (audit_path.is_symlink() or not audit_path.is_file()
            or audit_path.stat().st_size > 32*1024*1024):
        raise ValueError("Missing or unsafe original physical audit")
    audit = json.loads(audit_path.read_text())
    result = []
    for original in baseline["cases"]:
        case_name = original["case"]
        if not re.fullmatch(r"case_[0-9]{3}", case_name):
            raise ValueError("Unsafe original case identifier")
        original_case = audit["cases"][int(case_name[-3:])]
        if original_case.get("case") != case_name:
            raise ValueError("Source-attributed original case order mismatch")
        expected = (original_case.get("original_routed_report_sha256") or {}).get("drc.rpt")
        receipt_path = root / case_name / "vivado" / "vivado_invocation.json"
        if (receipt_path.is_symlink() or not receipt_path.is_file()
                or receipt_path.stat().st_size > 2*1024*1024):
            raise ValueError("Original Vivado invocation receipt missing or unsafe")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("completed_successfully") is not True
                or (receipt.get("fresh_report_sha256") or {}).get("drc.rpt")
                    != expected):
            raise ValueError("Original DRC digest does not match Vivado invocation receipt")
        report_path = root / case_name / "vivado" / "reports" / "drc.rpt"
        raw = _read_original_report(report_path, expected_sha256=expected)
        violations = parse_drc_report(raw)
        totals = Counter(x["rule"] for x in violations)
        summaries = Counter((x["rule"], x["severity"]) for x in violations)
        expected_rules = original["DRC_rule_details"]
        if (original["DRC_summary_verified"] is not True
                or original["DRC_violations"] != len(violations)
                or not isinstance(expected_rules, dict)
                or set(totals) != set(expected_rules)
                or any(
                    details.get("violations") != totals[rule]
                    or summaries[(rule, details.get("severity"))] != totals[rule]
                    for rule, details in expected_rules.items()
                )):
            raise ValueError("Raw DRC violations contradict original verified summary")
        result.append({
            "case": case_name,
            "source_storage": original["source_storage"],
            "original_drc_sha256": expected,
            "count": len(violations),
            "rule_counts": dict(sorted(totals.items())),
            "DSP_cell_witnesses": sorted(
                [v["rule"], v["dsp_cell"]] for v in violations
                if v["dsp_cell"] is not None
            ),
            "RTSTAT_no_routable_load_counts": [
                v["no_routable_load_net_count"] for v in violations
                if v["rule"] == "RTSTAT-10"
            ],
            "violations": violations,
            "strict_clean_DRC_E4": original["strict_clean_drc_E4"],
        })
    left, right = result
    return {
        "schema": SCHEMA,
        "physical_pair": pair,
        "original_graph": baseline["original_graph"],
        "original_mapping": baseline["original_mapping"],
        "original_HLS_source_revision": baseline["original_HLS_source_revision"],
        "cases": result,
        "same_rule_multiplicities": left["rule_counts"] == right["rule_counts"],
        "same_reported_DSP_cell_identities": (
            bool(left["DSP_cell_witnesses"])
            and left["DSP_cell_witnesses"] == right["DSP_cell_witnesses"]
        ),
        "same_RTSTAT_no_load_net_counts": (
            bool(left["RTSTAT_no_routable_load_counts"])
            and left["RTSTAT_no_routable_load_counts"]
                == right["RTSTAT_no_routable_load_counts"]
        ),
        "same_exact_RTSTAT_net_identities": None,
        "both_strict_clean_DRC_E4_qualified": baseline[
            "both_strict_clean_DRC_E4_qualified"],
        "new_original_GCC_Vitis_Vivado_calls": 0,
        "limitations": [
            "Identical warning identities do not demonstrate identical placement, physical resources, or root cause",
            "RTSTAT reported net counts are not exact net-name equivalence",
            "DSP pipeline warning guidance is not proof that new pipeline stages preserve matched numerical/latency behavior",
            "No DRC warnings waived; all E4 gates remain unchanged",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-study", required=True)
    parser.add_argument("--export-study", required=True)
    parser.add_argument("--pair", type=int, default=20)
    parser.add_argument("--output", help="Optional NEW create-only JSON report")
    args = parser.parse_args(argv)
    if args.output:
        target = Path(args.output)
        if target.exists() or target.is_symlink():
            raise FileExistsError("DRC forensics output must be create-only")
        if not target.parent.is_dir():
            raise ValueError("DRC forensics output parent directory does not exist")
    report = forensic_report(
        physical_study=args.physical_study,
        export_study=args.export_study,
        pair=args.pair,
    )
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
