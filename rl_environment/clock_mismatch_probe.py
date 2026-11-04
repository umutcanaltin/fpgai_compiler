"""Read-only analysis of ORIGINAL requested-vs-Vitis-reported clock periods.

This diagnostic never changes, re-qualifies, or re-runs original HLS cases.
It inspects known timing XML locations and the exact generated source Tcl, and
groups the differences without assuming that relaxed clocks fix them.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from math import isfinite
from pathlib import Path
from xml.etree import ElementTree as ET

from .timing import XML_REL


def _value(section, name):
    if section is None:
        return None
    rows = section.findall(name)
    if len(rows) != 1 or rows[0].text is None:
        return None
    try:
        result = float(rows[0].text)
        if isfinite(result):
            return result
    except ValueError:
        pass
    return None


def investigate(root, *, samples=5):
    root = Path(root).resolve(strict=True)
    campaign = json.loads((root / "campaign.json").read_text())
    if campaign.get("schema") != "fpgai-gym-native-plan/v1":
        raise ValueError("Expected a source-pinned original Gym study")
    counts = Counter()
    examples = []
    reported_values = Counter()
    desired_values = Counter()
    unique_pairs = Counter()
    journal = root / "native_steps.jsonl"
    for line in journal.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        ep = Path(row["private_episode_path"]).resolve(strict=True)
        if not ep.is_relative_to(root):
            raise ValueError("Original native episode escaped selected study")
        case = ep / f"case_{int(row['native_call_index']):04d}"
        manifest = json.loads((case / "manifest.json").read_text())
        requested = float(row["clock_ns"])
        if abs(requested - float(manifest["clock_ns"])) > 1e-8:
            raise ValueError("Original requested clock differs from HLS manifest")
        report = case / XML_REL
        if report.is_symlink() or not report.is_file():
            counts["missing_or_unsafe_original_xml"] += 1
            continue
        root_xml = ET.parse(report).getroot()
        user = root_xml.find("./UserAssignments")
        summary = root_xml.find(".//PerformanceEstimates/SummaryOfTimingAnalysis")
        target_user = _value(user, "TargetClockPeriod")
        target_summary = _value(summary, "TargetClockPeriod")
        estimate = _value(summary, "EstimatedClockPeriod")
        uncertainty = _value(user, "ClockUncertainty")
        if uncertainty is None:
            uncertainty = _value(summary, "ClockUncertainty")
        target = target_user if target_user is not None else target_summary
        counts["original_reports_read"] += 1
        if target is None or estimate is None:
            counts["missing_or_malformed_clock_fields"] += 1
            continue
        if target_user is not None and target_summary is not None and abs(
                target_user - target_summary) > 1e-4:
            counts["conflicting_xml_target_fields"] += 1
        if abs(target - requested) <= 1e-4:
            counts["matching_target_clock"] += 1
        else:
            counts["mismatching_target_clock"] += 1
            reported_values[f"{target:.3f}"] += 1
            desired_values[f"{requested:.3f}"] += 1
            unique_pairs[(f"{requested:.3f}", f"{target:.3f}")] += 1
            if len(examples) < samples:
                examples.append({
                    "case_relative_path": str(case.relative_to(root)),
                    "requested_manifest_clock_ns": requested,
                    "reported_xml_target_clock_ns": target,
                    "reported_xml_estimated_clock_ns": estimate,
                    "reported_xml_uncertainty_ns_or_null": uncertainty,
                    "difference_target_minus_requested_ns": round(target-requested, 6),
                    "reported_xml_user_target_ns_or_null": target_user,
                    "reported_xml_timing_target_ns_or_null": target_summary,
                })
        if uncertainty is None:
            counts["missing_xml_uncertainty"] += 1
        else:
            counts["xml_uncertainty_present"] += 1
    return {
        "status": "READ_ONLY_ORIGINAL_CLOCK_MISMATCH_ANALYSIS_NOT_QUALIFICATION",
        "campaign_source_revision": campaign["source_revision"],
        "native_vendor_calls": 0,
        "source_files_modified": 0,
        "counts": dict(sorted(counts.items())),
        "top_mismatched_requested_targets": reported_values.most_common(12),
        "top_mismatched_requested_clock_periods": desired_values.most_common(12),
        "top_clock_pairs_requested_reported": [
            {"requested_clock_ns": a, "reported_target_ns": b, "count": count}
            for (a,b),count in unique_pairs.most_common(16)],
        "example_mismatches": examples,
        "warning": "Clock mismatch is a source evidence failure; no report is accepted or rewritten by this diagnostic.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args(argv)
    if args.examples < 0 or args.examples > 20:
        parser.error("--examples must be 0..20")
    print(json.dumps(investigate(args.study, samples=args.examples),
                     sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
