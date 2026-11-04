"""R3c: read-only, complete-cohort publication table for original R3a CNN/Dense HLS.

This stage makes NO original GCC, Vitis, Vivado, or RL calls. It reuses
the source/receipt-attested R3b snapshot; qualified E3-only metrics are
compared only when *both* predeclared pair arms qualify. Rejected E3
clock cases are not turned into resource ties, zeros, or missing cases.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re

SCHEMA = "fpgai-research/R3c-complete-CNN-HLS-publication-tables/v1"
PAIR_IDS = ((0, 1), (2, 3), (4, 5))
METRICS = ("bram_18k", "dsp", "ff", "lut", "uram")
PAIR_METRIC_FIELDS = {
    "bram_18k": "HLS_estimated_BRAM18k_delta",
    "dsp": "HLS_estimated_DSP_delta",
    "ff": "HLS_estimated_FF_delta",
    "lut": "HLS_estimated_LUT_delta",
}
CASE_COLUMNS = (
    "case", "network", "status", "original_E2", "original_E3",
    "timing_status", "estimated_clock_ns", "effective_budget_ns",
    "timing_margin_ns", "worst_hls_cycles", "estimated_latency_ns",
    "bram_18k", "dsp", "ff", "lut", "uram",
    "source_cpp_sha256", "original_HLS_report_sha256",
)
PAIR_COLUMNS = (
    "network", "control", "independent_phi", "both_E2", "both_E3",
    "comparison_status", "cycles_control", "cycles_independent",
    "cycles_delta", "estimated_latency_delta_ns",
    "estimated_latency_change_percent", "bram_18k_delta", "dsp_delta",
    "ff_delta", "lut_delta", "uram_delta",
)


def _number(value, *, integer: bool = False, nonnegative: bool = False):
    if integer:
        if type(value) is not int:
            raise ValueError("R3c: non-integer HLS resource or cycle field")
    elif type(value) not in (int, float) or not isfinite(value):
        raise ValueError("R3c: non-finite or absent HLS numeric field")
    if nonnegative and value < 0:
        raise ValueError("R3c: negative HLS resource field")
    return value


def analyze(snapshot: dict, *, requested_clock_ns: float) -> dict:
    """Pure exact-denominator analysis; no original-study reads or writes.

    Callers must obtain the snapshot from the source-attributed R3b auditor.
    This function also checks internal count, pair, timing and delta invariants.
    """
    clock = _number(requested_clock_ns)
    if clock <= 0:
        raise ValueError("R3c: invalid source-prescribed requested clock")
    if snapshot.get("source_GCC_E1_qualified_inputs") != 12 or snapshot.get("original_source_packages") != 6:
        raise ValueError("R3c: original six-source/twelve-GCC denominator mismatch")
    if snapshot.get("original_Vitis_invocation_receipts") != 6:
        raise ValueError("R3c: complete original native cohort required: 6/6 attempts")
    revision = snapshot.get("source_revision")
    if not isinstance(revision, str) or re.fullmatch("[0-9a-f]{40}", revision) is None:
        raise ValueError("R3c: original full Git source revision unavailable")
    cases = snapshot.get("cases")
    pairs = snapshot.get("contrasts")
    if not isinstance(cases, list) or len(cases) != 6 or not isinstance(pairs, list) or len(pairs) != 3:
        raise ValueError("R3c: complete six-case/three-pair population required")

    case_rows = []
    for i, c in enumerate(cases):
        expected = f"case_{i:03d}"
        if c.get("case") != expected or c.get("original_vitis_invocation_present") is not True:
            raise ValueError("R3c: missing or out-of-order original vendor case")
        source_sha = c.get("source_cpp_sha256")
        functional_sha = c.get("functional_sha256")
        if (not isinstance(source_sha, str)
                or re.fullmatch("[0-9a-f]{64}", source_sha) is None
                or not isinstance(functional_sha, str)
                or re.fullmatch("[0-9a-f]{64}", functional_sha) is None):
            raise ValueError("R3c: original source and functional SHA lineage required")
        if c.get("state") not in ("QUALIFIED_HLS_E3_ESTIMATES_ONLY",
                                  "ATTEMPTED_BUT_NOT_E3_QUALIFIED"):
            raise ValueError("R3c: incomplete or unattributed original attempt")
        e2 = c.get("E2_original_CSim")
        e3 = c.get("E3_original_CSynth")
        if type(e2) is not bool or type(e3) is not bool or (e3 and not e2):
            raise ValueError("R3c: invalid E2/E3 qualification relation")
        if e3 != (c["state"] == "QUALIFIED_HLS_E3_ESTIMATES_ONLY"):
            raise ValueError("R3c: case status conflicts with E3")
        details = c.get("attested_HLS_timing_diagnostic")
        if e3:
            resources = c.get("HLS_estimated_resources")
            if not isinstance(resources, dict) or any(k not in resources for k in METRICS):
                raise ValueError("R3c: incomplete E3-qualified resources")
            resources = {k: _number(resources[k], integer=True, nonnegative=True)
                         for k in METRICS}
            cycles = _number(c.get("hls_worst_cycles"), integer=True)
            if cycles <= 0:
                raise ValueError("R3c: nonpositive worst HLS cycles")
            if not isinstance(details, dict) or details.get("feasible") is not True:
                raise ValueError("R3c: qualified HLS timing not independently reported")
            timing_status = "E3_QUALIFIED_HLS_ESTIMATES_ONLY"
            latency = cycles * clock
        else:
            if c.get("HLS_estimated_resources") is not None or c.get("hls_worst_cycles") is not None:
                raise ValueError("R3c: rejected E3 resources or cycles cannot be treated as qualified")
            resources = {k: None for k in METRICS}
            cycles = None
            latency = None
            timing_status = c.get("qualification_status") or "E3_NOT_QUALIFIED"
        if details is None:
            eclock = budget = margin = None
        else:
            if not isinstance(details, dict):
                raise ValueError("R3c: malformed timing diagnostic")
            eclock = _number(details.get("estimated_clock_ns"))
            budget = _number(details.get("effective_budget_ns"))
            margin = _number(details.get("margin_ns"))
            if abs((budget - eclock) - margin) > 1e-6:
                raise ValueError("R3c: inconsistent effective timing margin")
            if (e3 and (margin < 0 or details.get("feasible") is not True)) or (not e3 and
                    timing_status == "FAILED_HLS_EFFECTIVE_CLOCK_BUDGET"
                    and (details.get("feasible") is not False or margin >= 0)):
                raise ValueError("R3c: inconsistent E3 timing classification")
            if e3 and c.get("hls_estimated_clock_ns") != eclock:
                raise ValueError("R3c: qualified HLS reported clock mismatch")
        case_rows.append({
            "case": expected, "network": c.get("network"), "status": c["state"],
            "original_E2": e2, "original_E3": e3,
            "timing_status": timing_status,
            "estimated_clock_ns": eclock, "effective_budget_ns": budget,
            "timing_margin_ns": margin, "worst_hls_cycles": cycles,
            "estimated_latency_ns": latency,
            "source_cpp_sha256": c.get("source_cpp_sha256"),
            "original_HLS_report_sha256": (
                details.get("original_HLS_report_sha256") if details else None),
            **resources,
        })
    n_e2 = sum(r["original_E2"] for r in case_rows)
    n_e3 = sum(r["original_E3"] for r in case_rows)
    if (n_e2 != snapshot.get("original_CSim_E2_qualified") or
            n_e3 != snapshot.get("original_CSynth_E3_qualified")):
        raise ValueError("R3c: E2/E3 counts differ from source-attributed audit")

    pair_rows = []
    for (a, b), p in zip(PAIR_IDS, pairs):
        control = case_rows[a]
        independent = case_rows[b]
        if (p.get("control") != control["case"] or
                p.get("independent_Phi") != independent["case"] or
                p.get("network") != control["network"] or
                independent["network"] != control["network"]):
            raise ValueError("R3c: original pair identity/graph changed")
        if (cases[a]["functional_sha256"] != cases[b]["functional_sha256"]
                or cases[a]["source_cpp_sha256"] == cases[b]["source_cpp_sha256"]):
            raise ValueError("R3c: paired model identity or mapping source SHA mismatch")
        both_e2 = control["original_E2"] and independent["original_E2"]
        both_e3 = control["original_E3"] and independent["original_E3"]
        if p.get("both_E2") is not both_e2 or p.get("both_E3") is not both_e3:
            raise ValueError("R3c: original matched-pair E2/E3 classification mismatch")
        deltas = {k: independent[k] - control[k] if both_e3 else None
                  for k in METRICS}
        cycles_delta = (independent["worst_hls_cycles"] - control["worst_hls_cycles"]
                        if both_e3 else None)
        latency_delta = cycles_delta * clock if both_e3 else None
        percent = (100 * cycles_delta / control["worst_hls_cycles"]
                   if both_e3 else None)
        for k, field in PAIR_METRIC_FIELDS.items():
            if p.get(field) != deltas[k]:
                raise ValueError("R3c: pair resource delta disagrees with original R3a audit")
        if p.get("HLS_worst_latency_cycles_delta") != cycles_delta:
            raise ValueError("R3c: pair cycles disagree with original R3a audit")
        pair_rows.append({
            "network": control["network"], "control": control["case"],
            "independent_phi": independent["case"],
            "both_E2": both_e2, "both_E3": both_e3,
            "comparison_status": (
                "QUALIFIED_HLS_ESTIMATE_PAIR" if both_e3 else
                "REJECTED_OR_INCOMPLETE_E3_PAIR_NO_NUMERIC_DELTA"),
            "cycles_control": control["worst_hls_cycles"] if both_e3 else None,
            "cycles_independent": independent["worst_hls_cycles"] if both_e3 else None,
            "cycles_delta": cycles_delta,
            "estimated_latency_delta_ns": latency_delta,
            "estimated_latency_change_percent": percent,
            **{k + "_delta": v for k, v in deltas.items()},
        })
    valid_pairs = [p for p in pair_rows if p["both_E3"]]
    if len(valid_pairs) != snapshot.get("matched_E3_qualified_pairs") or (
            3 - len(valid_pairs) != snapshot.get("matched_E3_unknown_or_rejected_pairs")):
        raise ValueError("R3c: matched E3 pair denominator disagrees")
    rejected = sum(c["status"] == "ATTEMPTED_BUT_NOT_E3_QUALIFIED" for c in case_rows)
    if rejected != snapshot.get("original_native_attempted_not_E3"):
        raise ValueError("R3c: rejected native denominator disagrees")
    return {
        "schema": SCHEMA,
        "source_revision": revision,
        "requested_HLS_clock_ns": clock,
        "qualified_metric_type": "original Vitis HLS estimate, NOT routed FPGA QoR",
        "original_native_attempts": 6,
        "original_CSim_E2_qualified": n_e2,
        "original_CSynth_E3_qualified": n_e3,
        "original_CSynth_E3_rejected": rejected,
        "matched_pairs_prespecified": 3,
        "matched_pairs_both_E3_qualified": len(valid_pairs),
        "matched_pairs_not_both_E3_qualified": 3 - len(valid_pairs),
        "qualified_pairs_independent_phi_more_cycles": sum(
            p["cycles_delta"] > 0 for p in valid_pairs),
        "qualified_pairs_estimated_BRAM_ties": sum(
            p["bram_18k_delta"] == 0 for p in valid_pairs),
        "cases": case_rows,
        "pairs": pair_rows,
        "physical_E4": None,
        "measured_board_power": None,
        "learning_improvement": None,
        "scientific_novelty_established": False,
        "limits": [
            "Three deterministic small neural networks; two mappings each; not statistical generalization",
            "Worst-cycle times requested clock is a planning estimate, not achieved physical latency",
            "E3-rejected original HLS arms have null resource and latency comparisons",
            "Source-verified CSim numerical conformance does not prove bit-exact all-input equivalence",
            "Independent competing-artifact capability and routed clean-DRC E4 are not established",
        ],
        "original_vendor_calls_during_analysis": 0,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Original R3a CNN/Dense: full-cohort, read-only publication table",
        "",
        f"Original source revision: `{report['source_revision']}`.",
        "",
        f"Original Vitis attempts: **{report['original_native_attempts']}/6**; "
        f"CSim E2 **{report['original_CSim_E2_qualified']}/6**; "
        f"CSynth E3 **{report['original_CSynth_E3_qualified']}/6**; "
        f"E3 rejected **{report['original_CSynth_E3_rejected']}/6**.",
        f"Both-E3 matched pairs: **{report['matched_pairs_both_E3_qualified']}/3**; "
        "all others retain NULL comparisons.",
        "",
        "## Prespecified matched source pairs",
        "",
        "| Network | Control | Independent Phi | Both E3 | Worst HLS cycles control → independent | Estimated BRAM18k Δ | DSP Δ | LUT Δ | FF Δ |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for p in report["pairs"]:
        def fmt(v):
            return "UNKNOWN / REJECTED" if v is None else str(v)
        cycles = (f"{p['cycles_control']} → {p['cycles_independent']}"
                  if p["both_E3"] else "REJECTED E3 / NO DELTA")
        lines.append(
            f"| {p['network']} | {p['control']} | {p['independent_phi']} | "
            f"{p['both_E3']} | {cycles} | {fmt(p['bram_18k_delta'])} | "
            f"{fmt(p['dsp_delta'])} | {fmt(p['lut_delta'])} | {fmt(p['ff_delta'])} |")
    lines += [
        "",
        f"Among the **{report['matched_pairs_both_E3_qualified']}** "
        "original E3-qualified two-arm pairs, independent Phi increased "
        f"worst-case HLS cycle estimates in **{report['qualified_pairs_independent_phi_more_cycles']}** "
        "and tied estimated BRAM18k in "
        f"**{report['qualified_pairs_estimated_BRAM_ties']}**.",
        "",
        "## Original E3 effective-clock failures",
        "",
    ]
    for c in report["cases"]:
        if not c["original_E3"]:
            clock = c["estimated_clock_ns"]
            budget = c["effective_budget_ns"]
            lines.append(
                f"- {c['case']}: {c['timing_status']}; "
                f"source-attested estimated HLS period {clock if clock is not None else 'UNKNOWN'} ns; "
                f"effective HLS budget {budget if budget is not None else 'UNKNOWN'} ns. "
                "No qualified resource/cycle comparison.")
    lines += [
        "",
        "## Limitations and evidence boundary",
        "",
        *["- " + x for x in report["limits"]],
        "",
        "**No claim of physical FPGA BRAM savings, achieved Fmax, board power, "
        "trained-RL performance, or novelty superiority follows from these data.**",
        "",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_create_only(report: dict, *, output: str | Path) -> None:
    """Separate derived data; never mutate original six-case experiment."""
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R3c: existing export folder must not be overwritten")
    root.mkdir(parents=True, exist_ok=False)
    (root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "paper_results.md").write_text(markdown(report), encoding="utf-8")
    _write_csv(root / "case_population.csv", CASE_COLUMNS, report["cases"])
    _write_csv(root / "matched_pair_population.csv", PAIR_COLUMNS, report["pairs"])
    digests = {p.name: sha256(p.read_bytes()).hexdigest()
               for p in sorted(root.iterdir()) if p.is_file()}
    (root / "derived_export_sha256.json").write_text(
        json.dumps(digests, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    # Only actual source-pinned original results, not a JSON fixture, qualify
    # for an independent manuscript result export through this CLI.
    from fpgai_research.experiments import cst_r3b_publication_handoff as r3b
    actual = r3b.snapshot(study=args.study)
    verified = analyze(actual, requested_clock_ns=r3b.r3a.CLOCK_NS)
    export_create_only(verified, output=args.output)
    print(json.dumps({
        "schema": SCHEMA,
        "source_revision": verified["source_revision"],
        "original_Vitis_calls_during_analysis": 0,
        "original_E2": verified["original_CSim_E2_qualified"],
        "original_E3": verified["original_CSynth_E3_qualified"],
        "matched_E3_pairs": verified["matched_pairs_both_E3_qualified"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
