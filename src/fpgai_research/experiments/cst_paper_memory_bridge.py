"""Read-only paired analysis of HLS BRAM estimates and static RTL structure.

Scientific purpose: determine whether a change in the *source RTL type
graph* coincides with a change in *HLS-estimated BRAM18K* for the same
factorial arm pair. Neither observable is a placed RAM primitive. This is
a descriptive, matched analysis, not a causal proof or a new vendor run.

The source of truth is the original 32-case/64-contrast, SHA-attested
HLS cohort. Existing R2x analysis already rechecks original Vitis
receipts, report hashes, generated HDL hashes, and source provenance.
Original evidence directories are NEVER modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

SCHEMA = "fpgai-research/paper-memory-HLS-RTL-paired-bridge/v1"
R2X_SCHEMA = "fpgai-research/cst-R2x-read-only-RTL-static-module-hierarchy/v1"
SOURCE_REVISION = "0c6d85f189062df8d5157237f3c8ac74ac8da4b8"
FACTORS = ("storage", "address", "binding", "padding")
CATEGORIES = ("decrease", "tie", "increase", "unknown")
PAIR_FIELDS = (
    "graph_index", "factor", "arm_a", "arm_b",
    "HLS_E3_qualified", "static_RTL_known",
    "HLS_BRAM18K_delta", "RTL_memory_module_candidate_delta",
    "HLS_BRAM18K_class", "RTL_candidate_class", "comparison_class",
)


def _direction(value: int | None) -> str:
    if value is None:
        return "unknown"
    if type(value) is not int:
        raise ValueError("Measured differences must be integers or unknown")
    return "decrease" if value < 0 else "increase" if value > 0 else "tie"


def summarize(analysis: dict) -> dict:
    """Validate and classify *all* original matched factors, including unknowns.

    This function is deliberately pure to allow adversarial unit tests.
    The calling analyze() independently verifies the original source,
    invocation receipts and HDL files before supplying this summary.
    """
    if (
        analysis.get("schema") != R2X_SCHEMA
        or analysis.get("original_source_revision") != SOURCE_REVISION
        or analysis.get("original_R2u_cases") != 32
        or analysis.get("original_R2v_conditional_pairs") != 64
        or analysis.get("original_R2u_E3_qualified_cases") != 30
        or analysis.get("original_R2v_E3_qualified_pairs") != 57
        or analysis.get("original_R2v_E3_unknown_pairs") != 7
        or analysis.get("original_RTL_SHA_attested_cases") != 32
        or analysis.get("original_vendor_calls_during_analysis") != 0
        or analysis.get("physical_BRAM") is not None
        or analysis.get("achieved_routed_Fmax") is not None
    ):
        raise ValueError("Original HLS/RTL population or evidence tier changed")
    cases = analysis.get("cases")
    contrasts = analysis.get("contrasts")
    if not isinstance(cases, list) or len(cases) != 32:
        raise ValueError("Expected 32 original source and RTL arms")
    case_ids = {row.get("case") for row in cases if isinstance(row, dict)}
    if case_ids != {f"case_{i:03d}" for i in range(32)}:
        raise ValueError("Original source arm identity mismatch")
    if not isinstance(contrasts, list) or len(contrasts) != 64:
        raise ValueError("Expected all 64 conditional matched pairs")

    keys: set[tuple] = set()
    pairs = []
    by_factor = {f: Counter() for f in FACTORS}
    cross = Counter()
    for row in contrasts:
        if not isinstance(row, dict):
            raise ValueError("Malformed paired source comparison")
        factor, graph, a, b = (
            row.get("factor"), row.get("graph_index"),
            row.get("arm_a"), row.get("arm_b"),
        )
        key = (factor, graph, a, b)
        if (factor not in FACTORS or type(graph) is not int
                or graph not in (0, 1)
                or a not in case_ids or b not in case_ids
                or a == b or key in keys
                or int(a[-3:]) // 16 != graph
                or int(b[-3:]) // 16 != graph):
            raise ValueError("Unmatched, duplicated or cross-graph contrast")
        keys.add(key)
        e3 = row.get("original_E3_both_qualified")
        known = row.get("both_top_static_reachability_known")
        if type(e3) is not bool or type(known) is not bool:
            raise ValueError("Pair qualification must be explicit")
        if row.get("original_RTL_both_SHA_attested") is not True:
            raise ValueError("Unattested original RTL must not be analyzed")
        bram = row.get("estimated_BRAM_delta_or_null")
        rtl = row.get("reachable_memory_instance_count_delta")
        if e3 != (type(bram) is int):
            raise ValueError("Unqualified HLS resource comparison was imputed")
        if known != (type(rtl) is int):
            raise ValueError("Unsupported RTL static structure was imputed")
        if row.get("structural_pair_status") != (
            "STATIC_TYPE_GRAPH_COMPARISON_ONLY" if known
            else "UNKNOWN_UNSUPPORTED_STATIC_RTL"
        ):
            raise ValueError("RTL source type graph status was changed")
        hls_class, rtl_class = _direction(bram), _direction(rtl)
        if not e3:
            comparison = "unknown_HLS"
        elif not known:
            comparison = "unknown_RTL_structure"
        elif hls_class == "tie" and rtl_class != "tie":
            comparison = "RTL_changed_HLS_tied"
        elif hls_class != "tie" and rtl_class == "tie":
            comparison = "HLS_changed_RTL_tied"
        elif hls_class == "tie" and rtl_class == "tie":
            comparison = "both_tied"
        else:
            comparison = "both_changed"
        item = {
            "graph_index": graph, "factor": factor,
            "arm_a": a, "arm_b": b,
            "HLS_E3_qualified": e3, "static_RTL_known": known,
            "HLS_BRAM18K_delta": bram,
            "RTL_memory_module_candidate_delta": rtl,
            "HLS_BRAM18K_class": hls_class,
            "RTL_candidate_class": rtl_class,
            "comparison_class": comparison,
        }
        pairs.append(item)
        by_factor[factor][(graph, hls_class, rtl_class)] += 1
        cross[(hls_class, rtl_class)] += 1
    if len(keys) != 64 or any(
        sum(v for (g, _, _), v in by_factor[f].items() if g == graph) != 8
        for f in FACTORS for graph in (0, 1)
    ):
        raise ValueError("Original balanced factorial contrasts incomplete")
    hls = Counter(p["HLS_BRAM18K_class"] for p in pairs)
    if tuple(hls[c] for c in CATEGORIES) != (2, 47, 8, 7):
        raise ValueError("HLS estimated BRAM response differs from attested cohort")
    eligible = [p for p in pairs if p["HLS_E3_qualified"]
                and p["static_RTL_known"]]
    contingency = [
        {
            "HLS_BRAM18K_class": h,
            "RTL_candidate_class": s,
            "pair_count": cross[(h, s)],
        }
        for h in CATEGORIES for s in CATEGORIES
    ]
    grouped = [
        {
            "factor": f, "graph_index": graph,
            "planned_pairs": 8,
            "HLS_qualified_pairs": sum(
                v for (g, h, _), v in by_factor[f].items()
                if g == graph and h != "unknown"),
            "RTL_structure_known_pairs": sum(
                v for (g, _, s), v in by_factor[f].items()
                if g == graph and s != "unknown"),
            "RTL_changed_HLS_tied": sum(
                p["comparison_class"] == "RTL_changed_HLS_tied"
                for p in pairs if p["factor"] == f and p["graph_index"] == graph),
        }
        for f in FACTORS for graph in (0, 1)
    ]
    return {
        "schema": SCHEMA,
        "original_source_revision": SOURCE_REVISION,
        "original_source_cases": 32,
        "original_matched_pairs": 64,
        "HLS_E3_qualified_pairs": 57,
        "HLS_E3_unknown_pairs": 7,
        "HLS_BRAM18K_response": {c: hls[c] for c in CATEGORIES},
        "jointly_interpretable_HLS_RTL_pairs": len(eligible),
        "RTL_changed_HLS_tied_count": sum(
            p["comparison_class"] == "RTL_changed_HLS_tied" for p in pairs),
        "HLS_changed_RTL_tied_count": sum(
            p["comparison_class"] == "HLS_changed_RTL_tied" for p in pairs),
        "contingency": contingency,
        "by_factor_and_graph": grouped,
        "pairs": sorted(pairs, key=lambda p: (
            FACTORS.index(p["factor"]), p["graph_index"],
            p["arm_a"], p["arm_b"])),
        "original_vendor_calls_during_analysis": 0,
        "routed_FPGA_BRAM": None,
        "routed_FPGA_timing": None,
        "scientific_interpretation": (
            "Descriptive association between HLS BRAM estimates and "
            "statically reachable RTL memory-named module candidates. "
            "Neither structural counts nor HLS estimates establish "
            "physical BRAM allocation or a causal memory mechanism."
        ),
    }


def analyze(*, study: str | Path) -> dict:
    """Re-audit all original raw source, XML and RTL before deriving pairs."""
    from fpgai_research.experiments import cst_r2x_rtl_hierarchy_analysis as r2x

    return summarize(r2x.analyze(study=study))


def export_create_only(report: dict, *, output: str | Path) -> None:
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Paper memory translation output is create-only")
    if report.get("schema") != SCHEMA or report.get("routed_FPGA_BRAM") is not None:
        raise ValueError("Only source-attested, nonphysical bridge reports allowed")
    root.mkdir(parents=True, exist_ok=False)
    paths = ("summary.json", "paired_observations.csv")
    with (root / paths[0]).open("x", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, sort_keys=True, allow_nan=False)
        fp.write("\n")
    with (root / paths[1]).open("x", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(report["pairs"])
    digests = {name: sha256((root / name).read_bytes()).hexdigest()
               for name in paths}
    (root / "derived_sha256.json").write_text(
        json.dumps(digests, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = analyze(study=args.study)
    export_create_only(result, output=args.output)
    print(json.dumps({
        "schema": SCHEMA, "qualified_pairs": result["HLS_E3_qualified_pairs"],
        "jointly_interpretable_pairs":
            result["jointly_interpretable_HLS_RTL_pairs"],
        "new_vendor_calls": 0, "output": args.output,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
