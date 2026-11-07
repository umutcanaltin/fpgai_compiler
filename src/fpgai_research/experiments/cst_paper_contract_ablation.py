"""Controlled source-contract ablation on existing typed MatMul fanout traces.

The comparison holds graph, mapping and events constant and removes ONLY
the Phi issue-order predicate from the full source contract. The baseline checks source value readiness, last-reader lifetime AND
Phi owner coordinates, but omits the exact Phi READ issue order. It is an internal ablation, NOT another compiler,
whole-output equivalence oracle, or a measured HLS/physical speedup.

The original R3j study remains separate and is never overwritten.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

SCHEMA = "fpgai-research/paper-source-contract-predicate-ablation/v1"
INPUT_SCHEMA = "fpgai-research/R3j-C1-heldout-source-semantics-v1"
LABELS = (
    "canonical", "safe_wrong_issue_order", "early_h_read",
    "early_h_commit", "wrong_phi_owner", "missing_final_read",
    "duplicate_producer_term", "overlapping_reader_scratch",
    "unknown_final_tensor",
)
CSV_FIELDS = (
    "reader_count", "source_policy", "mutation_class",
    "value_safety_only_accepts", "full_contract_accepts",
    "declared_contract_valid",
)


def summarize(original: dict) -> dict:
    """Compare owner-aware source safety with the additional issue-order predicate."""
    if (
        original.get("schema") != INPUT_SCHEMA
        or original.get("prespecified_source_cases") != 24
        or original.get("total_trace_variants") != 216
        or original.get("variants_per_case") != 9
        or original.get("matched_independent_oracle") != 216
        or original.get("independent_oracle_disagreements") != 0
        or original.get("original_vitis_calls") != 0
        or original.get("original_vivado_calls") != 0
        or original.get("trained_agents") != 0
    ):
        raise ValueError("Incomplete or unqualified source-event population")
    rows = original.get("rows")
    if type(rows) is not list or len(rows) != 24:
        raise ValueError("Missing source topology/architecture profiles")
    seen_profiles = set()
    observations = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed source profile")
        config = row.get("case")
        if not isinstance(config, list) or len(config) != 6:
            raise ValueError("Invalid prespecified graph/mapping configuration")
        readers, m, k, n, q, policy = config
        profile = tuple(config)
        if (type(readers) is not int or readers not in (2, 3, 4)
                or any(type(v) is not int or v < 1 for v in (m, k, n, q))
                or policy not in ("scalar", "mixed_phi")
                or profile in seen_profiles):
            raise ValueError("Duplicate or unsupported graph/mapping profile")
        seen_profiles.add(profile)
        outcomes = row.get("outcomes")
        if (type(outcomes) is not list or len(outcomes) != 9
                or [a.get("name") for a in outcomes] != list(LABELS)):
            raise ValueError("Missing canonical, issue-order or fault controls")
        for observation in outcomes:
            name = observation["name"]
            safe = observation.get("safe")
            order = observation.get("phi_issue_conformant")
            if type(safe) is not bool or order not in (True, False, None):
                raise ValueError("Missing/invalid original semantic verdict")
            if name == "canonical" and not (safe and order is True):
                raise ValueError("Positive control did not satisfy both predicates")
            if name == "safe_wrong_issue_order" and not (
                    safe and order is False):
                raise ValueError("Issue-order negative control is not value-safe")
            if name not in ("canonical", "safe_wrong_issue_order") and safe:
                raise ValueError("Selected storage/readiness negative became safe")
            for field in ("source_trace_sha256", "source_offsets_sha256"):
                digest = observation.get(field)
                if not isinstance(digest, str) or not re.fullmatch(
                        r"[0-9a-f]{64}", digest):
                    raise ValueError("Trace/offset identity missing")
            valid = name == "canonical"
            observations.append({
                "reader_count": readers, "source_policy": policy,
                "mutation_class": name,
                "value_safety_only_accepts": safe,
                "full_contract_accepts": bool(safe and order),
                "declared_contract_valid": valid,
            })
    if len(observations) != 216 or len(seen_profiles) != 24:
        raise ValueError("Ablation trace population changed")
    value_accepted = sum(r["value_safety_only_accepts"] for r in observations)
    full_accepted = sum(r["full_contract_accepts"] for r in observations)
    value_invalid_accepted = sum(
        r["value_safety_only_accepts"] and not r["declared_contract_valid"]
        for r in observations)
    full_invalid_accepted = sum(
        r["full_contract_accepts"] and not r["declared_contract_valid"]
        for r in observations)
    if (value_accepted, full_accepted, value_invalid_accepted,
            full_invalid_accepted) != (48, 24, 24, 0):
        raise ValueError("Source-contract ablation no longer matches predeclared controls")
    by_class = [
        {
            "mutation_class": label,
            "population": sum(r["mutation_class"] == label for r in observations),
            "value_only_accepts": sum(
                r["mutation_class"] == label
                and r["value_safety_only_accepts"] for r in observations),
            "full_contract_accepts": sum(
                r["mutation_class"] == label
                and r["full_contract_accepts"] for r in observations),
        }
        for label in LABELS
    ]
    return {
        "schema": SCHEMA,
        "source_graph_mapping_profiles": len(seen_profiles),
        "compared_event_traces": len(observations),
        "declared_valid_traces": 24,
        "selected_invalid_traces": 192,
        "value_safety_only_accepts": value_accepted,
        "full_contract_accepts": full_accepted,
        "value_only_accepts_invalid_issue_controls": value_invalid_accepted,
        "full_contract_accepts_invalid_issue_controls": full_invalid_accepted,
        "valid_controls_rejected_by_full_contract": 0,
        "by_mutation_class": by_class,
        "observations": observations,
        "original_compiler_or_vendor_calls_during_summary": 0,
        "hardware_BRAM": None, "hardware_latency": None,
        "external_compiler_baseline": "NOT_RUN",
        "interpretation": (
            "Removing operator-local issue-order checks admits the 24 "
            "prespecified readiness-safe, mapping-order-invalid traces. "
            "These are generated controls under a shared bounded graph "
            "specification; not an independent competitor comparison."
        ),
    }


def analyze() -> dict:
    from fpgai_research.experiments import cst_r3j_c1_heldout_semantics as r3j

    return summarize(r3j.run())


def export_create_only(report: dict, *, output: str | Path,
                       source_revision: str) -> None:
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Contract ablation output is create-only")
    if (report.get("schema") != SCHEMA
            or not isinstance(source_revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None):
        raise ValueError("Missing verified experiment schema/source revision")
    root.mkdir(parents=True, exist_ok=False)
    with (root / "summary.json").open("x", encoding="utf-8") as fp:
        json.dump({**report, "experiment_source_revision": source_revision},
                  fp, indent=2, sort_keys=True, allow_nan=False)
        fp.write("\n")
    with (root / "observations.csv").open("x", encoding="utf-8",
                                          newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(report["observations"])
    digests = {name: sha256((root / name).read_bytes()).hexdigest()
               for name in ("summary.json", "observations.csv")}
    (root / "derived_sha256.json").write_text(
        json.dumps(digests, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from fpgai_research.experiments.cst_r3d_c1c2_finite_contract_ablation import (
        _git_revision,
    )

    revision = _git_revision()
    report = analyze()
    export_create_only(report, output=args.output, source_revision=revision)
    print(json.dumps({
        "schema": SCHEMA, "source_revision": revision,
        "value_only_accepted_invalid_controls":
            report["value_only_accepts_invalid_issue_controls"],
        "full_contract_accepted_invalid_controls":
            report["full_contract_accepts_invalid_issue_controls"],
        "new_vendor_calls": 0, "output": args.output,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
