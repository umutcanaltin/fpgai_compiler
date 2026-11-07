"""R4c: read-only numerical triage of original R4b matched Vitis CSim runs.

Uses the EXISTING source/receipt/hash-aware report parser and original R4a
population. Never runs a native tool, edits an original file, loosens a
threshold or promotes rejected CSim to HLS/physical evidence.
Matching aggregate errors do not prove pointwise identity or a cause.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from fpgai_research.experiments.cst_r4b_row_retention_native import (
    ARMS, audit_prepared,
)
from fpgai_research.experiments.results import summarize_case

SCHEMA = "fpgai-research/R4c-original-paired-csim-numerical-triage/v1"
_AGGREGATES = ("failures", "cosine", "rmse", "max_absolute_error")


def _valid_receipt(receipt: dict | None, *, samples: int, outputs: int) -> bool:
    if not isinstance(receipt, dict):
        return False
    if (type(receipt.get("samples")) is not int
            or receipt["samples"] != samples
            or type(receipt.get("outputs_per_sample")) is not int
            or receipt["outputs_per_sample"] != outputs
            or type(receipt.get("failures")) is not int
            or not 0 <= receipt["failures"] <= samples * outputs
            or type(receipt.get("passed")) is not bool
            or (receipt["passed"] and receipt["failures"] != 0)
            or not isinstance(receipt.get("source"), str)
            or not receipt["source"]):
        return False
    for key in ("cosine", "rmse", "max_absolute_error"):
        value = receipt.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            return False
    return (-1.000001 <= receipt["cosine"] <= 1.000001
            and receipt["rmse"] >= 0
            and receipt["max_absolute_error"] >= 0)


def paired_numeric_diagnostic(
    full: dict | None, row: dict | None, *,
    tolerance: float, samples: int, outputs: int,
) -> dict:
    """Interpret attested CSim receipts; keep causal claims strictly bounded."""
    if (type(tolerance) not in (int, float) or not math.isfinite(tolerance)
            or tolerance <= 0 or type(samples) is not int or samples < 1
            or type(outputs) is not int or outputs < 1):
        raise ValueError("Invalid original numerical study contract")
    total = samples * outputs
    records = {"full_H": full, "row_H": row}
    arms = {}
    for name, receipt in records.items():
        valid = _valid_receipt(receipt, samples=samples, outputs=outputs)
        arms[name] = {
            "attested_metrics_available": valid,
            "E2_passed": receipt["passed"] if valid else None,
            "failed_outputs": receipt["failures"] if valid else None,
            "failure_fraction": receipt["failures"] / total if valid else None,
            "rmse": receipt["rmse"] if valid else None,
            "max_absolute_error": receipt["max_absolute_error"] if valid else None,
            "cosine": receipt["cosine"] if valid else None,
            "max_error_to_tolerance_ratio": (
                receipt["max_absolute_error"] / tolerance if valid else None),
        }
    both = all(arm["attested_metrics_available"] for arm in arms.values())
    same = both and all(full[key] == row[key] for key in _AGGREGATES)
    if not both:
        status = "UNKNOWN_MISSING_OR_UNATTESTED_PAIR"
    elif same:
        status = "IDENTICAL_AGGREGATES_NOT_POINTWISE_IDENTICAL"
    else:
        status = "DIFFERENT_AGGREGATES_CAUSE_UNKNOWN"
    return {
        "status": status,
        "samples": samples,
        "outputs_per_sample": outputs,
        "total_outputs_per_arm": total,
        "original_absolute_tolerance": tolerance,
        "both_E2_qualified": (
            all(arm["E2_passed"] is True for arm in arms.values())
            if both else False),
        "identical_reported_aggregate_metrics": same if both else None,
        "arms": arms,
        "inference_limit": (
            "Matching aggregate errors cannot establish pointwise output "
            "identity, fixed-point causality, or hardware resource benefits."
        ),
        "E3_resource_delta": None,
        "physical_E4_delta": None,
    }


def inspect(*, study: str | Path, case_index: int = 3) -> dict:
    """Read ORIGINAL evidence only; independent of local git working tree."""
    evidence = audit_prepared(study=study)
    root = Path(study)
    summary = json.loads((root / "summary.json").read_text())
    rows = summary["rows"]
    if type(case_index) is not int or not 0 <= case_index < len(rows):
        raise ValueError("R4c case index outside original prespecified population")
    item = rows[case_index]
    expected = summary["configuration"]
    values = {}
    source_hashes = {}
    for arm in ARMS:
        folder = root / item["case"] / arm / "vitis_source"
        meta_path = folder / "manifest.json"
        if meta_path.is_symlink() or not meta_path.is_file():
            raise ValueError("Original HLS manifest unavailable or symlinked")
        meta = json.loads(meta_path.read_text())
        if (meta.get("part") != expected["part"]
                or meta.get("clock_ns") != expected["clock_ns"]
                or meta.get("numerical_tolerance")
                    != expected["csim_absolute_tolerance"]):
            raise ValueError("Original numeric/target contract changed")
        values[arm] = (meta, summarize_case(str(folder)))
        source_hashes[arm] = item[arm]["source_sha256"]
    a, b = values["full_H"][0], values["row_H"][0]
    for field in ("precision_bits", "integer_bits", "csim_sample_count",
                  "csim_outputs_per_sample", "functional_sha256"):
        if a.get(field) != b.get(field):
            raise ValueError("Unmatched original fixed-point/model controls: " + field)
    if source_hashes["full_H"] == source_hashes["row_H"]:
        raise ValueError("Full/row original source must differ")
    # summarize_case only exposes an E2 receipt when its native invocation,
    # source, Tcl, testbench, fresh log SHA and sample coverage all qualify.
    metrics = {
        arm: values[arm][1].get("csim_receipt")
        for arm in ARMS
    }
    paired = paired_numeric_diagnostic(
        metrics["full_H"], metrics["row_H"],
        tolerance=expected["csim_absolute_tolerance"],
        samples=a["csim_sample_count"],
        outputs=a["csim_outputs_per_sample"],
    )
    return {
        "schema": SCHEMA,
        "original_source_revision": evidence["source_revision"],
        "original_study": str(root),
        "original_population_pairs": len(rows),
        "inspected_case": item["case"],
        "prespecified_graph": item["graph"],
        "part": expected["part"],
        "clock_ns": expected["clock_ns"],
        "clock_uncertainty_ns": expected["clock_uncertainty_ns"],
        "precision_bits": a["precision_bits"],
        "integer_bits": a["integer_bits"],
        "original_source_sha256": source_hashes,
        "original_csim_receipts": metrics,
        "original_csim_warnings": {
            arm: values[arm][1].get("warnings", []) for arm in ARMS
        },
        "paired_numeric_diagnostic": paired,
        "native_vendor_calls_during_triage": 0,
        "original_source_files_modified": False,
        "original_tolerance_modified": False,
        "HLS_E3_qualified_from_this_diagnostic": False,
        "physical_E4_qualified_from_this_diagnostic": False,
    }


def run(*, study: str | Path, case_index: int = 3,
        output: str | Path | None = None) -> dict:
    result = inspect(study=study, case_index=case_index)
    if output is not None:
        path = Path(output)
        original = Path(study).resolve()
        target = path.resolve()
        if target == original or original in target.parents:
            raise ValueError("R4c must not write inside original R4a evidence")
        if path.exists() or path.is_symlink():
            raise FileExistsError("R4c diagnostic output must be create-only")
        path.mkdir(parents=True, exist_ok=False)
        report = json.dumps(result, indent=2, sort_keys=True,
                            allow_nan=False) + "\n"
        (path / "diagnostic.json").write_text(report)
        (path / "artifact_sha256.json").write_text(json.dumps({
            "diagnostic.json": sha256(report.encode()).hexdigest()
        }, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--case-index", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = run(study=args.study, case_index=args.case_index,
                 output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
