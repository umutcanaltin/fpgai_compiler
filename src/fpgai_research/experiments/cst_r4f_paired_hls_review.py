"""R4f: original-evidence-only paired HLS and numerical contrast report.

No tool execution, no rewriting R4a/R4d original studies, and no
post-hoc selection of faster/smaller arms. Uses the independent R4c
original E2 and R4e original E2/E3 audits, keeps all planned format
pairs and preserves unknowns as null. HLS estimates are not E4.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from fpgai_research.experiments import cst_r4c_csim_numerical_triage as r4c
from fpgai_research.experiments import cst_r4d_matched_numeric_formats as r4d
from fpgai_research.experiments import cst_r4e_numeric_format_native as r4e
from fpgai_research.experiments.cst_r4b_row_retention_native import ARMS, RESOURCES

SCHEMA = "fpgai-research/R4f-read-only-paired-HLS-numerical-review/v1"


def _number(value, *, nonnegative: bool = False) -> bool:
    return (type(value) in (int, float)
            and isfinite(value)
            and (not nonnegative or value >= 0))


def compare_pair(full: dict, row: dict, *, bits: int, integer: int,
                 requested_ns: float, uncertainty_ns: float) -> dict:
    """Derive a WITHIN-FORMAT E3 paired result, never inventing metrics."""
    if (not _number(requested_ns) or requested_ns <= 0
            or not _number(uncertainty_ns, nonnegative=True)
            or uncertainty_ns >= requested_ns
            or not isinstance(full, dict) or not isinstance(row, dict)
            or full.get("format") != [bits, integer]
            or row.get("format") != [bits, integer]
            or full.get("arm") != "full_H"
            or row.get("arm") != "row_H"):
        raise ValueError("R4f invalid paired source/clock identity")
    both_e2 = (full.get("E2_qualified") is True
               and row.get("E2_qualified") is True)
    both_e3 = (both_e2 and full.get("E3_qualified") is True
               and row.get("E3_qualified") is True)
    numerical = {
        arm: {
            "E2_qualified": obs.get("E2_qualified") is True,
            "failed_outputs": (obs.get("CSim_failed_outputs")
                               if obs.get("CSim_numerical_receipt_attested") is True
                               else None),
            "max_absolute_error": (obs.get("CSim_max_absolute_error")
                                   if obs.get("CSim_numerical_receipt_attested") is True
                                   else None),
            "rmse": (obs.get("CSim_rmse")
                     if obs.get("CSim_numerical_receipt_attested") is True else None),
            "status": obs.get("status"),
        } for arm, obs in (("full_H", full), ("row_H", row))
    }
    result = {
        "format_total_integer_bits": [bits, integer],
        "matched": True,
        "both_E2_qualified": both_e2,
        "both_E3_qualified": both_e3,
        "numerical": numerical,
        "status": ("QUALIFIED_E3_HLS_ESTIMATES_ONLY" if both_e3
                   else "UNKNOWN_OR_REJECTED_E3_PAIR"),
        "requested_clock_ns": requested_ns,
        "explicit_uncertainty_ns": uncertainty_ns,
        "effective_HLS_budget_ns": requested_ns - uncertainty_ns,
        "estimated_resource_control": {key: None for key in RESOURCES},
        "estimated_resource_intervention": {key: None for key in RESOURCES},
        "estimated_resource_delta_row_minus_full": {
            key: None for key in RESOURCES},
        "HLS_worst_cycles_control": None,
        "HLS_worst_cycles_intervention": None,
        "HLS_worst_cycles_delta_row_minus_full": None,
        "HLS_requested_period_latency_ns_control": None,
        "HLS_requested_period_latency_ns_intervention": None,
        "HLS_requested_period_latency_ns_delta_row_minus_full": None,
        "HLS_estimated_clock_ns_control": None,
        "HLS_estimated_clock_ns_intervention": None,
        "HLS_estimated_clock_ns_delta_row_minus_full": None,
        "HLS_effective_budget_margin_ns_control": None,
        "HLS_effective_budget_margin_ns_intervention": None,
        "physical_E4": None,
        "physical_resource_delta": None,
    }
    if not both_e3:
        return result
    resource_a = full.get("HLS_estimated_resources")
    resource_b = row.get("HLS_estimated_resources")
    if (not isinstance(resource_a, dict) or not isinstance(resource_b, dict)
            or any(not _number(resource_a.get(key), nonnegative=True)
                   or not _number(resource_b.get(key), nonnegative=True)
                   for key in RESOURCES)):
        raise ValueError("R4f qualified E3 result missing comparable resources")
    cycles_a, cycles_b = full.get("HLS_worst_cycles"), row.get("HLS_worst_cycles")
    clock_a = full.get("HLS_estimated_clock_ns")
    clock_b = row.get("HLS_estimated_clock_ns")
    if (type(cycles_a) is not int or type(cycles_b) is not int
            or cycles_a <= 0 or cycles_b <= 0
            or not _number(clock_a) or not _number(clock_b)
            or clock_a <= 0 or clock_b <= 0):
        raise ValueError("R4f qualified E3 result missing valid timing/cycles")
    # Independent internal guard: only make tradeoff claims for reports that
    # satisfy the SAME effective HLS clock constraint. Not physical Fmax.
    budget = requested_ns - uncertainty_ns
    if clock_a > budget + 1e-9 or clock_b > budget + 1e-9:
        raise ValueError("R4f claimed qualified E3 violates effective HLS budget")
    a = {key: resource_a[key] for key in RESOURCES}
    b = {key: resource_b[key] for key in RESOURCES}
    result.update({
        "estimated_resource_control": a,
        "estimated_resource_intervention": b,
        "estimated_resource_delta_row_minus_full": {
            key: b[key] - a[key] for key in RESOURCES},
        "HLS_worst_cycles_control": cycles_a,
        "HLS_worst_cycles_intervention": cycles_b,
        "HLS_worst_cycles_delta_row_minus_full": cycles_b - cycles_a,
        "HLS_requested_period_latency_ns_control": cycles_a * requested_ns,
        "HLS_requested_period_latency_ns_intervention": cycles_b * requested_ns,
        "HLS_requested_period_latency_ns_delta_row_minus_full": (
            (cycles_b - cycles_a) * requested_ns),
        "HLS_estimated_clock_ns_control": clock_a,
        "HLS_estimated_clock_ns_intervention": clock_b,
        "HLS_estimated_clock_ns_delta_row_minus_full": clock_b - clock_a,
        "HLS_effective_budget_margin_ns_control": budget - clock_a,
        "HLS_effective_budget_margin_ns_intervention": budget - clock_b,
    })
    return result


def inspect(*, study: str | Path, original_study: str | Path) -> dict:
    """Recheck original E2/E3 receipts; no vendor launches or evidence writes."""
    native = r4e.audit_native(study=study, original_study=original_study)
    earlier = r4c.inspect(study=original_study, case_index=r4d.CASE_INDEX)
    if (native["original_R4a_source_revision"]
            != earlier["original_source_revision"]):
        raise ValueError("R4f source revision differs between original studies")
    observations = native["observations"]
    if len(observations) != 2 * len(r4d.FORMATS):
        raise ValueError("R4f incomplete preregistered native observations")
    planned = r4d.plan()
    pairs = [
        compare_pair(
            observations[2 * i], observations[2 * i + 1],
            bits=bits, integer=integer,
            requested_ns=planned["clock_ns"],
            uncertainty_ns=planned["uncertainty_ns"])
        for i, (bits, integer) in enumerate(r4d.FORMATS)
    ]
    old = earlier["paired_numeric_diagnostic"]
    earlier_arms = old["arms"]
    # An old E2 failure is not a new E2 result and is never used to
    # impute a missing new 16/6 native run.
    prior = {
        "format_total_integer_bits": [16, 6],
        "case": earlier["inspected_case"],
        "E2_full_H": earlier_arms["full_H"]["E2_passed"],
        "E2_row_H": earlier_arms["row_H"]["E2_passed"],
        "CSim_failed_outputs_full_H": earlier_arms["full_H"]["failed_outputs"],
        "CSim_failed_outputs_row_H": earlier_arms["row_H"]["failed_outputs"],
        "CSim_max_abs_full_H": earlier_arms["full_H"]["max_absolute_error"],
        "CSim_max_abs_row_H": earlier_arms["row_H"]["max_absolute_error"],
        "same_aggregate_metrics_not_pointwise_proof":
            old["identical_reported_aggregate_metrics"],
        "HLS_E3_paired_resources_qualified": False,
    }
    return {
        "schema": SCHEMA,
        "original_R4a_source_revision": earlier["original_source_revision"],
        "new_R4d_source_revision": native["source_revision"],
        "original_R4a_case_003": prior,
        "prespecified_format_pairs": len(pairs),
        "qualified_E3_format_pairs": sum(p["both_E3_qualified"] for p in pairs),
        "rejected_or_missing_E3_format_pairs":
            sum(not p["both_E3_qualified"] for p in pairs),
        "original_R4e_invocation_files": native["original_native_invocation_files"],
        "pairs": pairs,
        "scientific_interpretation": [
            "A higher-precision numerical E2 pass alone does not isolate rounding, saturation or weights.",
            "Within-format row-retention effects are HLS estimates, not physical FPGA BRAM or Fmax.",
            "Unexecuted numerical formats remain UNKNOWN; preserve full population.",
            "Do not compare estimated periods as routed clocks or achieved throughput.",
            "No external novelty or cross-operator generalization established.",
        ],
        "new_native_Vitis_calls_during_report": 0,
        "new_native_Vivado_calls_during_report": 0,
        "physical_E4": None,
    }


def run(*, study: str | Path, original_study: str | Path,
        output: str | Path | None = None) -> dict:
    result = inspect(study=study, original_study=original_study)
    if output is None:
        return result
    out = Path(output)
    dest = out.resolve()
    roots = (Path(study).resolve(), Path(original_study).resolve())
    if any(dest == root or root in dest.parents for root in roots):
        raise ValueError("R4f must write outside both original evidence roots")
    if out.exists() or out.is_symlink():
        raise FileExistsError("R4f report output must be create-only")
    out.mkdir(parents=True, exist_ok=False)
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    (out / "report.json").write_text(payload)
    (out / "artifact_sha256.json").write_text(
        json.dumps({"report.json": sha256(payload.encode()).hexdigest()},
                   sort_keys=True, indent=2) + "\n")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--original-study", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    print(json.dumps(run(
        study=args.study, original_study=args.original_study,
        output=args.output), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
