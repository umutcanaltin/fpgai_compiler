"""R2s prespecified source-only Phi capacity scaling and bound falsification.

Source E0 only. No C++/GCC, Vitis/Vivado, learned RL or hardware memory
claims. Applies the *existing* fanout resolver, actual R2f event-order
checker, R2h source layout and R2q alternative lifetime accounting.

The reviewer's S_Phi <= R*(w-1) is TESTED here for each measured
scalar-versus-architecture pairing; it is not asserted as a theorem.
A violating case remains in the receipts and summary. All source
capacity effects and ratios must be interpreted under the chosen
committed-value and hypothetical in-place conventions separately.

Only the explicitly bounded template with 2..4 MatMul readers,
dimensions <=16 and PE widths in {1,2,4} is in scope.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import re
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import source_layout
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.source_allocation_baselines import (
    source_baseline_ladder,
)

SCHEMA = "fpgai-research/cst-R2s-bounded-phi-source-scaling-E0/v1"
AUDIT_SCHEMA = "fpgai-research/cst-R2s-recomputed-source-scaling-audit/v1"
SHAPES = (
    (2, 2, 3, 4),
    (2, 3, 5, 4),
    (3, 3, 4, 4),
    (4, 2, 4, 8),
    (8, 2, 8, 8),
    (16, 2, 16, 8),
    (16, 2, 16, 16),
)
FANOUTS = (2, 3, 4)
POLICIES = (
    "scalar", "first_pes2", "last_pes2", "all_pes2",
    "first_pes4", "last_pes4", "all_pes4",
)
MAX_CASES = len(SHAPES) * len(FANOUTS) * len(POLICIES)

CSV_COLUMNS = (
        "shape_index", "shape", "fanout_readers", "policy", "reader_pes",
        "scalar_committed_slots", "intervened_committed_slots",
        "scalar_hypothetical_inplace_slots", "intervened_hypothetical_inplace_slots",
        "committed_saved_slots", "hypothetical_inplace_saved_slots",
        "proposed_reader_width_bound",
        "committed_observed_satisfies_proposed_upper_bound",
        "committed_savings_share_of_scalar_total_declared_pct",
        "scalar_total_declared_mutable_elements",
        "source_scratch_nominal_16bit_bits", "source_event_sha256",
)


def clean_source_revision() -> str:
    repo = Path(__file__).resolve().parents[3]
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    if dirty.strip():
        raise RuntimeError("R2s original source requires clean tracked Git revision")
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("R2s requires exact full source Git SHA")
    return revision


def widths_for(readers: int, policy: str) -> tuple[int, ...]:
    if type(readers) is not int or readers not in FANOUTS or policy not in POLICIES:
        raise ValueError("R2s requires bounded readers and known PE policy")
    if policy == "scalar":
        return (1,) * readers
    direction, width = policy.split("_pes")
    width = int(width)
    if direction == "first":
        return (width,) + (1,) * (readers - 1)
    if direction == "last":
        return (1,) * (readers - 1) + (width,)
    if direction == "all":
        return (width,) * readers
    raise AssertionError("Unknown policy direction")


def _validate_shape_limit(shape_limit: int) -> None:
    if type(shape_limit) is not int or not 1 <= shape_limit <= len(SHAPES):
        raise ValueError("R2s shape_limit must be in 1..7")


def calculate_rows(*, shape_limit: int = 7) -> list[dict]:
    _validate_shape_limit(shape_limit)
    rows = []
    for shape_index, dims in enumerate(SHAPES[:shape_limit]):
        m, k, n, q = dims
        for readers in FANOUTS:
            graph = sample_fanout(readers, m=m, k=k, n=n, q=q)
            for policy in POLICIES:
                widths = widths_for(readers, policy)
                base = ArchitectureRequest()
                arch = resolve(graph, base, {
                    f"reader{i}": replace(base, pes=w)
                    for i, w in enumerate(widths) if w != 1
                })
                if tuple(
                    arch.points[idx+1].effective.pes
                    for idx in range(readers)
                ) != widths:
                    raise AssertionError(
                        "R2s requested reader PE policy not honored by resolver"
                    )
                spec, assigned, capacity, exact, events = source_layout(
                    arch, layout="per_element_optimal"
                )
                accounting = source_baseline_ladder(
                    events,
                    producer_tensor=spec.h,
                    reader_outputs=dict(zip(spec.readers, spec.reader_values)),
                    accumulator_elements=readers * m * q,
                    counter_elements=readers * m * q,
                )
                if capacity != exact.unit_slots or (
                    accounting["element_exact_committed_value_slots"] != capacity
                ):
                    raise AssertionError("R2s source compiler and accounting diverge")
                if len(assigned) != accounting["value_interval_count"]:
                    raise AssertionError("R2s incomplete original value population")
                rows.append({
                    "shape_index": shape_index,
                    "shape": list(dims),
                    "fanout_readers": readers,
                    "policy": policy,
                    "reader_pes": list(widths),
                    "max_reader_pes": max(widths),
                    "source_event_sha256": exact.schedule_sha256,
                    "source_value_count": accounting["value_interval_count"],
                    "source_disjoint_slots": accounting["disjoint_no_reuse_slots"],
                    "source_tensor_first_fit_slots": accounting[
                        "whole_tensor_first_fit_feasible_slots"
                    ],
                    "source_element_exact_committed_slots": capacity,
                    "hypothetical_first_mac_inplace_peak_slots": accounting[
                        "hypothetical_inplace_reader_value_slots"
                    ],
                    "original_r2h_separate_accumulator_elements": readers * m * q,
                    "original_r2h_separate_counter_elements": readers * m * q,
                    "original_r2h_declared_mutable_elements": (
                        capacity + 2 * readers * m * q
                    ),
                    "original_r2h_declared_mutable_elements_disjoint": (
                        accounting[
                            "actual_r2h_all_declared_mutable_elements_disjoint"
                        ]
                    ),
                    "original_r2h_scratch_nominal_16bit_bits": 16 * capacity,
                    "new_gcc_executions": 0,
                    "new_vendor_calls": 0,
                })
    if len(rows) != shape_limit * len(FANOUTS) * len(POLICIES):
        raise AssertionError("R2s original cohort missing cases")
    return rows


def compare_rows(rows: list[dict], *, shape_limit: int) -> list[dict]:
    _validate_shape_limit(shape_limit)
    expected = shape_limit * len(FANOUTS) * len(POLICIES)
    if len(rows) != expected:
        raise ValueError("R2s missing candidate policies or graphs")
    comparisons = []
    for shape_index in range(shape_limit):
        for readers in FANOUTS:
            group = [
                r for r in rows
                if r["shape_index"] == shape_index and
                   r["fanout_readers"] == readers
            ]
            if [r["policy"] for r in group] != list(POLICIES):
                raise ValueError("R2s noncanonical mapping policy cohort")
            scalar = group[0]
            for case in group[1:]:
                bound = readers * (case["max_reader_pes"] - 1)
                committed = (
                    scalar["source_element_exact_committed_slots"] -
                    case["source_element_exact_committed_slots"]
                )
                inplace = (
                    scalar["hypothetical_first_mac_inplace_peak_slots"] -
                    case["hypothetical_first_mac_inplace_peak_slots"]
                )
                total = scalar["original_r2h_declared_mutable_elements"]
                if total <= 0:
                    raise AssertionError("R2s invalid total mutable array population")
                comparisons.append({
                    "shape_index": shape_index,
                    "shape": scalar["shape"],
                    "fanout_readers": readers,
                    "policy": case["policy"],
                    "reader_pes": case["reader_pes"],
                    "scalar_committed_slots": scalar[
                        "source_element_exact_committed_slots"
                    ],
                    "intervened_committed_slots": case[
                        "source_element_exact_committed_slots"
                    ],
                    "scalar_hypothetical_inplace_slots": scalar[
                        "hypothetical_first_mac_inplace_peak_slots"
                    ],
                    "intervened_hypothetical_inplace_slots": case[
                        "hypothetical_first_mac_inplace_peak_slots"
                    ],
                    "committed_saved_slots": committed,
                    "hypothetical_inplace_saved_slots": inplace,
                    "proposed_reader_width_bound": bound,
                    "committed_observed_satisfies_proposed_upper_bound": committed <= bound,
                    "committed_savings_share_of_scalar_total_declared_pct": (
                        100.0 * committed / total
                    ),
                    "scalar_total_declared_mutable_elements": total,
                    "source_scratch_nominal_16bit_bits": 16 * case[
                        "source_element_exact_committed_slots"
                    ],
                    "source_event_sha256": case["source_event_sha256"],
                })
    if len(comparisons) != shape_limit * len(FANOUTS) * (len(POLICIES)-1):
        raise AssertionError("R2s unexpected controlled contrast denominator")
    return comparisons


def summarize(rows: list[dict], comparisons: list[dict], *,
              shape_limit: int, revision: str) -> dict:
    total = shape_limit * len(FANOUTS) * len(POLICIES)
    contrasts = shape_limit * len(FANOUTS) * (len(POLICIES)-1)
    if len(rows) != total or len(comparisons) != contrasts:
        raise ValueError("R2s incomplete source scaling denominator")
    def counts(key: str) -> dict:
        values = [c[key] for c in comparisons]
        return {
            "reduced": sum(v > 0 for v in values),
            "tied": sum(v == 0 for v in values),
            "increased": sum(v < 0 for v in values),
            "maximum_saved_slots": max(values),
            "minimum_saved_slots": min(values),
        }
    violations = [
        {"shape": c["shape"], "fanout_readers": c["fanout_readers"],
         "policy": c["policy"],
         "saved_slots": c["committed_saved_slots"],
         "proposed_upper_bound": c["proposed_reader_width_bound"]}
        for c in comparisons
        if not c["committed_observed_satisfies_proposed_upper_bound"]
    ]
    large = [c for c in comparisons if c["shape"] == [16,2,16,16]]
    return {
        "schema": SCHEMA, "source_revision": revision,
        "shape_limit": shape_limit, "prespecified_shapes": [
            list(v) for v in SHAPES[:shape_limit]
        ],
        "reader_degrees": list(FANOUTS),
        "pe_policies": list(POLICIES),
        "graph_count": shape_limit * len(FANOUTS),
        "source_configuration_count": total,
        "phi_intervention_count": contrasts,
        "committed_phi_effects": counts("committed_saved_slots"),
        "hypothetical_inplace_phi_effects": counts(
            "hypothetical_inplace_saved_slots"
        ),
        "proposed_upper_bound": "R * (max_reader_PE - 1), versus scalar, fixed source issue",
        "proposed_bound_violations": violations,
        "proposed_bound_all_tested_interventions_hold": not violations,
        "max_abs_committed_savings_percent_of_total_declared": max(
            abs(c["committed_savings_share_of_scalar_total_declared_pct"])
            for c in comparisons
        ),
        "full_size_16x2x16x16_contrasts": large,
        "max_disjoint_scratch_nominal_16bit_bits": 16 * max(
            r["source_disjoint_slots"] for r in rows
        ),
        "max_exact_scratch_nominal_16bit_bits": 16 * max(
            r["source_element_exact_committed_slots"] for r in rows
        ),
        "original_gcc_executions": 0, "new_vitis_calls": 0,
        "new_vivado_calls": 0, "rl_training_calls": 0,
        "limits": [
            "Finite selected R2f fanout graphs, not exhaustive up to dimensions 16",
            "Reported R*(w-1) is a tested conjecture, not an independent formal proof",
            "Actual R2h includes separate complete accumulator and counter arrays",
            "Hypothetical in-place first-MAC liveness does not describe emitted R2h C++",
            "Nominal 16-bit source scratch is not a mapped BRAM primitive or bandwidth",
            "Source E0 event/lifetime checks only; no new GCC/HLS/physical data",
        ],
    }


def run(*, output: str | Path, shape_limit: int = 7) -> dict:
    _validate_shape_limit(shape_limit)
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R2s original scaling source study strictly create-only")
    revision = clean_source_revision()
    rows = calculate_rows(shape_limit=shape_limit)
    comparisons = compare_rows(rows, shape_limit=shape_limit)
    summary = summarize(rows, comparisons,
                        shape_limit=shape_limit, revision=revision)
    # Validate the entire source evidence schema BEFORE creating the
    # immutable original study directory, never leave a partial cohort
    # because a new source field was omitted from the CSV export.
    if not comparisons or any(set(case) != set(CSV_COLUMNS) for case in comparisons):
        raise ValueError("R2s CSV schema does not match complete Phi contrasts")
    root.mkdir(parents=True)
    for name, data in (
        ("cases.json", rows),
        ("comparisons.json", comparisons),
        ("summary.json", summary),
    ):
        with (root/name).open("x", encoding="utf-8") as output_file:
            json.dump(data, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
    with (root/"comparisons.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for case in comparisons:
            writer.writerow({
                **case,
                "shape": json.dumps(case["shape"], separators=(",",":")),
                "reader_pes": json.dumps(
                    case["reader_pes"], separators=(",",":"
                )),
            })
    return summary


def audit(*, study: str | Path) -> dict:
    """Same-source re-derivation and receipt integrity; NO GCC/vendor calls."""
    root = Path(study)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("R2s original source study folder missing or symlinked")
    sources = {
        name: root/name
        for name in ("cases.json", "comparisons.json", "summary.json",
                     "comparisons.csv")
    }
    if any(not p.is_file() or p.is_symlink() for p in sources.values()):
        raise ValueError("R2s original source receipts missing or symlinked")
    summary = json.loads(sources["summary.json"].read_text())
    limit = summary.get("shape_limit")
    _validate_shape_limit(limit)
    if (summary.get("schema") != SCHEMA
            or summary.get("source_revision") != clean_source_revision()):
        raise ValueError("R2s source revision mismatch; checkout original commit")
    original_rows = json.loads(sources["cases.json"].read_text())
    computed_rows = calculate_rows(shape_limit=limit)
    if original_rows != computed_rows:
        raise ValueError("R2s original source event capacity rows modified")
    original_contrasts = json.loads(sources["comparisons.json"].read_text())
    computed_contrasts = compare_rows(computed_rows, shape_limit=limit)
    if original_contrasts != computed_contrasts:
        raise ValueError("R2s original Phi contrast rows modified")
    computed_summary = summarize(
        computed_rows, computed_contrasts,
        shape_limit=limit, revision=summary["source_revision"]
    )
    if summary != computed_summary:
        raise ValueError("R2s result denominator or bound claim modified")
    with sources["comparisons.csv"].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise ValueError("R2s original CSV columns differ from evidence schema")
        got = list(reader)
    if len(got) != len(computed_contrasts):
        raise ValueError("R2s CSV incomplete")
    for actual, expected in zip(got, computed_contrasts):
        if (int(actual["shape_index"]) != expected["shape_index"]
                or json.loads(actual["shape"]) != expected["shape"]
                or int(actual["fanout_readers"]) != expected["fanout_readers"]
                or actual["policy"] != expected["policy"]
                or json.loads(actual["reader_pes"]) != expected["reader_pes"]
                or any(int(actual[field]) != expected[field] for field in (
                    "scalar_committed_slots", "intervened_committed_slots",
                    "scalar_hypothetical_inplace_slots",
                    "intervened_hypothetical_inplace_slots",
                    "committed_saved_slots", "hypothetical_inplace_saved_slots",
                    "proposed_reader_width_bound",
                    "scalar_total_declared_mutable_elements",
                    "source_scratch_nominal_16bit_bits"))
                or (actual["committed_observed_satisfies_proposed_upper_bound"]
                    == "True") is not expected[
                    "committed_observed_satisfies_proposed_upper_bound"
                ]
                or float(actual[
                    "committed_savings_share_of_scalar_total_declared_pct"
                ]) != expected[
                    "committed_savings_share_of_scalar_total_declared_pct"
                ]
                or actual["source_event_sha256"] != expected["source_event_sha256"]):
            raise ValueError("R2s original CSV changed after source derivation")
    return {
        "schema": AUDIT_SCHEMA, "original_source_revision":summary["source_revision"],
        "recomputed_source_cases":len(computed_rows),
        "recomputed_phi_interventions":len(computed_contrasts),
        "committed_phi_effects":computed_summary["committed_phi_effects"],
        "hypothetical_inplace_phi_effects":computed_summary[
            "hypothetical_inplace_phi_effects"
        ],
        "proposed_bound_violations":computed_summary[
            "proposed_bound_violations"
        ],
        "new_original_gcc_calls":0, "new_vendor_calls":0,
        "rl_training_calls":0,
        "qualification":"Read-only same-compiler E0 source recomputation; not independent GCC, RTL or hardware",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--study")
    parser.add_argument("--audit-output")
    parser.add_argument("--shape-limit", type=int, default=len(SHAPES))
    args = parser.parse_args(argv)
    if bool(args.output) == bool(args.study):
        parser.error("Exactly one of --output or --study required")
    if args.study:
        result = audit(study=args.study)
        if args.audit_output:
            p = Path(args.audit_output)
            if p.exists() or p.is_symlink():
                raise FileExistsError("R2s audit receipt strictly create-only")
            with p.open("x",encoding="utf-8") as f:
                json.dump(result,f,indent=2,sort_keys=True)
                f.write("\n")
    else:
        if args.audit_output:
            parser.error("--audit-output only accompanies --study")
        result = run(output=args.output,shape_limit=args.shape_limit)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
