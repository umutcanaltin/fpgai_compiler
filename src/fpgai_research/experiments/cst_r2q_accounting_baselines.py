"""R2q independent-review response A/B: no-reuse, tensor, element, in-place.

Prospective 36-case source E0 accounting analysis. All 36 source
architectures reproduce the original R2f/R2h catalog but are newly
computed and source-revision pinned here. This is NOT a replay of
original frozen HLS/Vivado data, and not a new original GCC execution.

The in-place arm is hypothetical; it changes reader-output lifetime
semantics to include accumulation before COMMIT. The actual current
R2h C++ stores accumulators and scratch values separately, so one
MUST NOT claim that the in-place number is measured R2h utilisation.
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
from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import (
    emit_allocated_fanout_cpp, source_layout,
)
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r2f_fanout_campaign import (
    FANOUTS, POLICY_NAMES, SHAPES, policy_widths,
)
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.source_allocation_baselines import (
    source_baseline_ladder,
)

SCHEMA = "fpgai-research/cst-R2q-36-case-adversarial-memory-accounting/v1"
CASE_COUNT = len(FANOUTS) * len(SHAPES) * len(POLICY_NAMES)


def original_source_revision() -> str:
    repo = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    if status.strip():
        raise RuntimeError("R2q requires clean tracked Git source to pin new findings")
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    if re.fullmatch(r"[a-f0-9]{40}", revision) is None:
        raise RuntimeError("R2q full source revision pin unavailable")
    return revision


def calculate_catalog() -> list[dict]:
    rows = []
    for readers in FANOUTS:
        for dims in SHAPES:
            m, k, n, q = dims
            model = sample_fanout(readers, m=m, k=k, n=n, q=q)
            for policy in POLICY_NAMES:
                widths = policy_widths(readers, policy)
                default = ArchitectureRequest()
                arch = resolve(model, default, {
                    f"reader{i}": replace(default, pes=width)
                    for i, width in enumerate(widths) if width != 1
                })
                spec, _assigned, source_slots, optimum, events = source_layout(
                    arch, layout="per_element_optimal")
                accounting = source_baseline_ladder(
                    events,
                    producer_tensor=spec.h,
                    reader_outputs=dict(zip(spec.readers, spec.reader_values)),
                    accumulator_elements=readers * m * q,
                    counter_elements=readers * m * q,
                )
                # This is original emitted source CODE, not executed GCC.
                emitted_disjoint = emit_allocated_fanout_cpp(arch, layout="disjoint")
                emitted_exact = emit_allocated_fanout_cpp(
                    arch, layout="per_element_optimal")
                if (accounting["element_exact_committed_value_slots"] != source_slots
                        or optimum.unit_slots != source_slots
                        or accounting["disjoint_no_reuse_slots"]
                            != emitted_disjoint.manifest["source_scratch_array_elements"]
                        or source_slots != emitted_exact.manifest[
                            "source_scratch_array_elements"]
                        or accounting["actual_r2h_all_declared_mutable_elements_disjoint"]
                            != emitted_disjoint.manifest[
                                "source_all_declared_mutable_array_elements"]
                        or accounting["actual_r2h_all_declared_mutable_elements_element_exact"]
                            != emitted_exact.manifest[
                                "source_all_declared_mutable_array_elements"]):
                    raise AssertionError(
                        "R2q source-only memory counts disagree with real R2h emitter")
                rows.append({
                    "case": f"case_{len(rows):03d}",
                    "fanout_readers": readers,
                    "shape": list(dims),
                    "policy": policy,
                    "reader_pes": list(widths),
                    "original_r2h_disjoint_cpp_sha256":
                        emitted_disjoint.manifest["source_cpp_sha256"],
                    "original_r2h_optimized_cpp_sha256":
                        emitted_exact.manifest["source_cpp_sha256"],
                    **accounting,
                })
    if len(rows) != CASE_COUNT:
        raise AssertionError("R2q did not preserve all 36 configurations")
    return rows


def summarize(rows: list[dict], revision: str) -> dict:
    if len(rows) != CASE_COUNT:
        raise ValueError("Full original 36-case source catalog is required")
    phi = {
        "committed_reduced": 0, "committed_tied": 0, "committed_increased": 0,
        "inplace_reduced": 0, "inplace_tied": 0, "inplace_increased": 0,
    }
    max_delta_committed = max_delta_inplace = 0
    for readers in FANOUTS:
        for dims in SHAPES:
            group = [x for x in rows if x["fanout_readers"] == readers
                     and x["shape"] == list(dims)]
            if len(group) != len(POLICY_NAMES):
                raise ValueError("R2q missing original mapping policy")
            scalar = next(x for x in group if x["policy"] == "scalar")
            for case in group:
                if case is scalar:
                    continue
                for prefix, key in (
                        ("committed", "element_exact_committed_value_slots"),
                        ("inplace", "hypothetical_inplace_reader_value_slots")):
                    reduction = scalar[key] - case[key]
                    name = (prefix + "_reduced" if reduction > 0 else
                            prefix + "_increased" if reduction < 0 else
                            prefix + "_tied")
                    phi[name] += 1
                    if prefix == "committed":
                        max_delta_committed = max(max_delta_committed, abs(reduction))
                    else:
                        max_delta_inplace = max(max_delta_inplace, abs(reduction))
    gap_cases = [
        x for x in rows if x["disjoint_no_reuse_slots"] >
        x["element_exact_committed_value_slots"]
    ]
    first_fit_improvement = [
        x for x in rows if x["whole_tensor_first_fit_feasible_slots"] <
        x["disjoint_no_reuse_slots"]
    ]
    strictly_better_than_feasible_tensor = [
        x for x in rows if x["element_exact_committed_value_slots"] <
        x["whole_tensor_first_fit_feasible_slots"]
    ]
    lower_bound_gap = [
        x for x in rows if x["element_exact_committed_value_slots"] <
        x["whole_tensor_concurrency_lower_bound_slots"]
    ]
    return {
        "schema": SCHEMA,
        "source_revision": revision,
        "original_catalog_cases": CASE_COUNT,
        "fanout_degrees": list(FANOUTS),
        "original_shapes": [list(x) for x in SHAPES],
        "policies": list(POLICY_NAMES),
        "phi_intervention_count": len(FANOUTS) * len(SHAPES) *
            (len(POLICY_NAMES)-1),
        "phi_effects": phi,
        "largest_absolute_phi_change_in_committed_slots": max_delta_committed,
        "largest_absolute_phi_change_in_hypothetical_inplace_slots": max_delta_inplace,
        "disjoint_to_exact_positive_cases": len(gap_cases),
        "feasible_tensor_first_fit_beats_disjoint_cases":
            len(first_fit_improvement),
        "element_exact_beats_feasible_tensor_first_fit_cases":
            len(strictly_better_than_feasible_tensor),
        "element_exact_beats_tensor_concurrency_lower_bound_cases":
            len(lower_bound_gap),
        "source_event_and_cpp_count_audited_cases": len(rows),
        "new_original_GCC_compilations": 0,
        "new_original_GCC_executions": 0,
        "new_vitis_calls": 0,
        "new_vivado_calls": 0,
        "rl_training_calls": 0,
        "original_R2i_R2j_R2l_physical_artifacts_changed": False,
        "scope": [
            "Source-level emitted array counts; independent prospective R2q baseline interpretation",
            "The committed-value unit-slot optimum is classical fixed-order interval coloring",
            "The tensor-granular concurrency peak is ONLY a lower bound",
            "The hypothetical in-place reader accounting is NOT generated source, HLS or RTL",
            "No comparisons with external compilers or actual hardware BRAM here",
        ],
    }


def run(*, output: str | Path) -> dict:
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R2q original analysis output strictly create-only")
    revision = original_source_revision()
    rows = calculate_catalog()
    result = summarize(rows, revision)
    root.mkdir(parents=True)
    with (root / "cases.json").open("x", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, sort_keys=True)
        file.write("\n")
    with (root / "summary.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")
    keys = (
        "case", "fanout_readers", "shape", "policy", "reader_pes",
        "disjoint_no_reuse_slots",
        "whole_tensor_first_fit_feasible_slots",
        "whole_tensor_concurrency_lower_bound_slots",
        "element_exact_committed_value_slots",
        "hypothetical_inplace_reader_value_slots",
        "actual_r2h_all_declared_mutable_elements_disjoint",
        "actual_r2h_all_declared_mutable_elements_element_exact",
        "hypothetical_fused_inplace_plus_counters_elements",
        "original_r2h_disjoint_cpp_sha256",
        "original_r2h_optimized_cpp_sha256",
    )
    with (root / "baseline_ladder.csv").open("x", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "shape": json.dumps(row["shape"]),
                "reader_pes": json.dumps(row["reader_pes"]),
            })
    return result


def audit(study: str | Path) -> dict:
    """Read-only recomputation from current source, NOT original GCC execution."""
    root = Path(study)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("R2q original study missing or symlinked")
    files = [root / file for file in (
        "cases.json", "summary.json", "baseline_ladder.csv")]
    if any(not f.is_file() or f.is_symlink() for f in files):
        raise ValueError("R2q original source analysis files missing or symlinked")
    observed_rows = json.loads(files[0].read_text())
    observed_summary = json.loads(files[1].read_text())
    if (observed_summary.get("schema") != SCHEMA
            or not isinstance(observed_rows, list)
            or len(observed_rows) != CASE_COUNT
            or not isinstance(observed_summary.get("source_revision"), str)
            or re.fullmatch(r"[a-f0-9]{40}", observed_summary["source_revision"])
                is None):
        raise ValueError("R2q complete catalog or schema/source revision missing")
    # A source recomputation is meaningful only for the ORIGINAL revision:
    # comparing a changed compiler to a frozen study is not a receipt audit.
    if original_source_revision() != observed_summary["source_revision"]:
        raise ValueError("R2q original source revision differs from audit checkout")
    # Recompute every row rather than trusting its stored counts.
    current = calculate_catalog()
    if observed_rows != current:
        raise ValueError("R2q original source baseline results disagree with recomputation")
    if observed_summary != summarize(
            current, observed_summary.get("source_revision")):
        raise ValueError("R2q summary population or Phi effects were modified")
    with files[2].open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != CASE_COUNT or any(
            csv_rows[i]["case"] != observed_rows[i]["case"]
            or csv_rows[i]["policy"] != observed_rows[i]["policy"]
            or int(csv_rows[i]["fanout_readers"]) !=
                observed_rows[i]["fanout_readers"]
            or json.loads(csv_rows[i]["shape"]) != observed_rows[i]["shape"]
            or json.loads(csv_rows[i]["reader_pes"]) != observed_rows[i]["reader_pes"]
            or any(int(csv_rows[i][column]) != observed_rows[i][column]
                   for column in (
                       "disjoint_no_reuse_slots",
                       "whole_tensor_first_fit_feasible_slots",
                       "whole_tensor_concurrency_lower_bound_slots",
                       "element_exact_committed_value_slots",
                       "hypothetical_inplace_reader_value_slots",
                       "actual_r2h_all_declared_mutable_elements_disjoint",
                       "actual_r2h_all_declared_mutable_elements_element_exact",
                       "hypothetical_fused_inplace_plus_counters_elements",
                   ))
            or csv_rows[i]["original_r2h_disjoint_cpp_sha256"] !=
                observed_rows[i]["original_r2h_disjoint_cpp_sha256"]
            or csv_rows[i]["original_r2h_optimized_cpp_sha256"] !=
                observed_rows[i]["original_r2h_optimized_cpp_sha256"]
            for i in range(CASE_COUNT)):
        raise ValueError("R2q original baseline CSV mismatch")
    return {
        "schema": "fpgai-research/cst-R2q-source-accounting-read-only-recompute/v1",
        "original_source_revision": observed_summary["source_revision"],
        "audited_cases": CASE_COUNT,
        "audited_phi_interventions": observed_summary["phi_intervention_count"],
        "phi_effects": observed_summary["phi_effects"],
        "fresh_vitis_calls": 0, "fresh_vivado_calls": 0,
        "fresh_original_gcc_calls": 0,
        "audit_is_recomputed_source_not_independent_hardware": True,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--study")
    parser.add_argument("--audit-output")
    args = parser.parse_args(argv)
    if bool(args.output) == bool(args.study):
        parser.error("Choose exactly one of --output or --study")
    if args.study:
        result = audit(args.study)
        if args.audit_output:
            path = Path(args.audit_output)
            if path.exists() or path.is_symlink():
                raise FileExistsError("R2q audit output strictly create-only")
            with path.open("x", encoding="utf-8") as file:
                json.dump(result, file, indent=2, sort_keys=True)
                file.write("\n")
    else:
        if args.audit_output:
            parser.error("--audit-output is for --study only")
        result = run(output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
