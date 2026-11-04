"""R3e: bounded C1 independent source-event and scratch lifetime falsification.

A prespecified 2..4 reader MatMul-fanout family is checked with TWO
independent mechanisms: the existing typed event verifier and the separate
finite exhaustive interval-coloring oracle. Adversarial mutations test
readiness, incomplete producer reduction, wrong Phi ownership, issue
conformance, and overwrite of a value before its last reader. No claim of
arbitrary graph/RTL correctness, external novelty, or physical FPGA QoR.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.codegen.hls.r2h_allocated_fanout_cpp import (
    emit_allocated_fanout_cpp, source_layout,
)
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.element_scratch_allocation import (
    exhaustive_minimum_slots, extract_intervals, verify_allocation,
)
from fpgai_research.model_ir.fanout_contract import (
    disjoint_fanout_offsets, generate_fanout_events, verify_fanout_events,
    verify_fanout_issue,
)

SCHEMA = "fpgai-research/R3e-C1-independent-source-event-falsification/v1"
# Small enough for actual independent exhaustive enumeration, not a
# sampling-only re-run of the production allocator.
CASES = (
    (2, 1, 1, 2, 1, 1),
    (3, 1, 1, 2, 1, 1),
    (4, 1, 1, 2, 1, 1),
    (2, 1, 1, 2, 2, 1),
    (2, 1, 1, 2, 2, 2),
)
MUTANTS = (
    "early_consumer_read", "premature_producer_commit",
    "wrong_phi_owner", "wrong_phi_issue_order", "overwrite_live_value",
)


def _swap_to_front(events: tuple, index: int) -> tuple:
    seq = list(events)
    item = seq.pop(index)
    seq.insert(0, item)
    return tuple(seq)


def one_case(config: tuple[int, ...]) -> dict:
    if config not in CASES:
        raise ValueError("R3e: case not in prespecified bounded population")
    readers, m, k, n, q, reader0_pes = config
    model = sample_fanout(readers, m=m, k=k, n=n, q=q)
    default = ArchitectureRequest()
    overrides = ({"reader0": replace(default, pes=reader0_pes)}
                 if reader0_pes != 1 else {})
    arch = resolve(model, default, overrides)
    spec, assignments, capacity, optimal, projected = source_layout(
        arch, layout="per_element_optimal")
    emission = emit_allocated_fanout_cpp(arch, layout="per_element_optimal")
    if (emission.manifest["source_scratch_array_elements"] != capacity
            or emission.manifest["source_optimal_unit_slots_for_fixed_order"] != optimal.unit_slots
            or emission.manifest["source_event_order_sha256"] != optimal.schedule_sha256):
        raise AssertionError("R3e: E0 source certificate diverged from emitted HLS C++")
    independently_minimal = exhaustive_minimum_slots(projected, max_values=16)
    if independently_minimal != capacity:
        raise AssertionError("R3e: independent finite coloring oracle disagrees with compiler")
    if verify_allocation(projected, assignments, unit_slots=capacity)["status"] != (
            "EXACT_SOURCE_SLOT_REPLAY_VERIFIED"):
        raise AssertionError("R3e: source allocation failed independent replay")
    events = generate_fanout_events(arch)
    offsets = disjoint_fanout_offsets(spec)
    good = verify_fanout_events(arch, offsets, events)
    issue = verify_fanout_issue(arch, events)
    if good["status"] != "VERIFIED_E0" or issue["status"] != "VERIFIED_E0":
        raise AssertionError("R3e: legitimate original event stream rejected")
    first_read = next(i for i, e in enumerate(events)
                      if e.kind == "READ" and e.tensor == spec.h)
    first_commit = next(i for i, e in enumerate(events)
                        if e.kind == "COMMIT" and e.tensor == spec.h)

    # Move one existing, typed consumer READ to before producer readiness.
    early_read = _swap_to_front(events, first_read)
    early_evidence = verify_fanout_events(arch, offsets, early_read)
    if (early_evidence["status"] != "REJECTED"
            or early_evidence["reason"] != "FANOUT_READ_BEFORE_READY_OR_STALE"):
        raise AssertionError("R3e: premature source read escaped E0")
    # Move a producer COMMIT ahead of all required reduction contributions.
    premature_commit = _swap_to_front(events, first_commit)
    commit_evidence = verify_fanout_events(arch, offsets, premature_commit)
    if (commit_evidence["status"] != "REJECTED"
            or commit_evidence["reason"] != "PREMATURE_H_PRODUCER_COMMIT"):
        raise AssertionError("R3e: premature producer output escaped E0")

    wrong_owner = list(events)
    wrong_owner[first_read] = replace(
        wrong_owner[first_read], phi_owner=(99, 99, 99, 99))
    phi_evidence = verify_fanout_events(arch, offsets, tuple(wrong_owner))
    if (phi_evidence["status"] != "REJECTED"
            or phi_evidence["reason"] != "WRONG_FANOUT_PHI_COORDINATE_OR_OWNER"):
        raise AssertionError("R3e: incorrect Phi owner escaped E0")

    # Swap two otherwise valid reader READs. Both still read committed H,
    # but the finite independent forward-rank issue contract must reject.
    reads = [i for i, e in enumerate(events)
             if e.kind == "READ" and e.tensor == spec.h]
    if len(reads) < 2 or events[reads[0]].operation == events[reads[1]].operation:
        raise AssertionError("R3e: expected two independently mapped readers")
    changed = list(events)
    changed[reads[0]], changed[reads[1]] = changed[reads[1]], changed[reads[0]]
    changed = tuple(changed)
    safe_issue_reordered = verify_fanout_events(arch, offsets, changed)
    issue_fail = verify_fanout_issue(arch, changed)
    if (safe_issue_reordered["status"] != "VERIFIED_E0"
            or issue_fail["status"] != "REJECTED"
            or issue_fail["reason"] != "FANOUT_PHI_ISSUE_OR_OWNER_ORDER_MISMATCH"):
        raise AssertionError("R3e: independent issue-order distinction did not hold")

    # Independent alias oracle: force two simultaneously live values onto
    # the same address, keeping all other original slot bindings unchanged.
    intervals = extract_intervals(projected)
    overlap = next(((a, b) for i, a in enumerate(intervals)
                    for b in intervals[i+1:]
                    if a.commit_index <= b.last_read_index
                    and b.commit_index <= a.last_read_index), None)
    if overlap is None:
        raise AssertionError("R3e: positive overlapping-lifetime control missing")
    corrupted = dict(assignments)
    corrupted[overlap[1].key] = corrupted[overlap[0].key]
    try:
        verify_allocation(projected, corrupted, unit_slots=capacity)
    except ValueError as exc:
        alias_failure = str(exc)
    else:
        raise AssertionError("R3e: overlapping live scratch alias unexpectedly accepted")
    if not alias_failure.startswith(("OVERWRITE_LIVE_VALUE", "READ_STALE_OR_UNREADY_VALUE")):
        raise AssertionError("R3e: alias failed for unrelated reason: " + alias_failure)

    reasons = {
        "early_consumer_read": early_evidence["reason"],
        "premature_producer_commit": commit_evidence["reason"],
        "wrong_phi_owner": phi_evidence["reason"],
        "wrong_phi_issue_order": issue_fail["reason"],
        "overwrite_live_value": alias_failure.split(" ", 1)[0],
    }
    return {
        "source_family": "bounded-MatMul-multi-reader-fixed-order",
        "readers": readers, "shape": [m, k, n, q],
        "reader0_pes": reader0_pes,
        "source_cpp_sha256": emission.manifest["source_cpp_sha256"],
        "original_source_event_sha256": optimal.schedule_sha256,
        "event_count": len(events),
        "intermediate_value_count": len(intervals),
        "source_unit_slots": capacity,
        "independently_exhaustive_minimum_unit_slots": independently_minimal,
        "valid_typed_event_stream": True,
        "valid_phi_issue": True,
        "mutant_denominator": len(MUTANTS),
        "rejected_mutant_reasons": reasons,
        "source_only": True,
    }


def run() -> dict:
    rows = [one_case(case) for case in CASES]
    if len(rows) != len(CASES):
        raise AssertionError("R3e: incomplete fixed case population")
    return {
        "schema": SCHEMA,
        "cases_prespecified": len(CASES),
        "mutants_per_case": len(MUTANTS),
        "rejected_mutants": sum(r["mutant_denominator"] for r in rows),
        "oracle_agreement_cases": sum(
            r["source_unit_slots"] == r["independently_exhaustive_minimum_unit_slots"]
            for r in rows),
        "rows": rows,
        "new_gcc_calls": 0,
        "new_vitis_calls": 0,
        "new_vivado_calls": 0,
        "trained_rl_agents": 0,
        "formal_scope": "finite selected 2..4-reader MatMul fanout source event traces",
        "limitations": [
            "This is an independent tiny-population oracle, not a proof for arbitrary dynamic graphs",
            "The Phi event order is a source issue order, not a hardware clock schedule",
            "Unit-slot scratch is a source-level count, not routed BRAM or port feasibility",
            "Five selected mutation classes cannot prove all possible invalid programs are rejected",
            "Comparisons with independently inspected competing IR artifacts remain outstanding",
        ],
    }


def create_only(output: str | Path, report: dict, *, git_revision: str) -> None:
    folder = Path(output)
    if folder.exists() or folder.is_symlink():
        raise FileExistsError("R3e original bounded report is immutable")
    if (report.get("schema") != SCHEMA
            or re.fullmatch(r"[0-9a-f]{40}", git_revision) is None):
        raise ValueError("R3e: source revision or report schema invalid")
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "r3e_bounded_c1_results.json").write_text(
        json.dumps({**report, "source_revision": git_revision},
                   sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")


def _revision() -> str:
    repo = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True, timeout=10).stdout
    if status.strip():
        raise RuntimeError("R3e source original requires clean tracked Git")
    rev = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", rev) is None:
        raise RuntimeError("R3e exact tracked source SHA missing")
    return rev


def audit(study: str | Path) -> dict:
    """Read-only finite original receipt audit; never reruns the source compiler."""
    folder = Path(study)
    receipt = folder / "r3e_bounded_c1_results.json"
    if (folder.is_symlink() or not folder.is_dir()
            or receipt.is_symlink() or not receipt.is_file()):
        raise ValueError("R3e: original results absent or linked")
    data = json.loads(receipt.read_text())
    rows = data.get("rows")
    if (data.get("schema") != SCHEMA
            or re.fullmatch(r"[0-9a-f]{40}",
                            data.get("source_revision") or "") is None
            or data.get("cases_prespecified") != len(CASES)
            or data.get("mutants_per_case") != len(MUTANTS)
            or data.get("rejected_mutants") != len(CASES) * len(MUTANTS)
            or data.get("oracle_agreement_cases") != len(CASES)
            or not isinstance(rows, list) or len(rows) != len(CASES)
            or any(data.get(k) != 0 for k in (
                "new_gcc_calls", "new_vitis_calls", "new_vivado_calls",
                "trained_rl_agents"))):
        raise ValueError("R3e: original finite cohort receipt mismatch")
    for row, config in zip(rows, CASES):
        readers, m, k, n, q, pe = config
        reasons = row.get("rejected_mutant_reasons")
        if (row.get("readers") != readers
                or row.get("shape") != [m,k,n,q]
                or row.get("reader0_pes") != pe
                or row.get("mutant_denominator") != len(MUTANTS)
                or row.get("valid_typed_event_stream") is not True
                or row.get("valid_phi_issue") is not True
                or row.get("source_only") is not True
                or row.get("source_unit_slots") !=
                    row.get("independently_exhaustive_minimum_unit_slots")
                or not isinstance(row.get("source_cpp_sha256"), str)
                or re.fullmatch(r"[a-f0-9]{64}", row["source_cpp_sha256"]) is None
                or not isinstance(reasons, dict) or set(reasons) != set(MUTANTS)
                or reasons["early_consumer_read"] != "FANOUT_READ_BEFORE_READY_OR_STALE"
                or reasons["premature_producer_commit"] != "PREMATURE_H_PRODUCER_COMMIT"
                or reasons["wrong_phi_owner"] != "WRONG_FANOUT_PHI_COORDINATE_OR_OWNER"
                or reasons["wrong_phi_issue_order"] !=
                    "FANOUT_PHI_ISSUE_OR_OWNER_ORDER_MISMATCH"
                or reasons["overwrite_live_value"] not in (
                    "OVERWRITE_LIVE_VALUE","READ_STALE_OR_UNREADY_VALUE")):
            raise ValueError("R3e: saved original counterexample/slot evidence invalid")
    return {
        "schema": SCHEMA,
        "source_revision": data["source_revision"],
        "original_case_count": len(CASES),
        "rejected_mutants": len(CASES) * len(MUTANTS),
        "independent_minimum_slot_oracle_agreements": len(CASES),
        "read_only_receipt_sha256": sha256(receipt.read_bytes()).hexdigest(),
        "new_compiler_gcc_vitis_vivado_calls": 0,
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output")
    mode.add_argument("--study")
    args = parser.parse_args(argv)
    if args.study is not None:
        print(json.dumps(audit(args.study), indent=2, sort_keys=True))
        return
    if Path(args.output).exists() or Path(args.output).is_symlink():
        raise FileExistsError("R3e refuses to overwrite existing original output")
    revision = _revision()
    result = run()
    create_only(args.output, result, git_revision=revision)
    print(json.dumps({
        "schema": SCHEMA, "original_source_revision": revision,
        "cases": result["cases_prespecified"],
        "rejected_mutants": result["rejected_mutants"],
        "oracle_agreement_cases": result["oracle_agreement_cases"],
        "new_vendor_calls": 0,
        "output": args.output,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
