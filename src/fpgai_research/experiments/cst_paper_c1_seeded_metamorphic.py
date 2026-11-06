"""C1 seeded held-out metamorphic source-event challenges in admitted fanout domain.

Generates fresh (not R3j's 4 static shapes) MatMul fanout configurations
and mutates REAL typed event streams. On each trace, compare the separate
reference_event_oracle with the production value and Phi issue verifiers.
Selected mutation labels are generator-expected outcomes, NOT an unbiased
oracle for arbitrary graphs. Source-only; no Vitis/Vivado, no claims
about emitted C++ refinement, clock, resource or end-to-end latency.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from itertools import product
from pathlib import Path
import random
import re

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_fanout_models import sample_fanout
from fpgai_research.experiments.cst_r3d_c1c2_finite_contract_ablation import (
    _git_revision,
)
from fpgai_research.experiments.cst_r3g_c1_independent_event_semantics import (
    reference_event_oracle,
)
from fpgai_research.experiments.cst_r3j_c1_heldout_semantics import SHAPES
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.fanout_contract import (
    admit_fanout, disjoint_fanout_offsets, generate_fanout_events,
    verify_fanout_events, verify_fanout_issue,
)

SCHEMA = "fpgai-research/C1-seeded-heldout-metamorphic-source-events/v1"
SEEDS = (111, 222, 333)
READER_COUNTS = (2, 3, 4)
POLICIES = ("scalar", "mixed_phi")
SHAPES_PER_GROUP = 3
MUTATIONS = (
    "canonical", "safe_wrong_issue_order", "early_h_read",
    "early_h_commit", "wrong_phi_owner", "missing_final_read",
    "duplicate_producer_term", "overlapping_reader_scratch",
    "unknown_final_tensor",
)
# Bounded event-certificate study, NOT arbitrary graph sizes.
CANDIDATE_SHAPES = tuple(
    shape for shape in product(range(1, 4), range(2, 5),
                               range(2, 5), range(2, 5))
    if shape not in SHAPES
)


def configurations(*, seeds=SEEDS, shapes_per_group=SHAPES_PER_GROUP):
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(x) is not int for x in seeds)
            or type(shapes_per_group) is not int
            or not 1 <= shapes_per_group <= 3):
        raise ValueError("C1 heldout protocol seed or shape population invalid")
    cases = []
    for seed in seeds:
        shuffled = list(CANDIDATE_SHAPES)
        random.Random(seed).shuffle(shuffled)
        index = 0
        for readers in READER_COUNTS:
            for policy in POLICIES:
                for _ in range(shapes_per_group):
                    shape = shuffled[index]
                    index += 1
                    if shape in SHAPES:
                        raise AssertionError("C1 source heldout shape leakage")
                    cases.append((seed, readers, *shape, policy))
    return tuple(cases)


def _architecture(case):
    _, readers, m, k, n, q, policy = case
    model = sample_fanout(readers, m=m, k=k, n=n, q=q)
    default = ArchitectureRequest()
    changes = {}
    if policy == "mixed_phi":
        changes["reader0"] = replace(default, pes=2)
        changes["reader1"] = replace(default, simd=2)
        if readers >= 3:
            changes["reader2"] = replace(default, pes=2, simd=2)
    arch = resolve(model, default, changes)
    contract = admit_fanout(arch)
    return arch, contract, generate_fanout_events(arch), disjoint_fanout_offsets(contract)


def _mutants(arch, contract, original, offsets):
    h_reads = [i for i, e in enumerate(original)
               if e.kind == "READ" and e.tensor == contract.h]
    if len(h_reads) < 2 or original[h_reads[0]].operation == original[h_reads[1]].operation:
        raise ValueError("C1 heldout missing independent-reader H-read controls")
    h_commit = next(i for i,e in enumerate(original)
                    if e.kind == "COMMIT" and e.tensor == contract.h)
    # Explicit owner/order mutation, preserving data safety if the entire
    # original stream's first independent consumer H reads are swapped.
    swapped = list(original)
    a, b = h_reads[0], h_reads[1]
    swapped[a], swapped[b] = swapped[b], swapped[a]
    early_read = list(original)
    early_read.insert(0, early_read.pop(h_reads[0]))
    early_commit = list(original)
    early_commit.insert(0, early_commit.pop(h_commit))
    owner = list(original)
    owner[h_reads[0]] = replace(owner[h_reads[0]], phi_owner=(99, 99, 99, 99))
    alias = dict(offsets)
    alias[contract.reader_values[1]] = alias[contract.reader_values[0]]
    unknown = list(original)
    unknown[-1] = replace(unknown[-1], tensor="c1_unseen_tensor")
    return (
        ("canonical", original, dict(offsets), True, True),
        ("safe_wrong_issue_order", tuple(swapped), dict(offsets), True, False),
        ("early_h_read", tuple(early_read), dict(offsets), False, None),
        ("early_h_commit", tuple(early_commit), dict(offsets), False, None),
        ("wrong_phi_owner", tuple(owner), dict(offsets), False, False),
        ("missing_final_read", original[:-1], dict(offsets), False, None),
        ("duplicate_producer_term",
         (original[0],) + original, dict(offsets), False, None),
        ("overlapping_reader_scratch", original, alias, False, None),
        ("unknown_final_tensor", tuple(unknown), dict(offsets), False, None),
    )


def evaluate(case):
    arch, contract, original, offsets = _architecture(case)
    variants = _mutants(arch, contract, original, offsets)
    if tuple(v[0] for v in variants) != MUTATIONS:
        raise ValueError("C1 heldout mutations missing or out of order")
    observed = []
    for label, events, layout, expected_value, expected_issue in variants:
        ref_value, ref_issue = reference_event_oracle(arch, events, layout)
        source_value = verify_fanout_events(arch, layout, events)["status"] == "VERIFIED_E0"
        source_issue = verify_fanout_issue(arch, events)["status"] == "VERIFIED_E0"
        if (ref_value != source_value or ref_issue != source_issue
                or ref_value is not expected_value
                or (expected_issue is not None and ref_issue is not expected_issue)):
            raise AssertionError(
                f"C1 heldout independent model disagreement: {case}, {label}: "
                f"reference={(ref_value,ref_issue)}, "
                f"source={(source_value,source_issue)}, "
                f"expected={(expected_value,expected_issue)}")
        observed.append({
            "mutation": label,
            "value_safe": ref_value,
            "issue_conformant": ref_issue,
            "full_contract_admissible": bool(ref_value and ref_issue),
            "expected_value_safe": expected_value,
            "expected_issue_conformant": expected_issue,
            "event_trace_sha256": sha256(json.dumps(
                [e.__dict__ for e in events], sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "scratch_offsets_sha256": sha256(json.dumps(
                sorted(layout.items()), separators=(",", ":")).encode()).hexdigest(),
        })
    return {
        "seed": case[0], "readers": case[1],
        "shape": list(case[2:6]), "mapping": case[6],
        "source_value_oracle_disagreements": 0,
        "source_issue_oracle_disagreements": 0,
        "mutations": observed,
    }


def run(*, seeds=SEEDS, shapes_per_group=SHAPES_PER_GROUP):
    cases = configurations(seeds=seeds, shapes_per_group=shapes_per_group)
    rows = [evaluate(case) for case in cases]
    if len(rows) != len(seeds)*len(READER_COUNTS)*len(POLICIES)*shapes_per_group:
        raise AssertionError("C1 heldout population incomplete")
    by_type = Counter(v["mutation"] for row in rows for v in row["mutations"])
    if set(by_type) != set(MUTATIONS) or set(by_type.values()) != {len(cases)}:
        raise AssertionError("C1 missing a predefined mutation class")
    if any(row["mutations"][0]["full_contract_admissible"] is not True
           or row["mutations"][1]["value_safe"] is not True
           or row["mutations"][1]["full_contract_admissible"] is not False
           or any(item["value_safe"] for item in row["mutations"][2:])
           for row in rows):
        raise AssertionError("C1 heldout adversarial control unexpectedly changed")
    return {
        "schema": SCHEMA,
        "status": "BOUNDED_NEW_SHAPE_METAMORPHIC_CONTROLS_NOT_GENERAL_THEOREM",
        "seed_population": list(seeds),
        "prespecified_reader_counts": list(READER_COUNTS),
        "prespecified_mapping_policies": list(POLICIES),
        "prespecified_shapes_per_group_per_seed": shapes_per_group,
        "profiles": len(cases),
        "traces": len(cases) * len(MUTATIONS),
        "value_oracle_disagreements": 0,
        "issue_oracle_disagreements": 0,
        "value_safe_wrong_issue_controls": len(cases),
        "full_contract_accepts_wrong_issue_controls": 0,
        "canonical_valid_controls": len(cases),
        "other_selected_unsafe_controls": 7*len(cases),
        "source_reports": rows,
        "original_gcc_calls": 0,
        "original_vitis_calls": 0,
        "original_vivado_calls": 0,
        "trained_agents": 0,
        "hardware_latency": None,
        "limitations": [
            "All configurations remain the SAME bounded MatMul fanout topology",
            "Selected mutations and expectations encoded by the test generator",
            "Reference and production verifiers share admitted graph assumptions",
            "No generated C++ trace refinement or hardware/clock proof",
            "Seeded shapes are not independent external production workloads",
        ],
    }


def create_only(report, *, output: str | Path, source_revision: str):
    folder = Path(output)
    if folder.exists() or folder.is_symlink():
        raise FileExistsError("C1 heldout source output is create-only")
    if (report.get("schema") != SCHEMA
            or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
            or report.get("value_oracle_disagreements") != 0
            or report.get("issue_oracle_disagreements") != 0):
        raise ValueError("C1 heldout original receipt not qualified")
    folder.mkdir(parents=True, exist_ok=False)
    text = json.dumps(
        {**report, "experiment_source_revision":source_revision},
        sort_keys=True, indent=2, allow_nan=False) + "\n"
    (folder/"report.json").write_text(text)
    (folder/"artifact_sha256.json").write_text(
        json.dumps({"report.json":sha256(text.encode()).hexdigest()},
                   indent=2, sort_keys=True) + "\n")


def audit(study: str | Path):
    path = Path(study)
    p = path/"report.json"
    manifest = path/"artifact_sha256.json"
    if (path.is_symlink() or not path.is_dir()
            or p.is_symlink() or manifest.is_symlink()
            or not p.is_file() or not manifest.is_file()):
        raise ValueError("C1 heldout original study missing/symlinked")
    receipt = json.loads(p.read_text())
    if json.loads(manifest.read_text()) != {
        "report.json": sha256(p.read_bytes()).hexdigest()}:
        raise ValueError("C1 heldout original SHA drift")
    if (receipt.get("schema") != SCHEMA
            or len(receipt.get("source_reports", [])) != receipt.get("profiles")
            or receipt.get("traces") != receipt.get("profiles",0)*len(MUTATIONS)
            or not receipt.get("value_oracle_disagreements") == 0
            or not receipt.get("issue_oracle_disagreements") == 0
            or re.fullmatch(r"[0-9a-f]{40}",
                            receipt.get("experiment_source_revision") or "") is None):
        raise ValueError("C1 heldout original source report invalid")
    return {
        "schema": SCHEMA, "source_revision": receipt["experiment_source_revision"],
        "profiles": receipt["profiles"], "traces": receipt["traces"],
        "report_sha256": sha256(p.read_bytes()).hexdigest(),
        "oracle_disagreements": 0, "new_vendor_calls": 0,
        "hardware_latency": None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output")
    group.add_argument("--study")
    parser.add_argument("--quick", action="store_true",
                        help="One seed and shape per group: NOT publication data")
    args = parser.parse_args(argv)
    if args.study:
        result = audit(args.study)
    else:
        kwargs = {"seeds": (111,), "shapes_per_group": 1} if args.quick else {}
        report = run(**kwargs)
        create_only(report, output=args.output,
                    source_revision=_git_revision())
        result = {
            "schema": SCHEMA, "source_revision": _git_revision(),
            "profiles": report["profiles"], "traces": report["traces"],
            "oracle_disagreements": 0, "new_vendor_calls": 0,
            "hardware_latency": None, "output": args.output,
        }
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
