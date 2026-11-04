"""R3d: policy-neutral finite C1/C2 contract-gate and graph invariance study.

This is NOT a trained RL study or a hardware-performance comparison. Uses
the exact pre-existing R2p compiler-backed source transition for BOTH arms:
a blind proposal catalog and an E0-certified filtered catalog. Screening
itself incurs compiler calls. Its counts cannot imply trained-agent gains.
No source generator, GCC, Vitis, Vivado, or RL module is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from time import perf_counter_ns

from fpgai_research.experiments.cst_r2p_unified_interface_study import _cases
from fpgai_research.model_ir.unified_decision_interface import (
    unified_action_mask, unified_graph_observation,
    unified_state_hash, unified_transition,
)

SCHEMA = "fpgai-research/R3d-finite-contract-gate-ablation/v1"
GRAPH_SCHEMA = "fpgai-research/R3d-canonical-feature-fingerprint/v1"
DOMAINS = ("sequential", "multi_reader_fanout")
EXPECTED_RESTRICTED = (
    ("BOUNDED_CERTIFIED_E0", "REJECTED", "UNKNOWN_E0"),
    ("BOUNDED_CERTIFIED_E0", "REJECTED", "BOUNDED_CERTIFIED_E0"),
)


def _sha(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def structural_signature(graph: dict) -> dict:
    """Rename/order-invariant finite color-refinement feature signature.

    This is a testable invariant of the *public observation*; it is neither
    a graph-isomorphism proof nor a proof of neural functional equivalence.
    """
    if (not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list)
            or not isinstance(graph.get("edges"), list)
            or graph.get("schema") !=
                "fpgai-research/cst-R2p-unified-weight-private-graph/v1"):
        raise ValueError("R3d: unsupported public compiler graph")
    nodes = graph["nodes"]
    edges = graph["edges"]
    ids = [n.get("id") for n in nodes]
    if (not nodes or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("R3d: invalid or duplicated graph node IDs")
    node_by_id = dict(zip(ids, nodes))
    if any(not isinstance(e, dict)
           or e.get("src") not in node_by_id
           or e.get("dst") not in node_by_id
           for e in edges):
        raise ValueError("R3d: dangling or malformed graph edge")
    # Only the public feature fields; raw weights/CPP/status/rewards
    # are never inserted into this fingerprint.
    colors = {i: _sha({k: v for k, v in n.items() if k != "id"})
              for i, n in node_by_id.items()}
    for _ in range(len(nodes)):
        refined = {}
        for node_id in ids:
            adj = []
            for e in edges:
                attrs = {k: v for k, v in e.items() if k not in ("src", "dst")}
                if e["src"] == node_id:
                    adj.append(("out", attrs, colors[e["dst"]]))
                if e["dst"] == node_id:
                    adj.append(("in", attrs, colors[e["src"]]))
            # Color refinement is independent of name, node list order,
            # edge list order; fixed round count is deterministic.
            refined[node_id] = _sha({"self": colors[node_id],
                                    "neighbors": sorted(_sha(x) for x in adj)})
        colors = refined
    final_edges = sorted(_sha({
        "source_color": colors[e["src"]],
        "target_color": colors[e["dst"]],
        "edge_features": {k: v for k, v in e.items()
                          if k not in ("src", "dst")},
    }) for e in edges)
    return {
        "schema": GRAPH_SCHEMA,
        "domain": graph.get("domain"),
        "node_count": len(nodes), "edge_count": len(edges),
        "digest": _sha({
            "domain": graph.get("domain"),
            "node_colors": sorted(colors.values()),
            "edges": final_edges,
            "target_limits": graph.get("target_limits"),
            "source_domain": graph.get("source_domain"),
        }),
        "interpretation": "renaming/order-invariant typed public graph features only",
    }


def finite_ablation(*, domain: str, state, actions: tuple,
                    expected: tuple[str, ...]) -> dict:
    """Same candidate catalog and compiler for every screened/action arm."""
    if domain not in DOMAINS or len(actions) != 3 or len(expected) != 3:
        raise ValueError("R3d: restricted prespecified three-action cohort required")
    before = unified_state_hash(state)
    obs = unified_graph_observation(state)
    graph = structural_signature(obs)
    start = perf_counter_ns()
    mask = unified_action_mask(state, actions)
    mask_ns = perf_counter_ns() - start
    if (mask["parent_state_sha256"] != before
            or mask["candidate_count"] != len(actions)
            or mask["domain"] != domain
            or tuple(mask["statuses"]) != expected):
        raise ValueError("R3d: source compiler disagrees with fixed restricted oracle")
    # Blind baseline receives identical actions, one real compiler call
    # for each, without a mask. This is a correctness/control replay,
    # not a competing policy or an estimate of sample efficiency.
    start = perf_counter_ns()
    baseline = tuple(unified_transition(state, action) for action in actions)
    replay_ns = perf_counter_ns() - start
    if (tuple(d.status for d in baseline) != tuple(mask["statuses"])
            or [d.as_receipt() for d in baseline] != mask["rows"]
            or unified_state_hash(state) != before):
        raise ValueError("R3d: mask and unmasked real compiler transitions diverge")
    statuses = tuple(mask["statuses"])
    counts = Counter(statuses)
    assert sum(counts.values()) == 3
    return {
        "domain": domain,
        "original_state_sha256": before,
        "graph_signature": graph,
        "candidate_count": len(actions),
        "action_replay_statuses": list(statuses),
        "candidate_actions": [r["action"] for r in mask["rows"]],
        "action_source_cpp_sha256": [r["cpp_sha256"] for r in mask["rows"]],
        "status_counts": {k: counts.get(k, 0) for k in
                          ("BOUNDED_CERTIFIED_E0", "UNKNOWN_E0", "REJECTED")},
        "blind_unmasked_catalog_size": len(actions),
        "blind_invalid_or_uncertified": (
            counts.get("REJECTED", 0) + counts.get("UNKNOWN_E0", 0)),
        "source_admissible_candidates": sum(mask["source_admissible"]),
        "e0_certified_candidates": sum(mask["e0_certified"]),
        "e0_unverified_not_promoted": counts.get("UNKNOWN_E0", 0),
        "mask_build_compiler_transitions": len(actions),
        "blind_feedback_compiler_transitions": len(actions),
        "mask_build_wall_ns": mask_ns,
        "blind_feedback_wall_ns": replay_ns,
        "native_vitis_calls": 0,
        "original_gcc_calls": 0,
        "original_vivado_calls": 0,
        "rl_policies_trained": 0,
        "hls_timing_or_physical_feasibility": None,
    }


def run() -> dict:
    cases = _cases()  # Reuse existing exactly matched R2p compiler fixtures.
    if tuple(c[0] for c in cases) != DOMAINS:
        raise ValueError("R3d: unexpected original R2p compiler domains")
    rows = [finite_ablation(domain=domain, state=state, actions=actions,
                           expected=oracle)
            for (domain, state, actions, _x), oracle
            in zip(cases, EXPECTED_RESTRICTED)]
    total = sum(r["candidate_count"] for r in rows)
    assert total == 6
    return {
        "schema": SCHEMA,
        "supported_domains": list(DOMAINS),
        "finite_candidate_denominator": total,
        "e0_certified": sum(r["e0_certified_candidates"] for r in rows),
        "source_admitted_but_e0_unknown": sum(
            r["e0_unverified_not_promoted"] for r in rows),
        "rejected": sum(r["status_counts"]["REJECTED"] for r in rows),
        "mask_construction_compiler_transitions": sum(
            r["mask_build_compiler_transitions"] for r in rows),
        "blind_feedback_compiler_transitions": sum(
            r["blind_feedback_compiler_transitions"] for r in rows),
        "rows": rows,
        "native_vitis_calls": 0,
        "original_gcc_calls": 0,
        "original_vivado_calls": 0,
        "rl_policies_trained": 0,
        "observation_invariance_tested": "public rename/order feature signature only",
        "scientific_claim": "bounded E0 screening agreement; no learning or hardware advantage",
        "limitations": [
            "Six prespecified complete-design actions, two tiny compiler subdomains",
            "Restricted oracle expected statuses are human-specified, not an independent general correctness theorem",
            "Unmasked policy baseline still learns its feedback from the same compiler",
            "Eager E0 mask costs one actual compiler transition per candidate",
            "Wall-clock timings are uncalibrated exploratory CPU measurements, not a speedup claim",
            "E0 source certification is not E1 numeric or HLS E2/E3/E4 qualification",
            "No agent training, rewards, unseen-network generalization or learned sample efficiency",
        ],
    }


def _git_revision() -> str:
    root = Path(__file__).resolve().parents[3]
    tracked = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True, timeout=10).stdout
    if tracked.strip():
        raise RuntimeError("R3d original source study requires clean tracked Git")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("R3d: full source Git SHA required")
    return revision


def create_only(report: dict, *, output: str | Path, revision: str) -> None:
    folder = Path(output)
    if folder.exists() or folder.is_symlink():
        raise FileExistsError("R3d existing output is immutable: choose a new path")
    if report.get("schema") != SCHEMA or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("R3d: invalid report schema or source revision")
    folder.mkdir(parents=True, exist_ok=False)
    record = {**report, "source_revision": revision}
    (folder / "r3d_contract_gate.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")


def audit(study: str | Path) -> dict:
    """Read-only saved receipt consistency, not replay of source compiler."""
    folder = Path(study)
    receipt = folder / "r3d_contract_gate.json"
    if (folder.is_symlink() or not folder.is_dir()
            or receipt.is_symlink() or not receipt.is_file()):
        raise ValueError("R3d: missing/noncanonical original saved receipt")
    item = json.loads(receipt.read_text())
    rows = item.get("rows")
    if (item.get("schema") != SCHEMA
            or not isinstance(item.get("source_revision"), str)
            or re.fullmatch(r"[0-9a-f]{40}", item["source_revision"]) is None
            or not isinstance(rows, list) or len(rows) != 2
            or tuple(r.get("domain") for r in rows) != DOMAINS
            or item.get("finite_candidate_denominator") != 6
            or any(item.get(k) != 0 for k in
                   ("native_vitis_calls", "original_gcc_calls",
                    "original_vivado_calls", "rl_policies_trained"))):
        raise ValueError("R3d: study provenance/denominators invalid")
    totals = Counter()
    for row in rows:
        statuses = row.get("action_replay_statuses")
        if (not isinstance(statuses, list) or len(statuses) != 3
                or tuple(statuses) != EXPECTED_RESTRICTED[DOMAINS.index(row["domain"])]
                or not isinstance(row.get("graph_signature"), dict)
                or not re.fullmatch(r"[0-9a-f]{64}",
                                    row["graph_signature"].get("digest") or "")
                or row.get("mask_build_compiler_transitions") != 3
                or row.get("blind_feedback_compiler_transitions") != 3):
            raise ValueError("R3d: inconsistent action evidence or graph digest")
        totals.update(statuses)
    if (item.get("e0_certified") != totals["BOUNDED_CERTIFIED_E0"]
            or item.get("source_admitted_but_e0_unknown") != totals["UNKNOWN_E0"]
            or item.get("rejected") != totals["REJECTED"]
            or item.get("mask_construction_compiler_transitions") != 6
            or item.get("blind_feedback_compiler_transitions") != 6):
        raise ValueError("R3d: saved mask/outcome denominator altered")
    return {
        "schema": SCHEMA,
        "source_revision": item["source_revision"],
        "finite_candidate_denominator": 6,
        "e0_certified": item["e0_certified"],
        "unknown_e0": item["source_admitted_but_e0_unknown"],
        "rejected": item["rejected"],
        "saved_receipt_sha256": sha256(receipt.read_bytes()).hexdigest(),
        "original_compiler_or_vendor_calls_during_audit": 0,
        "rl_policies_trained": 0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--output")
    modes.add_argument("--study")
    args = parser.parse_args(argv)
    if args.output is not None:
        path = Path(args.output)
        if path.exists() or path.is_symlink():
            raise FileExistsError("R3d refuses to overwrite existing study")
        revision = _git_revision()
        result = run()
        create_only(result, output=path, revision=revision)
        print(json.dumps({**{k: result[k] for k in (
            "schema", "finite_candidate_denominator", "e0_certified",
            "source_admitted_but_e0_unknown", "rejected",
            "mask_construction_compiler_transitions")},
            "source_revision": revision}, sort_keys=True, indent=2))
    else:
        print(json.dumps(audit(args.study), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
