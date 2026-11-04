"""Auditable bounded instances of paper equation Phi_v for model-operator iterations.

The mapping encodes portable architecture coordinates, not actual FPGA cycles.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import product
from math import prod
import json

from fpgai_research.architecture.mapping import verify_mapping
from fpgai_research.architecture.resolver import coordinate


def iteration_mapping_receipt(point, *, max_iterations: int = 100_000) -> dict:
    """Hash all Phi coordinates, proving bounded coverage/collision properties."""
    check = verify_mapping(point, max_iterations=max_iterations)
    hasher = sha256()
    numerical = point.effective
    preview = []
    for index in product(*(range(dim) for dim in point.iteration_shape)):
        c = coordinate(point, index)
        # q includes the complete declared numeric representation, not just width.
        phi = {
            "q": {"signed": True, "width": numerical.precision_bits,
                  "integer_bits": numerical.integer_bits,
                  "rounding": "AP_RND", "overflow": "AP_SAT"},
            "kappa": c["tile"], "r": c["replica"], "p": c["pe"],
            "ell": c["lane"], "f": c["reuse_step"],
            "g": {"pipeline_ii_requested": c["pipeline_ii_request"],
                  "achieved_cycle_schedule": None},
        }
        entry = {"i": index, "phi": phi}
        hasher.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        hasher.update(b"\n")
        if len(preview) < 3:
            preview.append(entry)
    return {
        "schema": "fpgai-research/operator-phi-e0/v1",
        "operator": point.operator, "kind": point.kind,
        "iteration_shape": list(point.iteration_shape),
        "logical_iterations": prod(point.iteration_shape),
        "complete_mapping_sha256": hasher.hexdigest(),
        "mapping_preview": preview,
        "oracle": check,
        "evidence": "E0-bounded-semantic-enumeration",
        "note": "g preserves the II request; it does not report an achieved cycle schedule",
    }
