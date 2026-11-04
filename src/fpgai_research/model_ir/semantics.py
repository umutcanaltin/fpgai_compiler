"""Independently check semantic preservation across the three progressive Model IR stages.

These E0/E1 contracts make the paper's principal claim testable: architecture
refinement adds mapping and implementation semantics without replacing the
Functional Model IR's operators, tensor dependencies or constants.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any

from fpgai_research.model_ir.contracts import (
    ArchitecturalModel, FunctionalModel, LoweredModel,
)


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def functional_fingerprint(model: FunctionalModel) -> str:
    """Hash the complete mathematical graph contract, including constant values."""
    model.validate()
    return _digest(asdict(model))


def architecture_fingerprint(model: ArchitecturalModel, *, include_provenance: bool = False) -> str:
    """Hash operator-local effective mapping contracts, optionally with scope provenance."""
    model.functional.validate()
    points = [dict(operator=p.operator, kind=p.kind, iteration_shape=p.iteration_shape,
                   effective=asdict(p.effective)) for p in model.points]
    if include_provenance:
        for entry, point in zip(points, model.points):
            entry.update(requested=asdict(point.requested),
                         source_scope=point.source_scope,
                         decision_sources=point.decision_sources)
    return _digest({"functional": functional_fingerprint(model.functional), "points": points})


def verify_refinement(model: LoweredModel) -> dict:
    """Reject dropped/reordered operators or mechanisms inconsistent with a resolved architecture.

    This is an exact implementation-contract check, not a proof that Vitis
    realizes the requested pipeline II or maps structures to intended cells.
    """
    architectural = model.architectural
    functional = architectural.functional
    functional.validate()
    # Re-resolve saved architectural choices independently from saved
    # iteration shapes. Matching stage hashes alone do not establish legality.
    # The original device inventory is not needed for portable model-level
    # checks; widths are admitted only for this reconstruction.
    from fpgai_research.architecture.resolver import TargetLimits, resolve
    from fpgai_research.model_ir.contracts import ArchitectureRequest
    provided_points = architectural.points
    if provided_points:
        widths = tuple(sorted({p.effective.precision_bits for p in provided_points}))
        recomputed = resolve(
            functional,
            ArchitectureRequest(),
            {p.operator: p.effective for p in provided_points},
            limits=TargetLimits(supported_widths=widths),
        )
        expected_shapes = {p.operator: p.iteration_shape for p in recomputed.points}
        for point in provided_points:
            if point.iteration_shape != expected_shapes[point.operator]:
                raise ValueError(f"MODELIR-SHAPE: {point.operator} iteration geometry disagrees with the Functional Model IR")
            keys = [name for name, _ in point.decision_sources]
            if len(keys) != len(set(keys)):
                raise ValueError(f"MODELIR-PROVENANCE: duplicate decision source for {point.operator}")
    identities = tuple(op.name for op in functional.operators)
    resolved = tuple(p.operator for p in architectural.points)
    lowered = tuple(op.name for op in model.operators)
    if identities != resolved or identities != lowered:
        raise ValueError("MODELIR-STAGE: operators must preserve identity and topological order")
    for functional_op, point, lo in zip(functional.operators, architectural.points, model.operators):
        a = point.effective
        if (point.requested != a or point.kind != functional_op.kind or
                lo.kind != functional_op.kind or lo.iteration_shape != point.iteration_shape or
                lo.architecture != a):
            raise ValueError(f"MODELIR-SEMANTICS: altered mapping/operator identity: {lo.name}")
        expected = {
            "tile_shape": a.tile or point.iteration_shape,
            "replica_count": a.parallelism,
            "pe_count": a.pes,
            "simd_lanes": a.simd,
            "fold_factor": a.serialization,
            "pipeline_ii": a.pipeline_ii,
            "tail_policy": "bounds-guard",
            "numerical_type": f"ap_fixed<{a.precision_bits},{a.integer_bits},AP_RND,AP_SAT>",
            "decision_sources": point.decision_sources,
            "source_scope": point.source_scope,
        }
        for key, desired in expected.items():
            if getattr(lo, key) != desired:
                raise ValueError(f"MODELIR-LOWERING: {lo.name}.{key} does not realize the effective contract")
    return {
        "schema": "fpgai-research/model-ir-refinement-e0/v1",
        "functional_sha256": functional_fingerprint(functional),
        "effective_architecture_sha256": architecture_fingerprint(architectural),
        "scoped_architecture_sha256": architecture_fingerprint(architectural, include_provenance=True),
        "operators_preserved": len(identities),
        "tensor_identities_preserved": len(functional.tensors),
        "stage_invariants_verified": True,
        "evidence": "E0-typed-model-ir-contract-only",
        "limitations": "No physical realization or universal numerical equivalence is implied",
    }
