"""Reproducible E0/E1 experiment about representation, not FPGA speed.

The paper's scientific claim is tested by staged identity preservation,
architecture-coordinate differentiation, scoped decisions, legal blockers,
and a controlled global-only vs operator-local expressiveness ablation.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np

from fpgai_research.architecture.phi import iteration_mapping_receipt
from fpgai_research.architecture.resolver import ArchitectureError, resolve, TargetLimits
from fpgai_research.architecture.scopes import resolve_scoped
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.model_ir.contracts import ArchitectureRequest, Tensor, Operator, FunctionalModel
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import functional_fingerprint, verify_refinement
from fpgai_research.validation.reference import evaluate
from fpgai_research.validation.host import validate_host


def _fixture():
    """Deterministic two-operator DAG; 4 outputs permit observing PE=1/2/3."""
    rng = np.random.default_rng(2026)
    weight_a = tuple(map(float, rng.uniform(-0.3, .3, 16)))
    weight_b = tuple(map(float, rng.uniform(-0.3, .3, 16)))
    tensors = (Tensor("x", (1, 4)), Tensor("w_a", (4, 4), data=weight_a),
               Tensor("mid", (1, 4)), Tensor("w_b", (4, 4), data=weight_b),
               Tensor("y", (1, 4)))
    ops = (Operator("mm_a", "MatMul", ("x", "w_a"), ("mid",)),
           Operator("mm_b", "MatMul", ("mid", "w_b"), ("y",)))
    return FunctionalModel(tensors, ops, ("x",), ("y",)), np.array([[.2, -.1, .4, .3]], np.float32)


def _point_receipt(architecture):
    return {p.operator: iteration_mapping_receipt(p, max_iterations=1000)
            for p in architecture.points}


def _compile_case(model, x, *, label, base, overrides=None, scoped=None, host_cpp=False,
                  limits=TargetLimits()):
    record = {"name": label, "status": "BLOCKED", "blocker": None,
              "evidence": "E0-only"}
    try:
        if scoped is None:
            arch = resolve(model, base, overrides or {}, limits=limits)
        else:
            arch = resolve_scoped(model, scoped["default"],
                                  regions=scoped.get("regions"), operators=scoped.get("operators"),
                                  loops=scoped.get("loops"), limits=limits)
        lowered = lower(arch)
        stage = verify_refinement(lowered)
        mapping = _point_receipt(arch)
        emitted = emit_hls(lowered)
        record.update(status="PASS", evidence="E1-source-and-bounded-E0-oracle",
                      functional_sha256=stage["functional_sha256"],
                      effective_architecture_sha256=stage["effective_architecture_sha256"],
                      scoped_architecture_sha256=stage["scoped_architecture_sha256"],
                      mapping_sha256={name: item["complete_mapping_sha256"]
                                      for name, item in mapping.items()},
                      coverage={name: item["logical_iterations"] for name, item in mapping.items()},
                      requested={p.operator: asdict(p.requested) for p in arch.points},
                      effective={p.operator: asdict(p.effective) for p in arch.points},
                      provenance={p.operator: dict(p.decision_sources) for p in arch.points},
                      lowered={o.name: dict(tile_shape=list(o.tile_shape), replica_count=o.replica_count,
                                            pe_count=o.pe_count, simd_lanes=o.simd_lanes,
                                            fold_factor=o.fold_factor, pipeline_ii=o.pipeline_ii,
                                            numerical_type=o.numerical_type) for o in lowered.operators},
                      cpp_sha256=emitted.manifest["cpp_sha256"],
                      stage_invariants_verified=stage["stage_invariants_verified"])
        if host_cpp:
            host = validate_host(model, emitted, x)
            record["host_float"] = {key: host[key] for key in (
                "accepted", "rmse", "max_absolute_error", "cosine", "cpp_sha256")}
            if not host["accepted"]:
                record["status"] = "HOST_FLOAT_FAILED"
    except (ValueError, ArchitectureError) as exc:
        record["blocker"] = {"class": getattr(exc, "code", type(exc).__name__),
                              "message": str(exc)}
    return record


def run_model_ir_study(*, output: str | Path | None = None, host_cpp: bool = False) -> dict:
    """Run a bounded semantic research campaign; do not execute vendor tools."""
    if type(host_cpp) is not bool:
        raise ValueError("host_cpp must be a Boolean")
    model, x = _fixture()
    baseline = ArchitectureRequest()
    base = _compile_case(model, x, label="baseline", base=baseline, host_cpp=host_cpp)
    if base["status"] != "PASS":
        raise AssertionError("Scientific baseline failed its own semantic oracle")
    axes = {
        "precision_bits": (replace(baseline, precision_bits=12), {}),
        "parallelism": (baseline, {"mm_a": replace(baseline, parallelism=2)}),
        "pes": (baseline, {"mm_a": replace(baseline, pes=2)}),
        "simd": (baseline, {"mm_a": replace(baseline, simd=2)}),
        "serialization": (baseline, {"mm_a": replace(baseline, serialization=2)}),
        "tile": (baseline, {"mm_a": replace(baseline, tile=(1, 2, 3))}),
        "pipeline_ii": (baseline, {"mm_a": replace(baseline, pipeline_ii=2)}),
    }
    ablations = []
    for name, (default, overrides) in axes.items():
        row = _compile_case(model, x, label=name, base=default, overrides=overrides,
                            host_cpp=host_cpp)
        row["mapping_changed_operators"] = sorted(
            op for op, sig in row.get("mapping_sha256", {}).items()
            if sig != base["mapping_sha256"][op])
        row["lowered_changed_operators"] = sorted(
            op for op, mechanisms in row.get("lowered", {}).items()
            if mechanisms != base["lowered"][op])
        ablations.append(row)
    # Scope provenance is preserved even when the backend representation matches
    # an equivalent explicit operator-local contract.
    scoped = _compile_case(model, x, label="scoped_model_region_operator_loop",
              base=baseline, scoped={
                  "default": {"precision_bits": 16, "integer_bits": 6},
                  "regions": {"stem": {"operators": ["mm_a"], "architecture": {"parallelism": 2}}},
                  "operators": {"mm_a": {"pes": 2}},
                  "loops": {"mm_a.k": {"simd": 2, "pipeline_ii": 2, "tile_extent": 3}}},
              host_cpp=host_cpp)
    # One independent compute operator permits three PE settings.  Two independently
    # specialized instances admit all nine PE pairs; a flat global PE setting
    # can express only the three diagonal pairs. The Phi signatures are checked,
    # not inferred from the requested PE integers.
    pe_options = (1, 2, 3)
    profiles = []
    for first, second in product(pe_options, repeat=2):
        req = {"mm_a": replace(baseline, pes=first),
               "mm_b": replace(baseline, pes=second)}
        row = _compile_case(model, x, label=f"pe_{first}_{second}",
                            base=baseline, overrides=req, host_cpp=False)
        if row["status"] != "PASS":
            raise AssertionError(f"Global-vs-local experiment unexpectedly blocked: {row}")
        profiles.append({"pes": [first, second],
                         "mapping_pair": [row["mapping_sha256"]["mm_a"],
                                          row["mapping_sha256"]["mm_b"]]})
    distinct = {tuple(row["mapping_pair"]) for row in profiles}
    global_only = {tuple(row["mapping_pair"]) for row in profiles
                   if row["pes"][0] == row["pes"][1]}
    if len(distinct) != 9 or len(global_only) != 3:
        raise AssertionError("PE mapping expressiveness does not match observed Phi cardinality")
    rejects = [
        _compile_case(model, x, label="invalid_non_reduction_loop", base=baseline,
             scoped={"default": {}, "loops": {"mm_a.m": {"simd": 2}}}),
        _compile_case(model, x, label="resource_capacity", base=baseline,
             overrides={"mm_a": replace(baseline, pes=3)},
             limits=TargetLimits(max_parallel_units=2)),
        _compile_case(model, x, label="unsupported_mixed_precision", base=baseline,
             overrides={"mm_a": replace(baseline, precision_bits=12)}),
    ]
    baseline_hash = base["functional_sha256"]
    legal = [base, *ablations, scoped]
    result = {
        "schema": "fpgai-research/scientific-model-ir-study/v1",
        "evidence": "E0/E1-plus-optional-gcc-host-float",
        "vendor_tools_run": False, "fixed_point_bit_exact": False,
        "functional_sha256": functional_fingerprint(model),
        "input_sha256": sha256(x.tobytes()).hexdigest(),
        "numerical_reference_sha256": sha256(evaluate(model, {"x": x})["y"].tobytes()).hexdigest(),
        "baseline": base, "seven_dimension_ablations": ablations,
        "scoped_refinement": scoped,
        "expression_ablation": {
            "scope": "Two MatMul operator instances; independently selected PE count in {1,2,3}",
            "same_functional_model": True,
            "global_only_distinct_mapping_pairs": len(global_only),
            "operator_local_distinct_mapping_pairs": len(distinct),
            "additional_expressible_pairs": len(distinct) - len(global_only),
            "profiles": profiles,
            "not_hardware_qor": True,
        },
        "expected_blockers": rejects,
        "legal_cases": len(legal), "legal_cases_preserving_functional_hash": sum(
            row["functional_sha256"] == baseline_hash for row in legal if row["status"] == "PASS"),
        "legality_oracle_passed": all(row["status"] == "PASS" for row in legal),
        "blockers_observed": sum(row["status"] == "BLOCKED" for row in rejects),
        "limitations": "Bounded mapping enumeration and generated HLS. No physical PEs, cycle schedules, throughput or FPGA resource results have been established.",
    }
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result
