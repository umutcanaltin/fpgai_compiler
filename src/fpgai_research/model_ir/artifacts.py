"""Deterministic, lossless, independently verifiable progressive Model IR artifacts.

The three IR stages and emitted HLS manifest are preserved separately.
Artifacts include numeric constants, so they may contain model IP; publication
must receive independent clearance. E0/E1 evidence only, never vendor QoR.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from fpgai_research.model_ir.contracts import (
    Tensor, Operator, FunctionalModel, ArchitectureRequest, ArchitecturePoint,
    ArchitecturalModel, LoweredOperator, LoweredModel,
)
from fpgai_research.model_ir.semantics import verify_refinement, functional_fingerprint
from fpgai_research.codegen.hls.emitter import (
    emit_hls, LOOP_CODEGEN_STATIC, LOOP_CODEGEN_LEGACY,
    LOOP_CODEGEN_CONV_STATIC, LOOP_CODEGEN_VERSIONS,
)

SCHEMA = "fpgai-research/progressive-model-ir-artifact/v1"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=False).encode("utf-8")


def _hash(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _tuple_nested(value):
    if isinstance(value, list):
        return tuple(_tuple_nested(v) for v in value)
    return value


def _request(data: dict) -> ArchitectureRequest:
    return ArchitectureRequest(**{**data, "tile": tuple(data.get("tile", ()))})


def _decode(data: dict) -> LoweredModel:
    f = data["functional"]
    tensors = tuple(Tensor(t["name"], tuple(t["shape"]), t["dtype"],
                           tuple(t["data"]) if t["data"] is not None else None)
                    for t in f["tensors"])
    operators = tuple(Operator(op["name"], op["kind"], tuple(op["inputs"]),
                               tuple(op["outputs"]),
                               tuple((k, _tuple_nested(v)) for k, v in op["attributes"]))
                      for op in f["operators"])
    functional = FunctionalModel(tensors, operators, tuple(f["inputs"]),
                                 tuple(f["outputs"]), f["version"])
    points = tuple(ArchitecturePoint(
        operator=p["operator"], requested=_request(p["requested"]),
        effective=_request(p["effective"]),
        iteration_shape=tuple(p["iteration_shape"]),
        source_scope=p["source_scope"], kind=p["kind"],
        decision_sources=tuple(tuple(v) for v in p["decision_sources"]),
    ) for p in data["architectural"]["points"])
    architectural = ArchitecturalModel(functional, points)
    lowered = tuple(LoweredOperator(
        name=op["name"], kind=op["kind"],
        iteration_shape=tuple(op["iteration_shape"]),
        architecture=_request(op["architecture"]),
        source_scope=op["source_scope"],
        tile_shape=tuple(op["tile_shape"]), replica_count=op["replica_count"],
        pe_count=op["pe_count"], simd_lanes=op["simd_lanes"],
        fold_factor=op["fold_factor"], pipeline_ii=op["pipeline_ii"],
        tail_policy=op["tail_policy"], numerical_type=op["numerical_type"],
        decision_sources=tuple(tuple(v) for v in op["decision_sources"]),
    ) for op in data["lowered"]["operators"])
    return LoweredModel(architectural, lowered)


def export_ir_artifact(lowered: LoweredModel) -> dict:
    """Produce a full-weight snapshot for independently rebuilding HLS code."""
    verify_refinement(lowered)
    emitted = emit_hls(lowered)
    stages = {
        "functional": asdict(lowered.architectural.functional),
        "architectural": {"points": [asdict(p) for p in lowered.architectural.points]},
        "lowered": {"operators": [asdict(op) for op in lowered.operators]},
        "hls": {"manifest": emitted.manifest},
    }
    return {
        "schema": SCHEMA,
        "evidence": "E0-IR/E1-HLS-source-only",
        "contains_model_constant_values": True,
        "stages": stages,
        "stage_sha256": {k: _hash(v) for k, v in stages.items()},
        "functional_sha256": functional_fingerprint(lowered.architectural.functional),
        "not_hardware_evidence": True,
    }


def verify_ir_artifact(artifact: dict) -> LoweredModel:
    """Validate content hashes, reconstruct three stages, re-emit HLS exactly."""
    if artifact.get("schema") != SCHEMA or not artifact.get("contains_model_constant_values"):
        raise ValueError("Unsupported or incomplete progressive Model IR artifact")
    stages = artifact["stages"]
    if set(stages) != {"functional", "architectural", "lowered", "hls"}:
        raise ValueError("Stage artifact must preserve exactly all four stages")
    hashes = {k: _hash(v) for k, v in stages.items()}
    if hashes != artifact["stage_sha256"]:
        raise ValueError("Stage integrity hash mismatch")
    model = _decode(stages)
    verify_refinement(model)
    if functional_fingerprint(model.architectural.functional) != artifact["functional_sha256"]:
        raise ValueError("Functional meaning fingerprint mismatch")
    saved_manifest=stages["hls"]["manifest"]
    if not isinstance(saved_manifest,dict):
        raise ValueError("HLS artifact lacks a source manifest")
    # Old scientific snapshots were emitted before loop-codegen versions
    # existed. They MUST still regenerate byte-identical source under the
    # archived legacy path; evaluating them with today's HLS generator
    # would invalidate genuine source/vendor receipts.
    version=saved_manifest.get("loop_codegen_version",LOOP_CODEGEN_LEGACY)
    if version not in LOOP_CODEGEN_VERSIONS:
        raise ValueError("Unknown saved HLS loop-codegen version")
    if _canonical(emit_hls(model,loop_codegen=version).manifest) != _canonical(saved_manifest):
        raise ValueError("HLS source manifest cannot be regenerated from the Lowered Model IR")
    return model


def compare_ir_artifacts(left: dict, right: dict) -> dict:
    """Representation ablation: distinguish function, mapping, and HLS effects."""
    l = verify_ir_artifact(left)
    r = verify_ir_artifact(right)
    if left["functional_sha256"] != right["functional_sha256"]:
        raise ValueError("Cannot attribute an architecture ablation to an altered functional graph")
    lp = {p.operator: p for p in l.architectural.points}
    rp = {p.operator: p for p in r.architectural.points}
    lo = {op.name: op for op in l.operators}
    ro = {op.name: op for op in r.operators}
    if lp.keys() != rp.keys():
        raise ValueError("Operator identities differ")
    return {
        "schema": "fpgai-research/progressive-model-ir-comparison/v1",
        "functional_preserved": True,
        "functional_sha256": left["functional_sha256"],
        "changed_effective_mapping_operators": sorted(
            k for k in lp if lp[k].effective != rp[k].effective),
        "changed_scope_provenance_operators": sorted(
            k for k in lp if (lp[k].source_scope, lp[k].decision_sources)
            != (rp[k].source_scope, rp[k].decision_sources)),
        "changed_lowered_operators": sorted(k for k in lo if lo[k] != ro[k]),
        "hls_source_changed": left["stages"]["hls"]["manifest"]["cpp_sha256"]
                              != right["stages"]["hls"]["manifest"]["cpp_sha256"],
        "native_tools_executed": False,
        "evidence": "E0-IR/E1-generated-source-only",
    }


def read_ir_artifact(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_ir_artifact(payload)
    return payload


def write_ir_artifact(path: str | Path, lowered: LoweredModel) -> dict:
    artifact = export_ir_artifact(lowered)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing reproducibility artifact: {destination}")
    destination.write_bytes(json.dumps(artifact, indent=2, sort_keys=True,
                                       ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
    return artifact
