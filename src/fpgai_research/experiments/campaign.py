"""Reproducible seven-axis paper campaigns; preparation never runs vendor tools."""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence
import json

from fpgai_research.architecture.resolver import ArchitectureError, TargetLimits, resolve
from fpgai_research.model_ir.contracts import ArchitectureRequest, FunctionalModel
from fpgai_research.model_ir.lowering import lower
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.toolchains.vitis_hls.package import write_vitis_package
from fpgai_research.toolchains.vivado.project import write_vivado_package

DIMENSIONS = ("precision_bits", "parallelism", "pes", "simd", "serialization", "tile", "pipeline_ii")


def _normalized(field: str, value):
    if field == "tile":
        if not isinstance(value, (list, tuple)):
            raise ValueError("Tile values must be arrays")
        return tuple(value)
    if field == "pipeline_ii" and value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"{field} requires an integer")
    return value


def _architecture_dict(value: ArchitectureRequest) -> dict:
    return {**asdict(value), "tile": list(value.tile)}


def generate_points(default: ArchitectureRequest, variants: Mapping[str, Sequence],
                    interactions: Sequence[Mapping[str, object]] = ()) -> list[tuple[str, ArchitectureRequest]]:
    """Generate unique baseline, single-axis and explicit interaction contracts."""
    unexpected = set(variants) - set(DIMENSIONS)
    if unexpected:
        raise ValueError(f"Not a paper architecture dimension: {sorted(unexpected)}")
    points = [("baseline", default)]
    seen = {json.dumps(_architecture_dict(default), sort_keys=True)}
    for field in DIMENSIONS:
        for value in variants.get(field, ()):
            request = replace(default, **{field: _normalized(field, value)})
            key = json.dumps(_architecture_dict(request), sort_keys=True)
            if key not in seen:
                points.append((field, request))
                seen.add(key)
    for index, changes in enumerate(interactions):
        if not changes or set(changes) - set(DIMENSIONS):
            raise ValueError(f"Invalid interaction point {index}")
        request = replace(default, **{field: _normalized(field, value)
                                       for field, value in changes.items()})
        key = json.dumps(_architecture_dict(request), sort_keys=True)
        if key not in seen:
            points.append(("interaction", request))
            seen.add(key)
    return points


def prepare_campaign(*, model: FunctionalModel, runtime_inputs, directory: str | Path,
                     operator: str, default: ArchitectureRequest,
                     variants: Mapping[str, Sequence],
                     interactions: Sequence[Mapping[str, object]] = (),
                     limits: TargetLimits = TargetLimits(),
                     part: str = "xck26-sfvc784-2LV-c", clock_ns: float = 5.0,
                     tolerance: float = 0.1,
                     source_provenance: Mapping[str, object] | None = None,
                     runtime_labels=None) -> list[dict]:
    """Create independent HLS cases and Vivado scripts; never claim E2/E3/E4."""
    model.validate()
    if operator not in {op.name for op in model.operators}:
        raise ValueError(f"Unknown operator identity: {operator}")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (axis, requested) in enumerate(generate_points(default, variants, interactions)):
        case = f"case_{index:04d}"
        record = {"case": case, "dimension": axis, "operator": operator,
                  "requested": _architecture_dict(requested),
                  "status": "BLOCKED", "evidence": None,
                  "reason": None, "cpp_sha256": None}
        try:
            # Current implementation requires a single model-wide numeric format.
            numerical = replace(default, precision_bits=requested.precision_bits,
                                integer_bits=requested.integer_bits)
            arch = resolve(model, numerical, {operator: requested}, limits=limits)
            emitted = emit_hls(lower(arch))
            case_dir = root / case
            case_dir.mkdir(exist_ok=False)
            manifest = write_vitis_package(str(case_dir), model, emitted, runtime_inputs,
                                           part=part, clock_ns=clock_ns,
                                           tolerance=tolerance, labels=runtime_labels)
            write_vivado_package(case_dir, part=part, clock_ns=clock_ns)
            (case_dir / "inputs_provenance.json").write_text(json.dumps({
                "source_provenance": dict(source_provenance or {}),
                "source_provenance_is_caller_supplied": True,
                "architecture": _architecture_dict(requested)
            }, indent=2, sort_keys=True) + "\n")
            record.update(status="READY", evidence="E1-HLS-source",
                          cpp_sha256=manifest["cpp_sha256"])
        except (ArchitectureError, ValueError) as exc:
            record["reason"] = {"code": getattr(exc, "code", type(exc).__name__),
                                "detail": str(exc)}
        rows.append(record)
    (root / "campaign.json").write_text(json.dumps({
        "schema": "fpgai-research/paper-campaign/v1",
        "operator": operator, "part": part, "clock_ns": clock_ns,
        "tolerance": tolerance, "cases": rows,
        "generated_only": True, "no_measured_vendor_results": True
    }, indent=2, sort_keys=True) + "\n")
    return rows
