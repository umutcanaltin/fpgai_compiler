"""Deterministic structural ablations. Never fabricate Vitis or board metrics."""
from __future__ import annotations
from dataclasses import replace
from itertools import product
from pathlib import Path
import csv
import json
from fpgai_research.model_ir.contracts import ArchitectureRequest, FunctionalModel
from fpgai_research.architecture.resolver import ArchitectureError, resolve
from fpgai_research.model_ir.lowering import lower
from fpgai_research.codegen.hls.emitter import emit_hls


def sweep(model: FunctionalModel, base: ArchitectureRequest, operator: str,
          variants: dict[str, list], directory: str) -> list[dict]:
    """E0/E1-only campaigns. Each request is resolved and separately emitted or blocked."""
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in sorted(variants):
        if key not in base.__dataclass_fields__:
            raise ValueError(f"Unknown architecture dimension: {key}")
        for value in variants[key]:
            req = replace(base, **{key: tuple(value) if key == "tile" else value})
            row = {"dimension": key, "value": json.dumps(value), "operator": operator,
                   "requested": json.dumps(vars(req), sort_keys=True)}
            try:
                architecture = resolve(model, replace(base, **{key: value}) if key in ("precision_bits", "integer_bits") else base, {operator: req})
                emission = emit_hls(lower(architecture))
                row.update(evidence="E1", blocker="", cpp_sha256=emission.manifest["cpp_sha256"])
                label = f"{key}_{len(rows):03d}"
                (dest / f"{label}.cpp").write_text(emission.cpp)
                (dest / f"{label}.json").write_text(json.dumps(emission.manifest, indent=2, sort_keys=True))
            except (ArchitectureError, ValueError) as exc:
                row.update(evidence="BLOCKED", blocker=getattr(exc, "code", type(exc).__name__), cpp_sha256="")
            rows.append(row)
    with (dest / "ablations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, ["dimension", "value", "operator", "requested", "evidence", "blocker", "cpp_sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return rows
