"""Reusable compiler-derived architecture knob catalog for Gym HLS search.

Legacy module path is retained for compiler dependencies. Supports all eight ArchitectureRequest fields:
precision_bits/integer_bits are model-global due to absent casts;
parallelism, PEs, SIMD, reuse, tiling and pipeline II are operator-local.
Requested target clock period is an independent design-wide action.

Every proposal goes through FPGAI resolve_scoped -> lower ->
verify_refinement -> emit_hls. No ad-hoc direct HLS text mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from fpgai_research.architecture.resolver import (
    ArchitectureError, TargetLimits, iteration_shape,
)
from fpgai_research.architecture.scopes import resolve_scoped
from fpgai_research.compiler import compile_architecture
from fpgai_research.model_ir.semantics import (
    functional_fingerprint, verify_refinement,
)

FIELDS = (
    "precision_bits", "integer_bits",
    "parallelism", "pes", "simd", "serialization",
    "tile", "pipeline_ii",
)
GLOBAL_FIELDS = ("precision_bits", "integer_bits")
LAYER_FIELDS = tuple(x for x in FIELDS if x not in GLOBAL_FIELDS)
SCHEMA = "fpgai-research/compiler-knob-catalog-v2/v1"
DEFAULT_CAPACITY = 8


def _tile_options(model, operator):
    shape = tuple(iteration_shape(model, operator))
    if len(shape) != 3:
        raise ValueError("v2 pilot supports rank-three MatMul/Gemm iteration only")
    m, n, k = shape
    candidates = [
        (),
        (m, n, k),
        (m, 1, 1),
    ]
    # Explore each output/reduction extent independently, not one fixed tile.
    for extent in (2, 4, 8, 16):
        for a, b in ((extent, k), (n, extent), (extent, extent)):
            candidates.append((m, min(n, a), min(k, b)))
    # Deduplicate deterministic equal representations while leaving explicit
    # untiled mapping () available independently of a fully tiled request.
    return tuple(dict.fromkeys(candidates))


def _validate_domain(name, values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Every compiler knob must have at least one option")
    if len(values) != len(set(values)):
        raise ValueError("Duplicate value in knob " + name)


@dataclass(frozen=True)
class KnobCatalog:
    names: tuple[str, ...]
    options: tuple[tuple[tuple[object, ...], ...], ...]
    baseline: tuple[tuple[int, ...], ...]
    actions: tuple[tuple[object, ...], ...]
    max_parallel_units: int

    def values(self, candidate):
        if len(candidate) != len(self.names):
            raise ValueError("Wrong number of network operators")
        out = {}
        for i, indices in enumerate(candidate):
            if len(indices) != len(FIELDS):
                raise ValueError("Missing architecture knob index")
            mapped = {}
            for j, idx in enumerate(indices):
                if type(idx) is not int or not 0 <= idx < len(self.options[i][j]):
                    raise ValueError("Architecture index outside knob domain")
                mapped[FIELDS[j]] = self.options[i][j][idx]
            out[self.names[i]] = mapped
        # Numeric formats are globally synchronized until legal edge casts
        # and independent numerical validation are implemented.
        for field in GLOBAL_FIELDS:
            if len({out[name][field] for name in self.names}) != 1:
                raise ValueError("Mixed numeric formats are unsupported")
        return out

    def mutate(self, candidate, action):
        operator, field, change = action
        if field not in FIELDS or change not in (-1, 1):
            raise ValueError("Invalid catalog architecture action")
        axis = FIELDS.index(field)
        if operator is None:
            if field not in GLOBAL_FIELDS:
                raise ValueError("Non-numeric architecture knobs are operator-local")
            targets = range(len(self.names))
            if len({candidate[i][axis] for i in targets}) != 1:
                raise ValueError("Numeric format action requires synchronized state")
            active = 0
        else:
            if (field in GLOBAL_FIELDS or type(operator) is not int
                    or not 0 <= operator < len(self.names)):
                raise ValueError("Unsupported operator scope")
            targets = (operator,)
            active = operator
        level = candidate[active][axis] + change
        if level < 0 or level >= len(self.options[active][axis]):
            return None
        result = [list(row) for row in candidate]
        for i in targets:
            result[i][axis] = level
        return tuple(tuple(row) for row in result)


def build_catalog(model, *, max_parallel_units=DEFAULT_CAPACITY):
    if (type(max_parallel_units) is not int or max_parallel_units < 1
            or max_parallel_units > 1024):
        raise ValueError("Invalid positive bounded architecture capacity")
    if not 1 <= len(model.operators) <= 4 or any(
            op.kind not in ("MatMul", "Gemm") for op in model.operators):
        raise ValueError("v2 native-research pilot supports 1..4 MatMul/Gemm operators")
    names = tuple(op.name for op in model.operators)
    per_operator = []
    baselines = []
    for op in model.operators:
        domains = (
            (8, 12, 16, 32),
            (4, 6, 8, 10),
            (1, 2, 4, 8),
            (1, 2, 3, 4, 8),
            (1, 2, 4, 8),
            (1, 2, 4, 8),
            _tile_options(model, op),
            (None, 1, 2, 4),
        )
        for name, options in zip(FIELDS, domains):
            _validate_domain(name, options)
        per_operator.append(domains)
        baselines.append(tuple(
            domains[j].index(value) for j, value in enumerate(
                (16, 6, 1, 1, 1, 1, (), None)
            )
        ))
    actions = tuple(
        (operator, field, delta)
        for field in FIELDS
        for operator in ((None,) if field in GLOBAL_FIELDS
                         else range(len(names)))
        for delta in (-1, 1)
    )
    return KnobCatalog(
        names, tuple(per_operator), tuple(baselines), actions,
        max_parallel_units,
    )


def resolve_candidate(model, catalog, candidate):
    names = catalog.names
    requests = catalog.values(candidate)
    # Validate unsupported precision/integer combinations via the REAL
    # compiler, not with a hand-written architecture "cost oracle".
    arch = resolve_scoped(
        model, {"precision_bits": 16, "integer_bits": 6},
        operators=requests,
        limits=TargetLimits(max_parallel_units=catalog.max_parallel_units),
    )
    if tuple(p.operator for p in arch.points) != names:
        raise AssertionError("Compiler operator identity drifted")
    return arch


def legal_candidate(model, catalog, candidate):
    try:
        resolve_candidate(model, catalog, candidate)
    except (ArchitectureError, ValueError):
        return False
    return True


def compile_candidate(model, catalog, candidate):
    arch = resolve_candidate(model, catalog, candidate)
    compiled = compile_architecture(arch)
    lowered = compiled.lowered
    receipt = compiled.refinement_receipt
    if not receipt["stage_invariants_verified"]:
        raise AssertionError("Compiler failed Model IR refinement")
    actual_model_sha = functional_fingerprint(model)
    if receipt["functional_sha256"] != actual_model_sha:
        raise AssertionError("Architecture changed numerical functional model")
    emitted = compiled.emission
    if emitted.manifest["functional_sha256"] != actual_model_sha:
        raise AssertionError("Generated source changed functional model identity")
    return emitted, arch, lowered


def action_catalog_signature(model, catalog, periods_ns):
    payload = {
        "schema": SCHEMA,
        "functional_sha256": functional_fingerprint(model),
        "names": catalog.names,
        "axis_names": FIELDS,
        "options": catalog.options,
        "baseline": catalog.baseline,
        "actions": catalog.actions,
        "periods_ns": list(periods_ns),
        "max_parallel_units": catalog.max_parallel_units,
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
