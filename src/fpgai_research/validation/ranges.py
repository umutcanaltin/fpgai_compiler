"""E0 value-range preflight for signed fixed-point HLS formats.

The range checks use float32 ONNX/NumPy reference values, not HLS
arithmetic rounding, accumulator widths or achieved physical behavior.
They help identify possible saturation *before* expensive Vitis runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class FixedFormat:
    width: int
    integer: int

    def __post_init__(self):
        if type(self.width) is not int or type(self.integer) is not int:
            raise ValueError("Fixed format widths must be integers")
        if not 2 <= self.width <= 64 or not 1 <= self.integer <= self.width:
            raise ValueError("Expected 2<=width<=64 and 1<=integer<=width")

    @property
    def fractional(self) -> int:
        return self.width - self.integer

    @property
    def quantum(self) -> float:
        return 2.0 ** -self.fractional

    @property
    def minimum(self) -> float:
        return -2.0 ** (self.integer - 1)

    @property
    def maximum(self) -> float:
        return 2.0 ** (self.integer - 1) - self.quantum


def audit_ranges(model, samples, formats: Iterable[FixedFormat],
                 *, max_total_elements: int = 2_000_000) -> dict:
    """Inspect inputs, constants and all intermediate *reference* tensors.

    A possible saturation means a float32 value is outside the signed
    format's representable interval. It is not a simulated HLS overflow:
    each real HLS operation can round, saturate or widen independently.
    """
    import numpy as np
    from fpgai_research.validation.samples import normalize_samples
    from fpgai_research.validation.reference import evaluate

    if len(model.inputs) != 1:
        raise ValueError("Numerical range audit currently needs one ONNX input")
    model.validate()
    formats = tuple(formats)
    if not formats or len(set(formats)) != len(formats):
        raise ValueError("Supply unique nonempty fixed-point formats")
    if type(max_total_elements) is not int or max_total_elements < 1:
        raise ValueError("Invalid work budget")
    input_name = model.inputs[0]
    batch = normalize_samples(samples, model.tensor(input_name).shape)
    if batch.dtype != np.float32:
        raise ValueError("Range-audit inputs must be float32; avoid implicit casting")
    total = 0
    aggregates: dict[str, dict] = {}
    tensor_shapes = {t.name: t.shape for t in model.tensors}
    constants = {t.name for t in model.tensors if t.data is not None}
    outputs = set(model.outputs)
    for item in batch:
        trace = evaluate(model, {input_name: item}, return_all=True)
        for name, data in trace.items():
            arr = np.asarray(data)
            if tuple(arr.shape) != tensor_shapes[name] or not np.all(np.isfinite(arr)):
                raise ValueError(f"Invalid traced reference values: {name}")
            count = int(arr.size)
            total += count
            if total > max_total_elements:
                raise ValueError("Range audit exceeds requested element budget")
            entry = aggregates.setdefault(name, {
                "role": "constant" if name in constants else "input" if name == input_name
                        else "output" if name in outputs else "intermediate",
                "shape": list(arr.shape), "min": float("inf"),
                "max": -float("inf"), "values": 0,
            })
            entry["values"] += count
            entry["min"] = min(entry["min"], float(np.min(arr)))
            entry["max"] = max(entry["max"], float(np.max(arr)))
            for fmt in formats:
                key = f"W{fmt.width}_I{fmt.integer}"
                a = entry.setdefault("format_counts", {}).setdefault(key, 0)
                entry["format_counts"][key] = a + int(np.count_nonzero((arr < fmt.minimum) | (arr > fmt.maximum)))
    stats = {}
    for fmt in formats:
        key = f"W{fmt.width}_I{fmt.integer}"
        saturated = sum(entry["format_counts"][key] for entry in aggregates.values())
        stats[key] = {"width": fmt.width, "integer_bits": fmt.integer,
                      "fractional_bits": fmt.fractional,
                      "quantum": fmt.quantum,
                      "min_representable": fmt.minimum,
                      "max_representable": fmt.maximum,
                      "observed_out_of_range": saturated,
                      "observed_values": total,
                      "possible_saturation": saturated > 0}
    return {
        "schema": "fpgai-research/numeric-range-preflight/v1",
        "evidence": "E0-float32-reference-only",
        "not_bit_exact": True,
        "samples": int(len(batch)), "tensor_statistics": aggregates,
        "formats": stats,
        "limitations": (
            "Float32 reference values and signed representable ranges only; "
            "not an ap_fixed simulator, accumulator model or evidence of Vitis numerical agreement."
        ),
    }
