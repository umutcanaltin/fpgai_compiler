"""Immutable per-episode tasks; samples and weights never travel in public reports."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Literal
import numpy as np

from fpgai_research.model_ir.contracts import FunctionalModel
from fpgai_research.model_ir.semantics import functional_fingerprint
from .clock_actions import (CLOCK_MIN_NS, CLOCK_MAX_NS, CLOCK_RESOLUTION_NS,
                            CLOCK_ACTION_MODE, normalize_clock_period_ns)

RESOURCES = ("lut", "ff", "dsp", "bram_18k", "uram")
DEFAULT_PART = "xck26-sfvc784-2LV-c"
# Strict hard cap for new native Gym tasks; counts attempts, including failures.
MAX_GYM_VITIS_CALLS_PER_EPISODE = 30

# New Gym actions use a learnable bounded floating clock policy; there is no
# internally or externally enumerated list of clock targets.

@dataclass(frozen=True)
class Task:
    task_id: str
    model: FunctionalModel
    samples: np.ndarray
    part: str
    # Legacy pilot baseline only. The new Gym study deliberately sets None:
    # the target clock is *produced by the RL action*, never provided as a goal.
    clock_ns: float | None
    max_absolute_error: float
    resources: tuple[int, int, int, int, int]
    max_vitis_calls: int
    partition: Literal["train", "validation", "test"]
    legal_clocks_ns: tuple[float, ...] = ()  # Retired fixed-clock pilot only
    fixed_clock_ablation_ns: float | None = None
    freeze_numeric_format: bool = False  # Prospective architecture-only pilot
    beginner_pilot: bool = False  # Opt-in single-MatMul / four architecture knobs
    curriculum_stage: str | None = None

    def __post_init__(self):
        if type(self.freeze_numeric_format) is not bool:
            raise ValueError("freeze_numeric_format must be explicitly boolean")
        if type(self.beginner_pilot) is not bool:
            raise ValueError("beginner_pilot must be explicitly boolean")
        self.model.validate()
        from .curriculum import get_stage, stage_validation
        stage = get_stage(self.curriculum_stage)
        if stage is not None:
            stage_validation(
                stage, fixed_clock=self.fixed_clock_ablation_ns,
                frozen_numeric=self.freeze_numeric_format,
                beginner=self.beginner_pilot)
            depth = len(self.model.operators)
            allowed = {
                "train": stage.train_depths,
                "validation": stage.val_depths,
                "test": stage.test_depths,
            }
            if (depth not in allowed[self.partition]
                    or any(op.kind not in ("MatMul", "Gemm")
                           for op in self.model.operators)):
                raise ValueError("Model topology not permitted by selected curriculum stage")
        if self.beginner_pilot:
            if (self.fixed_clock_ablation_ns != 32.0
                    or not self.freeze_numeric_format
                    or len(self.model.operators) != 1
                    or self.model.operators[0].kind != "MatMul"):
                raise ValueError(
                    "Beginner pilot requires one MatMul, 32 ns clock and fixed 16/6 precision")
        if self.clock_ns is None:
            if self.legal_clocks_ns:
                raise ValueError("New Gym clocks are policy outputs, not input grids")
            clocks = ()
        else:
            clocks = self.legal_clocks_ns or (self.clock_ns,)
            if (not isinstance(clocks, tuple) or len(clocks) > 32 or not clocks
                    or any(type(t) not in (int, float) or not isfinite(t) or t <= 0
                           for t in clocks) or len(set(clocks)) != len(clocks)
                    or self.clock_ns not in clocks):
                raise ValueError("Legacy clock must lie within its legal clock grid")
        object.__setattr__(self, "legal_clocks_ns", tuple(float(t) for t in clocks))
        if self.fixed_clock_ablation_ns is not None:
            if self.clock_ns is not None:
                raise ValueError("Legacy clock cannot also specify an RL ablation")
            period = normalize_clock_period_ns(self.fixed_clock_ablation_ns)
            if abs(self.fixed_clock_ablation_ns - period) > 1e-8:
                raise ValueError("Fixed-clock ablation needs canonical millips")
            object.__setattr__(self, "fixed_clock_ablation_ns", period)
        if (not isinstance(self.task_id, str) or not self.task_id
                or not self.task_id.replace("_", "").isalnum()
                or self.partition not in ("train", "validation", "test")):
            raise ValueError("Invalid task ID or partition")
        if not isinstance(self.part, str) or not self.part or any(
                char in self.part for char in " \t\n;{}"):
            raise ValueError("Invalid FPGA part")
        for value in (self.max_absolute_error,):
            if type(value) not in (float, int) or not isfinite(value) or value <= 0:
                raise ValueError("Numerical error limit must be finite and positive")
        if self.clock_ns is not None and (
                type(self.clock_ns) not in (float, int)
                or not isfinite(self.clock_ns) or self.clock_ns <= 0):
            raise ValueError("Legacy clock must be finite and positive")
        if (len(self.resources) != 5 or any(type(x) is not int or x < 0 for x in self.resources)
                or type(self.max_vitis_calls) is not int
                or not 1 <= self.max_vitis_calls <= (
                    MAX_GYM_VITIS_CALLS_PER_EPISODE
                    if self.clock_ns is None else 64)):
            raise ValueError("Five usable resource limits and bounded call count required; "
                             "new Gym episodes permit at most 30 Vitis attempts")
        if len(self.model.inputs) != 1 or len(self.model.outputs) != 1:
            raise ValueError("Vitis C simulation requires one model input/output")
        if (not isinstance(self.samples, np.ndarray)
                or self.samples.dtype != np.float32 or self.samples.ndim < 2
                or self.samples.shape[1:] != self.model.tensor(self.model.inputs[0]).shape
                or not 1 <= self.samples.shape[0] <= 256
                or not np.isfinite(self.samples).all()):
            raise ValueError("Expected finite float32 samples shaped [N, *model_input_shape]")
        fixed = np.array(self.samples, copy=True)
        fixed.setflags(write=False)
        object.__setattr__(self, "samples", fixed)

    @property
    def clock_choices_ns(self):
        return self.legal_clocks_ns

    @property
    def budgets(self):
        return dict(zip(RESOURCES, self.resources))

    @property
    def model_sha(self):
        return functional_fingerprint(self.model)

    @property
    def sample_sha(self):
        return sha256(self.samples.tobytes()).hexdigest()

    @property
    def task_sha(self):
        payload = {
            "model": self.model_sha, "samples": self.sample_sha,
            "sample_shape": list(self.samples.shape), "part": self.part,
            "legacy_clock_ns": self.clock_ns,
            "clock_action_mode": CLOCK_ACTION_MODE if self.clock_ns is None
                else "legacy_fixed_clock",
            "clock_action_bounds_ns": [CLOCK_MIN_NS, CLOCK_MAX_NS],
            "clock_action_resolution_ns": CLOCK_RESOLUTION_NS,
            "fixed_clock_ablation_ns": self.fixed_clock_ablation_ns,
            "legacy_clock_choices_ns": self.legal_clocks_ns,
            "tolerance": self.max_absolute_error,
            "resources": self.budgets, "max_calls": self.max_vitis_calls
        }
        # Omit for archived original campaigns: old source-pinned task SHA
        # must not change when this new opt-in dimension is disabled.
        if self.freeze_numeric_format:
            payload["freeze_numeric_format"] = True
            payload["fixed_global_numeric_format"] = {
                "precision_bits": 16, "integer_bits": 6}
        if self.curriculum_stage is not None:
            from .curriculum import get_stage, stage_contract
            payload["curriculum"] = stage_contract(get_stage(self.curriculum_stage))
        if self.beginner_pilot:
            payload["beginner_pilot"] = "single_matmul_four_knobs_v1"
            payload["searchable_fields"] = [
                "parallelism", "pes", "simd", "serialization"]
        return sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def public_descriptor(self):
        return {
            "task_id": self.task_id, "task_sha256": self.task_sha,
            "model_sha256": self.model_sha, "sample_sha256": self.sample_sha,
            "operator_kinds": [op.kind for op in self.model.operators],
            "operator_count": len(self.model.operators),
            "part": self.part,
            "clock_is_agent_action": self.clock_ns is None,
            "clock_action_mode": CLOCK_ACTION_MODE if self.clock_ns is None
                else "legacy_fixed_clock",
            "clock_action_bounds_ns": [CLOCK_MIN_NS, CLOCK_MAX_NS],
            "clock_action_resolution_ns": CLOCK_RESOLUTION_NS,
            "fixed_clock_ablation_ns": self.fixed_clock_ablation_ns,
            "freeze_numeric_format": self.freeze_numeric_format,
            "beginner_pilot": self.beginner_pilot,
            "curriculum_stage": self.curriculum_stage,
            "beginner_searchable_fields": (
                ["parallelism", "pes", "simd", "serialization"]
                if self.beginner_pilot else None),
            "fixed_numeric_precision_bits": 16 if self.freeze_numeric_format else None,
            "fixed_numeric_integer_bits": 6 if self.freeze_numeric_format else None,
            "legacy_clock_choices_ns": list(self.legal_clocks_ns),
            "maximum_absolute_error": self.max_absolute_error,
            "resource_caps": self.budgets, "max_vitis_calls": self.max_vitis_calls,
            "partition": self.partition, "actual_hls_metrics": None,
            "native_vendor_calls_executed": 0,
        }
