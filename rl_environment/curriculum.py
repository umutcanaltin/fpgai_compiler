"""Immutable, opt-in REAL-Vitis Gym benchmark stages, ordered easy to hard.

Stage configuration is experiment provenance, not a synthetic performance
oracle. No stage uses a simulator-reward or changes the full compiler's default.
"""
from __future__ import annotations

from dataclasses import dataclass

FOUR_KNOBS = ("parallelism", "pes", "simd", "serialization")
SIX_KNOBS = (*FOUR_KNOBS, "tile", "pipeline_ii")
ALL_KNOBS = ("precision_bits", "integer_bits", *SIX_KNOBS)


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    family: str
    knob_fields: tuple[str, ...]
    fixed_clock_ns: float | None
    freeze_numeric_format: bool
    dense_reward: bool
    train_depths: tuple[int, ...]
    val_depths: tuple[int, ...]
    test_depths: tuple[int, ...]


STAGES = (
    Stage("L0", "Single small MatMul, four hardware knobs", "beginner",
          FOUR_KNOBS, 32., True, True, (1,), (1,), (1,)),
    Stage("L1", "Single variable-width MatMul, same four knobs", "single",
          FOUR_KNOBS, 32., True, True, (1,), (1,), (1,)),
    Stage("L2", "Two-operator MatMul/Gemm, four knobs per operator", "two",
          FOUR_KNOBS, 32., True, True, (2,), (2,), (2,)),
    Stage("L3", "One/two-operator MatMul/Gemm with tiling and II", "mixed",
          SIX_KNOBS, 32., True, True, (1, 2), (1, 2), (1, 2)),
    Stage("L4", "Unlock globally synchronized precision at fixed clock", "mixed",
          ALL_KNOBS, 32., False, True, (1, 2), (1, 2), (1, 2)),
    Stage("L5", "Unlock full continuous 2-32 ns target clock", "mixed",
          ALL_KNOBS, None, False, True, (1, 2), (1, 2), (1, 2)),
    Stage("L6", "Hard transfer from one/two to three/four operators", "transfer",
          ALL_KNOBS, None, False, True, (1, 2), (3,), (4,)),
)
BY_NAME = {s.name: s for s in STAGES}


def get_stage(name: str | None) -> Stage | None:
    if name is None:
        return None
    if type(name) is not str or name not in BY_NAME:
        raise ValueError("Unknown Gym curriculum stage (allowed L0..L6)")
    return BY_NAME[name]


def stage_contract(stage: Stage) -> dict:
    """Source-pinnable machine-readable stage contract."""
    return {
        "schema": "fpgai-original-vitis-gym-curriculum/v1",
        "stage": stage.name,
        "family": stage.family,
        "minimum_training_episodes_for_complete_result": 100,
        "training_hls_calls_per_episode": 10,
        "smoke_is_separate_from_scientific_training": True,
        "native_hls_clock_period_resolution_ns": 0.01,
        "native_hls_uncertainty_resolution_ns": 0.01,
        "allowed_architecture_knobs": list(stage.knob_fields),
        "fixed_clock_ns": stage.fixed_clock_ns,
        "freeze_numeric_format": stage.freeze_numeric_format,
        "training_reward": (
            "real_verified_hls_mac_normalized_latency_ns_efficiency/v1"
            if stage.dense_reward else "legacy_best_improvement/v1"),
        "train_depths": list(stage.train_depths),
        "validation_depths": list(stage.val_depths),
        "test_depths": list(stage.test_depths),
        "uses_original_vitis_hls_only": True,
        "strict_numerical_resource_timing_source_gates": True,
        "pretraining_between_stages": False,
        "policy_initialization": "from_scratch_per_stage_seed",
        "physical_fmax_measured": False,
    }


def stage_validation(stage: Stage, *, fixed_clock, frozen_numeric, beginner):
    if stage.fixed_clock_ns != fixed_clock:
        raise ValueError("Stage clock contract mismatch")
    if frozen_numeric != stage.freeze_numeric_format:
        raise ValueError("Stage precision contract mismatch")
    if beginner != (stage.name == "L0"):
        raise ValueError("Stage L0 requires beginner pilot; only L0 uses that flag")
