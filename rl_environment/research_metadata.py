"""Paper-method metadata captured at experiment creation; no vendor calls.

Defines descriptive provenance only; the authoritative executable reward is
reward.py at the campaign's source_revision. Never substitute this description
for the separately audited original HLS observations.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from fpgai_research.experiments.architecture_knobs_v2 import build_catalog

_SOURCE_FILES = (
    "rl_environment/reward.py",
    "rl_environment/gym_agent.py",
    "rl_environment/gym_env.py",
    "rl_environment/gym_study.py",
    "rl_environment/clock_actions.py",
    "rl_environment/post_hls_env.py",
    "rl_environment/gym_audit.py",
    # Source-pinned extension points, not writable runtime plugin discovery.
    "rl_environment/agents/registry.py",
    "rl_environment/experiments/registry.py",
    "rl_environment/environments/registry.py",
    "rl_environment/__init__.py",
    "rl_environment/runtime_contract.py",
    "src/fpgai_research/compiler/pipeline.py",
    "src/fpgai_research/toolchains/vitis_hls/tcl.py",
    "src/fpgai_research/toolchains/vitis_hls/package.py",
)


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def experiment_metadata(*, source_revision, use_message_passing,
                        beginner_pilot=False, curriculum_stage=None,
                        agent_algorithm="graph_actor_critic", experiment_id=None):
    root = Path(__file__).resolve().parents[1]
    if type(beginner_pilot) is not bool:
        raise ValueError("beginner_pilot flag must be boolean")
    from .curriculum import get_stage, stage_contract
    stage = get_stage(curriculum_stage)
    metadata = {
        "software_environment": {
            "python_version": sys.version,
            "host_platform": platform.platform(),
            "numpy_version": _package_version("numpy"),
            "torch_version": _package_version("torch"),
            "gymnasium_version": _package_version("gymnasium"),
            "matplotlib_version": _package_version("matplotlib"),
        },
        "schema": "fpgai-gym-paper-methods/v1",
        "exact_git_commit": source_revision,
        "research_extensions": {
            "agent_algorithm": agent_algorithm,
            "experiment_id": experiment_id,
            "environment_id": "graph_native_hls",
            "agent_initialized_from_scratch": True,
            "source_registry_pinned": True,
        },
        "source_file_sha256": {
            name: sha256((root / name).read_bytes()).hexdigest()
            for name in _SOURCE_FILES
        },
        "reward_definition": {
            "implementation": "rl_environment/reward.py::latency_only_reward",
            "qualification": "source-pinned original Vitis CSim numerical + HLS resources + uncertainty-adjusted HLS timing",
            "latency_unit": "nanoseconds",
            "latency_formula": "original_hls_worst_case_cycles * policy_requested_clock_period_ns",
            "improving_qualified_formula": "ln(previous_best_ns / updated_best_ns) - 0.005",
            "first_qualified_reward": -0.005,
            "qualified_no_improvement_reward": -0.005,
            "infeasible_reward": -0.05,
            "evaluation_penalty": 0.005,
            "reward_scope": "per-episode best qualified HLS estimated latency improvement",
            "no_reward_for_area_or_power": True,
        },
        "optimizer": {
            "policy": "one shared GNN plus explicit source-measured action-history attention + architecture Categorical and conditional floating-clock Beta",
            "initialization": "random from scratch per seed; updates across training episodes",
            "message_passing_enabled": bool(use_message_passing),
            "optimizer": "torch.optim.Adam",
            "learning_rate": 0.0005,
            "discount_gamma": 0.95,
            "critic_loss_weight": 0.5,
            "entropy_regularization_weight": 0.001,
            "max_gradient_norm": 5.0,
            "update_schedule": "one on-policy actor-critic plus real-HLS-labelled feasibility auxiliary gradient step per original Vitis Gym transition; no policy updates during frozen/controls",
            "history_window_steps": 30,
            "history_feature_dim": 41,
            "feasibility_auxiliary_weight": 0.2,
            "feasibility_auxiliary_positive_class_weight": 8.0,
            "feasibility_auxiliary_label_source": "qualified from original source-pinned Vitis CSim and CSynth",
            "feasibility_auxiliary_is_reward": False,
            "checkpoint_format": "private policy NPZ + trusted local optimizer state, both SHA256",
        },
        "metrics_schema": "fpgai-gym-policy-update-diagnostics/v1",
        "missing_metrics_mean": "not recorded or not applicable; do not impute",
        "physical_fmax_measured": False,
    }
    if stage is not None:
        metadata["curriculum_contract"] = stage_contract(stage)
        metadata["reward_definition"] = {
            "implementation": "rl_environment/reward.py::curriculum_real_hls_cycle_reward",
            "mode": "real_verified_hls_mac_ns_efficiency/v1",
            "normalizer_ns": "32.0 * sum(original_Model_IR_MAC_counts)",
            "feasible_reward_formula": (
                "normalizer_ns / (normalizer_ns + original_hls_worst_cycles * "
                "original_requested_clock_ns)"),
            "rejection_reward": -.05,
            "unqualified_cases_never_receive_imputed_latency": True,
            "no_reward_for_area_or_power": True,
            "strictly_decreasing_in_qualified_HLS_latency_ns_for_fixed_task": True,
            "not_equivalent_to_best_of_ten": True,
            "never_claims_post_route_Fmax": True,
        }
    elif beginner_pilot:
        metadata["reward_definition"] = {
            "implementation":
                "rl_environment/reward.py::beginner_real_hls_latency_reward",
            "mode": "beginner_dense_real_hls_cycle_efficiency_v1",
            "qualification":
                "original source-hashed Vitis CSim numerical, resources, fresh "
                "HLS latency/clock and executed Tcl uncertainty",
            "only_for": "one MatMul, 16/6 fixed numeric format, 32ns fixed clock",
            "exact_matmul_work": "input_width * output_width",
            "feasible_reward_formula":
                "exact_matmul_work/(exact_matmul_work + original_hls_worst_cycles)",
            "infeasible_reward": -0.05,
            "all_feasible_trials_receive_original_hls_latency_feedback": True,
            "strictly_monotone_with_lower_latency_on_same_task": True,
            "equivalent_to_best_of_ten_objective": False,
            "relative_latency_objective_in_outputs": "best qualified original HLS ns",
            "no_reward_for_area_or_power": True,
            "not_comparable_reward_scale_to_full_joint_clock_study": True,
        }
    return metadata


def summarize_trial(outcome, *, task, training_metrics):
    """Retain native case measurements/architecture for paper table generation.

    Source/testbench, HLS report, original logs and native invocation hashes
    remain in the immutable native case directory.
    """
    result = outcome["result"]
    catalog = build_catalog(task.model)
    candidate = tuple(tuple(row) for row in outcome["candidate"])
    # Convert tile tuples and other architecture values into the exact
    # canonical JSON-compatible representation independently regenerated by
    # gym_audit; Python tuple != JSON array after log serialization.
    per_operator_settings = json.loads(json.dumps(catalog.values(candidate)))
    network_shapes = [{
        "operator": op.name, "kind": op.kind,
        "input_tensor_shapes": [list(task.model.tensor(t).shape)
                                for t in op.inputs],
        "output_tensor_shapes": [list(task.model.tensor(t).shape)
                                 for t in op.outputs],
    } for op in task.model.operators]
    return {
        "schema": "fpgai-gym-trial-evidence/v1",
        "architecture_choices": outcome["candidate"],
        "architecture_compiler_knob_values": per_operator_settings,
        "network_operator_shapes": network_shapes,
        "requested_clock_ns": outcome["requested_clock_ns"],
        "hls_worst_case_cycles_if_qualified": result["hls_worst_case_cycles"],
        "hls_estimated_latency_ns_if_qualified": result["hls_estimated_latency_ns"],
        "csim_max_absolute_error": result["csim_max_absolute_error"],
        "allowed_max_absolute_error": task.max_absolute_error,
        "estimated_resources": result["hls_estimated_resources"],
        "available_resources": task.budgets,
        "hls_report_sha256": result["hls_report_sha256"],
        "vitis_invocation_sha256": result["vitis_invocation_sha256"],
        "hls_target_clock_ns": result["requested_clock_ns"],
        "hls_estimated_clock_ns": result["hls_estimated_clock_ns"],
        "hls_effective_scheduling_budget_ns": result[
            "hls_effective_scheduling_budget_ns"],
        "hls_scheduling_margin_ns": result["hls_scheduling_margin_ns"],
        "hls_timing_feasible": result["hls_timing_feasible"],
        "qualification_gates_failed": result["qualification_gates_failed"],
        "original_hls_timing_parse_issue": result["original_hls_timing_parse_issue"],
        "source_and_original_native_evidence_valid":
            result["source_and_original_native_evidence_valid"],
        "csim_passed": result["csim_passed"],
        "raw_hls_worst_case_cycles": result["raw_hls_worst_case_cycles"],
        "numerical_resource_timing_qualified": result["qualified"],
        "qualification_reason": result["reason"],
        "native_vendor_seconds": result["vendor_seconds"],
        "learning_metrics_recorded": training_metrics is not None,
    }
