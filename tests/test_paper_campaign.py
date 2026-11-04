"""No-Vitis paper campaign preflight and smoke acceptance gates."""
import pytest

from rl_environment.paper_campaign import (
    check_smoke_receipts, execute, study_arguments, study_plan,
)


def test_l0_smoke_is_bounded_source_only():
    p = study_plan()
    assert p["status"] == "SOURCE_ONLY_NOT_EXECUTED"
    assert p["original_vitis_calls_executed"] == 0
    assert p["native_budget"] == 22
    kw = study_arguments()
    assert kw["training"] == 2
    assert kw["calls_per_network"] == 2
    assert kw["agent_algorithm"] == "graph_actor_critic"


def test_l0_full_requires_smoke_and_1090_budget():
    p = study_plan(mode="full")
    assert p["native_budget"] == 1090
    assert p["full_run_requires_qualifying_smoke"]
    assert study_arguments(mode="full")["training"] == 100


@pytest.mark.parametrize("args", [
    {"stage": "L7"}, {"agent": "graph_ppo"}, {"mode": "unsafe"},
    {"seed": -1}, {"seed": 1.5},
])
def test_unavailable_or_malformed_study_rejected(args):
    with pytest.raises(ValueError):
        study_plan(**args)


def test_smoke_gate_needs_real_audit_22_and_14_qualified():
    audit = {"status": "INDEPENDENT_GYM_HLS_AUDIT_PASSED",
             "original_cases_recomputed": 22, "source_revision": "abc"}
    funnel = {"counts": {"original_cases": 22, "qualified": 14}}
    assert check_smoke_receipts(
        audit_report=audit, funnel=funnel)["status"] == "SMOKE_ACCEPTED_FOR_FULL_RESEARCH"
    for bad in (
        {**funnel, "counts": {"original_cases": 22, "qualified": 13}},
        {**funnel, "counts": {"original_cases": 22, "qualified": 20,
                              "timing_parse_exception": 1}},
        {**funnel, "counts": {"original_cases": 21, "qualified": 20}},
    ):
        assert check_smoke_receipts(
            audit_report=audit, funnel=bad)["status"] == "SMOKE_REJECTED_DO_NOT_START_FULL"
    assert check_smoke_receipts(
        audit_report={"status": "FAILED", "original_cases_recomputed": 22},
        funnel=funnel)["status"] == "SMOKE_REJECTED_DO_NOT_START_FULL"


def test_native_requires_exact_budget_before_import_or_native_dirs(tmp_path):
    destination = tmp_path / "never_created"
    with pytest.raises(ValueError, match="exactly 22"):
        execute(stage="L0", agent="graph_actor_critic",
                mode="smoke", seed=111, output=str(destination),
                max_total_vitis_calls=21)
    assert not destination.exists()


def test_native_refuses_overwrite_before_tool_use(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        execute(stage="L0", agent="graph_actor_critic",
                mode="smoke", seed=111, output=str(existing),
                max_total_vitis_calls=22)
