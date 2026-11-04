"""Source-only gates for the beginner native Vitis MatMul experiment.

These fixtures check configuration, strict evidence and the independent source
contract. They DO NOT create fabricated scientific FPGA latency results.
"""
from hashlib import sha256
import json

import numpy as np
import pytest

from rl_environment.cohort import make_cohort
from rl_environment.gym_study import plan
from rl_environment.post_hls_env import PostHLSEnv
from rl_environment.clock_actions import validate_clock_action
from rl_environment.timing import (
    XML_REL, parse_hls_timing, audited_hls_timing,
)
from fpgai_research.toolchains.vitis_hls.package import make_vitis_tcl


def beginner_cohort(**kwargs):
    return make_cohort(
        training=100, validation=2, test=2, seed=111,
        clock_ns=None, max_vitis_calls=10,
        fixed_clock_ablation_ns=32.0, freeze_numeric_format=True,
        beginner_pilot=True, **kwargs)


def dims(task):
    return (task.model.tensor(task.model.inputs[0]).shape[-1],
            task.model.tensor(task.model.outputs[0]).shape[-1])


def test_beginner_cohort_is_small_matmul_only_and_heldout_widths():
    study = beginner_cohort()
    assert {dims(t) for t in study["train"]} == {
        (4,4), (4,8), (8,4), (8,8)}
    assert {dims(t) for t in study["validation"]} == {(4,6),(6,4)}
    assert {dims(t) for t in study["test"]} == {(6,8),(8,6)}
    assert all(len(task.model.operators) == 1
               and task.model.operators[0].kind == "MatMul"
               and task.fixed_clock_ablation_ns == 32.
               and task.freeze_numeric_format
               and task.beginner_pilot
               for split in study.values() for task in split)
    assert len({task.model_sha for split in study.values()
                for task in split}) == 104
    assert all(task.max_vitis_calls == 10
               for split in study.values() for task in split)


def test_beginner_plan_and_flag_fail_closed_when_not_explicit():
    p = plan(training=100, validation=1, test=1,
             calls_per_network=10, seeds=(111,), clocks_ns=(32.,),
             freeze_numeric_format=True, beginner_pilot=True)
    assert p["total_episodes_planned"] == 109
    assert p["maximum_original_vitis_invocations"] == 1090
    assert p["beginner_pilot_schema"] == "single_matmul_four_knobs_v1"
    assert p["searchable_architecture_fields"] == [
        "parallelism", "pes", "simd", "serialization"]
    for clock in (None, (24.,), (10.,)):
        with pytest.raises(ValueError, match="Beginner pilot"):
            plan(training=1, validation=1, test=1, calls_per_network=2,
                 clocks_ns=clock, freeze_numeric_format=True,
                 beginner_pilot=True)
    with pytest.raises(ValueError, match="Beginner pilot"):
        plan(training=1, validation=1, test=1, calls_per_network=2,
             clocks_ns=(32.,), freeze_numeric_format=False,
             beginner_pilot=True)


def test_beginner_candidate_menu_is_compiler_legal_and_only_four_knobs(tmp_path):
    task = beginner_cohort()["train"][0]
    env = PostHLSEnv(task, tmp_path / "new_single_matmul")
    baseline = env.catalog.baseline
    candidates = env.candidates(max_candidates=128)
    assert baseline in candidates
    assert len(set(candidates)) >= 10  # 10 genuine distinct possible HLS trials
    for candidate in candidates:
        assert len(candidate) == 1
        assert all(candidate[0][axis] == baseline[0][axis]
                   for axis in (0,1,6,7))
        values = env.catalog.values(candidate)[env.catalog.names[0]]
        assert values["precision_bits"] == 16
        assert values["integer_bits"] == 6
        assert values["tile"] == ()
        assert values["pipeline_ii"] is None
    assert validate_clock_action(task, 32.) == 32.
    assert env.calls == 0
    changed = env.catalog.mutate(baseline, (None, "precision_bits", +1))
    assert changed not in candidates
    with pytest.raises(ValueError):
        env.step(changed, clock_ns=32., execute_native=False)
    assert env.calls == 0
    assert not (env.root / "case_0000").exists()


def test_beginner_gnn_observation_and_actor_clock_are_fixed(tmp_path):
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch")
    from rl_environment.gym_env import GymGraphEnv
    from rl_environment.gym_agent import OnlineGraphTrainer
    task = beginner_cohort()["train"][0]
    obs, _ = GymGraphEnv(task, tmp_path / "obs").reset(seed=111)
    assert tuple(obs["clock_bounds_ns"]) == (32., 32.)
    agent = OnlineGraphTrainer(seed=111)
    action, tape = agent.choose(obs, training=True)
    assert action[1] == 32.
    assert bool(obs["action_mask"][action[0]])
    assert tape is not None


def test_beginner_tcl_explicit_uncertainty_is_validated_and_unambiguous():
    script = make_vitis_tcl(
        clock_ns=32., clock_uncertainty_ns=8.64)
    assert "create_clock -period 32 -name default\n" in script
    assert script.count("set_clock_uncertainty 8.640\n") == 1
    with pytest.raises(ValueError, match="uncertainty"):
        make_vitis_tcl(clock_ns=32., clock_uncertainty_ns=32.)
    with pytest.raises(ValueError, match="uncertainty"):
        make_vitis_tcl(clock_ns=32., clock_uncertainty_ns=-.1)


def _original_xml(target="32", estimate="12", uncertainty_xml=None):
    uncertainty = (
        "<ClockUncertainty>" + uncertainty_xml + "</ClockUncertainty>"
        if uncertainty_xml is not None else "")
    return (f"<Profile><UserAssignments><TargetClockPeriod>{target}</TargetClockPeriod>"
            f"{uncertainty}</UserAssignments>"
            f"<PerformanceEstimates><SummaryOfTimingAnalysis>"
            f"<EstimatedClockPeriod>{estimate}</EstimatedClockPeriod>"
            f"</SummaryOfTimingAnalysis></PerformanceEstimates></Profile>")


def test_missing_xml_uncertainty_still_fails_without_executed_tcl(tmp_path):
    report = tmp_path / "timing.xml"
    report.write_text(_original_xml())
    with pytest.raises(ValueError, match="Missing HLS clock uncertainty"):
        parse_hls_timing(report, 32.)
    direct = parse_hls_timing(report, 32.,
                              source_attested_uncertainty_ns=8.64)
    assert direct["hls_timing_feasible"]
    assert direct["hls_effective_scheduling_budget_ns"] == pytest.approx(23.36)
    assert "hash-attested-executed-tcl" in direct["timing_source"]
    report.write_text(_original_xml(uncertainty_xml="7.8"))
    with pytest.raises(ValueError, match="disagree"):
        parse_hls_timing(
            report, 32., source_attested_uncertainty_ns=8.64)


def test_audited_source_attested_tcl_is_required_for_missing_xml_uncertainty(
        tmp_path):
    report = tmp_path / XML_REL
    report.parent.mkdir(parents=True)
    report.write_text(_original_xml())
    digest = sha256(report.read_bytes()).hexdigest()
    script = tmp_path / "run_hls.tcl"
    script.write_text(make_vitis_tcl(
        clock_ns=32., clock_uncertainty_ns=8.64))
    script_sha = sha256(script.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"explicit_clock_uncertainty_ns": 8.64}))
    summary = {
        "evidence": "E3-HLS-synthesis-report",
        "hls_report_sha256": digest}
    native = {
        "fresh_report_sha256": {"csynth_xml": digest},
        "script_sha256": script_sha,
        "input_hashes_before_execution": {"tcl": script_sha},
        "input_hashes_after_execution": {"tcl": script_sha},
        "inputs_stable_during_execution": True,
        "completed_successfully": True,
    }
    qualified = audited_hls_timing(tmp_path, native, summary, 32.)
    assert qualified["hls_timing_feasible"]
    assert qualified["hls_clock_uncertainty_ns"] == 8.64
    assert qualified["hls_effective_scheduling_budget_ns"] == pytest.approx(23.36)
    for change in ("script_sha256", "input_hashes_before_execution",
                   "input_hashes_after_execution", "completed_successfully"):
        bad = dict(native)
        if change == "completed_successfully":
            bad[change] = False
        elif change == "script_sha256":
            bad[change] = "x" * 64
        else:
            bad[change] = {"tcl": "x" * 64}
        with pytest.raises(ValueError, match="Tcl"):
            audited_hls_timing(tmp_path, bad, summary, 32.)
    report.write_text(_original_xml(uncertainty_xml="1"))
    newdigest = sha256(report.read_bytes()).hexdigest()
    native["fresh_report_sha256"]["csynth_xml"] = newdigest
    summary["hls_report_sha256"] = newdigest
    with pytest.raises(ValueError, match="disagree"):
        audited_hls_timing(tmp_path, native, summary, 32.)



def test_beginner_reward_is_dense_real_hls_latency_only_and_failure_safe():
    from rl_environment.reward import (
        beginner_real_hls_latency_reward, latency_only_reward)
    kwargs = dict(qualified=True, input_width=4, output_width=8)
    r100 = beginner_real_hls_latency_reward(hls_cycles=100, **kwargs)
    r50 = beginner_real_hls_latency_reward(hls_cycles=50, **kwargs)
    assert r50 > r100 > 0
    assert r100 == pytest.approx(32/132)
    assert beginner_real_hls_latency_reward(
        qualified=False, hls_cycles=None,
        input_width=4, output_width=8) == -.05
    with pytest.raises(ValueError, match="source-attested"):
        beginner_real_hls_latency_reward(hls_cycles=None, **kwargs)
    # Historical study reward and its first-feasible penalty remain unchanged.
    assert latency_only_reward(None, 3200., qualified=True) == -.005


def test_beginner_paper_method_marks_reward_noncomparable_to_joint_study():
    from rl_environment.research_metadata import experiment_metadata
    old = experiment_metadata(
        source_revision="0"*40, use_message_passing=True)
    new = experiment_metadata(
        source_revision="0"*40, use_message_passing=True,
        beginner_pilot=True)
    assert old["reward_definition"]["implementation"].endswith(
        "::latency_only_reward")
    assert new["reward_definition"]["implementation"].endswith(
        "::beginner_real_hls_latency_reward")
    assert new["reward_definition"][
        "not_comparable_reward_scale_to_full_joint_clock_study"]
    assert not new["reward_definition"]["equivalent_to_best_of_ten_objective"]
