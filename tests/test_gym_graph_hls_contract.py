"""New Gym/HLS contract tests. All hardware data below are FIXTURES, not evidence."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from rl_environment.cohort import make_cohort
from rl_environment.reward import qualified_latency_ns, latency_only_reward
from rl_environment.timing import parse_hls_timing, audited_hls_timing


def _xml(target="10", estimate="8.4", uncertainty="1.25"):
    return (f"<Profile><PerformanceEstimates><SummaryOfTimingAnalysis>"
            f"<TargetClockPeriod>{target}</TargetClockPeriod>"
            f"<EstimatedClockPeriod>{estimate}</EstimatedClockPeriod>"
            f"<ClockUncertainty>{uncertainty}</ClockUncertainty>"
            f"</SummaryOfTimingAnalysis></PerformanceEstimates></Profile>")


def test_hls_timing_strict_and_clock_aware(tmp_path):
    xml = tmp_path / "report.xml"
    xml.write_text(_xml())
    row = parse_hls_timing(xml, 10.)
    assert row["hls_timing_feasible"] is True
    assert row["hls_estimated_clock_ns"] == 8.4
    assert row["physical_fmax_mhz"] is None
    xml.write_text(_xml(target="8", estimate="9.4"))
    with pytest.raises(ValueError, match="different requested clock"):
        parse_hls_timing(xml, 10.)
    assert parse_hls_timing(xml, 8.)["hls_timing_feasible"] is False
    xml.write_text(_xml(estimate="undef"))
    with pytest.raises(ValueError, match="malformed"):
        parse_hls_timing(xml, 10.)
    xml.write_text(_xml().replace("</SummaryOfTimingAnalysis>",
                         "<EstimatedClockPeriod>9.7</EstimatedClockPeriod>"
                         "</SummaryOfTimingAnalysis>"))
    with pytest.raises(ValueError, match="malformed"):
        parse_hls_timing(xml, 10.)
    xml.write_text("<Profile></Profile>")
    with pytest.raises(ValueError, match="Missing"):
        parse_hls_timing(xml, 10.)


def test_timing_requires_exact_fresh_original_report_digest(tmp_path):
    path = tmp_path / "fpgai_hls_project" / "solution1" / "syn" / "report"
    path.mkdir(parents=True)
    xml = path / "fpgai_model_csynth.xml"
    xml.write_text(_xml())
    digest = sha256(xml.read_bytes()).hexdigest()
    source = {"evidence": "E3-HLS-synthesis-report",
              "hls_report_sha256": digest}
    native = {"fresh_report_sha256": {"csynth_xml": digest}}
    assert audited_hls_timing(tmp_path, native, source, 10.)["hls_timing_feasible"]
    with pytest.raises(ValueError, match="not fresh"):
        audited_hls_timing(tmp_path, {"fresh_report_sha256":
                                     {"csynth_xml": "wrong"}}, source, 10.)
    with pytest.raises(ValueError, match="not fresh"):
        audited_hls_timing(tmp_path, native, {"evidence": "E1-HLS-source"}, 10.)


def test_latency_objective_only_ns_never_resources():
    assert qualified_latency_ns(100, 10., feasible=True) == 1000.
    assert qualified_latency_ns(100, 8., feasible=True) == 800.
    assert qualified_latency_ns(100, 8., feasible=False) is None
    assert qualified_latency_ns(90, 12., feasible=True) == 1080.
    assert latency_only_reward(1000., 800., qualified=True) > 0
    assert latency_only_reward(1000., 1000., qualified=True) == pytest.approx(-.005)
    assert latency_only_reward(None, 800., qualified=True) == -.005
    assert latency_only_reward(None, None, qualified=False) == -.05
    with pytest.raises(ValueError, match="must not regress"):
        latency_only_reward(800., 900., qualified=True)


def test_tasks_keep_legal_clock_grid_immutable():
    task = make_cohort(training=2, validation=1, test=1, seed=42)["train"][0]
    with_clocks = replace(task, legal_clocks_ns=(8., 10., 12.))
    assert with_clocks.clock_choices_ns == (8., 10., 12.)
    assert with_clocks.task_sha != task.task_sha
    # Legacy pilot tasks retain a required clock; unlike Gym tasks, it must
    # remain inside any newly supplied search domain.
    with pytest.raises(ValueError, match="Legacy clock"):
        replace(task, legal_clocks_ns=(8., 12.))


def test_gym_reset_and_observation_never_launch_vendor(tmp_path, monkeypatch):
    pytest.importorskip("gymnasium")
    from rl_environment import post_hls_env
    from rl_environment.gym_env import GymGraphEnv
    invoked = []
    monkeypatch.setattr(post_hls_env, "run_vitis",
                        lambda *a, **k: invoked.append((a, k)))
    task = make_cohort(training=2, validation=1, test=1, seed=42,
                       clock_ns=None)["train"][1]
    env = GymGraphEnv(task, tmp_path / "gym", execute_native=False)
    first, info = env.reset(seed=9)
    assert env.observation_space.contains(first)
    assert info["original_vitis_calls"] == 0
    assert int(first["node_mask"].sum()) == len(task.model.operators)
    assert int(first["action_mask"].sum()) >= 1
    assert tuple(first["clock_bounds_ns"]) == (2., 32.)
    assert not invoked
    with pytest.raises(RuntimeError, match="explicit"):
        env.step((int(np.flatnonzero(first["action_mask"])[0]), 6.347))
    assert not invoked
    env.reset(seed=10)
    assert not invoked


def test_shared_message_passing_network_trains_only_online(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    from rl_environment.gym_env import GymGraphEnv
    from rl_environment.gym_agent import OnlineGraphTrainer
    cohort = make_cohort(training=2, validation=1, test=1, seed=42,
                         clock_ns=None)
    agent = OnlineGraphTrainer(seed=99)
    initial = None
    for i, task in enumerate(cohort["train"]):
        obs, _ = GymGraphEnv(task, tmp_path / f"ep_{i}").reset(seed=i)
        logits, value = agent.policy(obs)
        assert logits.shape[0] == len(obs["action_mask"])
        assert np.isfinite(float(value.detach()))
        action, tape = agent.choose(obs, training=True)
        assert obs["action_mask"][action[0]] == 1
        if initial is None:
            initial = agent.policy.mp1.self_layer.weight.detach().clone()
        loss = agent.update(tape, -.05, next_obs=obs)
        assert np.isfinite(loss)
    assert agent.updates == 2
    assert not (agent.policy.mp1.self_layer.weight.detach() == initial).all()
    obs, _ = GymGraphEnv(cohort["test"][0], tmp_path / "heldout").reset(seed=4)
    count = agent.updates
    _, frozen_tape = agent.choose(obs, training=False)
    assert frozen_tape is None and agent.updates == count
    agent.save_private(tmp_path / "checkpoint", {"source_revision": "fixture"})
    restored = OnlineGraphTrainer(seed=99)
    restored.load_private(tmp_path / "checkpoint")
    assert restored.updates == count


def test_new_plan_dry_run_creates_no_files(tmp_path):
    from rl_environment.gym_study import run
    output = tmp_path / "never_created"
    result = run(output=output, training=2, validation=1, test=1,
                 calls_per_network=3, execute_native=False)
    assert result["status"] == "DRY_RUN_ZERO_VENDOR_CALLS"
    assert result["maximum_original_vitis_invocations"] == 33
    assert not output.exists()
    with pytest.raises(ValueError, match="full --max-total"):
        run(output=output, training=2, validation=1, test=1,
            calls_per_network=3, execute_native=True, max_total_vitis_calls=1)
    assert not output.exists()


def test_matched_baseline_heuristics_only_consume_actual_native_labels():
    from rl_environment.baselines import BaselineChooser, METHODS
    features = np.zeros((8, 33), dtype=np.float32)
    for i in range(8):
        features[i, 0] = i / 8.
        features[i, -1] = 1.0 + i / 8.
    obs = {
        "action_mask": np.ones(8, dtype=np.int8),
        "action_features": features,
    }
    for method in METHODS:
        chooser = BaselineChooser(method, seed=1)
        first = chooser.choose(obs)
        assert 0 <= first < 8
        chooser.observe(features[first], 100.0)
        obs["action_mask"][first] = 0
        selected = chooser.choose(obs)
        assert 0 <= selected < 8 and selected != first
        chooser.observe(features[selected], None)
        assert chooser.best[1] == 100.
        obs["action_mask"][first] = 1


def test_zero_clock_uncertainty_is_valid_hls_metadata(tmp_path):
    path = tmp_path / "clock.xml"
    path.write_text(_xml(uncertainty="0"))
    assert parse_hls_timing(path, 10.)["hls_clock_uncertainty_ns"] == 0.


def test_clock_not_in_task_grid_rejected_without_vendor(tmp_path, monkeypatch):
    from rl_environment import post_hls_env
    from rl_environment.post_hls_env import PostHLSEnv
    invoked = []
    monkeypatch.setattr(post_hls_env, "run_vitis",
                        lambda *a, **kw: invoked.append((a, kw)))
    task = make_cohort(training=1, validation=1, test=1, seed=4)["train"][0]
    task = replace(task, legal_clocks_ns=(8., 10.))
    env = PostHLSEnv(task, tmp_path / "episode")
    with pytest.raises(ValueError, match="outside legacy task legal"):
        env.step(env.catalog.baseline, clock_ns=6., execute_native=True)
    assert not invoked
    assert env.calls == 0


def test_fixed_clock_and_no_message_ablation_plan():
    from rl_environment.gym_study import plan
    result = plan(training=2, validation=1, test=1, calls_per_network=3,
                  clocks_ns=(10.,), seeds=(111,), use_message_passing=False)
    assert result["clock_ablation_mode"] == "fixed_clock"
    assert result["message_passing_enabled"] is False
    assert result["maximum_original_vitis_invocations"] == 33


def test_no_message_policy_gradients_are_separate_from_gnn(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    from rl_environment.gym_env import GymGraphEnv
    from rl_environment.gym_agent import OnlineGraphTrainer
    task = make_cohort(training=2, validation=1, test=1, seed=40,
                       clock_ns=None)["train"][1]
    obs, _ = GymGraphEnv(task, tmp_path / "no_mp").reset(seed=5)
    agent = OnlineGraphTrainer(seed=88, use_message_passing=False)
    action, tape = agent.choose(obs, training=True)
    assert obs["action_mask"][action[0]] == 1
    assert np.isfinite(agent.update(tape, -.05, terminated=True))
    assert agent.updates == 1
    assert agent.policy.use_message_passing is False


def test_agent_clock_is_action_not_required_task_input(tmp_path):
    pytest.importorskip("gymnasium")
    from rl_environment.gym_env import GymGraphEnv
    task = make_cohort(training=1, validation=1, test=1, seed=103,
                       clock_ns=None)["train"][0]
    # No fixed requested clock value or list is supplied to the RL problem.
    assert task.clock_ns is None
    assert task.public_descriptor()["clock_is_agent_action"] is True
    assert "requested_clock_ns" not in task.public_descriptor()
    env = GymGraphEnv(task, tmp_path / "clock_free")
    obs, info = env.reset(seed=2)
    assert env.observation_space.contains(obs)
    offers = env.offers
    assert 1 <= len(offers) <= env.max_actions  # Architecture menu; clock is continuous.
    # No HLS timing knowledge is fabricated before the first original call.
    assert env.backend.last_result is None
    assert env.backend.best_latency_ns is None
    with pytest.raises(RuntimeError, match="explicit execute_native"):
        env.step((0, 6.347))


def test_clock_domain_has_no_forced_ten_ns_reference():
    from rl_environment.gym_study import plan
    p = plan(training=1, validation=1, test=1,
             calls_per_network=2, seeds=(123,))
    assert p["requested_clock_is_policy_output"] is True
    assert p["fixed_clock_task_input"] is False
    assert p["clock_action_bounds_ns"] == [2., 32.]
    assert p["clock_action_is_discrete"] is False
    assert p["clock_ablation_mode"] == "joint_clock"
    with pytest.raises(ValueError, match="Clock grids are prohibited"):
        plan(training=1, validation=1, test=1,
             calls_per_network=2, clocks_ns=(3., 4., 5.), seeds=(123,))


def test_cohort_generator_never_assigns_clock_in_gym_mode():
    cohort = make_cohort(training=2, validation=1, test=1,
                         seed=871, clock_ns=None)
    for split in cohort.values():
        for task in split:
            assert task.clock_ns is None
            assert task.clock_choices_ns == ()
            assert task.public_descriptor()["clock_action_bounds_ns"] == [2., 32.]
            assert task.public_descriptor()["clock_is_agent_action"] is True
            assert "requested_clock_ns" not in task.public_descriptor()


def test_read_only_existing_hls_timing_schema_probe(tmp_path, monkeypatch):
    import json
    from rl_environment.timing import XML_REL, probe_case_timing_schema
    from fpgai_research.toolchains.vitis_hls import runner as vendor
    monkeypatch.setattr(vendor, "run_vitis",
                        lambda *args, **kwargs: pytest.fail("Vendor tool must not execute"))
    case = tmp_path / "existing_case"
    report = case / XML_REL
    report.parent.mkdir(parents=True)
    report.write_text(_xml(target="10", estimate="8.4"))
    (case / "manifest.json").write_text(json.dumps({"clock_ns": 10.}))
    before = {p.relative_to(case): sha256(p.read_bytes()).hexdigest()
              for p in case.rglob("*") if p.is_file()}
    row = probe_case_timing_schema(case)
    assert row["status"] == "READ_ONLY_XML_SCHEMA_COMPATIBLE_NOT_NATIVE_QUALIFICATION"
    assert row["schema_timing_feasible_only"] is True
    assert row["qualified_for_new_gym_science"] is False
    assert row["source_csim_resource_attestation_performed"] is False
    assert row["native_vendor_invocations_during_probe"] == 0
    assert row["physical_fmax_mhz"] is None
    assert before == {p.relative_to(case): sha256(p.read_bytes()).hexdigest()
                      for p in case.rglob("*") if p.is_file()}
    report.write_text(_xml(target="10", estimate="10.1"))
    assert probe_case_timing_schema(case)["schema_timing_feasible_only"] is False


def test_read_only_timing_schema_probe_fails_closed(tmp_path):
    import json
    from rl_environment.timing import XML_REL, probe_case_timing_schema
    case = tmp_path / "no_report"
    case.mkdir()
    with pytest.raises(ValueError, match="manifest"):
        probe_case_timing_schema(case)
    (case / "manifest.json").write_text(json.dumps({"clock_ns": 8.}))
    with pytest.raises(ValueError, match="Missing or symlinked"):
        probe_case_timing_schema(case)
    report = case / XML_REL
    report.parent.mkdir(parents=True)
    report.write_text(_xml(target="10", estimate="8"))
    with pytest.raises(ValueError, match="different requested clock"):
        probe_case_timing_schema(case)
    (case / "manifest.json").write_text(json.dumps({"clock_ns": "10"}))
    with pytest.raises(ValueError, match="numeric"):
        probe_case_timing_schema(case)



def _vitis_xml_user_assignments(target="10", estimate="8.4",
                                 uncertainty="1.25"):
    # Common Vitis csynth XML layout also used by the hls4ml report reader:
    # target belongs to UserAssignments, estimate to SummaryOfTimingAnalysis.
    return (f"<Profile><UserAssignments>"
            f"<TargetClockPeriod>{target}</TargetClockPeriod>"
            f"<ClockUncertainty>{uncertainty}</ClockUncertainty>"
            f"</UserAssignments><PerformanceEstimates><SummaryOfTimingAnalysis>"
            f"<EstimatedClockPeriod>{estimate}</EstimatedClockPeriod>"
            f"</SummaryOfTimingAnalysis></PerformanceEstimates></Profile>")


def test_vitis_user_assignments_target_and_timing_estimate(tmp_path):
    path = tmp_path / "vitis_csynth.xml"
    path.write_text(_vitis_xml_user_assignments())
    row = parse_hls_timing(path, 10.)
    assert row["hls_target_clock_ns"] == 10.
    assert row["hls_estimated_clock_ns"] == 8.4
    assert row["hls_clock_uncertainty_ns"] == 1.25
    assert row["hls_timing_feasible"] is True
    assert row["physical_fmax_mhz"] is None
    path.write_text(_vitis_xml_user_assignments(target="8", estimate="9.4"))
    assert parse_hls_timing(path, 8.)["hls_timing_feasible"] is False
    with pytest.raises(ValueError, match="different requested clock"):
        parse_hls_timing(path, 10.)


def test_vitis_target_period_conflict_and_duplicate_fail_closed(tmp_path):
    path = tmp_path / "ambiguous.xml"
    good = _vitis_xml_user_assignments()
    # Both known target locations may occur; identical values are acceptable.
    identical = good.replace("<EstimatedClockPeriod>",
                             "<TargetClockPeriod>10</TargetClockPeriod>"
                             "<EstimatedClockPeriod>")
    path.write_text(identical)
    assert parse_hls_timing(path, 10.)["hls_timing_feasible"]
    path.write_text(identical.replace(
        "<TargetClockPeriod>10</TargetClockPeriod>"
        "<EstimatedClockPeriod>",
        "<TargetClockPeriod>12</TargetClockPeriod><EstimatedClockPeriod>"))
    with pytest.raises(ValueError, match="Conflicting HLS target periods"):
        parse_hls_timing(path, 10.)
    path.write_text(good.replace(
        "</UserAssignments>",
        "<TargetClockPeriod>10</TargetClockPeriod></UserAssignments>"))
    with pytest.raises(ValueError, match="malformed"):
        parse_hls_timing(path, 10.)
    path.write_text(_vitis_xml_user_assignments(target="NaN"))
    with pytest.raises(ValueError, match="malformed"):
        parse_hls_timing(path, 10.)
    path.write_text(_vitis_xml_user_assignments().replace(
        "<ClockUncertainty>1.25</ClockUncertainty>",
        "<ClockUncertainty>1.25</ClockUncertainty>"
        "<ClockUncertainty>2.0</ClockUncertainty>"))
    with pytest.raises(ValueError, match="uncertainty"):
        parse_hls_timing(path, 10.)


def test_readonly_vitis_layout_probe_does_not_qualify_old_report(tmp_path):
    import json
    from rl_environment.timing import probe_case_timing_schema, XML_REL
    case = tmp_path / "original_case"
    xml = case / XML_REL
    xml.parent.mkdir(parents=True)
    xml.write_text(_vitis_xml_user_assignments(target="10", estimate="8.0"))
    (case / "manifest.json").write_text(json.dumps({"clock_ns": 10.}))
    before = sha256(xml.read_bytes()).hexdigest()
    report = probe_case_timing_schema(case)
    assert report["status"] == "READ_ONLY_XML_SCHEMA_COMPATIBLE_NOT_NATIVE_QUALIFICATION"
    assert report["qualified_for_new_gym_science"] is False
    assert report["native_vendor_invocations_during_probe"] == 0
    assert report["schema_timing_feasible_only"] is True
    assert report["hls_target_clock_ns"] == 10
    assert report["hls_report_sha256"] == before
    assert sha256(xml.read_bytes()).hexdigest() == before



def test_clock_schema_inspector_is_readonly_and_exposes_only_known_fields(
        tmp_path, capsys):
    import json
    from rl_environment.timing import XML_REL, main, inspect_case_timing_schema
    case = tmp_path / "existing_with_private_material"
    report = case / XML_REL
    report.parent.mkdir(parents=True)
    report.write_text(_vitis_xml_user_assignments().replace(
        "</Profile>", "<PrivateModelName>DO_NOT_EXPORT_7265</PrivateModelName>"
        "</Profile>"))
    (case / "manifest.json").write_text(json.dumps({"clock_ns": 10.}))
    original = sha256(report.read_bytes()).hexdigest()
    record = inspect_case_timing_schema(case)
    assert record["recognized_clock_fields"][
        "UserAssignments/TargetClockPeriod"]["numeric_ns"] == 10.
    assert record["recognized_clock_fields"][
        "SummaryOfTimingAnalysis/EstimatedClockPeriod"]["numeric_ns"] == 8.4
    assert record["recognized_clock_fields"][
        "SummaryOfTimingAnalysis/TargetClockPeriod"]["count"] == 0
    assert record["qualified_for_new_gym_science"] is False
    assert record["original_vendor_invocations"] == 0
    main(["--case", str(case), "--inspect-schema"])
    displayed = capsys.readouterr().out
    assert "DO_NOT_EXPORT_7265" not in displayed
    assert "READ_ONLY_XML_FIELD_DIAGNOSTIC_NOT_QUALIFICATION" in displayed
    assert sha256(report.read_bytes()).hexdigest() == original


def test_vitis_2023_effective_clock_margin_rejects_too_slow_hls_estimate(tmp_path):
    # Actual observed private report values are reconstructed as a CPU fixture.
    # AMD UG1399 2023.2: effective scheduling budget = 10 - 2.7 = 7.3 ns.
    report = tmp_path / "margin.xml"
    report.write_text(_vitis_xml_user_assignments(
        target="10", estimate="7.664", uncertainty="2.7"))
    result = parse_hls_timing(report, 10.)
    assert result["hls_target_clock_ns"] == 10.
    assert result["hls_effective_scheduling_budget_ns"] == pytest.approx(7.3)
    assert result["hls_scheduling_margin_ns"] == pytest.approx(-.364)
    assert result["hls_timing_feasible"] is False
    report.write_text(_vitis_xml_user_assignments(
        target="10", estimate="7.299", uncertainty="2.7"))
    result = parse_hls_timing(report, 10.)
    assert result["hls_timing_feasible"] is True
    assert result["hls_scheduling_margin_ns"] == pytest.approx(.001)


def test_missing_hls_clock_uncertainty_fails_closed(tmp_path):
    report = tmp_path / "unknown_margin.xml"
    report.write_text(_vitis_xml_user_assignments(
        target="10", estimate="5", uncertainty="2.7").replace(
            "<ClockUncertainty>2.7</ClockUncertainty>", ""))
    with pytest.raises(ValueError, match="Missing HLS clock uncertainty"):
        parse_hls_timing(report, 10.)


def test_existing_case_probe_respects_effective_hls_scheduling_budget(tmp_path):
    import json
    from rl_environment.timing import probe_case_timing_schema, XML_REL
    case = tmp_path / "existing_from_vitis_2023"
    xml = case / XML_REL
    xml.parent.mkdir(parents=True)
    xml.write_text(_vitis_xml_user_assignments(
        target="10", estimate="7.664", uncertainty="2.7"))
    (case / "manifest.json").write_text(json.dumps({"clock_ns": 10.}))
    value = probe_case_timing_schema(case)
    assert value["schema_timing_feasible_only"] is False
    assert value["hls_scheduling_margin_ns"] == pytest.approx(-.364)
    assert value["hls_effective_scheduling_budget_ns"] == pytest.approx(7.3)
    assert value["qualified_for_new_gym_science"] is False
    assert value["native_vendor_invocations_during_probe"] == 0



def test_clock_action_domain_is_generated_internally_not_user_supplied():
    from rl_environment.clock_actions import CLOCK_ACTION_MODE
    from rl_environment.gym_study import plan
    cohort = make_cohort(training=1, validation=1, test=1, seed=104,
                         clock_ns=None)
    for items in cohort.values():
        for task in items:
            assert task.clock_ns is None
            assert task.clock_choices_ns == ()
            assert task.public_descriptor()["clock_action_mode"] == CLOCK_ACTION_MODE
            assert task.public_descriptor()["clock_is_agent_action"] is True
    p = plan(training=1, validation=1, test=1, calls_per_network=1)
    assert p["clock_action_mode"] == CLOCK_ACTION_MODE
    assert p["clock_action_bounds_ns"] == [2., 32.]
    assert p["clock_action_resolution_ns"] == .001
    assert p["clock_action_domain_origin"] == "learned_bounded_float"
    assert p["clock_action_is_discrete"] is False
    assert p["fixed_clock_task_input"] is False
    assert p["requested_clock_is_policy_output"] is True
    assert p["maximum_original_vitis_invocations"] == 10


def test_default_gym_study_cli_never_requires_clock_parameters(
        tmp_path, capsys, monkeypatch):
    import json
    from rl_environment import post_hls_env
    from rl_environment.gym_study import main, run
    monkeypatch.setattr(
        post_hls_env, "run_vitis",
        lambda *args, **kwargs: pytest.fail("Dry-run launched vendor process"))
    output = tmp_path / "not_staged"
    main(["--output", str(output), "--train-networks", "1",
          "--validation-networks", "1", "--test-networks", "1",
          "--calls-per-network", "1"])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "DRY_RUN_ZERO_VENDOR_CALLS"
    assert result["clock_action_domain_origin"] == "learned_bounded_float"
    assert result["clock_action_is_discrete"] is False
    assert result["clock_action_bounds_ns"] == [2., 32.]
    assert not output.exists()
    with pytest.raises(SystemExit):
        main(["--clocks-ns", "6", "8"])
    capsys.readouterr()  # CLI argument-error output is intentionally separate
    ablation = run(output=output, training=1, validation=1, test=1,
                   calls_per_network=1, clocks_ns=(10,), execute_native=False)
    assert ablation["clock_ablation_mode"] == "fixed_clock"
    assert ablation["clock_action_domain_origin"] == "explicit_fixed_clock_ablation"
    assert not output.exists()


def test_automatic_clock_action_choices_really_are_agent_outputs(tmp_path):
    pytest.importorskip("gymnasium")
    from rl_environment.gym_env import GymGraphEnv
    from rl_environment.clock_actions import CLOCK_MIN_NS, CLOCK_MAX_NS
    task = make_cohort(training=1, validation=1, test=1,
                       seed=105, clock_ns=None)["train"][0]
    env = GymGraphEnv(task, tmp_path / "auto_clock")
    obs, _ = env.reset(seed=33)
    assert env.observation_space.contains(obs)
    assert task.clock_ns is None
    assert 1 <= len(env.offers) <= env.max_actions
    assert tuple(obs["clock_bounds_ns"]) == (CLOCK_MIN_NS, CLOCK_MAX_NS)
    assert env.action_space.contains((0, np.asarray(3.141, dtype=np.float64)))
    # The declared Discrete space includes padded architecture slots.
    # Unmasked sampling is NOT guaranteed to choose a valid compiler action.
    assert int(obs["action_mask"].sum()) == len(env.offers)
    # Gymnasium Tuple.sample forwards per-child masks to Discrete and Box.
    # Only the architecture is masked; the floating clock stays continuous.
    for _ in range(20):
        sampled_slot, sampled_clock = env.action_space.sample(
            mask=(obs["action_mask"].astype(np.int8), None))
        assert env.action_space.contains((sampled_slot, sampled_clock))
        assert obs["action_mask"][int(sampled_slot)] == 1
        # The raw arbitrary floating value is canonicalized to 1 ps before
        # a possible native invocation, and reset/disabled steps run no Vitis.
        with pytest.raises(RuntimeError, match="explicit execute_native"):
            env.step((int(sampled_slot), sampled_clock))
    # Passing a padded, masked slot must fail closed *before* vendor execution.
    if len(env.offers) < env.max_actions:
        with pytest.raises(ValueError, match="Invalid or masked"):
            env.step((len(env.offers), np.asarray(3.141, dtype=np.float64)))
    assert env.backend.calls == 0
    assert not env.backend.tried_configs
    assert env.backend.last_result is None



def test_learned_clock_policy_samples_floating_subnanosecond_values(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    from rl_environment.gym_env import GymGraphEnv
    from rl_environment.gym_agent import OnlineGraphTrainer
    from rl_environment.clock_actions import (
        normalize_clock_period_ns, validate_clock_action)
    task = make_cohort(training=1, validation=1, test=1, seed=882,
                       clock_ns=None)["train"][0]
    obs, _ = GymGraphEnv(task, tmp_path / "continuous").reset(seed=25)
    trainer = OnlineGraphTrainer(seed=220)
    values = []
    original_head = trainer.policy.clock_head[-1].weight.detach().clone()
    for _ in range(15):
        (slot, period), tape = trainer.choose(obs, training=True)
        assert obs["action_mask"][slot]
        assert normalize_clock_period_ns(period) == period
        assert validate_clock_action(task, period) == period
        values.append(period)
        assert np.isfinite(trainer.update(tape, -.05, terminated=True))
    assert len(set(values)) >= 8
    assert any(abs(v - round(v)) > 0.001 for v in values)
    assert trainer.updates == 15
    assert not (trainer.policy.clock_head[-1].weight.detach()
                == original_head).all()
    (slot, period), tape = trainer.choose(obs, training=False)
    assert tape is None and obs["action_mask"][slot]
    assert validate_clock_action(task, period) == period


def test_float_clock_boundaries_and_noncanonical_pairs_reject_before_vendor(
        tmp_path, monkeypatch):
    from rl_environment.clock_actions import validate_clock_action
    from rl_environment import post_hls_env
    from rl_environment.post_hls_env import PostHLSEnv
    calls = []
    monkeypatch.setattr(post_hls_env, "run_vitis",
                        lambda *args, **kwargs: calls.append(1))
    task = make_cohort(training=1, validation=1, test=1,
                       seed=883, clock_ns=None)["train"][0]
    env = PostHLSEnv(task, tmp_path / "continuous_pair")
    for period in (1.999, 32.001, 9.12345, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            env.step(env.catalog.baseline, clock_ns=period, execute_native=True)
    assert not calls and env.calls == 0
    assert validate_clock_action(task, 3.141) == 3.141
    assert validate_clock_action(task, 8.527) == 8.527
    dry = env.step(env.catalog.baseline, clock_ns=3.141,
                   execute_native=False)
    assert dry["requested_clock_ns"] == 3.141
    assert not calls and env.calls == 0


def test_fixed_clock_ablation_is_explicit_and_not_normal_clock_input(tmp_path):
    from rl_environment.gym_agent import OnlineGraphTrainer
    from rl_environment.gym_env import GymGraphEnv
    from rl_environment.clock_actions import validate_clock_action
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    cohort = make_cohort(training=1, validation=1, test=1, seed=884,
                         clock_ns=None, fixed_clock_ablation_ns=7.125)
    task = cohort["train"][0]
    assert task.clock_ns is None
    assert task.fixed_clock_ablation_ns == 7.125
    obs, _ = GymGraphEnv(task, tmp_path / "fixed").reset(seed=3)
    assert tuple(obs["clock_bounds_ns"]) == (7.125, 7.125)
    agent = OnlineGraphTrainer(seed=22)
    (slot, period), tape = agent.choose(obs, training=True)
    assert period == 7.125
    assert obs["action_mask"][slot]
    assert np.isfinite(agent.update(tape, -.05, terminated=True))
    with pytest.raises(ValueError, match="fixed-clock ablation"):
        validate_clock_action(task, 7.124)
    assert validate_clock_action(task, 7.125) == 7.125


def test_floating_clock_exact_tcl_and_ns_reward_are_distinct(tmp_path):
    from fpgai_research.toolchains.vitis_hls.package import make_vitis_tcl
    from rl_environment.reward import qualified_latency_ns
    for period in (2.001, 3.141, 7.125, 31.999):
        tcl = make_vitis_tcl(clock_ns=period)
        assert f"create_clock -period {period:g} -name default" in tcl
        assert qualified_latency_ns(40, period, feasible=True) == 40 * period
    assert qualified_latency_ns(40, 7.125, feasible=True) != (
        qualified_latency_ns(40, 7.126, feasible=True))



def test_episode_count_plan_is_forecast_only_with_30_compile_hard_limit(tmp_path):
    from rl_environment.gym_study import plan, run
    from rl_environment.tasks import MAX_GYM_VITIS_CALLS_PER_EPISODE
    assert MAX_GYM_VITIS_CALLS_PER_EPISODE == 30
    p = plan(training=3000, validation=1, test=1, calls_per_network=30,
             estimated_seconds_per_compile=28.15)
    # 3000 train + 1 validation + 8 frozen/adaptive/control test arms +
    # 4 search-control tests; each distinct evaluation is one episode.
    assert p["episode_counts_by_method"]["train"] == 3000
    assert p["episode_counts_by_method"]["validation_frozen"] == 1
    assert p["episode_counts_by_method"]["test_frozen"] == 1
    assert p["episode_counts_by_method"]["test_frozen_stochastic"] == 1
    assert p["episode_counts_by_method"]["test_untrained_stochastic"] == 1
    assert p["episode_counts_by_method"]["test_adaptive"] == 1
    assert all(p["episode_counts_by_method"][f"test_{m}"] == 1
               for m in ("random", "greedy", "evolutionary", "parzen"))
    assert p["total_episodes_planned"] == 3009
    assert p["maximum_original_vitis_invocations"] == 90270
    assert p["hard_cap_vitis_compiles_per_episode"] == 30
    assert p["estimated_vitis_hours_at_max_calls"] == pytest.approx(
        90270 * 28.15 / 3600, abs=0.0005)
    assert p["wall_clock_limit_seconds"] is None
    assert p["estimates_are_not_stopping_conditions"] is True
    assert run(output=tmp_path / "never_created",
               training=3000, validation=1, test=1, calls_per_network=30,
               estimated_seconds_per_compile=28.15,
               execute_native=False)["status"] == "DRY_RUN_ZERO_VENDOR_CALLS"
    assert not (tmp_path / "never_created").exists()
    with pytest.raises(ValueError, match="Maximum 30"):
        plan(training=1, validation=1, test=1, calls_per_network=31)
    with pytest.raises(ValueError, match="Maximum 30"):
        plan(training=1, validation=1, test=1, calls_per_network=64)
    with pytest.raises(ValueError, match="finite and positive"):
        plan(training=1, validation=1, test=1, calls_per_network=30,
             estimated_seconds_per_compile=float("nan"))
    with pytest.raises(ValueError, match="Checkpoint interval"):
        run(output=tmp_path / "never_created", training=1, validation=1,
            test=1, calls_per_network=1, checkpoint_every_train_episodes=0)


def test_new_gym_task_rejects_more_than_30_original_vitis_attempts():
    cohort = make_cohort(training=1, validation=1, test=1,
                         clock_ns=None, max_vitis_calls=30)
    assert cohort["train"][0].max_vitis_calls == 30
    with pytest.raises(ValueError, match="at most 30 Vitis attempts"):
        make_cohort(training=1, validation=1, test=1,
                    clock_ns=None, max_vitis_calls=31)
    # The archived legacy fixed-clock pilot keeps its separate old contract.
    old = make_cohort(training=1, validation=1, test=1,
                      clock_ns=10., max_vitis_calls=31)
    assert old["train"][0].max_vitis_calls == 31


def test_episode_count_cli_is_alias_compatible_and_reports_estimates(
        tmp_path, capsys, monkeypatch):
    import json
    from rl_environment.gym_study import main
    from rl_environment import post_hls_env
    monkeypatch.setattr(post_hls_env, "run_vitis",
                        lambda *args, **kwargs: pytest.fail("Dry run launched Vitis"))
    output = tmp_path / "no_new_native_cases"
    main(["--output", str(output), "--train-episodes", "3000",
          "--validation-episodes", "1", "--test-tasks", "1",
          "--max-vitis-compiles-per-episode", "30",
          "--estimated-seconds-per-compile", "28.15",
          "--checkpoint-every-train-episodes", "10"])
    p = json.loads(capsys.readouterr().out)
    assert p["total_episodes_planned"] == 3009
    assert p["maximum_original_vitis_invocations"] == 90270
    assert p["checkpoint_every_train_episodes"] == 10
    assert p["estimated_seconds_per_compile_assumed"] == 28.15
    assert p["estimated_vitis_hours_at_max_calls"] is not None
    assert p["wall_clock_limit_seconds"] is None
    assert not output.exists()



def test_selected_100_training_episode_campaign_is_capped_and_dry_run_only(tmp_path):
    from rl_environment.gym_study import plan, run
    assumptions = dict(training=100, validation=1, test=1,
                       calls_per_network=30,
                       estimated_seconds_per_compile=28.15)
    p = plan(**assumptions)
    assert p["training_networks_per_seed"] == 100
    assert p["total_episodes_planned"] == 109
    assert p["episode_counts_by_method"]["train"] == 100
    assert p["maximum_original_vitis_invocations"] == 3270
    assert p["hard_cap_vitis_compiles_per_episode"] == 30
    assert p["estimated_vitis_hours_at_max_calls"] == pytest.approx(
        3270 * 28.15 / 3600, abs=0.0005)
    assert p["wall_clock_limit_seconds"] is None
    target = tmp_path / "no_output_created"
    result = run(output=target, execute_native=False, **assumptions)
    assert result["status"] == "DRY_RUN_ZERO_VENDOR_CALLS"
    assert result["maximum_original_vitis_invocations"] == 3270
    assert result["checkpoint_every_train_episodes"] == 10
    assert not target.exists()
    with pytest.raises(ValueError, match="full --max-total"):
        run(output=target, execute_native=True,
            max_total_vitis_calls=3269, **assumptions)
    assert not target.exists()



def test_paper_reward_formula_and_exact_source_pinning_metadata():
    from hashlib import sha256
    from pathlib import Path
    from rl_environment.research_metadata import experiment_metadata, summarize_trial
    from rl_environment.reward import latency_only_reward
    root = Path(__file__).resolve().parents[1]
    meta = experiment_metadata(source_revision="a" * 40,
                               use_message_passing=True)
    assert meta["exact_git_commit"] == "a" * 40
    assert meta["optimizer"]["learning_rate"] == .0005
    assert meta["optimizer"]["discount_gamma"] == .95
    assert meta["optimizer"]["max_gradient_norm"] == 5.
    assert meta["reward_definition"]["evaluation_penalty"] == .005
    assert meta["reward_definition"]["infeasible_reward"] == -.05
    assert meta["reward_definition"]["first_qualified_reward"] == -.005
    assert meta["reward_definition"]["no_reward_for_area_or_power"] is True
    assert latency_only_reward(1000., 800., qualified=True) == pytest.approx(
        np.log(1000. / 800.) - .005)
    for filename, digest in meta["source_file_sha256"].items():
        assert digest == sha256((root / filename).read_bytes()).hexdigest()
    task = make_cohort(training=1, validation=1, test=1,
                       clock_ns=None)["train"][0]
    result = {
        "hls_worst_case_cycles": 40,
        "hls_estimated_latency_ns": 125.64,
        "csim_max_absolute_error": .001,
        "hls_estimated_resources": {"lut": 20, "ff": 41, "dsp": 2,
                                    "bram_18k": 0, "uram": 0},
        "hls_report_sha256": "a" * 64,
        "vitis_invocation_sha256": "b" * 64,
        "requested_clock_ns": 3.141,
        "hls_estimated_clock_ns": 2.6,
        "hls_effective_scheduling_budget_ns": 2.85,
        "hls_scheduling_margin_ns": .25,
        "hls_timing_feasible": True,
        "qualified": True,
        "qualification_gates_failed": [],
        "original_hls_timing_parse_issue": None,
        "source_and_original_native_evidence_valid": True,
        "csim_passed": True,
        "raw_hls_worst_case_cycles": 40,
        "reason": "NUMERIC_RESOURCE_HLS_TIMING_ACCEPTED",
        "vendor_seconds": 13.2,
    }
    from fpgai_research.experiments.architecture_knobs_v2 import build_catalog
    native = {"candidate": [list(x) for x in build_catalog(task.model).baseline],
              "requested_clock_ns": 3.141, "result": result}
    row = summarize_trial(native, task=task, training_metrics=None)
    assert row["requested_clock_ns"] == 3.141
    assert row["csim_max_absolute_error"] == .001
    assert row["estimated_resources"]["lut"] == 20
    assert row["architecture_compiler_knob_values"]["op_0"]["precision_bits"] == 16
    assert row["network_operator_shapes"][0]["kind"] == "MatMul"
    assert row["learning_metrics_recorded"] is False


def test_online_policy_gradient_diagnostics_are_separate_from_reward(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    from rl_environment.gym_env import GymGraphEnv
    from rl_environment.gym_agent import OnlineGraphTrainer
    task = make_cohort(training=1, validation=1, test=1,
                       clock_ns=None)["train"][0]
    obs, _ = GymGraphEnv(task, tmp_path / "metrics").reset(seed=7)
    agent = OnlineGraphTrainer(seed=11)
    _action, tape = agent.choose(obs, training=True)
    metrics = agent.update(tape, -.05, terminated=True, return_metrics=True)
    assert metrics["schema"] == "fpgai-gym-policy-update-diagnostics/v1"
    assert metrics["training_updates"] == 1
    for key in (
        "loss_total", "loss_actor", "loss_critic",
        "loss_entropy_term", "policy_entropy", "log_probability",
        "value_prediction", "discounted_td_target", "advantage",
        "gradient_l2_before_clip",
    ):
        assert np.isfinite(metrics[key])
    assert metrics["loss_total"] == pytest.approx(
        metrics["loss_actor"] + metrics["loss_critic"] +
        metrics["loss_entropy_term"], abs=1e-6)
    assert metrics["advantage"] == pytest.approx(
        metrics["discounted_td_target"] - metrics["value_prediction"],
        abs=1e-6)
    _action2, frozen_tape = agent.choose(obs, training=False)
    assert frozen_tape is None
    assert agent.updates == 1


def test_paper_evidence_export_is_create_only_source_bound_and_unverified_until_audit(
        tmp_path, monkeypatch):
    import csv
    import json
    from rl_environment import paper_evidence
    study = tmp_path / "original_incomplete_study"
    native_episode = study / "seed_101" / "train" / "train_00000" / "episode_000000"
    native_episode.mkdir(parents=True)
    # A second native case was staged but interrupted, with no completed step.
    (native_episode / "case_0001").mkdir()
    (native_episode / "case_0001" / "action.json").write_text("{}")
    (study / "campaign.json").write_text(json.dumps({
        "schema": "fpgai-gym-native-plan/v1",
        "source_revision": "a" * 40,
    }))
    record = {
        "task_id": "train_00000", "task_sha": "b" * 64,
        "private_episode_path": str(native_episode),
        "method": "online_train", "native_call_index": 0,
        "clock_ns": 3.141, "qualified": False,
        "hls_estimated_latency_ns": None, "hls_cycles": None,
        "best_feasible_latency_ns": None,
        "reward": -.05, "cumulative_episode_reward": -.05,
        "wall_seconds": 11.7, "hls_report_sha256": None,
        "vitis_invocation_sha256": None,
        "training_metrics": None,
        "trial_evidence": {"csim_max_absolute_error": None,
                           "architecture_choices": [[0]],
                           "estimated_resources": {"lut": None},
                           "qualification_reason": "FAILED_OR_MISSING_CSIM_RESOURCE_HLS_TIMING_EVIDENCE"},
    }
    (study / "native_steps.jsonl").write_text(
        json.dumps(record) + "\n")
    monkeypatch.setattr(paper_evidence, "_git_snapshot",
                        lambda rev, path: b"source-pinned test fixture only")
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="unfinished"):
        paper_evidence.export(study, output, allow_incomplete=False)
    assert not output.exists()
    result = paper_evidence.export(study, output,
                                   allow_incomplete=True, plots=False)
    assert result["status"] == "PROVISIONAL_INCOMPLETE_NOT_INDEPENDENTLY_AUDITED"
    assert result["full_independent_audit_receipt"] is None
    assert result["publication_claim_ready"] is False
    assert result["trial_rows"] == 1
    assert result["episode_rows"] == 1
    assert result["observed_original_case_directories"] == 1
    assert result["native_cases_without_completed_step_journal"] == 1
    with (output / "native_attempt_inventory.csv").open() as stream:
        attempts = list(csv.DictReader(stream))
    assert attempts[0]["has_completed_step_journal"] == "False"
    assert attempts[0]["has_staged_action"] == "True"
    with (output / "native_trial_table.csv").open() as stream:
        values = list(csv.DictReader(stream))
    assert len(values) == 1
    assert values[0]["reward"] == "-0.05"
    assert values[0]["hls_estimated_latency_ns"] == ""
    assert values[0]["requested_clock_ns"] == "3.141"
    with pytest.raises(FileExistsError, match="create-only"):
        paper_evidence.export(study, output, allow_incomplete=True,
                              plots=False)


def test_paper_evidence_does_not_merge_same_task_id_across_seeds(
        tmp_path, monkeypatch):
    import csv
    import json
    from rl_environment import paper_evidence
    root = tmp_path / "cross_seed"
    root.mkdir()
    (root / "campaign.json").write_text(json.dumps({
        "schema": "fpgai-gym-native-plan/v1",
        "source_revision": "a"*40,
    }))
    records = []
    episodes = []
    for seed in (101, 102):
        path = root / f"seed_{seed}" / "train" / "train_00000" / "episode_000000"
        path.mkdir(parents=True)
        records.append({
            "task_id": "train_00000", "task_sha": "b"*64,
            "private_episode_path": str(path),
            "method": "online_train", "native_call_index": 0,
            "clock_ns": 4.125, "qualified": True,
            "hls_estimated_latency_ns": float(seed),
            "hls_cycles": 10, "best_feasible_latency_ns": float(seed),
            "reward": -.005, "cumulative_episode_reward": -.005,
            "wall_seconds": 2.5, "hls_report_sha256": "a"*64,
            "vitis_invocation_sha256": "b"*64,
            "training_metrics": None,
        })
        episodes.append({
            "task_id": "train_00000", "task_sha": "b"*64,
            "method": "online_train", "private_episode_path": str(path),
            "original_vitis_calls": 1, "episode_return": -.005,
            "best_feasible_latency_ns": float(seed),
        })
    (root / "native_steps.jsonl").write_text(
        "".join(json.dumps(x)+"\n" for x in records))
    (root / "completed_episodes.jsonl").write_text(
        "".join(json.dumps(x)+"\n" for x in episodes))
    monkeypatch.setattr(paper_evidence, "_git_snapshot",
                        lambda revision, path: b"fixture code only")
    result = paper_evidence.export(root, tmp_path / "export",
                                   allow_incomplete=True, plots=False)
    assert result["episode_rows"] == 2
    with (tmp_path / "export" / "episode_learning_curve.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row["seed"]) for row in rows} == {101, 102}
    assert all(row["complete_in_journal"] == "True" for row in rows)



def test_revised_gym_history_and_global_initial_actions_are_real_source_only(tmp_path):
    pytest.importorskip("gymnasium")
    from rl_environment.gym_env import (
        GymGraphEnv, MAX_HISTORY, HISTORY_FEATURES)
    task = make_cohort(training=1, validation=1, test=1,
                       clock_ns=None, max_vitis_calls=10)["train"][0]
    env = GymGraphEnv(task, tmp_path / "history")
    obs, _ = env.reset(seed=57)
    assert env.observation_space.contains(obs)
    assert obs["history"].shape == (MAX_HISTORY, HISTORY_FEATURES)
    assert obs["history_mask"].sum() == 0
    assert len(env.offers) >= 2  # no longer baseline-only at step one
    baseline = env.backend.catalog.baseline
    assert baseline in env.offers
    assert any(candidate != baseline for candidate in env.offers)
    # Production backend re-evaluates the legal set at its larger authorization
    # bound: every UI-offered legal architecture must remain allowed.
    native_authorized = env.backend.candidates(max_candidates=128)
    assert native_authorized[:12] == env.backend.candidates(max_candidates=12)
    assert all(candidate in native_authorized for candidate in env.offers)
    assert env.backend.calls == 0
    # Never invent Vitis calls; an inspected historical row must be real.
    fake = dict(
        qualified=False, csim_max_absolute_error=.09,
        hls_estimated_resources={"lut": 8, "ff": 11, "dsp": 2,
                                 "bram_18k": 1, "uram": 0},
        hls_estimated_clock_ns=None, hls_estimated_latency_ns=None,
        evidence="E3-HLS-synthesis-report",
    )
    env.backend.history.append((baseline, 9.127, fake))
    updated = env._observation()
    assert env.observation_space.contains(updated)
    assert updated["history_mask"].sum() == 1
    assert updated["history"][0, 4*8] == pytest.approx(9.127 / 32.)
    assert updated["history"][0, 4*8+1] == 0
    assert updated["history"][0, 4*8+3] == 1


def test_frozen_stochastic_uses_policy_sampling_without_backprop(tmp_path):
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch")
    import torch
    from rl_environment.gym_env import GymGraphEnv
    from rl_environment.gym_agent import OnlineGraphTrainer
    task = make_cohort(training=1, validation=1, test=1,
                       clock_ns=None)["train"][0]
    obs, _ = GymGraphEnv(task, tmp_path / "frozen").reset(seed=8)
    policy = OnlineGraphTrainer(seed=31)
    torch.manual_seed(88)
    sampled, frozen_tape = policy.choose(obs, training=False, sample=True)
    torch.manual_seed(88)
    matched, train_tape = policy.choose(obs, training=True, sample=True)
    assert sampled == matched
    assert frozen_tape is None
    assert train_tape is not None
    assert policy.updates == 0


def test_real_hls_feasibility_auxiliary_receives_both_labels(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    from rl_environment.gym_env import GymGraphEnv
    from rl_environment.gym_agent import OnlineGraphTrainer
    task = make_cohort(training=1, validation=1, test=1,
                       clock_ns=None)["train"][0]
    obs, _ = GymGraphEnv(task, tmp_path / "aux").reset(seed=11)
    agent = OnlineGraphTrainer(seed=87)
    for i, qualified in enumerate((False, True)):
        action, tape = agent.choose(obs, training=True)
        assert obs["action_mask"][action[0]]
        metrics = agent.update(tape, -.05, terminated=True,
                               qualified=qualified, return_metrics=True)
        assert metrics["native_feasibility_label"] is qualified
        p = metrics["feasibility_prediction_before_hls"]
        assert 0 < p < 1
        assert metrics["feasibility_brier_score"] == pytest.approx(
            (p - float(qualified)) ** 2)
        assert metrics["loss_auxiliary_feasibility"] > 0
        assert metrics["loss_total"] == pytest.approx(
            metrics["loss_actor"] + metrics["loss_critic"] +
            metrics["loss_entropy_term"] +
            .2 * metrics["loss_auxiliary_feasibility"], abs=1e-5)
        assert metrics["training_updates"] == i + 1


def test_100_train_ten_native_calls_is_1090_cap_in_revised_protocol(tmp_path):
    from rl_environment.gym_study import plan
    p = plan(training=100, validation=1, test=1, calls_per_network=10)
    assert p["total_episodes_planned"] == 109
    assert p["maximum_original_vitis_invocations"] == 1090
    assert p["episode_counts_by_method"]["test_frozen_stochastic"] == 1
    assert p["calls_per_episode_max"] == 10


def test_baselines_differ_after_all_infeasible_feedback():
    from rl_environment.baselines import BaselineChooser
    features = np.zeros((6, 32), dtype=np.float32)
    for i in range(6):
        features[i, 0] = i / 5.
    obs = {"action_mask": np.ones(6, dtype=np.int8),
           "action_features": features,
           "clock_bounds_ns": np.array([2., 32.])}
    preferences = {}
    for method in ("greedy", "evolutionary", "parzen"):
        chooser = BaselineChooser(method, seed=11)
        chooser.observe(features[0], None, clock_ns=10.)
        chooser.observe(features[2], None, clock_ns=10.)
        slot = chooser.choose(obs)
        assert 0 <= slot < 6
        preferences[method] = slot
    # Distinct rules rather than silently identical no-incumbent fallback.
    assert len(set(preferences.values())) >= 2


def test_qualification_funnel_reports_no_vendor_calls_for_empty_case_journal(
        tmp_path):
    import json
    from rl_environment.qualification_funnel import diagnose
    study = tmp_path / "empty_original"
    study.mkdir()
    (study / "campaign.json").write_text(json.dumps({
        "schema": "fpgai-gym-native-plan/v1",
        "source_revision": "a" * 40,
        "training_seeds": [101],
        "training_networks_per_seed": 1,
        "validation_networks_per_seed": 1,
        "test_networks_per_seed": 1,
        "calls_per_episode_max": 10,
        "fixed_clock_ablation_ns": None,
    }))
    (study / "native_steps.jsonl").write_text("")
    result = diagnose(study)
    assert result["vendor_invocations"] == 0
    assert result["counts"] == {}
    assert result["source_study_unchanged"] is True



def test_qualification_funnel_counts_exact_original_case_rejection_gates(
        tmp_path, monkeypatch):
    import json
    from rl_environment import qualification_funnel
    seed = 101
    task = make_cohort(training=1, validation=1, test=1,
                       seed=seed, max_vitis_calls=10, clock_ns=None)["train"][0]
    study = tmp_path / "old_native_campaign"
    case = (study / "seed_101" / "train" / task.task_id /
            "episode_000000" / "case_0000")
    case.mkdir(parents=True)
    (study / "campaign.json").write_text(json.dumps({
        "schema": "fpgai-gym-native-plan/v1",
        "source_revision": "a" * 40,
        "training_seeds": [seed],
        "training_networks_per_seed": 1,
        "validation_networks_per_seed": 1,
        "test_networks_per_seed": 1,
        "calls_per_episode_max": 10,
        "fixed_clock_ablation_ns": None,
    }))
    (study / "native_steps.jsonl").write_text(json.dumps({
        "private_episode_path": str(case.parent),
        "task_id": task.task_id, "task_sha": task.task_sha,
        "native_call_index": 0, "qualified": False,
    }) + "\n")
    seen = []
    def original_native_receipt(candidate, loaded_task):
        seen.append((candidate, loaded_task.task_sha))
        return {
            "qualified": False,
            "qualification_gates_failed": [
                "HLS_WORST_CYCLES_MISSING_OR_NONINTEGER",
                "CSIM_NOT_VERIFIED_OR_FAILED",
            ],
            "original_hls_timing_parse_issue": None,
            "raw_hls_worst_case_cycles": None,
            "csim_passed": False,
        }
    monkeypatch.setattr(qualification_funnel, "_receipt",
                        original_native_receipt)
    result = qualification_funnel.diagnose(study)
    assert len(seen) == 1
    assert seen[0][0] == case
    assert seen[0][1] == task.task_sha
    assert result["counts"]["original_cases"] == 1
    assert result["counts"]["infeasible_or_unqualified"] == 1
    assert result["counts"]["missing_original_worst_cycles"] == 1
    assert result["counts"]["failure/CSIM_NOT_VERIFIED_OR_FAILED"] == 1
    assert result["counts"]["failure/HLS_WORST_CYCLES_MISSING_OR_NONINTEGER"] == 1
    assert result["vendor_invocations"] == 0


def test_frozen_duplicate_clock_is_resolved_before_native_call():
    from types import SimpleNamespace
    import numpy as np
    from rl_environment.gym_study import _choose_unused_hls_action
    arch = ((2, 0, 0, 0, 0, 0, 0, 0),)
    env = SimpleNamespace(
        offers=(arch,),
        backend=SimpleNamespace(tried_configs={(arch, 12.500)}))
    observation = {"clock_bounds_ns": np.asarray([2., 32.], np.float32)}
    selected, tape = _choose_unused_hls_action(
        agent=None, obs=observation, env=env, method="frozen",
        action=(0, 12.500), tape=None)
    assert selected == (0, 12.501)
    assert tape is None
    assert (arch, selected[1]) not in env.backend.tried_configs
    # This native-free uniqueness convention must not silently change
    # an already fresh chosen original clock.
    unchanged, tape2 = _choose_unused_hls_action(
        agent=None, obs=observation, env=env, method="frozen",
        action=(0, 15.247), tape=None)
    assert unchanged == (0, 15.247) and tape2 is None


def test_stochastic_duplicate_resamples_and_preserves_selected_update_tape():
    from types import SimpleNamespace
    import numpy as np
    from rl_environment.gym_study import _choose_unused_hls_action
    arch = ((2, 0, 0, 0, 0, 0, 0, 0),)
    class Agent:
        def __init__(self):
            self.n = 0
        def choose(self, obs, *, training, sample):
            self.n += 1
            assert training and sample
            return (0, 7.200 + self.n / 1000.), ("actual_tape", self.n)
    agent = Agent()
    env = SimpleNamespace(
        offers=(arch,),
        backend=SimpleNamespace(tried_configs={
            (arch, 7.200), (arch, 7.201)}))
    selected, tape = _choose_unused_hls_action(
        agent=agent,
        obs={"clock_bounds_ns": np.asarray([2., 32.], np.float32)},
        env=env, method="online_train",
        action=(0, 7.200), tape=("old_tape", 0))
    assert selected == (0, 7.202)
    assert tape == ("actual_tape", 2)
    assert agent.n == 2


def test_duplicated_hls_clock_detection_uses_backend_canonical_1ps_key():
    from types import SimpleNamespace
    import numpy as np
    from rl_environment.gym_study import _choose_unused_hls_action
    arch = ((2, 0, 0, 0, 0, 0, 0, 0),)
    env = SimpleNamespace(
        offers=(arch,),
        backend=SimpleNamespace(tried_configs={(arch, 7.201)}))
    obs = {"clock_bounds_ns": np.asarray([2., 32.], dtype=np.float32)}
    # This is precisely the binary-float representation that caused the
    # previous resampling unit test to misclassify an already tried design.
    selected, tape = _choose_unused_hls_action(
        agent=None, obs=obs, env=env, method="frozen",
        action=(0, 7.2010000000000005), tape=None)
    assert selected == (0, 7.202)
    assert tape is None

    # Native backend must only receive canonical keys even when an agent
    # returns a slightly inexact but otherwise NEW HLS clock proposal.
    selected2, tape2 = _choose_unused_hls_action(
        agent=None, obs=obs, env=env, method="frozen",
        action=(0, 8.501000000000001), tape=None)
    assert selected2 == (0, 8.501)
    assert tape2 is None


def test_easier_fixed_24ns_architecture_only_100x10_protocol_is_explicit(tmp_path):
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch")
    import numpy as np
    from rl_environment.gym_study import plan
    from rl_environment.cohort import make_cohort
    from rl_environment.gym_env import GymGraphEnv
    from rl_environment.gym_agent import OnlineGraphTrainer
    from rl_environment.clock_actions import validate_clock_action
    p = plan(training=100, validation=1, test=1,
             calls_per_network=10, clocks_ns=(24.,), seeds=(111,))
    assert p["clock_ablation_mode"] == "fixed_clock"
    assert p["fixed_clock_ablation_ns"] == 24.
    assert p["maximum_original_vitis_invocations"] == 1090
    assert p["total_episodes_planned"] == 109
    task = make_cohort(training=1, validation=1, test=1,
                       seed=111, max_vitis_calls=10,
                       clock_ns=None, fixed_clock_ablation_ns=24.)["train"][0]
    env = GymGraphEnv(task, tmp_path / "fixed_24_arch_only")
    obs, info = env.reset(seed=111)
    assert env.observation_space.contains(obs)
    assert tuple(obs["clock_bounds_ns"]) == (24., 24.)
    assert len(env.offers) > 1
    agent = OnlineGraphTrainer(seed=111)
    (slot, period), tape = agent.choose(obs, training=True)
    assert period == 24.
    assert obs["action_mask"][slot] == 1
    assert tape is not None
    assert np.isfinite(agent.update(tape, -.05, terminated=True))
    assert validate_clock_action(task, 24.) == 24.
    with pytest.raises(ValueError, match="fixed-clock"):
        validate_clock_action(task, 24.001)
    # Source plan/reset/policy has not yet invoked any original HLS tool.
    assert env.backend.calls == 0



def test_fixed32_frozen_numeric_format_architecture_only_no_vitis(tmp_path):
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch")
    from rl_environment.gym_study import plan
    from rl_environment.cohort import make_cohort
    from rl_environment.gym_env import GymGraphEnv
    from rl_environment.gym_agent import OnlineGraphTrainer
    from rl_environment.clock_actions import validate_clock_action
    p = plan(training=100, validation=1, test=1, calls_per_network=10,
             clocks_ns=(32.,), seeds=(111,), freeze_numeric_format=True)
    assert p["fixed_clock_ablation_ns"] == 32.
    assert p["freeze_numeric_format"] is True
    assert p["fixed_numeric_precision_bits"] == 16
    assert p["fixed_numeric_integer_bits"] == 6
    assert p["total_episodes_planned"] == 109
    assert p["maximum_original_vitis_invocations"] == 1090
    frozen = make_cohort(training=2, validation=1, test=1, seed=111,
                         clock_ns=None, fixed_clock_ablation_ns=32.,
                         freeze_numeric_format=True, max_vitis_calls=10)
    default = make_cohort(training=2, validation=1, test=1, seed=111,
                          clock_ns=None, fixed_clock_ablation_ns=32.,
                          freeze_numeric_format=False, max_vitis_calls=10)
    task = frozen["train"][0]
    assert task.task_sha != default["train"][0].task_sha
    assert task.model_sha == default["train"][0].model_sha
    assert task.public_descriptor()["fixed_numeric_precision_bits"] == 16
    assert task.public_descriptor()["fixed_numeric_integer_bits"] == 6
    env = GymGraphEnv(task, tmp_path / "numeric_fixed_32ns")
    obs, _ = env.reset(seed=111)
    catalog = env.backend.catalog
    assert tuple(obs["clock_bounds_ns"]) == (32., 32.)
    assert len(env.offers) >= 2
    for candidate in env.backend.candidates(max_candidates=128):
        assert all(row[:2] == catalog.baseline[i][:2]
                   for i, row in enumerate(candidate))
        values = catalog.values(candidate)
        assert values[catalog.names[0]]["precision_bits"] == 16
        assert values[catalog.names[0]]["integer_bits"] == 6
    (slot, clock), tape = OnlineGraphTrainer(seed=111).choose(obs, training=True)
    assert obs["action_mask"][slot] and clock == 32.
    assert tape is not None
    assert validate_clock_action(task, 32.) == 32.
    assert env.backend.calls == 0


def test_numeric_format_lock_rejects_changed_precision_without_hls(tmp_path):
    from rl_environment.cohort import make_cohort
    from rl_environment.post_hls_env import PostHLSEnv
    task = make_cohort(training=1, validation=1, test=1, seed=222,
                       max_vitis_calls=10, clock_ns=None,
                       fixed_clock_ablation_ns=32.,
                       freeze_numeric_format=True)["train"][0]
    backend = PostHLSEnv(task, tmp_path / "format_lock")
    baseline = backend.catalog.baseline
    changed = backend.catalog.mutate(
        baseline, (None, "precision_bits", +1))
    assert changed is not None
    assert all(row[0] != baseline[i][0]
               for i, row in enumerate(changed))
    assert changed not in backend.candidates(max_candidates=128)
    with pytest.raises(ValueError):
        backend.step(changed, clock_ns=32., execute_native=False)
    assert backend.calls == 0
    assert not (backend.root / "case_0000").exists()
