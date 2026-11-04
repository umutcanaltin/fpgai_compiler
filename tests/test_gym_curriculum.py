"""Native-free L0-L6 original Vitis Gym curriculum gates and contracts.

All metrics below are CPU fixtures. No test invokes Vitis/Vivado or fabricates
qualifying source evidence.
"""
import pytest
from rl_environment.curriculum import (
    STAGES, FOUR_KNOBS, SIX_KNOBS, ALL_KNOBS, get_stage, stage_contract)
from rl_environment.gym_curriculum import config, plans
from rl_environment.cohort import make_cohort
from rl_environment.post_hls_env import PostHLSEnv
from rl_environment.reward import curriculum_real_hls_cycle_reward


def _taskset(name, seed=111):
    stage=get_stage(name)
    return make_cohort(
        training=4, validation=2, test=2, seed=seed,
        max_vitis_calls=10, clock_ns=None,
        fixed_clock_ablation_ns=stage.fixed_clock_ns,
        freeze_numeric_format=stage.freeze_numeric_format,
        beginner_pilot=name=="L0", curriculum_stage=name)


def test_all_seven_real_hls_stages_have_unique_scientific_contracts():
    assert tuple(s.name for s in STAGES) == tuple(f"L{i}" for i in range(7))
    assert STAGES[0].knob_fields == FOUR_KNOBS
    assert STAGES[2].knob_fields == FOUR_KNOBS
    assert STAGES[3].knob_fields == SIX_KNOBS
    assert STAGES[4].knob_fields == ALL_KNOBS
    assert STAGES[5].fixed_clock_ns is None
    assert STAGES[6].test_depths == (4,)
    for stage in STAGES:
        contract=stage_contract(stage)
        assert contract["stage"]==stage.name
        assert contract["pretraining_between_stages"] is False
        assert contract["uses_original_vitis_hls_only"] is True
        assert contract["strict_numerical_resource_timing_source_gates"] is True
    with pytest.raises(ValueError, match="Unknown"):
        get_stage("L7")


def test_separate_100_by_10_stage_plans_require_no_vendor_calls():
    result=plans()
    assert len(result)==7
    for row in result:
        p=row["research_plan"]
        assert p["total_episodes_planned"]==109
        assert p["maximum_original_vitis_invocations"]==1090
        assert row["native_vendor_calls_executed"]==0
        assert p["curriculum_stage"]==row["stage"]["stage"]
        assert p["curriculum_contract"]==row["stage"]
    assert config("L0")["beginner_pilot"] is True
    assert config("L6")["clocks_ns"] is None
    assert config("L3")["freeze_numeric_format"] is True


@pytest.mark.parametrize("stage_name", [s.name for s in STAGES])
def test_real_functional_model_cohorts_obey_stage_topology(stage_name):
    stage=get_stage(stage_name)
    cohorts=_taskset(stage_name)
    for split, depths in (
        ("train",stage.train_depths),("validation",stage.val_depths),
        ("test",stage.test_depths)):
        assert all(len(t.model.operators) in depths
                   and t.curriculum_stage==stage_name
                   and t.fixed_clock_ablation_ns==stage.fixed_clock_ns
                   for t in cohorts[split])
    assert len({t.model_sha for rows in cohorts.values()
                for t in rows})==8
    assert all(op.kind in ("MatMul","Gemm")
               for rows in cohorts.values() for t in rows
               for op in t.model.operators)
    if stage_name in ("L0","L1"):
        assert all(len(t.model.operators)==1
                   for rows in cohorts.values() for t in rows)
    if stage_name=="L6":
        assert len(cohorts["test"][0].model.operators)==4


@pytest.mark.parametrize("stage_name", [s.name for s in STAGES])
def test_real_backend_stage_actions_are_restricted_to_allowed_knobs_without_native(
        tmp_path, stage_name):
    from fpgai_research.experiments.architecture_knobs_v2 import FIELDS
    stage=get_stage(stage_name)
    task=_taskset(stage_name)["train"][0]
    env=PostHLSEnv(task,tmp_path / stage_name)
    choices=env.candidates(max_candidates=32)
    assert choices
    assert env.calls==0
    for candidate in choices:
        assert all(
            all(row[j]==env.catalog.baseline[i][j]
                for j,field in enumerate(FIELDS)
                if field not in stage.knob_fields)
            for i,row in enumerate(candidate))
    forbidden=[field for field in FIELDS if field not in stage.knob_fields]
    if forbidden:
        field=forbidden[0]
        target=None
        for action in env.catalog.actions:
            if action[1]==field:
                modified=env.catalog.mutate(env.catalog.baseline,action)
                if modified is not None:
                    target=modified
                    break
        assert target is not None
        with pytest.raises(ValueError):
            env.step(target,
                     clock_ns=stage.fixed_clock_ns or 10.,
                     execute_native=False)
        assert env.calls==0
    assert not (env.root/"case_0000").exists()


def test_stage_dense_reward_uses_real_clock_dependent_ns_and_never_proxy():
    t=_taskset("L0")["train"][0]
    r1=curriculum_real_hls_cycle_reward(
        qualified=True,hls_cycles=40,requested_clock_ns=32.,model=t.model)
    r2=curriculum_real_hls_cycle_reward(
        qualified=True,hls_cycles=80,requested_clock_ns=32.,model=t.model)
    r3=curriculum_real_hls_cycle_reward(
        qualified=True,hls_cycles=40,requested_clock_ns=16.,model=t.model)
    assert r1>r2>0 and r3>r1
    assert curriculum_real_hls_cycle_reward(
        qualified=False,hls_cycles=None,requested_clock_ns=32.,model=t.model)==-.05
    with pytest.raises(ValueError,match="true bounded"):
        curriculum_real_hls_cycle_reward(
            qualified=True,hls_cycles=None,requested_clock_ns=32.,model=t.model)


@pytest.mark.parametrize("stage_name", [s.name for s in STAGES])
def test_each_full_stage_requires_at_least_100_training_episodes_and_ten_hls_calls(stage_name):
    from rl_environment.gym_study import plan
    kw=config(stage_name,train=100,calls=10)
    p=plan(**kw)
    assert p["training_episode_minimum"]==100
    assert p["required_original_train_calls_per_episode"]==10
    assert p["curriculum_mode"]=="full"
    assert p["clock_report_resolution_ns"]==.01
    for count in (1,2,50,99):
        with pytest.raises(ValueError,match=">=100"):
            plan(**config(stage_name,train=count,calls=10))
    for calls in (1,2,9,11):
        with pytest.raises(ValueError,match="exactly 10"):
            plan(**config(stage_name,train=100,calls=calls))
    assert plan(**config(stage_name,train=101,calls=10))["training_episode_minimum"]==100


@pytest.mark.parametrize("stage_name", [s.name for s in STAGES])
def test_short_stage_smoke_is_explicit_separate_not_learning(stage_name):
    from rl_environment.gym_study import plan
    smoke=plan(**config(stage_name,train=2,calls=2),
               curriculum_mode="smoke")
    assert smoke["curriculum_mode"]=="smoke"
    assert smoke["training_episode_minimum"] is None
    assert smoke["maximum_original_vitis_invocations"]==22
    with pytest.raises(ValueError,match="Curriculum smoke"):
        plan(**config(stage_name,train=99,calls=2),
             curriculum_mode="smoke")


def test_original_legacy_millips_clock_remains_but_curriculum_clock_representable():
    from rl_environment.clock_actions import (
        normalize_clock_period_ns,validate_clock_action)
    old=_taskset("L5")["train"][0]
    from dataclasses import replace
    assert normalize_clock_period_ns(17.677)==17.677
    assert normalize_clock_period_ns(17.677,decimals=2)==17.68
    assert validate_clock_action(old,17.68)==17.68
    with pytest.raises(ValueError,match="0.010"):
        validate_clock_action(old,17.677)
    legacy=replace(old,curriculum_stage=None)
    assert validate_clock_action(legacy,17.677)==17.677


def test_stage_scorecard_rejects_smoke_and_incomplete_even_before_native_audit(tmp_path):
    import json
    from rl_environment.gym_curriculum_report import scorecard,export
    root=tmp_path/"smoke"
    root.mkdir()
    (root/"campaign.json").write_text(json.dumps({
        "curriculum_stage":"L0","curriculum_mode":"smoke",
        "training_networks_per_seed":2,"calls_per_episode_max":2}))
    with pytest.raises(ValueError,match="Only independently audited"):
        scorecard(root)
    with pytest.raises(ValueError,match="requires exactly L0..L6"):
        export([("L0",root)],tmp_path/"report",require_all_seven=True)
    assert not (tmp_path/"report").exists()



def test_new_continuous_clock_stage_resolves_original_xml_precision_before_duplicate_native_call():
    from types import SimpleNamespace
    import numpy as np
    from rl_environment.gym_study import _choose_unused_hls_action
    arch=((2,0,0,0,0,0,0,0),)
    # Original Vitis 2023.2 XML only recorded 2 decimals, so do not issue
    # 17.677 as a new action after 17.68 has already been synthesized.
    env=SimpleNamespace(
        task=SimpleNamespace(curriculum_stage="L5"),
        offers=(arch,),
        backend=SimpleNamespace(tried_configs={(arch,17.68)}))
    action,tape=_choose_unused_hls_action(
        agent=None,
        obs={"clock_bounds_ns":np.asarray([2.,32.],dtype=np.float32)},
        env=env,method="frozen",action=(0,17.677),tape=None)
    assert action==(0,17.69)
    assert tape is None
    other=SimpleNamespace(
        task=SimpleNamespace(curriculum_stage=None),
        offers=(arch,),
        backend=SimpleNamespace(tried_configs={(arch,17.68)}))
    original,tape=_choose_unused_hls_action(
        agent=None,
        obs={"clock_bounds_ns":np.asarray([2.,32.],dtype=np.float32)},
        env=other,method="frozen",action=(0,17.677),tape=None)
    assert original==(0,17.677)



def test_staged_2023_2_source_clock_and_uncertainty_round_to_native_xml_precision(tmp_path):
    from rl_environment.timing import parse_hls_timing
    from fpgai_research.toolchains.vitis_hls.package import make_vitis_tcl
    target=17.68
    uncertainty=round(target*.27,2)
    assert uncertainty==4.77
    script=make_vitis_tcl(
        clock_ns=target,clock_uncertainty_ns=uncertainty)
    assert "create_clock -period 17.68 -name default" in script
    assert "set_clock_uncertainty 4.770" in script
    xml=tmp_path/"original_vitis_clock.xml"
    xml.write_text(
        "<Profile><UserAssignments>"
        "<TargetClockPeriod>17.68</TargetClockPeriod>"
        "<ClockUncertainty>4.77</ClockUncertainty>"
        "</UserAssignments><PerformanceEstimates>"
        "<SummaryOfTimingAnalysis><EstimatedClockPeriod>11.517</EstimatedClockPeriod>"
        "</SummaryOfTimingAnalysis></PerformanceEstimates></Profile>")
    timing=parse_hls_timing(
        xml,target,source_attested_uncertainty_ns=uncertainty)
    assert timing["hls_timing_feasible"]
    assert timing["hls_clock_uncertainty_ns"]==4.77
    with pytest.raises(ValueError,match="different requested clock"):
        parse_hls_timing(xml,17.677,
                         source_attested_uncertainty_ns=uncertainty)
