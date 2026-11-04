"""CPU-only guarantees; NO real vendor tools executed by this test module."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rl_environment.cohort import make_cohort
from rl_environment.tasks import Task
from rl_environment.post_hls_env import PostHLSEnv
from rl_environment.study import plan, run


def tiny():
    return make_cohort(training=2,validation=1,test=1,
                       seed=123,max_vitis_calls=2)


def test_100_network_cohort_uses_real_ir_and_heldout_topologies():
    tasks=make_cohort(training=100,validation=2,test=4,
                      seed=123,max_vitis_calls=3)
    assert len(tasks["train"])==100
    assert [len(tasks[x]) for x in ("validation","test")]==[2,4]
    assert {len(t.model.operators) for t in tasks["train"]}=={1,2}
    assert {len(t.model.operators) for t in tasks["validation"]}=={3}
    assert {len(t.model.operators) for t in tasks["test"]}=={4}
    assert len({t.model_sha for group in tasks.values() for t in group})==106
    other=make_cohort(training=100,validation=2,test=4,
                      seed=123,max_vitis_calls=3)
    assert [t.task_sha for t in tasks["train"]]==[
        t.task_sha for t in other["train"]]
    assert tasks["train"][0].samples.flags.writeable is False
    assert "actual_hls_metrics" in tasks["train"][0].public_descriptor()


def test_native_campaign_default_is_dry_run_and_budgeted(tmp_path):
    result=run(output=tmp_path/"must_not_exist",training=100,
               validation=2,test=4,calls_per_network=3,
               validate_every=25,execute_native=False)
    assert result["status"]=="DRY_RUN_NO_VENDOR_EXECUTION"
    assert result["maximum_original_vitis_invocations"]==372
    assert result["planned_training_calls"]==300
    assert result["planned_validation_calls"]==48
    assert result["planned_final_test_and_random_calls"]==24
    assert not (tmp_path/"must_not_exist").exists()
    with pytest.raises(ValueError,match="explicit"):
        run(output=tmp_path/"blocked",training=2,validation=1,test=1,
            calls_per_network=2,validate_every=1,
            execute_native=True,max_total_vitis_calls=1)
    assert not (tmp_path/"blocked").exists()


def test_no_implicit_native_run_no_source_only_reward(tmp_path,monkeypatch):
    from rl_environment import post_hls_env as envmod
    invoked=[]
    monkeypatch.setattr(envmod,"run_vitis",
                        lambda *a,**k: invoked.append((a,k)))
    task=tiny()["train"][0]
    env=PostHLSEnv(task,tmp_path/"episode",seed=33)
    assert env.calls==0
    assert env.best_cycles is None
    assert env.observe()["best_qualified_hls_cycles"] is None
    proposals=env.candidates()
    assert proposals==env.candidates()
    # The revised policy must be able to propose a legal nonbaseline
    # architecture on its *first* HLS attempt, without invoking Vitis.
    assert env.catalog.baseline in proposals
    assert len(proposals) >= 2
    assert any(candidate != env.catalog.baseline for candidate in proposals)
    from fpgai_research.experiments.architecture_knobs_v2 import legal_candidate
    assert all(legal_candidate(task.model, env.catalog, candidate)
               for candidate in proposals)
    status=env.step(proposals[0])
    assert status["status"]=="DRY_RUN_NO_VENDOR_EXECUTION"
    assert env.calls==0 and not invoked
    assert not (tmp_path/"episode"/"case_0000").exists()
    assert json.loads((tmp_path/"episode"/"task.json").read_text()
                      )["original_vendor_calls"]==0
    with pytest.raises(FileExistsError):
        PostHLSEnv(task,tmp_path/"episode")


def test_source_only_stage_may_be_created_only_on_explicit_native_call(
        tmp_path,monkeypatch):
    from rl_environment import post_hls_env as envmod
    # Fault injection: real tool is intentionally NOT invoked and no
    # fraudulent synthesis/CSim outcome is supplied.
    def stop_before_vitis(*args,**kwargs):
        raise RuntimeError("SENTINEL_NOT_REAL_HLS")
    monkeypatch.setattr(envmod,"run_vitis",stop_before_vitis)
    task=tiny()["train"][0]
    env=PostHLSEnv(task,tmp_path/"private",seed=123)
    baseline=env.candidates()[0]
    with pytest.raises(RuntimeError,match="SENTINEL"):
        env.step(baseline,execute_native=True)
    assert env.done
    case=tmp_path/"private"/"case_0000"
    assert (case/"model.cpp").is_file()
    assert (case/"tb.cpp").is_file()
    assert (case/"run_hls.tcl").is_file()
    assert (case/"action.json").is_file()
    assert not (case/"online_step.json").exists()
    with pytest.raises(RuntimeError,match="complete"):
        env.step(baseline,execute_native=True)


def test_invalid_tensor_shape_rejected_immutably():
    task=tiny()["train"][0]
    with pytest.raises(ValueError,match="float32"):
        Task(task.task_id,task.model,task.samples.astype(np.float64),
             task.part,task.clock_ns,task.max_absolute_error,
             task.resources,task.max_vitis_calls,"train")


def test_agent_is_shared_and_trains_only_when_called_online():
    torch=pytest.importorskip("torch")
    from rl_environment.agent import OnlineAgent
    task=tiny()["train"][0]
    from fpgai_research.experiments.architecture_knobs_v2 import build_catalog
    cat=build_catalog(task.model)
    agent=OnlineAgent(seed=19)
    candidates=[cat.baseline]
    selected,tape=agent.choose(task,cat,candidates,None,0,training=True)
    assert selected==cat.baseline
    assert agent.updates==0
    # The legacy agent receives one legal baseline in this unit fixture.
    # The newer Gym study offers global alternatives from the first step.
    # Only an explicit online update with verified feedback changes weights.
    loss=agent.update(tape,-0.05)
    assert np.isfinite(loss) and agent.updates==1
    _,tape=agent.choose(task,cat,candidates,None,0,training=False)
    assert tape is None
    assert agent.update(tape,0.0) is None
    assert agent.updates==1


def test_policy_uses_only_previous_audited_hls_features():
    torch=pytest.importorskip("torch")
    from rl_environment.agent import encode, OnlineAgent
    from fpgai_research.experiments.architecture_knobs_v2 import build_catalog
    task=tiny()["train"][0]
    catalog=build_catalog(task.model)
    before=encode(task,catalog,catalog.baseline,None,0)
    report={
        "qualified":True,"hls_worst_case_cycles":123,
        "csim_max_absolute_error":0.01,"vendor_seconds":2.4,
        "hls_estimated_resources":{
            "lut":100,"ff":230,"dsp":2,"bram_18k":0,"uram":0},
        "evidence":"E3-HLS-synthesis-report",
    }
    after=encode(task,catalog,catalog.baseline,123,1,report)
    assert before.shape==(56,)
    assert after.shape==(56,)
    assert not np.array_equal(before,after)
    agent=OnlineAgent(seed=77)
    chosen,tape=agent.choose(
        task,catalog,[catalog.baseline],None,0,training=True)
    estimate=agent.estimate_next_value(
        task,catalog,[catalog.baseline],123,1,last_result=report)
    assert np.isfinite(estimate)
    loss=agent.update(tape,0.1,next_value=estimate,terminated=False)
    assert np.isfinite(loss)
    assert agent.updates==1


def test_candidate_menu_is_reproducible_and_not_fixed_catalog_prefix(tmp_path):
    from fpgai_research.experiments.architecture_knobs_v2 import (
        FIELDS, legal_candidate,
    )
    task=tiny()["train"][0]
    e1=PostHLSEnv(task,tmp_path/"menu_a",seed=33)
    e2=PostHLSEnv(task,tmp_path/"menu_b",seed=33)
    for e in (e1,e2):
        e.current=e.catalog.baseline
        e.tried.add(e.catalog.baseline)
        e.calls=1
    choices=e1.candidates(max_candidates=12)
    assert choices==e1.candidates(max_candidates=12)
    assert choices==e2.candidates(max_candidates=12)
    assert len(choices)==12
    assert len(set(choices))==len(choices)
    assert all(legal_candidate(task.model,e1.catalog,c) for c in choices)
    # A constrained menu should expose a diversity of hardware dimensions,
    # rather than always returning only the first declared catalog axes.
    changed_axes=set()
    for candidate in choices:
        for current,proposed in zip(e1.catalog.baseline,candidate):
            for index,(a,b) in enumerate(zip(current,proposed)):
                if a!=b:
                    changed_axes.add(FIELDS[index])
    assert len(changed_axes)>=3
