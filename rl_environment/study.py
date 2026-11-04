"""Foreground native online HLS study; dry-run by default.

Commands:
  python -m rl_environment.study --train-networks 100 --plan
  python -m rl_environment.study --train-networks 2 --validation-networks 1 \
     --test-networks 1 --calls-per-network 2 --validate-every 1 \
     --max-total-vitis-calls 12 --output build/hls_online_smoke_v1 \
     --execute-native --vitis-executable /absolute/path/to/vitis_hls

No background/native work without explicit command. A campaign cannot
be resumed automatically; preserve partial folders and source identity.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import subprocess

import numpy as np

from .cohort import make_cohort
from .post_hls_env import PostHLSEnv
from .agent import OnlineAgent


def _source_revision():
    base=Path(__file__).resolve().parents[1]
    rev=subprocess.run(["git","rev-parse","HEAD"],cwd=base,
                       capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    if len(rev)!=40 or any(x not in "0123456789abcdef" for x in rev):
        raise ValueError("Invalid full source revision")
    dirty=subprocess.run(
        ["git","status","--porcelain","--untracked-files=no"],
        cwd=base,capture_output=True,text=True,check=True,timeout=15).stdout
    if dirty.strip():
        raise ValueError("Original native experiment requires a clean source-pinned worktree")
    return rev


def _write_json_new(path,doc):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(doc,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")


def _append_jsonl(path,doc):
    with Path(path).open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(doc,sort_keys=True,allow_nan=False)+"\n")


def plan(*,training,validation,test,calls_per_network,validate_every):
    if (any(type(n) is not int or not 1<=n<=10000 for n in
            (training,validation,test))
            or type(calls_per_network) is not int or
            not 1<=calls_per_network<=64
            or type(validate_every) is not int or validate_every<1):
        raise ValueError("Invalid bounded campaign size")
    rounds=(training+validate_every-1)//validate_every
    total=(training+2*validation*rounds+2*test)*calls_per_network
    return {
        "training_networks":training,
        "heldout_validation_networks":validation,
        "heldout_test_networks":test,
        "vitis_calls_per_episode_including_baseline":calls_per_network,
        "frozen_validation_rounds":rounds,
        "planned_training_calls":training*calls_per_network,
        "planned_validation_calls":2*validation*rounds*calls_per_network,
        "planned_final_test_and_random_calls":2*test*calls_per_network,
        "validation_uses_matched_random_control":True,
        "maximum_original_vitis_invocations":total,
        "native_execution_default":False,
        "post_vivado_included":False,
        "source_proxy_rewards":False,
    }


def _run_episode(root,task,agent,*,seed,training,random_search,
                 vitis_executable,timeout_seconds,callback):
    env=PostHLSEnv(task,root,seed=seed)
    rng=Random(seed+950)
    baseline_cycles=None
    while not env.done:
        candidates=env.candidates(max_candidates=12)
        if not candidates:
            break
        if env.calls==0:
            candidate=candidates[0]
            tape=None
        elif random_search:
            candidate=rng.choice(candidates)
            tape=None
        else:
            candidate,tape=agent.choose(
                task,env.catalog,candidates,env.best_cycles,
                env.calls,training=training,last_result=env.last_result)
        result=env.step(
            candidate,execute_native=True,
            vitis_executable=vitis_executable,
            timeout_seconds=timeout_seconds)
        if env.calls==1:
            baseline_cycles=result["result"]["hls_worst_case_cycles"]
        # Updating immediately AFTER the audit, BEFORE the next choice,
        # is the defining online RL mechanism.
        next_menu=env.candidates(max_candidates=12) if not env.done else []
        future_value=(
            agent.estimate_next_value(
                task,env.catalog,next_menu,env.best_cycles,env.calls,
                last_result=env.last_result)
            if training and tape is not None and next_menu else 0.0)
        loss=(
            agent.update(tape,result["reward"],
                         next_value=future_value,
                         terminated=(env.done or not next_menu))
            if training else None)
        callback({
            "task_id":task.task_id,"task_sha256":task.task_sha,
            "partition":task.partition,"method":"random" if random_search
                else "online_drl",
            "native_call":env.calls,
            "reward":result["reward"],"qualified":result["result"]["qualified"],
            "best_hls_worst_case_cycles":env.best_cycles,
            "measured_hls_cycles":result["result"]["hls_worst_case_cycles"],
            "native_vendor_seconds":result["result"]["vendor_seconds"],
            "online_gradient_loss":loss,
            "global_online_updates":agent.updates,
            "native_receipt_sha256":result["result"]["vitis_invocation_sha256"],
            "source_proxy_reward":False,
        })
    return {
        "task_id":task.task_id,
        "method":"random" if random_search else "online_drl",
        "partition":task.partition,
        "calls":env.calls,
        "best_hls_worst_case_cycles":env.best_cycles,
        "baseline_qualified":baseline_cycles is not None,
        "baseline_hls_worst_case_cycles":baseline_cycles,
        "normalized_hls_improvement":(
            (baseline_cycles-env.best_cycles)/baseline_cycles
            if baseline_cycles is not None and env.best_cycles is not None
            else None),
        "fully_budgeted":env.calls==task.max_vitis_calls,
    }


def run(*,output,training=100,validation=2,test=4,
        calls_per_network=3,validate_every=25,seed=20261030,
        vitis_executable="vitis_hls",timeout_seconds=7200,
        execute_native=False,max_total_vitis_calls=None):
    budget=plan(training=training,validation=validation,test=test,
                calls_per_network=calls_per_network,
                validate_every=validate_every)
    if not execute_native:
        return {"status":"DRY_RUN_NO_VENDOR_EXECUTION",**budget}
    if type(max_total_vitis_calls) is not int or (
            max_total_vitis_calls<budget["maximum_original_vitis_invocations"]):
        raise ValueError(
            "Set explicit --max-total-vitis-calls to the full planned "
            "training + validation + random-test budget")
    raise RuntimeError("Legacy cycles-only native campaign retired; use rl_environment.gym_study")
    if type(timeout_seconds) is not int or not 60<=timeout_seconds<=14400:
        raise ValueError("Vitis timeout must be within 60..14400 seconds")
    import torch
    from .tasks import DEFAULT_PART
    cohorts=make_cohort(
        training=training,validation=validation,test=test,seed=seed,
        max_vitis_calls=calls_per_network,part=DEFAULT_PART)
    path=Path(output)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Always use a NEW immutable experiment directory")
    path.mkdir(parents=True)
    source_revision=_source_revision()
    _write_json_new(path/"campaign.json",{
        "schema":"fpgai-research/real-online-post-hls-campaign/v1",
        "source_revision":source_revision,
        "cohort_seed":seed,
        "budget":budget,
        "original_tool":vitis_executable,
        "part":DEFAULT_PART,
        "real_vendor_execution_explicit_opt_in":True,
        "source_proxy_reward_used":False,
        "no_post_vivado_execution":True,
        "status":"IN_PROGRESS",
    })
    _write_json_new(path/"cohort_hashes.json",{
        split:[task.public_descriptor() for task in cases]
        for split,cases in cohorts.items()})
    agent=OnlineAgent(seed=seed)
    rows=[]
    attempts=0

    def log(value):
        nonlocal attempts
        attempts+=1
        if attempts>max_total_vitis_calls:
            raise RuntimeError("Explicit original Vitis process cap exceeded")
        _append_jsonl(path/"native_steps.jsonl",value)

    def episode(task,subdir,*,index,training=False,random_search=False):
        if _source_revision()!=source_revision:
            raise RuntimeError("Native campaign source revision changed; preserve all original cases")
        return _run_episode(
            path/"episodes"/subdir/task.task_id,task,agent,
            seed=seed+10000*index,training=training,
            random_search=random_search,
            vitis_executable=vitis_executable,
            timeout_seconds=timeout_seconds,callback=log)

    for i,task in enumerate(cohorts["train"]):
        rows.append(episode(task,"train",index=i,training=True))
        if (i+1)%validate_every==0 or i+1==training:
            checkpoint=path/"private_checkpoints"/f"after_{i+1:05d}"
            agent.save_private(checkpoint,{
                "source_revision":source_revision,
                "training_networks_completed":i+1})
            frozen=[]
            random_control=[]
            for j,item in enumerate(cohorts["validation"]):
                frozen.append(episode(
                    item,f"validation_after_{i+1:05d}",
                    index=100000+i*100+j))
                random_control.append(episode(
                    item,f"validation_random_after_{i+1:05d}",
                    index=100000+i*100+j,random_search=True))
            qualified=[x["best_hls_worst_case_cycles"] for x in frozen
                       if x["best_hls_worst_case_cycles"] is not None]
            random_qualified=[
                x["best_hls_worst_case_cycles"] for x in random_control
                if x["best_hls_worst_case_cycles"] is not None]
            paired=[
                (a["normalized_hls_improvement"],b["normalized_hls_improvement"])
                for a,b in zip(frozen,random_control)
                if a["normalized_hls_improvement"] is not None
                and b["normalized_hls_improvement"] is not None
            ]
            record={
                "training_networks_completed":i+1,
                "validation_networks":len(frozen),
                "qualified_validation_networks":len(qualified),
                "mean_best_qualified_hls_cycles":sum(qualified)/len(qualified)
                    if qualified else None,
                "online_training_updates":agent.updates,
                "no_validation_gradient_updates":True,
                "paired_normalized_valid_networks":len(paired),
                "drl_mean_normalized_hls_improvement":(
                    sum(a for a,b in paired)/len(paired) if paired else None),
                "random_mean_normalized_hls_improvement":(
                    sum(b for a,b in paired)/len(paired) if paired else None),
                "random_qualified_validation_networks":len(random_qualified),
                "random_mean_best_qualified_hls_cycles":(
                    sum(random_qualified)/len(random_qualified)
                    if random_qualified else None),
            }
            _append_jsonl(path/"learning_curve.jsonl",record)
    test_rows=[]
    for j,task in enumerate(cohorts["test"]):
        test_rows.append(episode(task,"test_rl",index=200000+j))
        test_rows.append(episode(
            task,"test_random",index=300000+j,random_search=True))
    _write_json_new(path/"summary.json",{
        "schema":"fpgai-research/real-online-post-hls-campaign/v1",
        "status":"COMPLETED_WITH_ORIGINAL_VITIS_ATTEMPTS",
        "training_networks":training,
        "validation_networks":validation,
        "test_networks":test,
        "online_gradient_updates":agent.updates,
        "original_vitis_calls_audited":attempts,
        "original_vitis_calls_limit":max_total_vitis_calls,
        "training_results":rows,
        "heldout_comparison":test_rows,
        "source_proxy_reward_used":False,
        "vendor_timing_is_post_hls_only":True,
        "physical_fmax_measured":False,
    })
    return {
        "status":"COMPLETED_WITH_ORIGINAL_VITIS_ATTEMPTS",
        "output":str(path),"native_steps":attempts,
        "online_updates":agent.updates,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-networks",type=int,default=100)
    p.add_argument("--validation-networks",type=int,default=2)
    p.add_argument("--test-networks",type=int,default=4)
    p.add_argument("--calls-per-network",type=int,default=3)
    p.add_argument("--validate-every",type=int,default=25)
    p.add_argument("--seed",type=int,default=20261030)
    p.add_argument("--output",default="build/real_online_hls_100_v1")
    p.add_argument("--plan",action="store_true")
    p.add_argument("--execute-native",action="store_true")
    p.add_argument("--vitis-executable",default="vitis_hls")
    p.add_argument("--timeout-seconds",type=int,default=7200)
    p.add_argument("--max-total-vitis-calls",type=int)
    a=p.parse_args(argv)
    result=run(
        output=a.output,training=a.train_networks,
        validation=a.validation_networks,test=a.test_networks,
        calls_per_network=a.calls_per_network,validate_every=a.validate_every,
        seed=a.seed,execute_native=(a.execute_native and not a.plan),
        vitis_executable=a.vitis_executable,
        timeout_seconds=a.timeout_seconds,
        max_total_vitis_calls=a.max_total_vitis_calls)
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
