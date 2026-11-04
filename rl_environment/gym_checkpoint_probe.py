"""Read-only, no-Vitis GNN learning probe on a fixed future native-HLS cohort.

Compare *the same* ORIGINAL Vitis-label examples from training episodes
91..100 across checkpoints 0,10,...,90. All checkpoint weights preceded the
heldout observational data. This evaluates the shared GNN feasibility HEAD,
NOT the policy's causal search effectiveness or HLS latency improvement.
The logged original source cases are never modified or replayed in hardware.
"""
from __future__ import annotations

import argparse
import csv
import json
from math import log
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.experiments.architecture_knobs_v2 import build_catalog
from .cohort import make_cohort
from .gym_env import GymGraphEnv, MAX_NODES, KNOBS
from .gym_agent import OnlineGraphTrainer
from .gym_eval_recovery import training_reference, _read_journal
from .gym_learning_diagnostics import _correct_weighted_probability, _auc

try:
    import torch
except ImportError:
    torch = None


def _metrics(probs, labels):
    n = len(labels)
    prevalence = sum(labels) / n
    return {
        "cases": n, "qualified": sum(labels),
        "feasibility_fraction": prevalence,
        "raw_weighted_brier": sum((p-y)**2 for p, y in zip(probs, labels))/n,
        "corrected_brier": sum(
            (_correct_weighted_probability(p)-y)**2
            for p,y in zip(probs,labels))/n,
        "roc_auc": _auc(list(zip(probs, labels))),
        "constant_zero_brier_reference": prevalence,
        "empirical_class_prevalence_brier_reference_in_sample":
            prevalence*(1-prevalence),
    }


def probe(study, output, *, holdout_first_episode=91):
    if holdout_first_episode != 91:
        raise ValueError("Only preregistered episodes 91..100 permitted")
    # Validate source provenance before checking optional ML dependencies.
    # A missing original study is a file error even on compiler-only installs.
    original = training_reference(study)
    if torch is None:
        raise RuntimeError("PyTorch required for native-free checkpoint inference")
    root = original["root"]
    seed = original["seed"]
    old = original["campaign"]
    source_revision = old["source_revision"]
    # All 1000 HLS calls already existed; their original journal is the
    # only source of feasibility labels, never learned or proxy HLS QoR.
    rows = [
        row for row in _read_journal(root / "native_steps.jsonl")
        if row["method"] == "online_train"
        and row["task_id"] in {f"train_{i:05d}" for i in range(90,100)}
    ]
    if len(rows) != 100:
        raise ValueError("Last-ten-task fixed observational test requires 100 real HLS cases")
    cohort = make_cohort(
        training=100,
        validation=old["validation_networks_per_seed"],
        test=old["test_networks_per_seed"],
        max_vitis_calls=10, seed=seed, clock_ns=None,
        fixed_clock_ablation_ns=old.get("fixed_clock_ablation_ns"))
    tasks = {t.task_id: t for t in cohort["train"]}
    data = []
    for row in rows:
        task = tasks[row["task_id"]]
        if row["task_sha"] != task.task_sha:
            raise ValueError("Source task descriptor differs from original HLS case")
        candidate = tuple(tuple(v) for v in row["trial_evidence"]["architecture_choices"])
        clock = float(row["clock_ns"])
        if not 2.0 <= clock <= 32.0:
            raise ValueError("Original native clock outside study domain")
        catalog = build_catalog(task.model)
        if len(candidate) != len(task.model.operators):
            raise ValueError("Original candidate operator count mismatch")
        features = np.zeros(MAX_NODES*KNOBS, dtype=np.float32)
        for i, knobs in enumerate(candidate):
            for j, value in enumerate(knobs):
                if value < 0 or value >= len(catalog.options[i][j]):
                    raise ValueError("Original candidate index not valid for source task")
                features[i*KNOBS+j] = value/max(1,len(catalog.options[i][j])-1)
        data.append((task.task_id, task, features, clock, bool(row["qualified"])))
    # The *same* empty-history input context for all checkpoints. This is a
    # deliberately restricted fixed-candidate feasibility probe, never a
    # reconstruction of the agent's original adaptive episode history.
    with TemporaryDirectory(prefix="fpgai_checkpoint_probe_") as temp:
        base = Path(temp)
        obs_for_task = {}
        for task_id, task, _, _, _ in data:
            if task_id not in obs_for_task:
                env = GymGraphEnv(task, base / task_id)
                obs, _ = env.reset(seed=seed)
                obs_for_task[task_id] = obs
        policy_variants = [(0, None)]
        for episode in range(10,100,10):
            checkpoint = (
                root / f"seed_{seed}" / "training_checkpoints" /
                f"after_{episode:06d}_episodes")
            if not checkpoint.is_dir():
                raise ValueError(f"Checkpoint missing: {checkpoint}")
            policy_variants.append((episode, checkpoint))
        report = []
        all_predictions = []
        for episode, checkpoint in policy_variants:
            agent = OnlineGraphTrainer(
                seed=seed,
                use_message_passing=old["message_passing_enabled"])
            if checkpoint is not None:
                meta = agent.load_private(checkpoint, load_optimizer=False)
                if (meta["online_updates"] != 10*episode or
                    meta["training_episodes_completed"] != episode or
                    meta["source_revision"] != source_revision):
                    raise ValueError("Checkpoint source/update provenance drift")
            predictions, labels = [], []
            for task_id, task, features, clock, label in data:
                obs = dict(obs_for_task[task_id])
                menu = np.zeros_like(obs["action_features"])
                menu[0] = features
                mask = np.zeros_like(obs["action_mask"])
                mask[0] = 1
                obs["action_features"] = menu
                obs["action_mask"] = mask
                with torch.no_grad():
                    _, _, joint = agent.policy(
                        obs, with_arch_context=True)
                    fraction = torch.tensor(
                        [(clock-2.)/30.], dtype=torch.float32)
                    logit = agent.policy.feasibility_head(
                        torch.cat([joint[0],fraction])).squeeze()
                    raw = float(torch.sigmoid(logit).item())
                predictions.append(raw)
                labels.append(int(label))
                all_predictions.append({
                    "checkpoint_after_episodes": episode,
                    "task_id": task_id,
                    "original_native_case_clock_ns": clock,
                    "native_qualified": int(label),
                    "raw_positive_weighted_sigmoid": raw,
                    "analytic_class_weight_corrected_probability":
                        _correct_weighted_probability(raw),
                })
            summary = {"checkpoint_after_episodes": episode,
                       **_metrics(predictions,labels)}
            report.append(summary)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Checkpoint learning probe output must be create-only")
    output.mkdir(parents=True)
    with (output/"checkpoint_results.csv").open("x",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    with (output/"checkpoint_case_predictions.csv").open("x",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=list(all_predictions[0]))
        writer.writeheader()
        writer.writerows(all_predictions)
    manifest = {
        "status": "FIXED_FUTURE_ORIGINAL_HLS_OBSERVATIONAL_FEASIBILITY_PROBE",
        "source_training_revision": source_revision,
        "checkpoint_episodes": [x["checkpoint_after_episodes"] for x in report],
        "fixed_original_training_task_ids": [f"train_{i:05d}" for i in range(90,100)],
        "fixed_task_candidate_outcomes": 100,
        "source_cases_modified": 0, "vitis_invocations": 0,
        "original_HLS_latency_not_predicted": True,
        "not_policy_search_performance": True,
        "not_external_unseen_holdout": True,
        "selected_actions_from_later_policy_may_cause_observational_bias": True,
        "empty_trial_history_for_all_checkpoints": True,
        "trained_checkpoint_never_saw_fixed_cases": True,
        "not_claiming_physical_timing_or_transfer_superiority": True,
        "results": report,
    }
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        x=[r["checkpoint_after_episodes"] for r in report]
        for key,ylabel,filename in (
            ("roc_auc","Feasibility discrimination ROC AUC (higher better)",
             "checkpoint_feasibility_roc_auc.png"),
            ("corrected_brier","Analytic weight-corrected Brier (lower better)",
             "checkpoint_feasibility_corrected_brier.png"),
        ):
            y=[r[key] for r in report]
            if all(v is not None for v in y):
                fig,ax=plt.subplots(figsize=(8,4))
                ax.plot(x,y,marker="o")
                if key=="roc_auc":
                    ax.axhline(.5,color="gray",linestyle="--",label="Chance discrimination")
                else:
                    ax.axhline(report[0]["constant_zero_brier_reference"],
                               color="gray",linestyle="--",label="Always infeasible")
                ax.set(xlabel="Training episodes completed before fixed future cases",
                       ylabel=ylabel,
                       title="Same 100 logged native-HLS outcomes; offline observational probe")
                ax.legend()
                fig.tight_layout()
                fig.savefig(output/filename,dpi=180)
                plt.close(fig)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    result=probe(args.study,args.output)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
