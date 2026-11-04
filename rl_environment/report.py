"""Read-only plots from completed ORIGINAL online Vitis evidence.

Never convert missing/unqualified native observations to artificial zeros.
Never modify original study or rewrite previous reports.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from math import isfinite


def export(study, output):
    root=Path(study)
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Report must be written to a fresh output folder")
    campaign=json.loads((root/"campaign.json").read_text())
    if campaign.get("source_proxy_reward_used") is not False:
        raise ValueError("Native-only original-receipt study required")
    if not (root/"summary.json").is_file():
        raise ValueError("Study is incomplete; preserve and inspect native_steps.jsonl")
    points=[json.loads(line) for line in
            (root/"learning_curve.jsonl").read_text().splitlines()]
    observations=[json.loads(line) for line in
                  (root/"native_steps.jsonl").read_text().splitlines()]
    if not points or not observations:
        raise ValueError("Missing native learning curve or original steps")
    if any(row.get("source_proxy_reward") is not False or
           not row.get("native_receipt_sha256") for row in observations):
        raise ValueError("Original vendor provenance is missing")
    dest.mkdir(parents=True)
    headers=["training_networks_completed","validation_networks",
             "qualified_validation_networks",
             "mean_best_qualified_hls_cycles","online_training_updates",
             "random_qualified_validation_networks",
             "random_mean_best_qualified_hls_cycles",
             "paired_normalized_valid_networks",
             "drl_mean_normalized_hls_improvement",
             "random_mean_normalized_hls_improvement"]
    with (dest/"native_learning_curve.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=headers,extrasaction="ignore")
        w.writeheader()
        w.writerows(points)
    made_plot=False
    curve_metric='none'
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        xs=[p["training_networks_completed"] for p in points]
        normalized=all(
            p.get("paired_normalized_valid_networks",0)>0
            and type(p.get("drl_mean_normalized_hls_improvement")) in (int,float)
            and type(p.get("random_mean_normalized_hls_improvement")) in (int,float)
            and isfinite(p["drl_mean_normalized_hls_improvement"])
            and isfinite(p["random_mean_normalized_hls_improvement"])
            for p in points)
        if normalized:
            ys=[p["drl_mean_normalized_hls_improvement"] for p in points]
            random_ys=[p["random_mean_normalized_hls_improvement"] for p in points]
            label="Mean paired improvement from each task's verified HLS baseline"
            curve_metric="paired_normalized_hls_improvement"
        else:
            ys=[p["mean_best_qualified_hls_cycles"] for p in points]
            random_ys=[p.get("random_mean_best_qualified_hls_cycles") for p in points]
            label="Unseen validation mean best qualified HLS cycles"
            curve_metric="mean_qualified_hls_worst_case_cycles"
        if all(y is not None for y in ys) and all(y is not None for y in random_ys):
            fig,ax=plt.subplots(figsize=(9,5))
            ax.plot(xs,ys,marker="o",label="Frozen DRL policy")
            ax.plot(xs,random_ys,marker="s",linestyle="--",label="Matched random")
            ax.set(xlabel="Different networks trained ONLINE (original Vitis)",
                   ylabel=label,
                   title="FPGAI Post-HLS online learning (qualified original Vitis only)")
            ax.grid(True,alpha=.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(dest/"native_validation_learning_curve.png",dpi=160)
            plt.close(fig)
            made_plot=True
    summary=json.loads((root/"summary.json").read_text())
    test=summary["heldout_comparison"]
    methods={}
    for method in ("online_drl","random"):
        matching=[row for row in test if row["method"]==method]
        qualified=[row["best_hls_worst_case_cycles"] for row in matching
                   if row["best_hls_worst_case_cycles"] is not None]
        methods[method]={
            "cohort_size":len(matching),
            "qualified_cohort_size":len(qualified),
            "mean_best_qualified_hls_cycles":sum(qualified)/len(qualified)
                if qualified else None,
        }
    paired_by_task={}
    for row in test:
        paired_by_task.setdefault(row["task_id"],{})[row["method"]]=row
    paired_test=[
        (v["online_drl"]["normalized_hls_improvement"],
         v["random"]["normalized_hls_improvement"])
        for v in paired_by_task.values()
        if "online_drl" in v and "random" in v
        and type(v["online_drl"].get("normalized_hls_improvement")) in (int,float)
        and type(v["random"].get("normalized_hls_improvement")) in (int,float)]
    vendor_seconds=[r["native_vendor_seconds"] for r in observations
                    if type(r.get("native_vendor_seconds")) in (int,float)]
    report={
        "schema":"fpgai-research/online-post-hls-report/v1",
        "real_vitis_receipt_step_count":len(observations),
        "total_original_vitis_seconds_recorded":sum(vendor_seconds),
        "native_calls_with_recorded_duration":len(vendor_seconds),
        "validation_checkpoints":len(points),
        "graph_created":made_plot,
        "validation_curve_metric":curve_metric if made_plot else "not_plotted",
        "paired_test_tasks_with_qualified_baselines":len(paired_test),
        "paired_test_mean_drl_normalized_improvement":(
            sum(a for a,b in paired_test)/len(paired_test)
            if paired_test else None),
        "paired_test_mean_random_normalized_improvement":(
            sum(b for a,b in paired_test)/len(paired_test)
            if paired_test else None),
        "heldout_test":methods,
        "only_qualified_HLS_observations":True,
        "original_source_proxy_reward":False,
        "physical_Fmax_claimed":False,
        "study_sha256":__import__("hashlib").sha256(
            (root/"summary.json").read_bytes()).hexdigest(),
    }
    with (dest/"report.json").open("x") as f:
        json.dump(report,f,sort_keys=True,indent=2)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--output",required=True)
    a=parser.parse_args(argv)
    print(json.dumps(export(a.study,a.output),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
