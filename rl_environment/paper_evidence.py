"""Source-pinned paper evidence export from immutable original Gym/Vitis cases.

--allow-incomplete creates *PROVISIONAL* descriptive snapshots while a campaign
is in progress; it cannot claim a native audit or publishable science. A
completed study MUST pass independent gym_audit before the report is generated.
Never invokes Vitis, modifies source study files, or invents missing metrics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

from .gym_audit import audit


TRIAL_FIELDS = (
    "seed", "split", "method", "task_id", "task_sha", "episode_path",
    "native_call_index", "global_call_index", "requested_clock_ns",
    "hls_cycles_if_qualified", "hls_estimated_latency_ns",
    "best_feasible_latency_ns", "qualified", "reward", "cumulative_episode_reward",
    "vendor_seconds", "csim_max_absolute_error", "allowed_max_absolute_error",
    "hls_estimated_clock_ns", "hls_effective_scheduling_budget_ns",
    "hls_scheduling_margin_ns", "hls_timing_feasible",
    "resource_lut", "resource_ff", "resource_dsp", "resource_bram_18k",
    "resource_uram", "resource_limits_json", "architecture_choices_json",
    "architecture_knob_values_json", "network_operator_shapes_json",
    "hls_report_sha256", "vitis_invocation_sha256", "qualification_reason",
    "qualification_gates_failed_json", "csim_passed",
    "raw_hls_worst_case_cycles", "timing_parse_issue",
    "online_update_index", "loss_total", "loss_actor", "loss_critic",
    "loss_entropy_term", "policy_entropy", "log_probability",
    "value_prediction", "discounted_td_target", "advantage",
    "gradient_l2_before_clip", "loss_auxiliary_feasibility",
    "native_feasibility_label", "feasibility_prediction_before_hls",
    "feasibility_brier_score",
)
EPISODE_FIELDS = (
    "seed", "split", "method", "task_id", "task_sha", "episode_path",
    "training_episode_index", "recorded_native_calls", "qualified_calls",
    "qualification_fraction", "episode_reward_total", "best_feasible_hls_ns",
    "first_feasible_hls_ns", "normalized_improvement_over_first_feasible",
    "native_vendor_seconds", "last_training_update", "complete_in_journal",
)


def _lines(path):
    if not path.is_file():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed source journal {path.name}:{line_no}") from exc
    return rows


def _write_csv(path, fields, records):
    with path.open("x", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _git_snapshot(revision, path):
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Campaign source revision must be exact 40-character Git SHA")
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, check=False, timeout=20,
    )
    if result.returncode != 0:
        raise ValueError(f"Cannot fetch exact source-pinned file from Git: {path}")
    return result.stdout


def _native_episode(row, study_root):
    ep = Path(row["private_episode_path"]).resolve(strict=True)
    if not ep.is_relative_to(study_root):
        raise ValueError("Log references an episode outside its own study")
    parts = ep.relative_to(study_root).parts
    if len(parts) != 4 or not parts[0].startswith("seed_"):
        raise ValueError("Unexpected campaign episode layout")
    return ep, int(parts[0][5:]), parts[1]


def export(study, output, *, allow_incomplete=False, plots=True):
    root = Path(study).resolve(strict=True)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Paper evidence export output must be create-only")
    campaign = json.loads((root / "campaign.json").read_text())
    if campaign.get("schema") != "fpgai-gym-native-plan/v1":
        raise ValueError("Not an original source-pinned Gym study")
    summary_path = root / "summary.json"
    completed = summary_path.is_file()
    if completed:
        audit_receipt = audit(root)  # original source HLS & reward replay: read-only
        confidence = "FULL_INDEPENDENT_ORIGINAL_HLS_AUDIT_PASSED"
    else:
        if not allow_incomplete:
            raise ValueError("Campaign unfinished: use --allow-incomplete only for provisional inspection")
        audit_receipt = None
        confidence = "PROVISIONAL_INCOMPLETE_NOT_INDEPENDENTLY_AUDITED"
    observations = _lines(root / "native_steps.jsonl")
    episode_journal = _lines(root / "completed_episodes.jsonl")
    # Each task can have multiple arms; exact case/seed/path disambiguates.
    trials = []
    per_episode = defaultdict(list)
    for i, row in enumerate(observations, 1):
        ep, seed, split = _native_episode(row, root)
        evidence = row.get("trial_evidence") or {}
        training = row.get("training_metrics") or {}
        metrics = evidence.get("estimated_resources") or {}
        item = {
            "seed": seed, "split": split, "method": row["method"],
            "task_id": row["task_id"], "task_sha": row["task_sha"],
            "episode_path": str(ep.relative_to(root)),
            "native_call_index": int(row["native_call_index"]) + 1,
            "global_call_index": i,
            "requested_clock_ns": row["clock_ns"],
            "hls_cycles_if_qualified": row["hls_cycles"],
            "hls_estimated_latency_ns": row["hls_estimated_latency_ns"],
            "best_feasible_latency_ns": row["best_feasible_latency_ns"],
            "qualified": row["qualified"],
            "reward": row["reward"],
            "cumulative_episode_reward": row.get("cumulative_episode_reward"),
            "vendor_seconds": row["wall_seconds"],
            "csim_max_absolute_error": evidence.get("csim_max_absolute_error"),
            "allowed_max_absolute_error": evidence.get("allowed_max_absolute_error"),
            "hls_estimated_clock_ns": row.get("hls_estimated_clock_ns"),
            "hls_effective_scheduling_budget_ns": row.get(
                "hls_effective_scheduling_budget_ns"),
            "hls_scheduling_margin_ns": row.get("hls_scheduling_margin_ns"),
            "hls_timing_feasible": row.get("hls_timing_feasible"),
            "resource_lut": metrics.get("lut"),
            "resource_ff": metrics.get("ff"),
            "resource_dsp": metrics.get("dsp"),
            "resource_bram_18k": metrics.get("bram_18k"),
            "resource_uram": metrics.get("uram"),
            "resource_limits_json": json.dumps(
                evidence.get("available_resources"), sort_keys=True),
            "architecture_choices_json": json.dumps(
                evidence.get("architecture_choices"), sort_keys=True),
            "architecture_knob_values_json": json.dumps(
                evidence.get("architecture_compiler_knob_values"), sort_keys=True),
            "network_operator_shapes_json": json.dumps(
                evidence.get("network_operator_shapes"), sort_keys=True),
            "hls_report_sha256": row.get("hls_report_sha256"),
            "vitis_invocation_sha256": row.get("vitis_invocation_sha256"),
            "qualification_reason": evidence.get("qualification_reason"),
            "qualification_gates_failed_json": json.dumps(
                evidence.get("qualification_gates_failed"), sort_keys=True),
            "csim_passed": evidence.get("csim_passed"),
            "raw_hls_worst_case_cycles": evidence.get("raw_hls_worst_case_cycles"),
            "timing_parse_issue": evidence.get("original_hls_timing_parse_issue"),
            "online_update_index": training.get("training_updates"),
        }
        for name in (
            "loss_total", "loss_actor", "loss_critic", "loss_entropy_term",
            "policy_entropy", "log_probability", "value_prediction",
            "discounted_td_target", "advantage", "gradient_l2_before_clip",
            "loss_auxiliary_feasibility",
            "feasibility_prediction_before_hls", "feasibility_brier_score",
        ):
            item[name] = training.get(name)
        item["native_feasibility_label"] = training.get(
            "native_feasibility_label")
        trials.append(item)
        per_episode[(seed, split, row["task_id"], row["method"])].append(item)
    finished = {
        Path(ep["private_episode_path"]).resolve(strict=True)
        for ep in episode_journal
    }
    episodes = []
    for (seed, split, task_id, method), rows in per_episode.items():
        earliest = next((r["hls_estimated_latency_ns"]
                         for r in rows if r["qualified"]), None)
        best = rows[-1]["best_feasible_latency_ns"]
        training_idx = (int(task_id.rsplit("_", 1)[-1]) + 1
                        if split == "train" and method == "online_train" else None)
        episodes.append({
            "seed": seed, "split": split, "method": method,
            "task_id": task_id, "task_sha": rows[0]["task_sha"],
            "episode_path": rows[0]["episode_path"],
            "training_episode_index": training_idx,
            "recorded_native_calls": len(rows),
            "qualified_calls": sum(bool(r["qualified"]) for r in rows),
            "qualification_fraction": sum(bool(r["qualified"]) for r in rows)/len(rows),
            "episode_reward_total": sum(float(r["reward"]) for r in rows),
            "best_feasible_hls_ns": best,
            "first_feasible_hls_ns": earliest,
            "normalized_improvement_over_first_feasible":
                (earliest-best)/earliest if earliest is not None and best is not None else None,
            "native_vendor_seconds": sum(r["vendor_seconds"] or 0 for r in rows),
            "last_training_update": rows[-1]["online_update_index"],
            "complete_in_journal":
                (root / rows[0]["episode_path"]).resolve(strict=True) in finished,
        })
    # Preserve truly complete zero-call episodes: no fabricated rewards or latency.
    populated_paths = {
        (root / r["episode_path"]).resolve(strict=True)
        for r in episodes if r["episode_path"] is not None
    }
    for ep in episode_journal:
        ep_path = Path(ep["private_episode_path"]).resolve(strict=True)
        if ep_path in populated_paths:
            continue
        rel = ep_path.relative_to(root).parts
        if len(rel) != 4 or not rel[0].startswith("seed_"):
            raise ValueError("Completed episode escaped expected campaign layout")
        partition = rel[1]
        training_idx = (
            int(ep["task_id"].rsplit("_", 1)[-1]) + 1
            if partition == "train" and ep["method"] == "online_train" else None)
        episodes.append({
            "seed": int(rel[0][5:]), "split": partition, "method": ep["method"],
            "task_id": ep["task_id"], "task_sha": ep["task_sha"],
            "episode_path": str(ep_path.relative_to(root)),
            "training_episode_index": training_idx,
            "recorded_native_calls": 0, "qualified_calls": 0,
            "qualification_fraction": None, "episode_reward_total": ep.get("episode_return"),
            "best_feasible_hls_ns": ep["best_feasible_latency_ns"],
            "first_feasible_hls_ns": None,
            "normalized_improvement_over_first_feasible": None,
            "native_vendor_seconds": 0,
            "last_training_update": ep.get("online_updates_after_episode"),
            "complete_in_journal": True,
        })
    episodes.sort(key=lambda r: (
        -1 if r["seed"] is None else r["seed"], r["split"] or "",
        r["training_episode_index"] or 0, r["task_id"], r["method"]))
    # Inventory every native case directory, including an interrupted/failed
    # subprocess whose original logs exist but no finished study-step receipt
    # was ever appended. These are NOT silently counted as qualified trials.
    attested_cases = {
        (root / r["episode_path"] /
         f"case_{r['native_call_index'] - 1:04d}").resolve()
        for r in trials
    }
    case_inventory = []
    for case in sorted(root.rglob("case_*")):
        if not case.is_dir() or case.is_symlink():
            continue
        if not case.parent.name.startswith("episode_"):
            continue
        native_receipt = case / "vitis_invocation.json"
        invocation = json.loads(native_receipt.read_text()) if native_receipt.is_file() else {}
        case_inventory.append({
            "native_case_path": str(case.relative_to(root)),
            "has_completed_step_journal": case.resolve() in attested_cases,
            "has_staged_action": (case / "action.json").is_file(),
            "has_original_vitis_invocation_receipt": native_receipt.is_file(),
            "vendor_returncode": invocation.get("returncode"),
            "vendor_timed_out": invocation.get("timed_out"),
            "vendor_completed_successfully": invocation.get("completed_successfully"),
            "vendor_seconds": invocation.get("duration_seconds"),
            "vitis_tool_version": invocation.get("tool_version_output"),
        })
    # Snapshot immutable reward/actor/plan implementation from the *campaign*
    # commit, even if this exporter is invoked by a newer checkout.
    meta = campaign.get("paper_evidence")
    provenance = {}
    source_files = (
        "rl_environment/reward.py", "rl_environment/gym_agent.py",
        "rl_environment/gym_study.py", "rl_environment/clock_actions.py",
    )
    for name in source_files:
        data = _git_snapshot(campaign["source_revision"], name)
        digest = sha256(data).hexdigest()
        expected = (meta or {}).get("source_file_sha256", {}).get(name)
        if expected is not None and expected != digest:
            raise ValueError(f"Source-pin digest mismatch for {name}")
        provenance[name] = {"sha256": digest, "verified_against_campaign": expected is not None}
    output.mkdir(parents=True, exist_ok=False)
    snaps = output / "source_snapshots"
    snaps.mkdir()
    for name in source_files:
        (snaps / name.rsplit("/", 1)[-1]).write_bytes(
            _git_snapshot(campaign["source_revision"], name))
    _write_csv(output / "native_trial_table.csv", TRIAL_FIELDS, trials)
    _write_csv(output / "native_attempt_inventory.csv", (
        "native_case_path", "has_completed_step_journal", "has_staged_action",
        "has_original_vitis_invocation_receipt", "vendor_returncode",
        "vendor_timed_out", "vendor_completed_successfully",
        "vendor_seconds", "vitis_tool_version"), case_inventory)
    _write_csv(output / "episode_learning_curve.csv", EPISODE_FIELDS, episodes)
    training_trials = [r for r in trials
                       if r["method"] == "online_train"]
    _write_csv(output / "policy_update_curve.csv", TRIAL_FIELDS,
               [r for r in trials if r["online_update_index"] is not None])
    _write_csv(output / "reward_curve.csv", TRIAL_FIELDS, training_trials)
    made_figures = []
    if plots and episodes:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pass
        else:
            for field, name, ylabel in (
                ("episode_reward_total", "training_reward_by_episode.png",
                 "Summed reward per training episode"),
                ("qualification_fraction", "training_feasibility_by_episode.png",
                 "Fraction of original HLS candidates qualified"),
                ("normalized_improvement_over_first_feasible",
                 "training_relative_latency_improvement.png",
                 "Relative best HLS latency improvement within episode"),
            ):
                fig, ax = plt.subplots()
                for seed in sorted({r["seed"] for r in episodes if r["seed"] is not None}):
                    data = sorted(
                        (r for r in episodes if r["method"] == "online_train"
                         and r["seed"] == seed
                         and r["training_episode_index"] is not None
                         and r[field] is not None),
                        key=lambda r: r["training_episode_index"])
                    if data:
                        ax.plot([r["training_episode_index"] for r in data],
                                [r[field] for r in data], label=f"seed {seed}")
                ax.set_xlabel("Online training episode (tasks differ)")
                ax.set_ylabel(ylabel)
                if ax.lines:
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(output / name, dpi=180)
                    made_figures.append(name)
                plt.close(fig)
            for field in ("loss_total", "loss_actor", "loss_critic",
                          "policy_entropy", "loss_auxiliary_feasibility",
                          "feasibility_brier_score"):
                data = [r for r in training_trials
                        if r["online_update_index"] is not None
                        and r[field] is not None]
                if data:
                    fig, ax = plt.subplots()
                    ax.plot([r["online_update_index"] for r in data],
                            [r[field] for r in data])
                    ax.set_xlabel("Online policy updates")
                    ax.set_ylabel(field.replace("_", " "))
                    fig.tight_layout()
                    name = f"training_{field}.png"
                    fig.savefig(output / name, dpi=180)
                    made_figures.append(name)
                    plt.close(fig)
    manifest = {
        "schema": "fpgai-gym-paper-evidence-export/v1",
        "status": confidence,
        "campaign_original_source_commit": campaign["source_revision"],
        "full_independent_audit_receipt": audit_receipt,
        "source_snapshots": provenance,
        "campaign_paper_method_metadata": meta,
        "reward_defined_in": "source_snapshots/reward.py",
        "results_are_original_vitis_hls_estimates_not_physical_latency": True,
        "training_episode_count_is_not_cross_task_latency_comparability": True,
        "trial_rows": len(trials),
        "observed_original_case_directories": len(case_inventory),
        "native_cases_without_completed_step_journal": sum(
            not item["has_completed_step_journal"] for item in case_inventory),
        "episode_rows": len(episodes),
        "completed_episode_journal_rows": len(episode_journal),
        "figures": made_figures,
        "publication_claim_ready": False,
        "missing_diagnostics_are_null_not_imputed": True,
        "not_git_tracked_raw_native_artifacts": True,
    }
    (output / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "campaign_snapshot.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-incomplete", action="store_true",
                   help="Only a provisional, *non-audited*, non-publication preview")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)
    print(json.dumps(export(
        a.study, a.output, allow_incomplete=a.allow_incomplete,
        plots=not a.no_plots), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
