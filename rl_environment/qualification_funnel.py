"""Read-only diagnostic of WHY original native HLS cases did not qualify.

Not a replacement for the source-pinned independent scientific audit.
Can inspect immutable earlier-revision campaigns as long as the task generator
and model identity remain compatible. Never writes to source or invokes Vitis.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .cohort import make_cohort
from .post_hls_env import _receipt


def _timing_exception_category(message):
    """Classify existing original-Vitis timing failures without disclosing XML.

    Return a stable category; never infer qualification or modify vendor cases.
    """
    if message is None:
        return None
    if not isinstance(message, str):
        raise ValueError("Original timing parse issue must be string or null")
    signatures = (
        ("missing_clock_uncertainty", "Missing HLS clock uncertainty"),
        ("missing_or_malformed_period", "Missing or malformed HLS target/estimated period"),
        ("different_reported_target", "Vitis reported a different requested clock"),
        ("freshness_or_hash_mismatch", "HLS timing report not fresh and source-attributed"),
        ("missing_timing_section", "Missing or ambiguous HLS timing section"),
        ("conflicting_target_fields", "Conflicting HLS target periods"),
        ("conflicting_uncertainty_fields", "Conflicting HLS clock uncertainty"),
        ("invalid_uncertainty", "Invalid or conflicting HLS clock uncertainty"),
        ("invalid_effective_budget", "Invalid effective HLS scheduling clock budget"),
        ("missing_or_symlinked_xml", "Missing or symlinked original HLS XML"),
        ("ambiguous_user_assignments", "Ambiguous Vitis HLS user assignments"),
        ("invalid_requested_clock", "Invalid requested HLS clock"),
    )
    for label, signature in signatures:
        if signature in message:
            return label
    return "other_unclassified"


def diagnose(study, *, examples=3):
    root = Path(study).resolve(strict=True)
    campaign = json.loads((root / "campaign.json").read_text())
    tasks = {}
    for seed in campaign["training_seeds"]:
        cohort = make_cohort(
            training=campaign["training_networks_per_seed"],
            validation=campaign["validation_networks_per_seed"],
            test=campaign["test_networks_per_seed"],
            max_vitis_calls=campaign["calls_per_episode_max"],
            seed=seed, clock_ns=None,
            fixed_clock_ablation_ns=campaign.get("fixed_clock_ablation_ns"),
            freeze_numeric_format=campaign.get("freeze_numeric_format", False),
            beginner_pilot=campaign.get("beginner_pilot", False),
            curriculum_stage=campaign.get("curriculum_stage"))
        tasks.update({(seed, task.task_id): task
                      for split in cohort.values() for task in split})
    cases = Counter()
    examples_by_reason = defaultdict(list)
    timing_exception_categories = Counter()
    timing_exception_examples = defaultdict(list)
    records = (root / "native_steps.jsonl").read_text().splitlines()
    for row_text in records:
        row = json.loads(row_text)
        ep = Path(row["private_episode_path"]).resolve(strict=True)
        if not ep.is_relative_to(root):
            raise ValueError("Episode directory escapes source-pinned study")
        parts = ep.relative_to(root).parts
        if len(parts) != 4 or not parts[0].startswith("seed_"):
            raise ValueError("Unexpected original study layout")
        seed = int(parts[0][5:])
        task = tasks.get((seed, row["task_id"]))
        if task is None or task.task_sha != row["task_sha"]:
            raise ValueError("Study task fingerprint cannot be reconstructed")
        case = ep / f"case_{int(row['native_call_index']):04d}"
        result = _receipt(case, task)
        if result["qualified"] != row["qualified"]:
            raise ValueError("Reconstructed qualification differs from original step")
        cases["original_cases"] += 1
        if result["qualified"]:
            cases["qualified"] += 1
            continue
        cases["infeasible_or_unqualified"] += 1
        failures = result["qualification_gates_failed"]
        if not failures:
            raise ValueError("Rejected result without an auditable failure reason")
        for reason in failures:
            cases["failure/" + reason] += 1
            if len(examples_by_reason[reason]) < examples:
                examples_by_reason[reason].append(str(case.relative_to(root)))
        if result.get("original_hls_timing_parse_issue") is not None:
            cases["timing_parse_exception"] += 1
            category = _timing_exception_category(
                result["original_hls_timing_parse_issue"])
            timing_exception_categories[category] += 1
            if len(timing_exception_examples[category]) < examples:
                timing_exception_examples[category].append(
                    str(case.relative_to(root)))
        if result.get("raw_hls_worst_case_cycles") is None:
            cases["missing_original_worst_cycles"] += 1
        if result.get("csim_passed") is True:
            cases["csim_passed"] += 1
    return {
        "status": "READ_ONLY_QUALIFICATION_FUNNEL_NOT_FULL_SOURCE_REPLAY_AUDIT",
        "campaign_source_revision": campaign["source_revision"],
        "source_study_unchanged": True,
        "vendor_invocations": 0,
        "failure_stages_are_not_mutually_exclusive": True,
        "counts": dict(sorted(cases.items())),
        "timing_exception_category_counts": dict(
            sorted(timing_exception_categories.items())),
        "example_case_paths_by_timing_exception_category": dict(
            sorted(timing_exception_examples.items())),
        "example_case_paths_by_failure": dict(sorted(examples_by_reason.items())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--examples-per-failure", type=int, default=3)
    args = parser.parse_args(argv)
    if not 0 <= args.examples_per_failure <= 20:
        parser.error("examples-per-failure must be 0..20")
    print(json.dumps(diagnose(args.study, examples=args.examples_per_failure),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
