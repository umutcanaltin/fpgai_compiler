"""Read-only, SOURCE-LOCKED interpretation of the original C2 CPU cost run.

This module NEVER calls the compiler, replay algorithms, GCC, Vitis, Vivado,
or GitHub Actions and NEVER changes either original study or its reports.
It checks the author's original 6 Nov 2026 report SHA before summarizing.

CPU source-checking wall time is NOT FPGA inference latency or RL policy
quality. Correlated seeds/repeated requests are descriptive trials, NOT
independent randomly sampled graph populations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median

SCHEMA = "fpgai-research/C2-original-fair-source-cost-readonly-analysis/v1"
INPUT_SCHEMA = "fpgai-research/C2-direct-lazy-eager-source-compiler-cost/v1"
ORIGINAL_SOURCE_REVISION = "637f2745721b32d6f5a7e02c4659b0623bc9eee7"
ORIGINAL_REPORT_SHA256 = (
    "be2adaa4acf30196a2ccb17114f2caa0af3e6679dc2685872e27cb086ca86ec1"
)
ORIGINAL_COHORTS = {
    ("chain_d2", 2), ("chain_d3", 3), ("chain_d4", 4),
    ("fanout_r2", 2), ("fanout_r3", 3), ("fanout_r4", 4),
}
METHODS = ("direct", "lazy", "eager")
STREAMS = ("repeat_certified", "cycle_certified")
SEEDS = (111, 222, 333)
QUERIES = (1, 8, 32)
EXPECTED_ROWS = 6 * 3 * 3 * 2 * 3


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _finite_wall(value: object) -> bool:
    return type(value) is int and value >= 0


def _load(root: Path, *, pinned_report_sha256: str) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("C2 original research directory missing or symlinked")
    receipt = root / "report.json"
    manifest = root / "artifact_sha256.json"
    if (receipt.is_symlink() or manifest.is_symlink()
            or not receipt.is_file() or not manifest.is_file()):
        raise ValueError("C2 original research report or SHA manifest invalid")
    if receipt.stat().st_size > 8_000_000 or manifest.stat().st_size > 100_000:
        raise ValueError("C2 original source report unexpectedly large")
    content = receipt.read_bytes()
    digest = _sha(content)
    if digest != pinned_report_sha256:
        raise ValueError("C2 original report SHA differs from author's verified receipt")
    sha_index = json.loads(manifest.read_text())
    if sha_index != {"report.json": digest}:
        raise ValueError("C2 original SHA inventory differs from report")
    report = json.loads(content)
    if (report.get("schema") != INPUT_SCHEMA
            or report.get("experiment_source_revision") != ORIGINAL_SOURCE_REVISION
            or report.get("cohorts") != len(ORIGINAL_COHORTS)
            or report.get("measured_cases") != EXPECTED_ROWS
            or report.get("seeds") != list(SEEDS)
            or report.get("query_counts") != list(QUERIES)
            or report.get("streams") != list(STREAMS)
            or report.get("original_source_preflight_compiler_calls_excluded") != 48
            or report.get("all_compiled_source_results_matched") is not True
            or any(report.get(field) != 0 for field in (
                "native_gcc_calls", "native_vitis_calls",
                "native_vivado_calls", "trained_rl_agents"))
            or report.get("hardware_latency") is not None):
        raise ValueError("C2 original source study population or provenance changed")
    return report


def _validate_records(report: dict) -> dict:
    rows = report.get("records")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise ValueError("C2 original data row population not complete")
    by_trial = defaultdict(dict)
    cohort_source = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("C2 original data row invalid")
        cohort = (row.get("cohort"), row.get("width"))
        method = row.get("method")
        stream = row.get("stream")
        seed = row.get("seed")
        queries = row.get("queries")
        if (cohort not in ORIGINAL_COHORTS
                or method not in METHODS or stream not in STREAMS
                or seed not in SEEDS or queries not in QUERIES):
            raise ValueError("C2 original row includes unknown cohort/protocol")
        indices = row.get("request_indices")
        if (not isinstance(indices, list) or len(indices) != queries
                or any(type(i) is not int or not 0 <= i < 8 for i in indices)
                or row.get("distinct_requested") != len(set(indices))
                or row.get("matched_compiled_results") is not True
                or not _finite_wall(row.get("wall_ns"))
                or type(row.get("execution_order")) is not int
                or not 0 <= row["execution_order"] < 3
                or type(row.get("source_compiler_calls")) is not int):
            raise ValueError("C2 original request/measurement row malformed")
        hash_value = row.get("result_sha256")
        state_hash = row.get("source_parent_sha256")
        if (not isinstance(hash_value, str) or len(hash_value) != 64
                or not isinstance(state_hash, str) or len(state_hash) != 64
                or any(c not in "0123456789abcdef" for c in hash_value + state_hash)):
            raise ValueError("C2 original compiler identity SHA malformed")
        if cohort in cohort_source and cohort_source[cohort] != state_hash:
            raise ValueError("C2 original cohort source state changed between methods")
        cohort_source[cohort] = state_hash
        correct_calls = (
            queries if method == "direct" else
            len(set(indices)) if method == "lazy" else 8
        )
        if row["source_compiler_calls"] != correct_calls:
            raise ValueError("C2 original method compiler invocation count incorrect")
        key = (*cohort, stream, seed, queries)
        if method in by_trial[key]:
            raise ValueError("C2 original duplicate strategy in paired trial")
        by_trial[key][method] = row
    expected_keys = len(ORIGINAL_COHORTS)*len(STREAMS)*len(SEEDS)*len(QUERIES)
    if len(by_trial) != expected_keys:
        raise ValueError("C2 original missing paired workload trials")
    for cohort, source in cohort_source.items():
        if not source:
            raise ValueError("C2 original source state missing")
    for _, methods in by_trial.items():
        if set(methods) != set(METHODS):
            raise ValueError("C2 original paired trial missing a strategy")
        if {methods[m]["execution_order"] for m in METHODS} != {0, 1, 2}:
            raise ValueError("C2 original strategy execution order duplicated")
        if len({tuple(methods[m]["request_indices"]) for m in METHODS}) != 1:
            raise ValueError("C2 original requests were not identical between methods")
        if len({methods[m]["result_sha256"] for m in METHODS}) != 1:
            raise ValueError("C2 original actual compiled source results did not match")
    return by_trial


def analyze(root: str | Path, *, pinned_report_sha256: str = ORIGINAL_REPORT_SHA256) -> dict:
    study = Path(root)
    report = _load(study, pinned_report_sha256=pinned_report_sha256)
    by_trial = _validate_records(report)
    rows = report["records"]
    conditions = []
    for stream in STREAMS:
        for queries in QUERIES:
            paired = [
                methods for key, methods in by_trial.items()
                if key[2] == stream and key[4] == queries
            ]
            if len(paired) != 18:
                raise ValueError("C2 original subgroup lacks 6×3 matched trials")
            stats = {}
            for method in METHODS:
                wall = [m[method]["wall_ns"] for m in paired]
                calls = [m[method]["source_compiler_calls"] for m in paired]
                stats[method] = {
                    "n_descriptive_correlated_observations": len(wall),
                    "median_wall_ms": round(median(wall) / 1_000_000, 6),
                    "min_wall_ms": round(min(wall) / 1_000_000, 6),
                    "max_wall_ms": round(max(wall) / 1_000_000, 6),
                    "median_compiler_calls": median(calls),
                }
            comparisons = {}
            for left, right in (("lazy", "eager"), ("direct", "eager"),
                                ("lazy", "direct")):
                # Count paired CPU comparisons; do not promote them into
                # independent sample statistics or significance tests.
                comparisons[f"{left}_vs_{right}"] = {
                    "left_faster_pairs": sum(
                        1 for m in paired
                        if m[left]["wall_ns"] < m[right]["wall_ns"]),
                    "equal_time_pairs": sum(
                        1 for m in paired
                        if m[left]["wall_ns"] == m[right]["wall_ns"]),
                    "right_faster_pairs": sum(
                        1 for m in paired
                        if m[left]["wall_ns"] > m[right]["wall_ns"]),
                }
            conditions.append({
                "stream": stream, "queries": queries,
                "paired_workloads": len(paired),
                "strategies": stats, "paired_wall_time_comparisons": comparisons,
            })
    totals = {m: sum(row["source_compiler_calls"] for row in rows
                     if row["method"] == m) for m in METHODS}
    sample_counts = Counter(x["method"] for x in rows)
    if any(sample_counts[m] != EXPECTED_ROWS // 3 for m in METHODS):
        raise ValueError("C2 source method sample sizes not equal")
    if totals["lazy"] > totals["eager"]:
        raise ValueError("C2 lazy source cost impossible on same 8-action parent")
    return {
        "schema": SCHEMA,
        "status": "SOURCE_PINNED_ORIGINAL_CPU_DESCRIPTIVE_EVIDENCE_NOT_FPGA_LATENCY",
        "original_source_revision": ORIGINAL_SOURCE_REVISION,
        "original_report_sha256": pinned_report_sha256,
        "original_measured_method_rows": EXPECTED_ROWS,
        "paired_trial_count": len(by_trial),
        "cohort_count": len(ORIGINAL_COHORTS),
        "methods": list(METHODS),
        "original_source_preflight_calls_excluded": 48,
        "total_measured_compiler_calls": totals,
        "by_condition": conditions,
        "all_compiler_results_matched": True,
        "new_source_compiler_calls_during_analysis": 0,
        "new_vendor_calls_during_analysis": 0,
        "hardware_latency": None,
        "limitations": [
            "CPU wall milliseconds are NOT FPGA inference latency",
            "6 fixed source graph cohorts, repeated seeds and query streams are correlated",
            "Certified-only action requests selected using excluded 48-call preflight",
            "Lazy cache cannot require more source compilations than eager catalog",
            "These source-only measurements do not demonstrate RL/search superiority",
        ],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True,
                        help="Existing original 6-Nov 2026 C2 source-only output")
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.study), sort_keys=True,
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
