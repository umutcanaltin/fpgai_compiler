"""Read-only and SHA-pinned interpretation of the author's original full C1 E1 run.

DOES NOT COMPILE OR EXECUTE C++, Python compiler operations, Vitis, Vivado,
or GitHub Actions. DOES NOT MODIFY ORIGINAL STUDY EVIDENCE.

Use only on author-verified 54-profile E0-matched original E1 evidence.
Reports source SCRATCH SLOT savings, numerical source comparisons and actual
address witnesses. These are NOT routed BRAM/FF resources, Fmax, FPGA compute
latency, throughput, energy, formal verification, or randomized graph effects.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median

from fpgai_research.experiments.cst_paper_c1_heldout_cpp_bridge import (
    SCHEMA as E1_SCHEMA,
    ORIGINAL_C1_REPORT_SHA256 as ORIGINAL_E0_SHA,
    ORIGINAL_C1_SOURCE_REVISION as ORIGINAL_E0_REVISION,
    FULL_PROFILES,
    LAYOUTS,
    audit as audit_e1,
)

SCHEMA = "fpgai-research/C1-original-full-E1-GCC-readonly-descriptive-analysis/v1"
ORIGINAL_E1_REVISION = "3bd1e36ff49864d89fa4c676e61da8ba0b936fce"
ORIGINAL_MANIFEST_SHA256 = (
    "a79e700208e5d7cada93dd1f488809b6622f52cf6b14fa7b7d7b05a8d624d83e"
)
EXPECTED_ARTIFACTS = FULL_PROFILES * 5 + 1
SEEDS = (111, 222, 333)
READERS = (2, 3, 4)
MAPPINGS = ("scalar", "mixed_phi")
NUMERIC_FIELDS = (
    "intermediate_numeric_max_absolute_error",
    "final_numeric_max_absolute_error",
)


def _integer(value, minimum=0) -> bool:
    return type(value) is int and value >= minimum


def _real(value) -> bool:
    return type(value) in (int, float) and isfinite(value) and value >= 0


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("C1 original E1 manifest missing/symlinked")
    return sha256(path.read_bytes()).hexdigest()


def _summary(root: Path, expected_manifest_sha: str) -> tuple[dict, dict]:
    if not isinstance(expected_manifest_sha, str) or len(expected_manifest_sha) != 64:
        raise ValueError("C1 expected original E1 manifest hash malformed")
    manifest = root / "artifact_sha256.json"
    if _sha(manifest) != expected_manifest_sha:
        raise ValueError("C1 original 54-profile E1 manifest SHA mismatch")
    # Existing E1 verifier must check the manifest against every original
    # generated C++/stdout/case JSON and the source-locked summary.
    receipt = audit_e1(root)
    if (receipt.get("profiles") != FULL_PROFILES
            or receipt.get("source_gcc_compilations") != FULL_PROFILES * len(LAYOUTS)
            or receipt.get("source_gcc_executions") != FULL_PROFILES * len(LAYOUTS)
            or receipt.get("source_revision") != ORIGINAL_E1_REVISION
            or receipt.get("original_e0_revision") != ORIGINAL_E0_REVISION
            or receipt.get("original_e0_sha256") != ORIGINAL_E0_SHA
            or receipt.get("source_artifact_count") != EXPECTED_ARTIFACTS
            or receipt.get("source_savings_profiles") != FULL_PROFILES
            or receipt.get("source_tie_profiles") != 0
            or receipt.get("new_compiler_or_vendor_calls") != 0
            or receipt.get("physical_compute_latency") is not None
            or receipt.get("board_qualified") is not False):
        raise ValueError("C1 original full E1 audit population/provenance mismatch")
    data = json.loads((root / "summary.json").read_text())
    if (data.get("schema") != E1_SCHEMA
            or data.get("experiment_source_revision") != ORIGINAL_E1_REVISION
            or data.get("original_E0_report_sha256") != ORIGINAL_E0_SHA
            or data.get("original_E0_source_revision") != ORIGINAL_E0_REVISION
            or data.get("smoke_only") is not False
            or len(data.get("cases", [])) != FULL_PROFILES):
        raise ValueError("C1 original E1 summary source/smoke scope mismatch")
    return data, receipt


def _profile(case: dict) -> dict:
    if (not isinstance(case, dict)
            or case.get("seed") not in SEEDS
            or case.get("readers") not in READERS
            or case.get("mapping") not in MAPPINGS
            or not isinstance(case.get("shape"), list)
            or len(case["shape"]) != 4
            or any(not _integer(x, 1) for x in case["shape"])):
        raise ValueError("C1 original E1 source profile malformed")
    canonical_sha = case.get("original_c1_e0_event_sha256")
    if (not isinstance(canonical_sha, str)
            or len(canonical_sha) != 64
            or any(x not in "0123456789abcdef" for x in canonical_sha)):
        raise ValueError("C1 original E0 event identity invalid")
    arms = case.get("layouts")
    if not isinstance(arms, dict) or set(arms) != set(LAYOUTS):
        raise ValueError("C1 original E1 source layouts missing")
    for layout, arm in arms.items():
        if (not isinstance(arm, dict)
                or arm.get("status") !=
                "R2H_ACTUAL_SOURCE_SCRATCH_ALIAS_AND_NUMPY_VERIFIED"
                or not _integer(arm.get("source_scratch_unit_slots"), 1)
                or not _integer(arm.get("original_actual_address_witness_count"), 1)
                or arm.get("original_GCC_compilations") != 1
                or arm.get("original_GCC_executions") != 1
                or any(not _real(arm.get(f)) for f in NUMERIC_FIELDS)):
            raise ValueError("C1 original E1 source values/witness/numeric fields invalid")
        for sha_field in (
            "generated_cpp_sha256", "actual_stdout_sha256",
            "runtime_source_event_sha256", "functional_sha256",
            "architecture_sha256", "input_sha256",
        ):
            value = arm.get(sha_field)
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError("C1 E1 original source witness identity invalid")
    full = arms["disjoint"]
    compact = arms["per_element_optimal"]
    for field in ("functional_sha256", "architecture_sha256", "input_sha256"):
        if full[field] != compact[field]:
            raise ValueError("C1 E1 storage arms are not matched source conditions")
    full_slots = full["source_scratch_unit_slots"]
    compact_slots = compact["source_scratch_unit_slots"]
    delta = full_slots - compact_slots
    if (delta <= 0 or case.get("new_e1_reuse_source_slot_savings") != delta):
        raise ValueError("C1 E1 source allocation improvement failed original contrast")
    return {
        "seed": case["seed"],
        "readers": case["readers"],
        "mapping": case["mapping"],
        "shape": case["shape"],
        "original_E0_canonical_event_sha256": canonical_sha,
        "disjoint_slots": full_slots,
        "reuse_slots": compact_slots,
        "saved_slots": delta,
        "source_slot_reduction_percent": round(100.0 * delta / full_slots, 6),
        "actual_address_witnesses_disjoint":
            full["original_actual_address_witness_count"],
        "actual_address_witnesses_reuse":
            compact["original_actual_address_witness_count"],
        "max_intermediate_numeric_absolute_error_both_layouts":
            max(full["intermediate_numeric_max_absolute_error"],
                compact["intermediate_numeric_max_absolute_error"]),
        "max_final_numeric_absolute_error_both_layouts":
            max(full["final_numeric_max_absolute_error"],
                compact["final_numeric_max_absolute_error"]),
    }


def _aggregates(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("C1 original summary has no rows")
    sums = {
        "n_descriptive_profiles": len(rows),
        "sum_disjoint_source_slots_over_distinct_models":
            sum(r["disjoint_slots"] for r in rows),
        "sum_reused_source_slots_over_distinct_models":
            sum(r["reuse_slots"] for r in rows),
        "sum_saved_source_slots_over_distinct_models":
            sum(r["saved_slots"] for r in rows),
        "median_saved_source_slots":
            median(r["saved_slots"] for r in rows),
        "min_saved_source_slots": min(r["saved_slots"] for r in rows),
        "max_saved_source_slots": max(r["saved_slots"] for r in rows),
        "median_source_slot_reduction_percent":
            median(r["source_slot_reduction_percent"] for r in rows),
        "min_source_slot_reduction_percent":
            min(r["source_slot_reduction_percent"] for r in rows),
        "max_source_slot_reduction_percent":
            max(r["source_slot_reduction_percent"] for r in rows),
        "total_actual_address_witnesses_disjoint":
            sum(r["actual_address_witnesses_disjoint"] for r in rows),
        "total_actual_address_witnesses_reuse":
            sum(r["actual_address_witnesses_reuse"] for r in rows),
        "max_intermediate_numeric_absolute_error":
            max(r["max_intermediate_numeric_absolute_error_both_layouts"]
                for r in rows),
        "max_final_numeric_absolute_error":
            max(r["max_final_numeric_absolute_error_both_layouts"]
                for r in rows),
        "all_source_slot_contrasts_positive": all(
            r["saved_slots"] > 0 for r in rows),
    }
    # Summed slots are *distinct generated programs*, not an aggregated
    # deployed design or a real FPGA resource saving. Median percent is per
    # profile; do not quietly average denominators across source sizes.
    return sums


def analyze(study: str | Path, *, pinned_manifest_sha256: str =
            ORIGINAL_MANIFEST_SHA256) -> dict:
    root = Path(study)
    data, receipt = _summary(root, pinned_manifest_sha256)
    observations = [_profile(case) for case in data["cases"]]
    identities = [
        (r["seed"], r["readers"], tuple(r["shape"]), r["mapping"])
        for r in observations
    ]
    if len(set(identities)) != FULL_PROFILES:
        raise ValueError("C1 original E1 static source profiles duplicated")
    count = Counter((r["seed"], r["readers"], r["mapping"])
                    for r in observations)
    expected_count = {
        (seed, readers, mapping): 3
        for seed in SEEDS for readers in READERS for mapping in MAPPINGS
    }
    if count != expected_count:
        raise ValueError("C1 original E1 seed/reader/mapper strata incomplete")
    conditions = []
    for readers in READERS:
        for mapping in MAPPINGS:
            rows = [r for r in observations
                    if r["readers"] == readers and r["mapping"] == mapping]
            if len(rows) != 9:
                raise ValueError("C1 source mapping/reader group missing")
            conditions.append({
                "reader_count": readers, "mapping": mapping,
                **_aggregates(rows),
            })
    summary = _aggregates(observations)
    if summary["all_source_slot_contrasts_positive"] is not True:
        raise ValueError("C1 original 54/54 positive source allocation false")
    return {
        "schema": SCHEMA,
        "status": "SIGNED_ORIGINAL_E1_SOURCE_SLOT_AND_NUMERIC_DESCRIPTIVES_ONLY",
        "original_E0_source_revision": ORIGINAL_E0_REVISION,
        "original_E0_report_sha256": ORIGINAL_E0_SHA,
        "original_E1_source_revision": ORIGINAL_E1_REVISION,
        "original_E1_manifest_sha256": pinned_manifest_sha256,
        "profiles": FULL_PROFILES,
        "emitted_GCC_compilations": FULL_PROFILES * len(LAYOUTS),
        "emitted_GCC_executions": FULL_PROFILES * len(LAYOUTS),
        "source_artifacts_attested": receipt["source_artifact_count"],
        "source_profile_aggregates": summary,
        "by_reader_and_mapping": conditions,
        "per_profile": observations,
        "new_compiler_GCC_or_vendor_calls": 0,
        "physical_inference_latency": None,
        "physical_BRAM_reduction": None,
        "board_qualified": False,
        "limitations": [
            "Original 54 profiles belong to one bounded MatMul fanout family",
            "Three fixed seeds and generated shapes are not independent deployed workloads",
            "Source scratch unit slots are not routed FPGA BRAM tiles, bytes or power",
            "Summed slots combine distinct source designs and do not represent one hardware allocation",
            "Generated host-GCC correctness at chosen inputs is not RTL/source all-input formal proof",
            "Address witness counts alone are not hardware cycle timing or memory bank-port proof",
            "Allo/CODO semantic parity and FPGA compute latency remain unestablished",
        ],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True,
                        help="Immutable author-original full 54-profile E1 root")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print overall and six subgroup statistics, omit "
                             "54 per-profile rows from stdout; original data untouched")
    args = parser.parse_args(argv)
    report = analyze(args.study)
    if args.summary_only:
        report = {k: v for k, v in report.items() if k != "per_profile"}
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
