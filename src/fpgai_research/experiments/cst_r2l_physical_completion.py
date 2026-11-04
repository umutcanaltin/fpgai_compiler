"""R2l: preregistered read-only six-positive-pair physical completion ledger.

Preserve the frozen R2i HLS and R2j original Vivado experiments. A separately
prepared follow-up study is permitted for the five *unobserved* HLS-positive
pairs only. The CLI never executes Vivado; its vendor commands are suggestions
for explicit, bounded foreground invocation of the canonical R2j runner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "fpgai-research/cst-R2l-positive-HLS-physical-completion/v1"
ORIGINAL_HLS_SOURCE_SHA = "b2d5dcc4f18fad7cb1cd2c342101dd33185e817f"
ORIGINAL_HLS_POSITIVE_PAIRS = (20, 23, 28, 31, 32, 35)
TOTAL_CASES = 72
TOTAL_PAIRS = 36


def _validate_audit(name: str, audit: dict, source_revision: str) -> list[dict]:
    if (audit.get("prespecified_cases") != TOTAL_CASES
            or audit.get("original_hls_source_revision") != source_revision
            or type(audit.get("cases")) is not list
            or len(audit["cases"]) != TOTAL_CASES):
        raise ValueError(name + " has invalid 72-case source/physical identity")
    for index, row in enumerate(audit["cases"]):
        if (row.get("case") != f"case_{index:03d}"
                or row.get("group") != index // 2
                or row.get("source_storage") != (
                    "disjoint" if index % 2 == 0 else "per_element_optimal")
                or not isinstance(row.get("source_cpp_sha256"), str)
                or len(row["source_cpp_sha256"]) != 64):
            raise ValueError(name + " has mismatched case order or source")
    return audit["cases"]


def _paired_routed(a: dict, b: dict) -> dict:
    observed = (a.get("routed_observation_valid") is True
                and b.get("routed_observation_valid") is True)
    strict = (a.get("routed_qualified") is True
              and b.get("routed_qualified") is True)
    if strict and not observed:
        raise ValueError("Strict physical pair is not a routed observation")
    if not observed:
        return {"both_routed_observation_valid": False,
                "both_strict_clean_DRC_E4": False,
                "routed_resource_delta_disjoint_minus_optimal": None,
                "optimized_minus_disjoint_WNS_ns": None,
                "DRC_rule_ids_by_arm": None,
                "source_classification": "UNKNOWN_NOT_BOTH_ROUTED"}
    ua = a.get("physical_resources") or {}
    ub = b.get("physical_resources") or {}
    from math import isfinite
    keys = ("bram", "lut", "ff", "dsp")
    if any(type(u.get(k)) not in (int, float) or not isfinite(u[k])
           or u[k] < 0 for u in (ua, ub) for k in keys):
        raise ValueError("Matched routed observation lacks valid physical resources")
    wa, wb = a.get("routed_wns_ns"), b.get("routed_wns_ns")
    if (type(wa) not in (int, float) or type(wb) not in (int, float)
            or not isfinite(wa) or not isfinite(wb)):
        raise ValueError("Matched routed observation lacks numeric WNS")
    warnings = [
        sorted(((case.get("drc_warning_summary") or {}).get("rules") or {}).keys())
        for case in (a, b)
    ]
    if strict and (a.get("drc_verified_clean") is not True
                   or b.get("drc_verified_clean") is not True
                   or any(warnings)):
        raise ValueError("Physical E4 requires both clean original DRC receipts")
    return {
        "both_routed_observation_valid": True,
        "both_strict_clean_DRC_E4": strict,
        "routed_resource_delta_disjoint_minus_optimal": {
            k: ua[k] - ub[k] for k in keys
        },
        "optimized_minus_disjoint_WNS_ns": wb - wa,
        "DRC_rule_ids_by_arm": warnings,
        "source_classification": ("MATCHED_STRICT_CLEAN_DRC_E4" if strict
                                  else "MATCHED_ROUTED_DESCRIPTIVE_ONLY"),
    }


def build_completion_ledger(*, native: dict, effects: dict,
                            original_physical: dict,
                            followup_physical: dict | None = None) -> dict:
    """Compare already-audited reports, without touching any source or vendor."""
    source_revision = native.get("original_source_revision")
    if source_revision != ORIGINAL_HLS_SOURCE_SHA:
        raise ValueError("R2l requires the frozen original R2i source SHA")
    native_cases = native.get("per_case_status")
    effect_pairs = effects.get("cases")
    if (native.get("prespecified_source_cases") != TOTAL_CASES
            or native.get("prespecified_matched_pairs") != TOTAL_PAIRS
            or type(native_cases) is not list
            or len(native_cases) != TOTAL_CASES
            or effects.get("original_source_revision") != source_revision
            or type(effect_pairs) is not list
            or len(effect_pairs) != TOTAL_PAIRS):
        raise ValueError("Original native E3 denominator or source revision invalid")
    if (native.get("original_CSim_qualified_cases") != 72
            or native.get("original_CSynth_timing_resource_qualified_cases") != 48
            or effects.get("both_E3_qualified_matched_pairs") != 24
            or effects.get("not_both_E3_qualified_matched_pairs") != 12):
        raise ValueError("R2l requires exactly the frozen original R2i E2/E3 population")
    original = _validate_audit("original pilot", original_physical, source_revision)
    followup = (_validate_audit("new follow-up", followup_physical, source_revision)
                if followup_physical is not None else None)
    rows = []
    hls_positive = []
    original_observed = []
    followup_observed = []
    ready = []
    for pair in range(TOTAL_PAIRS):
        x, y = pair * 2, pair * 2 + 1
        effects_row = effect_pairs[pair]
        na, nb = native_cases[x], native_cases[y]
        oa, ob = original[x], original[y]
        if (effects_row.get("group") != pair
                or effects_row.get("cases") != [f"case_{x:03d}", f"case_{y:03d}"]
                or (na.get("case"), nb.get("case")) != (oa["case"], ob["case"])):
            raise ValueError("R2l original case identity mismatches R2i effects")
        for native_case, phys_case in ((na, oa), (nb, ob)):
            if any(native_case.get(k) != phys_case.get(k)
                   for k in ("source_cpp_sha256", "graph", "mapping", "source_storage")):
                raise ValueError("R2l original HLS/RTL exact-source identity mismatch")
        if (effects_row["both_original_E3_qualified"] is not (
                na["original_CSynth_timing_resources_qualified"] is True
                and nb["original_CSynth_timing_resources_qualified"] is True)):
            raise ValueError("R2l HLS qualification differs from native case audit")
        delta = effects_row["HLS_estimated_delta_disjoint_minus_optimal"]["bram_18k"]
        positive = effects_row["both_original_E3_qualified"] is True and delta is not None and delta > 0
        if positive:
            hls_positive.append(pair)
        primary = _paired_routed(oa, ob)
        if primary["both_routed_observation_valid"]:
            original_observed.append(pair)
        if followup is not None:
            fa, fb = followup[x], followup[y]
            for base, other in ((oa, fa), (ob, fb)):
                if any(base.get(k) != other.get(k)
                       for k in ("source_cpp_sha256", "graph", "mapping",
                                 "source_storage", "requested_clock_ns")):
                    raise ValueError("New physical study changed original source or clock")
            second = _paired_routed(fa, fb)
            if second["both_routed_observation_valid"]:
                followup_observed.append(pair)
        else:
            fa, fb, second = None, None, None
        if positive and primary["both_routed_observation_valid"] and second and second["both_routed_observation_valid"]:
            raise ValueError("Follow-up duplicated an already observed original HLS-positive pair")
        status = "NOT_HLS_POSITIVE"
        if positive:
            if primary["both_routed_observation_valid"]:
                status = "ORIGINAL_PHYSICAL_PILOT_OBSERVED"
            elif second and second["both_routed_observation_valid"]:
                status = "SEPARATE_FOLLOWUP_PAIR_OBSERVED"
            elif followup is None:
                status = "NEW_STUDY_REQUIRED"
            elif (fa.get("status") == "PREPARED_NOT_RUN"
                  and fb.get("status") == "PREPARED_NOT_RUN"
                  and fa.get("original_hls_qualified") is True
                  and fb.get("original_hls_qualified") is True):
                status = "READY_FOR_SEPARATE_FOREGROUND_VIVADO"
                ready.append(pair)
            else:
                status = "PARTIAL_FAILED_OR_BLOCKED_NO_AUTOMATIC_RERUN"
        chosen = primary if primary["both_routed_observation_valid"] else second
        rows.append({
            "pair": pair,
            "cases": [f"case_{x:03d}", f"case_{y:03d}"],
            "graph": na["graph"],
            "mapping": na["mapping"],
            "both_original_E3_qualified": effects_row["both_original_E3_qualified"],
            "HLS_estimated_delta_BRAM_18K_disjoint_minus_optimal": delta,
            "prespecified_HLS_positive": positive,
            "original_observation": primary,
            "separate_followup_observation": second,
            "selected_physical_observation": chosen,
            "followup_execution_status": status,
        })
    if tuple(hls_positive) != ORIGINAL_HLS_POSITIVE_PAIRS:
        raise ValueError("HLS-positive pair preregistration no longer matches original R2i")
    if 20 not in original_observed:
        raise ValueError("Original R2j pair20 physical pilot missing or not fully observed")
    if any(pair in followup_observed for pair in original_observed):
        raise ValueError("Separate follow-up duplicated an original observed pair")
    if any(pair not in ORIGINAL_HLS_POSITIVE_PAIRS for pair in followup_observed):
        raise ValueError("Separate follow-up executed an unregistered HLS-positive control")
    physical_positive_observed = [r for r in rows if r["prespecified_HLS_positive"]
                                 and r["selected_physical_observation"]
                                 and r["selected_physical_observation"]["both_routed_observation_valid"]]
    physical_positive_strict = [r for r in physical_positive_observed
                                if r["selected_physical_observation"]["both_strict_clean_DRC_E4"]]
    return {
        "schema": SCHEMA,
        "original_HLS_source_revision": source_revision,
        "original_prespecified_cases": TOTAL_CASES,
        "original_prespecified_pairs": TOTAL_PAIRS,
        "original_both_E3_qualified_pairs": effects["both_E3_qualified_matched_pairs"],
        "original_E3_unknown_pairs": effects["not_both_E3_qualified_matched_pairs"],
        "original_HLS_positive_pairs": list(hls_positive),
        "original_HLS_positive_pair_count": len(hls_positive),
        "original_physical_pair20_observed": True,
        "original_physical_observation_pair_indices": original_observed,
        "separate_followup_observation_pair_indices": followup_observed,
        "original_HLS_positive_pairs_with_routed_observations": len(physical_positive_observed),
        "original_HLS_positive_pairs_with_strict_E4": len(physical_positive_strict),
        "positive_pairs_with_routed_BRAM_reduction_descriptive": sum(
            r["selected_physical_observation"]["routed_resource_delta_disjoint_minus_optimal"]["bram"] > 0
            for r in physical_positive_observed
        ),
        "positive_pairs_with_routed_BRAM_tie_descriptive": sum(
            r["selected_physical_observation"]["routed_resource_delta_disjoint_minus_optimal"]["bram"] == 0
            for r in physical_positive_observed
        ),
        "positive_pairs_with_routed_BRAM_regression_descriptive": sum(
            r["selected_physical_observation"]["routed_resource_delta_disjoint_minus_optimal"]["bram"] < 0
            for r in physical_positive_observed
        ),
        "HLS_positive_unobserved_pairs": len(hls_positive) - len(physical_positive_observed),
        "ready_for_next_original_pair_in_new_study": ready[0] if ready else None,
        "all_ready_original_pair_indices": ready,
        "new_original_GCC_Vitis_Vivado_calls": 0,
        "physical_cohorts_are_separate": True,
        "rows": rows,
        "limitations": [
            "The first R2j study remains frozen; do not rerun or overwrite original case040/041",
            "Follow-up original Vivado cases belong to a separately prepared, source-pinned study",
            "HLS BRAM estimates and routed BRAM tiles are not interchangeable",
            "Missing/failed and warning-only physical cases are not strict clean-DRC E4",
            "OOC routed results do not establish bitstream deployment, achieved Fmax, power or firstness",
        ],
    }


def run_cli(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hls-study", required=True)
    parser.add_argument("--original-physical-study", required=True)
    parser.add_argument("--followup-study")
    parser.add_argument("--prepare-followup-output", help="New create-only physical RTL study; zero vendor calls")
    parser.add_argument("--output", help="New create-only forensic completion JSON")
    args = parser.parse_args(argv)
    if args.followup_study and args.prepare_followup_output:
        parser.error("Existing and new follow-up study choices are mutually exclusive")
    if args.prepare_followup_output:
        target = Path(args.prepare_followup_output)
        if target.resolve() == Path(args.original_physical_study).resolve():
            raise ValueError("Refusing to replace frozen original R2j physical study")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Follow-up physical experiment must be create-only")
    if args.output:
        out = Path(args.output)
        if out.exists() or out.is_symlink():
            raise FileExistsError("R2l completion ledger output must be create-only")
        if not out.parent.is_dir():
            raise ValueError("R2l completion ledger parent must already exist")
    from fpgai_research.experiments.cst_r2i_fanout_native_execute import audit_native
    from fpgai_research.experiments.cst_r2i_fanout_native_effects import compute_effects
    from fpgai_research.experiments.cst_r2j_fanout_physical import (
        audit_routed_study, prepare_routed_study,
    )
    native = audit_native(args.hls_study, require_current_source=False)
    effects = compute_effects(native)
    original = audit_routed_study(args.original_physical_study)
    # Validate frozen original first, BEFORE creating any new physical study.
    build_completion_ledger(native=native, effects=effects, original_physical=original)
    followup_study = args.followup_study
    if args.prepare_followup_output:
        prepare_routed_study(native_study=args.hls_study, output=args.prepare_followup_output)
        followup_study = args.prepare_followup_output
    followup = audit_routed_study(followup_study) if followup_study else None
    result = build_completion_ledger(native=native, effects=effects,
                                     original_physical=original, followup_physical=followup)
    result["original_physical_study_path"] = str(Path(args.original_physical_study))
    result["followup_study_path"] = str(Path(followup_study)) if followup_study else None
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> None:
    run_cli(argv)


if __name__ == "__main__":
    main()
