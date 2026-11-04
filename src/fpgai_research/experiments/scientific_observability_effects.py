"""Conservative scientific-effect summary from a QUALIFIED cross-stage pair receipt.

Descriptive effects of four preregistered, source/context-audited E3/E4
contrasts. The JSON is an existing independent audit's output, not raw
vendor measurement. This module does not execute tools or infer speedup,
statistical significance, netlist equivalence or population-level rates.
"""
from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.cross_stage_observations import SCHEMA as SOURCE_SCHEMA
from fpgai_research.experiments.independent_e4_pairs import PAIR_SPECS

SCHEMA = "fpgai-research/qualified-cross-stage-effect-summary/v1"
_PAIRS = {entry[3] for entry in PAIR_SPECS}
_FLAGS = (
    "source_cpp_file_bytes_equal",
    "fresh_generated_rtl_file_set_bytes_equal",
    "hls_six_objective_e3_estimates_equal",
    "routed_e4_wns_and_five_resources_equal",
    "e3_estimate_tie_hides_e4_qor_difference",
)


def analyze_effects(receipt: dict) -> dict:
    if (not isinstance(receipt, dict) or receipt.get("schema") != SOURCE_SCHEMA
            or receipt.get("case_count") != 8
            or receipt.get("preregistered_pair_count") != 4
            or receipt.get("e3_tied_pair_count") is None
            or receipt.get("native_vendor_tools_executed_by_analysis") is not False
            or receipt.get("formal_logic_equivalence_claimed") is not False
            or receipt.get("measured_board_inference_latency_claimed") is not False
            or receipt.get("private_case_hashes_or_paths_exported") is not False):
        raise ValueError("Require complete independently attested cross-stage study")
    pairs = receipt.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 4:
        raise ValueError("Four preregistered physical contrasts required")
    ids = [r.get("pair_id") for r in pairs if isinstance(r, dict)]
    if len(ids) != 4 or len(set(ids)) != 4 or set(ids) != _PAIRS:
        raise ValueError("Wrong or duplicate matched pairs")

    e3_ties = physical_ties = hidden = rtl_collapses = 0
    summaries = []
    for row in pairs:
        if (any(type(row.get(key)) is not bool for key in _FLAGS)
                or row.get("formal_circuit_equivalence_verified") is not False
                or row.get("measured_inference_latency_verified") is not False):
            raise ValueError("Partial or overclaimed pair observation")
        expected_hidden = (
            row["hls_six_objective_e3_estimates_equal"]
            and not row["routed_e4_wns_and_five_resources_equal"]
        )
        if row["e3_estimate_tie_hides_e4_qor_difference"] != expected_hidden:
            raise ValueError("E3/E4 tie relation is inconsistent")
        wns = row.get("e4_postroute_wns_delta_b_minus_a_ns")
        lut = row.get("e4_postroute_lut_delta_b_minus_a")
        dsp = row.get("e4_postroute_dsp_delta_b_minus_a")
        if (any(type(x) not in (int, float) or not isfinite(x)
                for x in (wns, lut, dsp))
                or type(lut) is not int or type(dsp) is not int):
            raise ValueError("Incomplete physical evidence or non-integer resource counts")
        if row["routed_e4_wns_and_five_resources_equal"] and any(
            x != 0 for x in (wns, lut, dsp)
        ):
            raise ValueError("Physical tie conflicts with measured deltas")
        e3_ties += int(row["hls_six_objective_e3_estimates_equal"])
        physical_ties += int(row["routed_e4_wns_and_five_resources_equal"])
        hidden += int(expected_hidden)
        rtl_collapse = (not row["source_cpp_file_bytes_equal"]
                        and row["fresh_generated_rtl_file_set_bytes_equal"])
        rtl_collapses += int(rtl_collapse)
        summaries.append({
            "pair_id": row["pair_id"],
            "e3_estimates_equal": row["hls_six_objective_e3_estimates_equal"],
            "e4_observables_equal": row["routed_e4_wns_and_five_resources_equal"],
            "different_hls_cpp_same_fresh_rtl_bytes": rtl_collapse,
            "e3_tie_masked_e4_difference": expected_hidden,
            "delta_wns_ns": wns, "delta_lut": lut, "delta_dsp": dsp,
        })
    if (receipt.get("e3_tied_pair_count") != e3_ties
            or receipt.get("physical_qor_tie_pair_count") != physical_ties
            or receipt.get("e3_tie_with_e4_difference_count") != hidden):
        raise ValueError("Summary counts disagree with individual audited pairs")
    return {
        "schema": SCHEMA,
        "preregistered_contrasts": 4,
        "original_model_cases": 8,
        "e3_estimate_ties": e3_ties,
        "e4_qor_observational_ties": physical_ties,
        "e3_ties_masking_physical_difference": hidden,
        "different_hls_cpp_but_identical_fresh_rtl_bytes_pairs": rtl_collapses,
        "e3_tie_miss_fraction_within_observed_e3_ties":
            {"numerator": hidden, "denominator": e3_ties},
        "measured_pairs": summaries,
        "observational_only": True,
        "independent_implementation_seeds_repeated": False,
        "statistical_p_value_or_population_inference_available": False,
        "logic_equivalence_or_on_device_speedup_proven": False,
        "native_vendor_tools_executed_by_summary": False,
        "crosscompiler_superiority_eligible": False,
        "note": "Finite descriptive counts over four fixed contrasts, not an "
                "unbiased estimate of future architectures. Exact E3 and E4 "
                "observables differ in evidentiary strength.",
    }


def effects_latex(summary: dict) -> str:
    if (summary.get("schema") != SCHEMA or summary.get("preregistered_contrasts") != 4
            or summary.get("observational_only") is not True
            or summary.get("statistical_p_value_or_population_inference_available") is not False
            or summary.get("logic_equivalence_or_on_device_speedup_proven") is not False):
        raise ValueError("Cannot typeset an unqualified effect summary")
    metrics = [
        ("Source-distinct C++ / byte-identical fresh RTL",
         summary["different_hls_cpp_but_identical_fresh_rtl_bytes_pairs"]),
        ("Matched pairs with equal E3 estimates", summary["e3_estimate_ties"]),
        ("Matched pairs with equal routed E4 observables",
         summary["e4_qor_observational_ties"]),
        ("E3-estimate ties masking an E4 difference",
         summary["e3_ties_masking_physical_difference"]),
    ]
    if any(type(v) is not int or not 0 <= v <= 4 for _, v in metrics):
        raise ValueError("Invalid descriptive pair counts")
    lines = [
        "% From independently qualified four-pair E3/E4 observational audit.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Descriptive observability outcomes across four prespecified "
        r"matched architecture contrasts. Ties describe reported observables, "
        r"not formal hardware equivalence or statistical population rates.}",
        r"\label{tab:fpgai-scientific-effects}",
        r"\begin{tabular}{@{}lr@{}}",
        r"\toprule",
        r"Source-to-physical observation & Contrasts \\",
        r"\midrule",
    ]
    for label, n in metrics:
        lines.append(f"{label} & {n}/4 " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def verify_existing_effects(audited_pairs_path, existing_output):
    """Verify, without writing, an immutable effects export against its input.

    The source receipt is itself a cross-stage observational audit; the
    strict manuscript builder independently reaudits its native ancestry.
    """
    src = Path(audited_pairs_path)
    target = Path(existing_output)
    if (src.is_symlink() or not src.is_file()
            or not 0 < src.stat().st_size <= 2 * 1024 * 1024):
        raise ValueError("Missing safe independently audited cross-stage JSON")
    if target.is_symlink() or not target.is_dir():
        raise ValueError("Missing safe existing effects directory")
    raw = src.read_bytes()
    expected = analyze_effects(json.loads(raw))
    expected["audited_cross_stage_receipt_sha256"] = sha256(raw).hexdigest()
    filenames = {"effects.json", "effects.tex"}
    if {part.name for part in target.iterdir()} != filenames:
        raise ValueError("Existing effects export has missing or unexpected files")
    json_path = target / "effects.json"
    tex_path = target / "effects.tex"
    for part in (json_path, tex_path):
        if (part.is_symlink() or not part.is_file()
                or not 0 < part.stat().st_size <= 2 * 1024 * 1024):
            raise ValueError("Existing effects export has unsafe files")
    if json.loads(json_path.read_text(encoding="utf-8")) != expected:
        raise ValueError("Existing effects JSON differs from reaudited scientific results")
    if tex_path.read_text(encoding="utf-8") != effects_latex(expected):
        raise ValueError("Existing effects LaTeX differs from reaudited scientific results")
    return expected


def export_effects(audited_pairs_path, new_output):
    src = Path(audited_pairs_path)
    target = Path(new_output)
    if (src.is_symlink() or not src.is_file() or
            not 0 < src.stat().st_size <= 2 * 1024 * 1024):
        raise ValueError("Missing safe independently audited cross-stage JSON")
    if target.is_symlink() or target.exists():
        raise FileExistsError("Scientific effects export requires a new destination")
    raw = src.read_bytes()
    effect = analyze_effects(json.loads(raw))
    effect["audited_cross_stage_receipt_sha256"] = sha256(raw).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-effects-", dir=target.parent) as tmp:
        stage = Path(tmp) / "effects"
        stage.mkdir()
        (stage / "effects.json").write_text(
            json.dumps(effect, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        (stage / "effects.tex").write_text(effects_latex(effect), encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Scientific effects destination appeared")
        stage.rename(target)
    return effect
