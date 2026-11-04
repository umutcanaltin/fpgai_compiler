"""R2v independent read-only interpretation of the FROZEN R2u native cohort.

Reuses original SHA-attested Vitis E2/E3 qualification without running
the vendor tools. Covers every original 32 source arm and each of the
64 predeclared pairwise factor contrasts. Missing/unqualified E3 pairs
are UNKNOWN, never imputed as a zero delta or a BRAM tie.

R2u originals were source-pinned at 0c6d85f: R2v may analyze the
same byte-identical archived source packages after main advances,
while re-deriving the original 32 generated C++ sources and checking
all original model/Tcl/testbench/vendor report SHA receipts.

No physical routed BRAM, port inference, binding implementation,
bit-exactness or broad causal effect can be proved from this report.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

from fpgai_research.experiments.cst_r2u_hls_mechanism_prepare import (
    CASES, ARMS, FACTOR_NAMES, FACTORS, GRAPHS,
)
from fpgai_research.experiments.cst_r2u_hls_mechanism_execute import (
    RESOURCES, SCHEMA as ORIGINAL_NATIVE_SCHEMA, audit_native,
)

SCHEMA = "fpgai-research/cst-R2v-read-only-R2u-factorial-mechanism-review/v1"
ORIGINAL_R2U_SOURCE_REVISION = "0c6d85f189062df8d5157237f3c8ac74ac8da4b8"
HEADERS = (
    "graph_index", "factor", "arm_a", "arm_b",
    "e2_both_qualified", "e3_both_qualified", "classification",
    "bram_18k_b_minus_a", "lut_b_minus_a", "ff_b_minus_a",
    "dsp_b_minus_a", "uram_b_minus_a", "worst_cycles_b_minus_a",
)


def _classify(c: dict) -> str:
    if not c["matched_original_CSynth_qualified"]:
        return "UNKNOWN_NOT_BOTH_E3"
    delta = c["estimated_resource_delta_b_minus_a"]["bram_18k"]
    if type(delta) is not int:
        return "UNKNOWN_NO_BRAM_ESTIMATE"
    if delta > 0:
        return "INCREASED_HLS_ESTIMATED_BRAM"
    if delta < 0:
        return "REDUCED_HLS_ESTIMATED_BRAM"
    return "TIED_HLS_ESTIMATED_BRAM"


def review(original: dict, *,
           expected_source_revision: str = ORIGINAL_R2U_SOURCE_REVISION) -> dict:
    """Deterministic summary of an already source-attested R2u native audit.

    The input must have been obtained from audit_native, not hand-entered
    vendor numbers. Standalone synthetic tests test accounting logic.
    """
    if (not isinstance(expected_source_revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}", expected_source_revision)):
        raise ValueError("R2v requires an exact expected original source SHA")
    if (original.get("schema") != ORIGINAL_NATIVE_SCHEMA
            or original.get("original_source_revision") != expected_source_revision
            or original.get("planned_source_cases") != CASES
            or original.get("observed_source_cases") != CASES
            or original.get("prespecified_matched_factor_contrasts")
                    != len(GRAPHS) * len(ARMS) * len(FACTOR_NAMES)//2
            or original.get("original_vendor_calls_during_audit") != 0
            or original.get("physical_BRAM_measurements") is not None
            or original.get("physical_Fmax_measurements") is not None):
        raise ValueError("R2v original R2u provenance/population/evidence tier mismatch")
    cases = original.get("cases")
    contrasts = original.get("contrasts")
    if (type(cases) is not list or len(cases) != CASES
            or type(contrasts) is not list or len(contrasts) != 64):
        raise ValueError("R2v full immutable arm/contrast population missing")
    e2 = e3 = invocations = 0
    for index, row in enumerate(cases):
        levels = dict(zip(FACTOR_NAMES, ARMS[index % len(ARMS)]))
        if (row.get("case") != f"case_{index:03d}"
                or row.get("graph_index") != index // len(ARMS)
                or any(row.get(key) != level for key, level in levels.items())
                or not isinstance(row.get("source_cpp_sha256"), str)
                or not re.fullmatch("[a-f0-9]{64}", row["source_cpp_sha256"])
                or type(row.get("original_E2_qualified")) is not bool
                or type(row.get("original_E3_qualified")) is not bool):
            raise ValueError("R2v arm/factor/source-case provenance mismatch")
        if row["original_E3_qualified"] and not row["original_E2_qualified"]:
            raise ValueError("R2v E3 cannot qualify without original E2")
        e2 += int(row["original_E2_qualified"])
        e3 += int(row["original_E3_qualified"])
        invocations += int(row.get("actual_native_invocation_receipt") is True)
    if (original.get("native_CSim_E2_qualified_cases") != e2
            or original.get("native_CSynth_E3_qualified_cases") != e3
            or original.get("original_vitis_invocation_receipts_present") != invocations):
        raise ValueError("R2v E2/E3/invocation original denominator mismatch")
    expected_ids = set()
    status_rows = []
    for factor in FACTOR_NAMES:
        for gi in range(len(GRAPHS)):
            for n,arm in enumerate(ARMS):
                if arm[FACTOR_NAMES.index(factor)] != FACTORS[factor][0]:
                    continue
                adjacent = list(arm)
                adjacent[FACTOR_NAMES.index(factor)] = FACTORS[factor][1]
                partner = ARMS.index(tuple(adjacent))
                expected_ids.add((factor,gi,f"case_{gi*16+n:03d}",
                                  f"case_{gi*16+partner:03d}"))
    if len(expected_ids) != 64:
        raise AssertionError("R2v invalid planned 64 contrast denominator")
    seen_ids = set()
    for c in contrasts:
        key = (c.get("factor"), c.get("graph_index"),
               c.get("arm_a"), c.get("arm_b"))
        if key not in expected_ids or key in seen_ids:
            raise ValueError("R2v wrong, missing or duplicate paired HLS comparison")
        seen_ids.add(key)
        ca, cb = cases[int(c["arm_a"].split("_")[1])], cases[int(c["arm_b"].split("_")[1])]
        if (c.get("matched_original_CSim_qualified") is not
                    (ca["original_E2_qualified"] and cb["original_E2_qualified"])
                or c.get("matched_original_CSynth_qualified") is not
                    (ca["original_E3_qualified"] and cb["original_E3_qualified"])):
            raise ValueError("R2v forged qualified pair status")
        delta = c.get("estimated_resource_delta_b_minus_a")
        if not isinstance(delta, dict) or set(delta) != set(RESOURCES):
            raise ValueError("R2v missing complete estimated resource vector")
        qualifies = c["matched_original_CSynth_qualified"]
        for resource in RESOURCES:
            actual = delta[resource]
            a = ca["estimated_resources"][resource]
            b = cb["estimated_resources"][resource]
            expected = b-a if qualifies and a is not None and b is not None else None
            if actual != expected or (
                    actual is not None and type(actual) is not int):
                raise ValueError("R2v resource delta does not match original Vitis cases")
        cycles = c["estimated_latency_worst_cycles_delta_b_minus_a"]
        expected_cycles = (
            cb["hls_worst_cycles"] - ca["hls_worst_cycles"]
            if qualifies else None
        )
        if cycles != expected_cycles:
            raise ValueError("R2v HLS cycle delta does not match original cases")
        if c.get("physical_resource_delta") is not None or \
                c.get("achieved_Fmax_difference") is not None:
            raise ValueError("R2v HLS evidence illegally promoted to physical QoR")
        status_rows.append({
            "graph_index": c["graph_index"],
            "factor": c["factor"],
            "arm_a": c["arm_a"], "arm_b": c["arm_b"],
            "e2_both_qualified": c["matched_original_CSim_qualified"],
            "e3_both_qualified": qualifies,
            "classification": _classify(c),
            "bram_18k_b_minus_a": delta["bram_18k"],
            "lut_b_minus_a": delta["lut"],
            "ff_b_minus_a": delta["ff"],
            "dsp_b_minus_a": delta["dsp"],
            "uram_b_minus_a": delta["uram"],
            "worst_cycles_b_minus_a": cycles,
        })
    if seen_ids != expected_ids:
        raise ValueError("R2v paired factor population incomplete")
    by_factor = []
    for factor in FACTOR_NAMES:
        for gi in range(len(GRAPHS)):
            group = [r for r in status_rows
                     if r["factor"] == factor and r["graph_index"] == gi]
            if len(group) != 8:
                raise AssertionError("R2v expected eight conditional pairs per graph/factor")
            counts = {key: sum(r["classification"] == key for r in group)
                      for key in (
                        "REDUCED_HLS_ESTIMATED_BRAM",
                        "TIED_HLS_ESTIMATED_BRAM",
                        "INCREASED_HLS_ESTIMATED_BRAM",
                        "UNKNOWN_NOT_BOTH_E3",
                        "UNKNOWN_NO_BRAM_ESTIMATE",
                      )}
            if sum(counts.values()) != 8:
                raise AssertionError("R2v missing conditional pair outcome")
            changes = [r["bram_18k_b_minus_a"] for r in group
                       if type(r["bram_18k_b_minus_a"]) is int]
            by_factor.append({
                "factor": factor, "graph_index": gi,
                "planned_conditional_pairs": 8,
                "qualified_bram_comparisons": len(changes),
                "status_counts": counts,
                "bram_18k_b_minus_a_sum": sum(changes) if changes else None,
                "bram_18k_b_minus_a_mean":
                    sum(changes)/len(changes) if changes else None,
                "physical_bram_measurement": None,
            })
    failed = [
        {"case": row["case"], "graph_index": row["graph_index"],
         "status": row["qualification_status"],
         "factors": {name: row[name] for name in FACTOR_NAMES}}
        for row in cases if not row["original_E3_qualified"]
    ]
    return {
        "schema": SCHEMA,
        "original_source_revision": expected_source_revision,
        "original_vitis_invocation_receipts_present": invocations,
        "original_E2_CSim_qualified_cases": e2,
        "original_E3_CSynth_qualified_cases": e3,
        "original_E3_unqualified_cases": CASES-e3,
        "original_E3_unqualified_case_details": failed,
        "prespecified_conditional_HLS_factor_pairs": len(status_rows),
        "E3_qualified_conditional_factor_pairs":
            sum(r["e3_both_qualified"] for r in status_rows),
        "E3_unknown_conditional_factor_pairs":
            sum(not r["e3_both_qualified"] for r in status_rows),
        "by_factor_and_graph": by_factor,
        "contrasts": status_rows,
        "new_original_Vitis_calls_during_review": 0,
        "new_original_Vivado_calls_during_review": 0,
        "physical_BRAM": None, "physical_Fmax": None,
        "limits": [
            "Only 2 fixed model graphs and 32 original 2026-11-03 HLS cases",
            "Source-attested Vitis E2 and E3 estimates, never routed E4",
            "Binding pragma and padding are requested source interventions",
            "Missing, failed or nonnumeric E3 outcomes remain UNKNOWN, not ties",
            "No causal claim beyond same-graph conditional source contrast",
            "No independent host/native recompilation in this read-only review",
        ],
    }


def _original_report_provenance(study: Path, original: dict) -> list[dict]:
    """Read-only SHA audit of CSynth XML present even for failed-clock E3.

    Each retained XML must match the original Vitis invocation's
    freshness hash to be called source-attested. Missing or unattested
    reports are never silently counted as qualified measurements.
    """
    result=[]
    for case in original["cases"]:
        folder=study/case["case"]
        xml=folder/"fpgai_hls_project"/"solution1"/"syn"/"report"/"fpgai_model_csynth.xml"
        receipt=folder/"vitis_invocation.json"
        if xml.is_symlink() or receipt.is_symlink():
            raise ValueError("R2v refuses symlinked original HLS evidence")
        observed=None
        attested=False
        if xml.is_file():
            observed=sha256(xml.read_bytes()).hexdigest()
            if receipt.is_file():
                data=json.loads(receipt.read_text())
                fresh=data.get("fresh_report_sha256") or {}
                attested=(fresh.get("csynth_xml")==observed)
        if case["original_E3_qualified"] and not attested:
            raise ValueError("R2v E3 claimed without matching source-attested CSynth XML")
        result.append({
            "case":case["case"],
            "raw_csynth_xml_present":observed is not None,
            "raw_csynth_xml_sha256":observed,
            "original_invocation_sha_attested":attested,
            "e3_qualified":case["original_E3_qualified"],
        })
    return result


def run(*, study: str | Path, output: str | Path,
        expected_source_revision: str=ORIGINAL_R2U_SOURCE_REVISION) -> dict:
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R2v read-only review output must be create-only")
    study_path = Path(study)
    if study_path.is_symlink() or not study_path.is_dir():
        raise ValueError("R2v original R2u source folder missing or symlinked")
    # Use original SHA check and regenerated C++ identity for all arms;
    # original SHA differs from current checkout once future docs/code commit.
    original = audit_native(
        study=study_path,require_current_source=False
    )
    report = review(original,expected_source_revision=expected_source_revision)
    report["raw_csynth_report_provenance"] = _original_report_provenance(
        study_path,original)
    report["original_attested_csynth_xml_reports_present"] = sum(
        item["original_invocation_sha_attested"]
        for item in report["raw_csynth_report_provenance"]
    )
    root.mkdir(parents=True)
    with (root/"summary.json").open("x",encoding="utf-8") as out:
        json.dump(report,out,sort_keys=True,indent=2)
        out.write("\n")
    with (root/"contrasts.csv").open("x",newline="",encoding="utf-8") as out:
        writer = csv.DictWriter(out,fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(report["contrasts"])
    return {k:v for k,v in report.items() if k!="contrasts"}


def main(argv: list[str] | None=None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--expected-source-revision",
                        default=ORIGINAL_R2U_SOURCE_REVISION)
    args=parser.parse_args(argv)
    print(json.dumps(run(
        study=args.study,output=args.output,
        expected_source_revision=args.expected_source_revision,
    ),indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
