"""Within-model E4 architecture ablation comparisons from an attested suite snapshot.

Input must be the collected suite_results.json produced by collect_suite_tables.
No native programs are run; this reuses only previously source-attributed data.
A cross-model MLP vs CNN ratio is *not* a controlled architecture ablation.
"""
from __future__ import annotations

import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory


SCHEMA = "fpgai-research/full-paper-suite-comparison/v1"
MEASUREMENTS = (
    ("hls_latency_best", "hls_latency_cycles"),
    ("hls_ii_min", "hls_ii_cycles"),
    ("postroute_lut", "lut"),
    ("postroute_ff", "ff"),
    ("postroute_dsp", "dsp"),
    ("postroute_bram", "bram_tiles"),
    ("postroute_uram", "uram"),
    ("postroute_wns_ns", "wns_ns"),
)
COLUMNS = (
    "model", "family", "case", "axis", "precision_bits", "evidence",
    "integer_bits", "fractional_bits", "numerical_tolerance",
    "csim_failures", "cosine", "rmse", "max_abs_error",
    "reference_top1", "compiled_top1",
    "comparison_status", "blocker", "postroute_implementation_mode",
    "numerical_pass", "postroute_timing_met",
    "postroute_drc_critical_warnings", "postroute_io_unconstrained",
    "class_disagreement_rate", "labeled_samples",
    *(f"baseline_{name}" for _, name in MEASUREMENTS),
    *(f"observed_{name}" for _, name in MEASUREMENTS),
    *(f"delta_{name}" for _, name in MEASUREMENTS),
)


def _number(value):
    if type(value) is bool or value is None or type(value) not in (int, float, str):
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    return x if isfinite(x) else None


def _qualifying(row):
    if row["evidence"] != "E4-Vivado-implementation":
        return "No source-attested E4 routed reports"
    if row.get("numerical_pass") is not True:
        return "Native fixed-point C simulation did not pass"
    if row.get("postroute_timing_met") is not True:
        return "Post-route timing not met"
    if row.get("postroute_implementation_mode") != "out_of_context":
        return "Implementation is not a matched out-of-context study"
    if row.get("postroute_drc_critical_warnings") != 0:
        return "DRC critical/error count missing or nonzero"
    if row.get("postroute_io_unconstrained") is not False:
        return "Unconstrained I/O DRC status is missing or positive"
    if not all(_number(row.get(field)) is not None for field, _ in MEASUREMENTS):
        return "Native HLS or routed metrics incomplete"
    return None


def compare_suite_tables(snapshot: dict) -> dict:
    if snapshot.get("schema") != "fpgai-research/full-paper-suite-tables/v1":
        raise ValueError("Expected attested full-paper-suite-tables/v1 snapshot")
    rows = snapshot.get("rows")
    if not isinstance(rows, list) or len(rows) != snapshot.get("case_count"):
        raise ValueError("Missing or inconsistent collected-case inventory")
    models = {}
    identities = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid suite table row")
        model, case = row.get("model"), row.get("case")
        if not isinstance(model, str) or not model or not isinstance(case, str) or not case:
            raise ValueError("Every measured case requires model and case identity")
        key = (model, case)
        if key in identities:
            raise ValueError("Duplicated model/case identity")
        identities.add(key)
        models.setdefault(model, []).append(row)
    out = []
    for model, study in models.items():
        baseline = next((row for row in study
                         if row["axis"] == "baseline" and row["case"] == "case_0000"), None)
        if baseline is None:
            raise ValueError(f"Missing identified baseline for {model}")
        if sum(row.get("axis") == "baseline" for row in study) != 1:
            raise ValueError(f"Ambiguous baseline for {model}")
        base_blocker = _qualifying(baseline)
        for row in study:
            blocker = _qualifying(row)
            # Reject heterogeneous physical methodology, even with all numbers.
            if blocker is None and base_blocker is None and (
                row["postroute_implementation_mode"]
                != baseline["postroute_implementation_mode"]
                or row.get("family") != baseline.get("family")
                or row.get("functional_sha256") != baseline.get("functional_sha256")
            ):
                blocker = "Model fingerprint, family or implementation mode differs from baseline"
            if base_blocker is not None:
                blocker = "Baseline not E4/numerically/physically qualified: " + base_blocker
            eligible = blocker is None
            record = {
                "model": model, "family": row.get("family"),
                "case": row["case"], "axis": row.get("axis"),
                "precision_bits": row.get("precision_bits"),
                "integer_bits": row.get("integer_bits"),
                "fractional_bits": row.get("fractional_bits"),
                "numerical_tolerance": row.get("numerical_tolerance"),
                "csim_failures": row.get("csim_failures"),
                "cosine": row.get("cosine"),
                "rmse": row.get("rmse"),
                "max_abs_error": row.get("max_abs_error"),
                "reference_top1": row.get("reference_top1"),
                "compiled_top1": row.get("compiled_top1"),
                "evidence": row.get("evidence"),
                "comparison_status": "QUALIFIED" if eligible else "INCOMPLETE",
                "blocker": blocker,
                "postroute_implementation_mode": row.get("postroute_implementation_mode"),
                "numerical_pass": row.get("numerical_pass"),
                "postroute_timing_met": row.get("postroute_timing_met"),
                "postroute_drc_critical_warnings": row.get("postroute_drc_critical_warnings"),
                "postroute_io_unconstrained": row.get("postroute_io_unconstrained"),
                "class_disagreement_rate": row.get("class_disagreement_rate"),
                "labeled_samples": row.get("labeled_samples"),
            }
            for field, name in MEASUREMENTS:
                # An E1-only record must never display fabricated/stale
                # HLS or physical values as observations, even if an
                # inconsistent table row happened to contain numeric data.
                # E3 can report measured HLS estimates, but only E4 can
                # report routed utilization and timing measurements.
                base_native=(baseline.get("evidence") in
                             ("E3-HLS-synthesis-report", "E4-Vivado-implementation")
                             if field.startswith("hls_") else
                             baseline.get("evidence") == "E4-Vivado-implementation")
                row_native=(row.get("evidence") in
                            ("E3-HLS-synthesis-report", "E4-Vivado-implementation")
                            if field.startswith("hls_") else
                            row.get("evidence") == "E4-Vivado-implementation")
                b=_number(baseline.get(field)) if base_native else None
                current=_number(row.get(field)) if row_native else None
                record[f"baseline_{name}"] = b if base_blocker is None else None
                record[f"observed_{name}"] = current
                record[f"delta_{name}"] = (
                    round(current - b, 9) if eligible and b is not None and current is not None
                    else None)
            out.append(record)
    exclusions={}
    for result in out:
        if result["comparison_status"]=="INCOMPLETE":
            reason=result["blocker"]
            exclusions[reason]=exclusions.get(reason,0)+1
    return {
        "schema": SCHEMA,
        "source_schema": snapshot["schema"],
        "source_suite_sha256": snapshot.get("suite_sha256"),
        "total_cases": len(out),
        "e4_routed_cases": sum(row["evidence"] == "E4-Vivado-implementation" for row in out),
        "qualified_cases": sum(row["comparison_status"] == "QUALIFIED" for row in out),
        "exclusion_reasons": exclusions,
        "rows": out,
        "limitations": (
            "Same-model OOC baseline-relative measurements only; no cross-model "
            "architecture rankings. Deltas are absolute, not percentages. "
            "HLS latency/II are synthesis estimates, postroute utilization/ "
            "WNS are Vivado measurements. BRAM means Vivado Block RAM tiles, "
            "not HLS BRAM_18K. Numerical failures use the case's original "
            "absolute-error acceptance threshold; a 16-sample CSim pass "
            "is not full-dataset fixed-point accuracy or board deployment."
        ),
    }


def export_suite_comparisons(tables_dir, output_dir) -> dict:
    source = Path(tables_dir)/"suite_results.json"
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("Need a regular collect-suite suite_results.json")
    data = source.read_bytes()
    snapshot = json.loads(data)
    result = compare_suite_tables(snapshot)
    result["source_table_sha256"] = sha256(data).hexdigest()
    dest = Path(output_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Refusing to overwrite scientific comparison snapshot")
    with TemporaryDirectory(prefix=".fpgai-comparison-", dir=dest.parent) as folder:
        stage = Path(folder)
        (stage/"comparisons.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
        with (stage/"comparisons.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows({column:("TBD" if row.get(column) is None else row[column])
                             for column in COLUMNS} for row in result["rows"])
        from fpgai_research.experiments.suite_outcomes import write_architecture_outcomes
        write_architecture_outcomes(result, stage)
        stage.rename(dest)
    return result
