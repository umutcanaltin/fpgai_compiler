"""Generate manuscript-ready tables only from validated E0/E1 experiment receipts.

Do not turn generated structural evidence into invented HLS cycles/area.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

_AXES = ("precision_bits", "parallelism", "pes", "simd", "serialization", "tile", "pipeline_ii")


def semantic_table_rows(receipt: dict) -> list[dict]:
    if receipt.get("schema") != "fpgai-research/scientific-model-ir-study/v1":
        raise ValueError("Expected model-IR scientific receipt schema v1")
    cases = [receipt.get("baseline"), *receipt.get("seven_dimension_ablations", []),
             receipt.get("scoped_refinement")]
    if (len(cases) != 9 or any(not isinstance(c, dict) or c.get("status") != "PASS" or
            c.get("functional_sha256") != receipt.get("functional_sha256") or
            not c.get("stage_invariants_verified") for c in cases)):
        raise ValueError("Missing or mismatched Model IR stage-preservation evidence")
    ablations = receipt["seven_dimension_ablations"]
    if tuple(c.get("name") for c in ablations) != _AXES:
        raise ValueError("Missing or reordered seven-axis materialization evidence")
    if any(not c.get("mapping_changed_operators") or not c.get("lowered_changed_operators")
           or not c.get("cpp_sha256") for c in ablations):
        raise ValueError("Missing actual mapping/lowering/HLS evidence")
    blocked = receipt.get("expected_blockers", [])
    if len(blocked) != 3 or any(c.get("status") != "BLOCKED" or
                                not (c.get("blocker") or {}).get("class") for c in blocked):
        raise ValueError("Missing explicit blocked-architecture evidence")
    express = receipt.get("expression_ablation", {})
    if (express.get("global_only_distinct_mapping_pairs") != 3 or
            express.get("operator_local_distinct_mapping_pairs") != 9 or
            len({tuple(p["mapping_pair"]) for p in express.get("profiles", [])}) != 9):
        raise ValueError("Scope-expressiveness result lacks nine distinct Phi witnesses")
    if receipt.get("vendor_tools_run") is not False or receipt.get("fixed_point_bit_exact") is not False:
        raise ValueError("E0/E1 receipt must not claim vendor-tool or fixed-point results")
    host = [c.get("host_float") for c in cases]
    host_count = sum(isinstance(row, dict) and row.get("accepted") is True
                     and row.get("cpp_sha256") == case["cpp_sha256"]
                     for row, case in zip(host, cases))
    if any(row is not None for row in host) and host_count != len(cases):
        raise ValueError("Incomplete or mismatched optional GCC evidence")
    rows = [
        {"property": "Legal stage refinements preserving the Functional IR", "observed": "9/9", "level": "E0"},
        {"property": "Changed seven-axis mappings and Lowered IR mechanisms", "observed": "7/7", "level": "E0/E1"},
        {"property": "Explicitly blocked illegal requests", "observed": "3/3", "level": "E0"},
        {"property": "Global-only distinct PE mapping pairs", "observed": "3", "level": "E0"},
        {"property": "Operator-local distinct PE mapping pairs", "observed": "9", "level": "E0"},
    ]
    if host_count == 9:
        rows.append({"property": "GCC host-float checks (not fixed-point)", "observed": "9/9", "level": "host-float"})
    return rows


def export_semantic_tables(receipt_file, output_directory) -> list[dict]:
    source=Path(receipt_file)
    rows=semantic_table_rows(json.loads(source.read_text()))
    root=Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root/"model_ir_results.csv").open("w", newline="") as dest:
        writer=csv.DictWriter(dest,fieldnames=("property","observed","level"))
        writer.writeheader()
        writer.writerows(rows)
    with (root/"model_ir_results.tex").open("w") as dest:
        dest.write("% Generated E0/E1 representation evidence, not FPGA performance.\n")
        dest.write("\\begin{table}[t]\n\\centering\n\\small\n")
        dest.write("\\caption{Bounded Model IR scientific evidence; no HLS synthesis results.}\n")
        dest.write("\\label{tab:model-ir-representation-generated}\n")
        dest.write("\\begin{tabular}{@{}p{0.63\\columnwidth}rl@{}}\n\\toprule\n")
        dest.write("Representation property & Observed & Evidence \\\\\n\\midrule\n")
        for row in rows:
            dest.write(f'{row["property"]} & {row["observed"]} & {row["level"]} \\\\\n')
        dest.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return rows
