"""Source-pinned exhaustive *differential* CST schedule research experiment.

Enumerates every permutation of 6 distinct row-stage operations in five
independent small-domain configurations, including different numbers of
physical scratch row slots. Labels are computed by a constructive finite
language oracle, not by the primary checker or by hand-picked mutants.
All disagreements and first-counterexamples are retained; fail the campaign
only *after* writing its complete observations.
"""
from __future__ import annotations

import argparse
import csv
from itertools import permutations
import json
from pathlib import Path
import subprocess
from time import perf_counter_ns

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.cst_chain_campaign import make_chain
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.row_chain import resolve_row_chain
from fpgai_research.model_ir.temporal_schedule import ComputeEvent, check_schedule
from fpgai_research.validation.schedule_reference_oracle import (
    reference_accepts, reference_valid_sequences,
)

CASES = (
    (2, (3, 5, 4, 2), 1), (2, (3, 5, 4, 2), 2),
    (3, (3, 5, 2), 1), (3, (3, 5, 2), 2), (3, (3, 5, 2), 3),
)


def run(*, output: str | Path, domains: int = 5) -> dict:
    if type(domains) is not int or not 1 <= domains <= len(CASES):
        raise ValueError("domains must be 1..5")
    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Create-only independent oracle study exists")
    rows_out = []
    counts = []
    for case_id, (nrows, widths, slots) in enumerate(CASES[:domains]):
        graph = make_chain(nrows, widths)
        ir = resolve_row_chain(resolve(graph, ArchitectureRequest()))
        stages = len(ir.chain)
        oracle = reference_valid_sequences(nrows, stages, slots)
        candidates = tuple(
            ComputeEvent(r, s) for r in range(nrows) for s in range(stages))
        matched = accepted = false_accepts = false_rejects = 0
        for perm_id, proposal in enumerate(permutations(candidates)):
            expected = reference_accepts(proposal, oracle)
            start = perf_counter_ns()
            checked = check_schedule(ir, proposal, row_slots=slots)
            elapsed = perf_counter_ns() - start
            observed = checked["status"] == "VERIFIED"
            matched += int(expected == observed)
            accepted += int(expected)
            false_accepts += int(observed and not expected)
            false_rejects += int(expected and not observed)
            rows_out.append({
                "domain": case_id,
                "permutation_id": perm_id,
                "rows": nrows,
                "stages": stages,
                "slots": slots,
                "widths": "x".join(map(str, widths)),
                "reference_legal": expected,
                "actual_status": checked["status"],
                "false_accept": observed and not expected,
                "false_reject": expected and not observed,
                "first_counterexample_reason": (
                    checked.get("counterexample") or {}).get("reason"),
                "check_elapsed_ns": elapsed,
                "native_vitis_calls": 0,
            })
        counts.append({
            "domain": case_id, "rows": nrows, "stages": stages, "slots": slots,
            "tested": len(rows_out)-sum(x["tested"] for x in counts),
            "oracle_legal": accepted,
            "agreed": matched, "false_accepts": false_accepts,
            "false_rejects": false_rejects,
        })
    source = subprocess.run(["git", "rev-parse", "HEAD"],
                            capture_output=True, text=True, timeout=5)
    summary = {
        "schema": "fpgai-research/cst-independent-oracle-study/v1",
        "source_revision": source.stdout.strip() if source.returncode == 0 else None,
        "domains": counts,
        "complete_permutations": len(rows_out),
        "reference_legal": sum(x["oracle_legal"] for x in counts),
        "reference_illegal": len(rows_out)-sum(x["oracle_legal"] for x in counts),
        "false_accepts": sum(x["false_accepts"] for x in counts),
        "false_rejects": sum(x["false_rejects"] for x in counts),
        "all_legalities_agree": all(x["agreed"] == x["tested"] for x in counts),
        "original_vitis_calls": 0,
        "evidence": "E0-exhaustive-finite-differential",
        "limitations": [
            "Independent oracle and checker share the stated abstract atomic-stage specification",
            "Exhaustive only for three or fewer rows and four or fewer stages in tested six-event domains",
            "Does not establish behavior of arbitrary generated HLS, RTL or vendor hardware",
            "Does not prove priority/novelty versus other hardware compilers",
        ],
    }
    dest.mkdir(parents=True)
    with (dest/"cases.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    (dest/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--domains", type=int, default=5)
    args = p.parse_args(argv)
    print(json.dumps(run(output=args.output, domains=args.domains),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
