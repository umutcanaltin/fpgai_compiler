"""Read-only integrity audit of the prospective R2n C3 GCC action study.

This checks case completeness, result-status/denominator honesty and saved
SHA lineage. It does NOT independently rerun original GCC, recompute the
numeric reference or prove certificates; the original executed source and
its provenance must be inspected separately for such a stronger claim.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re

import numpy as np

from fpgai_research.experiments.cst_r2n_e1_action_study import (
    ACTION_ROLES, CATALOG, SCHEMA as STUDY_SCHEMA,
)

SCHEMA = "fpgai-research/cst-R2n-e1-action-source-receipt-audit/v1"
STATUSES = (
    "REJECTED",
    "UNKNOWN_E0_CERTIFICATE",
    "FAILED_E1_SOURCE_EXECUTION",
    "E1_GCC_SOURCE_QUALIFIED",
)


def _sha(value):
    return (isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def audit(study: str | Path) -> dict:
    root = Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R2n original study directory missing or symlinked")
    files = (root / "summary.json", root / "case_evidence.json")
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError("R2n source summary or case evidence missing/symlinked")
    summary = json.loads(files[0].read_text())
    records = json.loads(files[1].read_text())
    if (summary.get("schema") != STUDY_SCHEMA
            or type(summary.get("graphs")) is not int
            or not 1 <= summary["graphs"] <= len(CATALOG)
            or not isinstance(records, list)
            or len(records) != summary["graphs"]
            or summary.get("graph_depths")
               != [x[0] for x in CATALOG[:summary["graphs"]]]
            or not isinstance(summary.get("source_revision"), str)
            or re.fullmatch(r"[0-9a-f]{40}",
                            summary["source_revision"]) is None
            or summary.get("native_vitis_calls") != 0
            or summary.get("new_vivado_calls") != 0
            or summary.get("rl_policies_trained") != 0):
        raise ValueError("R2n study schema, source pin or original cohort invalid")
    counts = {}
    observed = []
    e1_attempts = 0
    for record, (expected_depth, expected_rows, widths) in zip(
            records, CATALOG):
        if (record.get("graph_depth") != expected_depth
                or record.get("roles") != list(ACTION_ROLES)
                or type(record.get("e1_attempts")) is not int
                or record["e1_attempts"] < 0
                or record.get("native_vitis_calls") != 0
                or record.get("rl_policies_trained") != 0
                or not _sha(record.get("fixed_input_sha256"))
                or not _sha(record.get("original_state_sha256"))
                or not isinstance(record.get("actions"), list)
                or len(record["actions"]) != 3):
            raise ValueError("R2n incomplete graph/action identities")
        expected_input_sha = sha256(np.linspace(
            -.125, .125, expected_rows * widths[0],
            dtype=np.float32).tobytes()).hexdigest()
        if record["fixed_input_sha256"] != expected_input_sha:
            raise ValueError("R2n original deterministic GCC input SHA disagrees")
        e1_attempts += record["e1_attempts"]
        status_list = []
        for index, case in enumerate(record["actions"]):
            status = case.get("status")
            if (case.get("action_index") != index
                    or status not in STATUSES
                    or type(case.get("source_admissible")) is not bool
                    or type(case.get("e0_mapped_certified")) is not bool
                    or type(case.get("e1_host_source_qualified")) is not bool):
                raise ValueError("R2n action result malformed")
            action = case.get("action")
            if not isinstance(action, dict):
                raise ValueError("R2n typed action missing")
            if index in (0, 1):
                if (action.get("scope") != "operator"
                        or action.get("field") != "pes"
                        or action.get("value") != (2 if index == 0 else 0)):
                    raise ValueError("R2n original Phi treatment changed")
            elif (action.get("scope") != "exchange"
                    or action.get("target") != "model"
                    or action.get("field") != "exchange_policy"
                    or action.get("value") != "whole_tensor"):
                raise ValueError("R2n original exchange control changed")
            present_source = (case.get("source_admissible") is True
                              and _sha(case.get("state_sha256"))
                              and _sha(case.get("cpp_sha256")))
            if status == "REJECTED":
                if (case["source_admissible"]
                        or case["e0_mapped_certified"]
                        or case["e1_host_source_qualified"]
                        or case.get("state_sha256") is not None
                        or case.get("cpp_sha256") is not None
                        or case.get("qualification_sha256") is not None
                        or case.get("source_trace_sha256") is not None
                        or not case.get("reason")):
                    raise ValueError("R2n rejected action carries forged source acceptance")
            elif status == "UNKNOWN_E0_CERTIFICATE":
                if (not present_source or case["e0_mapped_certified"]
                        or case["e1_host_source_qualified"]
                        or case.get("qualification_sha256") is not None
                        or case.get("source_trace_sha256") is not None
                        or not case.get("reason")):
                    raise ValueError("R2n unknown E0 action promoted to certified")
            elif status == "FAILED_E1_SOURCE_EXECUTION":
                if (not present_source or
                        not case["e0_mapped_certified"] or
                        case["e1_host_source_qualified"] or
                        case.get("qualification_sha256") is not None or
                        not case.get("reason")):
                    raise ValueError("R2n original failed E1 action misclassified")
            else:
                if (not present_source or
                        not case["e0_mapped_certified"] or
                        not case["e1_host_source_qualified"] or
                        not _sha(case.get("qualification_sha256")) or
                        not _sha(case.get("source_trace_sha256")) or
                        case.get("host_input_sha256")
                            != record["fixed_input_sha256"] or
                        type(case.get("last_reader_releases_verified"))
                            is not int or
                        case["last_reader_releases_verified"] < 1):
                    raise ValueError("R2n E1 success lacks original GCC evidence identity")
                for key in ("host_max_absolute_error", "host_rmse"):
                    value = case.get(key)
                    if (type(value) not in (int, float)
                            or not isfinite(value) or value < 0):
                        raise ValueError("R2n invalid GCC numerical error measurement")
            counts[status] = counts.get(status, 0) + 1
            status_list.append(status)
        if record["e1_attempts"] != sum(
                x in ("FAILED_E1_SOURCE_EXECUTION", "E1_GCC_SOURCE_QUALIFIED")
                for x in status_list):
            raise ValueError("R2n original GCC attempt count disagrees with statuses")
        observed.append({
            "graph_depth": record["graph_depth"],
            "status": status_list,
            "e1_attempts": record["e1_attempts"],
        })
    if (summary.get("all_candidate_actions") != 3 * len(records)
            or summary.get("status_denominators") != counts
            or summary.get("e1_host_executions_attempted") != e1_attempts
            or summary.get("e1_verified_actions") != counts.get(
                "E1_GCC_SOURCE_QUALIFIED", 0)
            or summary.get("source_only_uncertified_actions") != counts.get(
                "UNKNOWN_E0_CERTIFICATE", 0)
            or summary.get("rejected_actions") != counts.get("REJECTED", 0)
            or summary.get("failed_e1_actions") != counts.get(
                "FAILED_E1_SOURCE_EXECUTION", 0)):
        raise ValueError("R2n summary misreports original complete action denominator")
    return {
        "schema": SCHEMA,
        "source_revision": summary["source_revision"],
        "original_graphs": len(records),
        "original_action_cases": 3 * len(records),
        "statuses": counts,
        "source_e1_qualification_attempts": e1_attempts,
        "source_e1_qualifications": counts.get("E1_GCC_SOURCE_QUALIFIED", 0),
        "case_receipts": observed,
        "native_vendor_calls_during_audit": 0,
        "rl_policies_trained": 0,
        "evidence": "Read-only JSON consistency/source SHA audit, NOT replayed GCC",
        "limitations": [
            "This read-only audit does not independently execute C++",
            "Local source and host trace attestations remain bounded evidence",
            "No original Vitis/Vivado or physical feasibility claim",
            "Policy-neutral certificate interface, not learned RL performance",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    record = audit(args.study)
    if args.output:
        out = Path(args.output)
        if out.exists() or out.is_symlink():
            raise FileExistsError("R2n read-only audit output create-only")
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
