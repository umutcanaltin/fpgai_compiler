"""R2o independent read-only audit of original 2..4 fanout C3 GCC receipts.

Checks complete prespecified case/action counts, input/source identity,
original source bytes, compiler manifest, original GCC stdout digests, and
E0/E1/UNKNOWN/REJECTED status separation. It does not replay executable
GCC C++ or independently establish its numeric semantics or FPGA timing.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re

import numpy as np

from fpgai_research.experiments.cst_r2o_multi_reader_action_study import (
    ACTION_ROLES, CATALOG, SCHEMA as STUDY_SCHEMA,
)

SCHEMA = "fpgai-research/cst-R2o-original-C3-GCC-read-only-audit/v1"
ACTION_STATUSES = (
    "REJECTED", "UNKNOWN_E1_NOT_RUN",
    "FAILED_E1_SOURCE_EXECUTION", "E1_GCC_MULTI_READER_QUALIFIED",
)


def _sha(value, length=64):
    return isinstance(value, str) and re.fullmatch(
        rf"[0-9a-f]{{{length}}}", value) is not None


def _load(root: Path, name: str) -> dict:
    f = root / name
    if f.is_symlink() or not f.is_file():
        raise ValueError("R2o original source artifact missing or symlinked: " + name)
    obj = json.loads(f.read_text())
    if not isinstance(obj, dict):
        raise ValueError("R2o JSON evidence must be an object")
    return obj


def audit(study: str | Path) -> dict:
    root = Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R2o original study missing or symlinked")
    summary = _load(root, "summary.json")
    count = summary.get("graph_count")
    revision = summary.get("original_source_revision")
    if (summary.get("schema") != STUDY_SCHEMA
            or type(count) is not int or not 1 <= count <= len(CATALOG)
            or not _sha(revision, 40)
            or summary.get("actions_per_graph") != 4
            or summary.get("original_Vitis_calls") != 0
            or summary.get("original_Vivado_calls") != 0
            or summary.get("trained_RL_policies") != 0
            or summary.get("prespecified_graphs") != [
                {"fanout_readers": r, "shape": list(d)}
                for r,d in CATALOG[:count]]):
        raise ValueError("R2o source cohort/source SHA is invalid")
    counts = {}
    actual_attempts = 0
    accepted = 0
    per_graph = []
    for i, (readers, shape) in enumerate(CATALOG[:count]):
        folder = root / f"graph_{i:03d}"
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("R2o original graph folder missing or symlinked")
        data = _load(folder, "receipt.json")
        if (data.get("graph_case") != f"graph_{i:03d}"
                or data.get("fanout_readers") != readers
                or data.get("shape") != list(shape)
                or data.get("original_source_revision") != revision
                or not _sha(data.get("initial_source_certificate_sha256"))
                or data.get("action_roles") != list(ACTION_ROLES)):
            raise ValueError("R2o original graph/action treatment altered")
        m,k,_,_ = shape
        from fpgai_research.experiments.cst_r2b_source_campaign import _input
        reference = np.asarray(_input(m,k), dtype=np.float32)
        input_sha = sha256(reference.tobytes()).hexdigest()
        if data.get("input_float32_sha256") != input_sha:
            raise ValueError("R2o fixed original GCC input SHA mismatch")
        evidence = data.get("evidence")
        if (not isinstance(evidence, dict)
                or evidence.get("schema")
                    != "fpgai-research/cst-R2o-source-qualified-multi-reader-mask/v1"
                or evidence.get("fanout_readers") != readers
                or not _sha(evidence.get("parent_state_sha256"))
                or evidence.get("candidate_count") != 4
                or evidence.get("original_vitis_calls") != 0
                or evidence.get("original_vivado_calls") != 0
                or evidence.get("rl_policies_trained") != 0
                or evidence.get("hardware_feasibility_checked") is not False
                or not isinstance(evidence.get("rows"), list)
                or len(evidence["rows"]) != 4):
            raise ValueError("R2o original source action evidence malformed")
        expect_actions = (
            ("operator", "reader0", "pes", 2),
            ("operator", f"reader{readers-1}", "pes", 2),
            ("operator", "producer", "pes", 2),
            ("memory", "model", "layout", "per_element_optimal"),
        )
        statuses = []
        e0_values = []
        e1_values = []
        e1_count = 0
        receipt_counts = []
        for n, row in enumerate(evidence["rows"]):
            if (row.get("index") != n
                    or not isinstance(row.get("action"), dict)
                    or tuple(row["action"].get(k) for k in
                             ("scope","target","field","value")) != expect_actions[n]
                    or row.get("status") not in ACTION_STATUSES
                    or type(row.get("source_admissible")) is not bool
                    or type(row.get("e0_certified")) is not bool
                    or type(row.get("e1_qualified")) is not bool):
                raise ValueError("R2o original typed action or status malformed")
            status = row["status"]
            counts[status] = counts.get(status, 0)+1
            statuses.append(status)
            e0_values.append(row["e0_certified"])
            e1_values.append(row["e1_qualified"])
            if status == "REJECTED":
                if (row["source_admissible"] or row["e0_certified"]
                        or row["e1_qualified"] or row.get("gcc_receipt") is not None
                        or row.get("state_sha256") is not None
                        or row.get("cpp_sha256") is not None
                        or not row.get("reason")):
                    raise ValueError("R2o rejected original action misclassified")
            else:
                if (not row["source_admissible"]
                        or not row["e0_certified"]
                        or not _sha(row.get("state_sha256"))
                        or not _sha(row.get("cpp_sha256"))
                        or not _sha(row.get("certificate_sha256"))):
                    raise ValueError("R2o E0-admitted original action lacks certificate")
            if status in ("UNKNOWN_E1_NOT_RUN", "FAILED_E1_SOURCE_EXECUTION"):
                if row["e1_qualified"] or row.get("gcc_receipt") is not None:
                    raise ValueError("R2o unknown or failed GCC action promoted to E1")
                if not row.get("reason"):
                    raise ValueError("R2o absent reason for unqualified candidate")
            if status in ("FAILED_E1_SOURCE_EXECUTION", "E1_GCC_MULTI_READER_QUALIFIED"):
                e1_count += 1
            if status == "E1_GCC_MULTI_READER_QUALIFIED":
                if row["e1_qualified"] is not True:
                    raise ValueError("R2o qualified GCC action missing E1 certification")
                raw = row.get("gcc_receipt")
                if not isinstance(raw, dict):
                    raise ValueError("R2o original GCC receipt absent")
                for name in ("source_cpp_sha256", "original_stdout_sha256",
                             "original_input_sha256",
                             "original_runtime_source_event_sha256"):
                    if not _sha(raw.get(name)):
                        raise ValueError("R2o original GCC source receipt digest absent")
                if (raw.get("source_cpp_sha256") != row.get("cpp_sha256")
                        or raw["original_input_sha256"] != input_sha
                        or type(raw.get("original_actual_address_witness_count")) is not int
                        or raw["original_actual_address_witness_count"] < 1):
                    raise ValueError("R2o original source GCC qualification mismatch")
                for name in (
                    "original_all_intermediate_numeric_max_absolute_error",
                    "original_final_numeric_max_absolute_error"):
                    value = raw.get(name)
                    if type(value) not in (int,float) or not isfinite(value) or value < 0:
                        raise ValueError("R2o original numeric measurement malformed")
                source_name = f"case_{n:03d}_source.cpp"
                manifest_name = f"case_{n:03d}_manifest.json"
                stdout_name = f"case_{n:03d}_original_stdout.txt"
                if (row.get("source_file") != source_name
                        or row.get("source_manifest_file") != manifest_name
                        or row.get("original_stdout_file") != stdout_name):
                    raise ValueError("R2o original source paths changed")
                for file_name in (source_name, stdout_name):
                    p = folder / file_name
                    if p.is_symlink() or not p.is_file():
                        raise ValueError("R2o actual source/GCC stdout missing or symlinked")
                source_sha = sha256((folder/source_name).read_bytes()).hexdigest()
                stdout_sha = sha256((folder/stdout_name).read_bytes()).hexdigest()
                manifest = _load(folder, manifest_name)
                if (source_sha != raw["source_cpp_sha256"]
                        or stdout_sha != raw["original_stdout_sha256"]
                        or manifest.get("source_cpp_sha256") != source_sha
                        or manifest.get("source_layout")
                            != ("per_element_optimal" if n==3 else "disjoint")):
                    raise ValueError("R2o actual GCC stdout/source SHA or layout mismatch")
                receipt_counts.append(raw["original_actual_address_witness_count"])
                accepted += 1
            elif row["e1_qualified"] or row.get("gcc_receipt") is not None:
                raise ValueError("R2o nonqualified action has forged GCC success")
        if (statuses != evidence.get("statuses")
                or e0_values != evidence.get("e0_certified")
                or e1_values != evidence.get("e1_qualified")
                or evidence.get("gcc_e1_attempts") != e1_count):
            raise ValueError("R2o per-graph action denominator/status mismatch")
        actual_attempts += e1_count
        per_graph.append({
            "graph": i, "fanout_readers": readers,
            "outcome_statuses": statuses,
            "gcc_address_witness_counts": receipt_counts,
        })
    if (summary.get("original_action_denominator") != count*4
            or summary.get("status_denominators") != counts
            or summary.get("original_GCC_source_qualification_attempts")
                != actual_attempts
            or summary.get("original_GCC_qualified_actions") != accepted
            or summary.get("rejected_actions") != counts.get("REJECTED",0)
            or summary.get("unknown_actions") != counts.get("UNKNOWN_E1_NOT_RUN",0)
            or summary.get("failed_e1_actions")
                != counts.get("FAILED_E1_SOURCE_EXECUTION",0)):
        raise ValueError("R2o original source summary population/status mismatch")
    return {
        "schema": SCHEMA,
        "original_source_revision": revision,
        "original_graphs": count,
        "original_actions": count*4,
        "observed_statuses": counts,
        "source_GCC_e1_attempts": actual_attempts,
        "source_GCC_qualified_actions": accepted,
        "original_vendor_calls_during_audit": 0,
        "RL_training_calls": 0,
        "per_graph": per_graph,
        "evidence": "read-only source/GCC transcript SHA and outcome audit",
        "limits": [
            "Not a second GCC execution or independent numeric recomputation",
            "No original Vitis/Vivado, RTL, physical BRAM or trained RL",
            "Only original 2..4-reader MatMul + Add + Relu source domain",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output")
    a = parser.parse_args(argv)
    result = audit(a.study)
    if a.output is not None:
        dest = Path(a.output)
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("R2o read-only audit output strictly create-only")
        with dest.open("x", encoding="utf-8") as out:
            json.dump(result, out, indent=2, sort_keys=True)
            out.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
