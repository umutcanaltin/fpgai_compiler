"""Read-only immutable provenance/trace audit for original prospective R2b E1.

No execution of GCC, Vitis or Vivado. Original input/source/report
digests are checked without regenerating or changing case outputs.
Publication novelty, hardware quality and source-level generalization
are not inferred from a passing audit.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

from fpgai_research.experiments.cst_r2b_source_campaign import SCHEMA
from fpgai_research.validation.r2b_interleaved_cpp import _decode_trace
from fpgai_research.model_ir.branch_matmul_contract import _hash
from dataclasses import asdict


def _blob(path:Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked original source study input: "+str(path))
    if path.stat().st_size>8*1024*1024:
        raise ValueError("Unbounded original source artifact")
    return path.read_bytes()


def audit(study: str | Path)->dict:
    root=Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Original E1 R2b study must be a real directory")
    report=json.loads(_blob(root/"summary.json"))
    rows=list(csv.DictReader(_blob(root/"cases.csv").decode().splitlines()))
    if (report.get("schema")!=SCHEMA or
            report.get("prespecified_cases")!=24 or
            len(rows)!=24 or
            report.get("original_gcc_compilations")!=24 or
            report.get("original_gcc_executions")!=24):
        raise ValueError("Original R2b GCC case denominator/summary corrupted")
    seen=set()
    for i,row in enumerate(rows):
        case=f"case_{i:03d}"
        folder=root/case
        if row.get("case")!=case or folder.is_symlink() or not folder.is_dir():
            raise ValueError("Original R2b source case order/directories changed")
        meta=json.loads(_blob(folder/"original_case.json"))
        verified=json.loads(_blob(folder/"verified_e1.json"))
        cpp=_blob(folder/"source.cpp")
        raw=_blob(folder/"original_gcc_trace.txt")
        source_sha=sha256(cpp).hexdigest()
        raw_sha=sha256(raw).hexdigest()
        if ((folder/"FAILED.json").exists() or
                meta.get("source_revision")!=report.get("source_revision") or
                row.get("source_revision")!=report.get("source_revision") or
                meta.get("case")!=case or
                meta.get("source_cpp_sha256")!=source_sha or
                meta.get("source_manifest",{}).get("cpp_sha256")!=source_sha or
                row.get("source_cpp_sha256")!=source_sha or
                verified.get("source_cpp_sha256")!=source_sha or
                verified.get("raw_original_gcc_stdout_sha256")!=raw_sha or
                row.get("original_gcc_stdout_sha256")!=raw_sha or
                verified.get("runtime_input_sha256")!=meta.get("input_float32_sha256") or
                row.get("input_sha256")!=meta.get("input_float32_sha256") or
                verified.get("status")!=
                    "E1_ORIGINAL_GCC_SOURCE_EVENT_AND_NUMERIC_VERIFIED" or
                verified.get("original_gcc_executions")!=1 or
                row.get("source_executed_e1")!="True"):
            raise ValueError("Original R2b source/trace/numerical lineage mismatch "+case)
        # Read actual saved stdout and recompute the observed trace hash.
        outputs=int(verified["original_source_output_elements"])
        events,y=_decode_trace(raw.decode(),n_outputs=outputs,max_events=100_000)
        if (_hash([asdict(e) for e in events])!=
                    verified.get("observed_event_sha256") or
                row.get("actual_source_event_sha256")!=
                    verified.get("observed_event_sha256")):
            raise ValueError("Original R2b source observed event trace corrupted "+case)
        h=meta["source_manifest"]["scratch_offsets"]
        # The H identity and logical element count derive from the fixed
        # source model fixture, not from user-controlled text in stdout.
        if set(h)!={"a","b","g","h"}:
            raise ValueError("Unexpected bounded source scratch tensors")
        last={}
        for j,ev in enumerate(events):
            if ev.kind=="READ" and ev.tensor=="h":
                last[ev.element]=j
        if (_hash(tuple(sorted(last.items())))!=
                    verified.get("observed_h_last_use_sha256") or
                row.get("actual_source_h_last_use_sha256")!=
                    verified.get("observed_h_last_use_sha256") or
                [list(x) for x in sorted(last.items())]!=
                    [list(x) for x in verified["observed_h_last_use_indices"]]):
            raise ValueError("Original R2b observed last-reader release corrupted "+case)
        read_rank,final_reader=0,{}
        for ev in events:
            if ev.kind=="READ" and ev.tensor=="h":
                final_reader[ev.element]=(
                    read_rank,ev.consumer,ev.output_element)
                read_rank+=1
        signature=tuple(sorted((key,*identity) for key,identity
                               in final_reader.items()))
        if (_hash(signature)!=verified.get(
                    "observed_h_final_read_rank_and_owner_sha256") or
                row.get("actual_source_h_final_read_only_sha256")!=
                    verified.get("observed_h_final_read_rank_and_owner_sha256") or
                [list(row) for row in signature]!=[
                    list(row) for row in verified[
                        "observed_h_final_read_rank_and_owner"]]):
            raise ValueError("Original R2b H-only final-reader identity/rank corrupted "+case)
        slot=(row["shape"],row["policy"],row["layout"])
        if slot in seen:
            raise ValueError("Duplicate original R2b source case")
        seen.add(slot)
    return {
        "schema":"fpgai-research/cst-R2b-original-E1-artifact-audit/v1",
        "source_revision":report["source_revision"],
        "original_cases_verified":len(rows),
        "original_gcc_compilations_in_receipts":len(rows),
        "all_case_cpp_and_stdout_sha256_verified":True,
        "original_stdout_events_redecoded":True,
        "source_executed_last_use_recomputed":True,
        "independent_H_only_read_rank_and_last_reader_recomputed":True,
        "original_vendor_tool_calls_during_audit":0,
        "source_execution_valid_for_bounded_graphs":True,
        "rl_advantage_proven":False,
        "physical_hardware_qor_proven":False,
        "external_compiler_firstness_proven":False,
        "limitations":[
            "Static receipts plus reparse are not fresh recompilation",
            "Audit does not independently re-run NumPy numerical calculations",
            "The original source-runner still controls input and compiled source",
            "Fixed complete-design MatMul graph, not arbitrary source or FPGA implementation",
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",required=True)
    args=p.parse_args(argv)
    print(json.dumps(audit(args.study),sort_keys=True,indent=2))


if __name__=="__main__":
    main()
