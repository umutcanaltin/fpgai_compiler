"""R2w: provenance-checked, READ-ONLY HLS RTL/memory forensic inventory.

Original input: the frozen 32-case R2u Vitis HLS cohort, revision 0c6d85f.
Never run Vitis or Vivado. The old source/Tcl/testbench, invocation,
CSynth XML, generated HDL and ROM are read but NEVER modified.
Output goes to a new create-only R2w directory.

This is intentionally lexical and structural evidence, NOT a claim
that a source pragma became an FPGA RAMB primitive, that HLS estimates
match routed resource use, or that an array was retained physically.
Missing HDL is UNKNOWN, not proof that storage was optimized away.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

from fpgai_research.experiments.cst_r2u_hls_mechanism_execute import (
    audit_native,
)
from fpgai_research.experiments.cst_r2v_hls_factorial_review import (
    ORIGINAL_R2U_SOURCE_REVISION, review,
)
from fpgai_research.toolchains.vivado.rtl_inputs import inventory_vitis_rtl

SCHEMA = "fpgai-research/cst-R2w-source-attested-read-only-HLS-RTL-forensics/v1"
CASE_FIELDS = (
    "case","graph_index","storage","address","binding","padding",
    "E2_qualified","E3_qualified","HLS_estimated_BRAM_18K",
    "rtl_sha_attested","rtl_file_count","rom_file_count",
    "scratch_name_rtl_file_count","memory_name_rtl_file_count",
    "primitive_token_occurrences","report_scratch_line_count",
    "rtl_observation_status",
)
PAIR_FIELDS = (
    "factor","graph_index","arm_a","arm_b","original_E3_both_qualified",
    "both_RTL_sha_attested","observation_status","estimated_BRAM_18K_delta",
    "rtl_file_count_delta","scratch_name_file_count_delta",
    "primitive_token_occurrences_delta","report_scratch_line_count_delta",
)
# Strict lexical matching on ORIGINAL generated HDL. These words may
# occur in modules/comments or be design artifacts, not physical cells.
_MEMORY_NAME = re.compile(
    r"scratch|(?:^|[_\W])(?:ram|rom|bram|uram|mem|fifo)(?:[_\W]|$)", re.I
)
_PRIMITIVE = re.compile(
    r"\b(?:RAMB18E[12]|RAMB36E[12]|URAM288|"
    r"RAM32M|RAM64M|xpm_memory_[A-Za-z0-9_]+)\b", re.I
)
_REPORT_SCRATCH = re.compile(r"\bscratch\b|scratch[_\[\d]", re.I)
_MAX_TEXT_BYTES = 12*1024*1024
_MAX_SNIPPETS = 12


def _digest(file: Path) -> str:
    if file.is_symlink() or not file.is_file():
        raise ValueError("R2w invalid/symlinked original evidence path: "+str(file))
    return sha256(file.read_bytes()).hexdigest()


def _limited_text(file: Path) -> str:
    if file.is_symlink() or not file.is_file():
        raise ValueError("R2w refuses symlinked original HLS text")
    if file.stat().st_size > _MAX_TEXT_BYTES:
        raise ValueError("R2w original RTL/report exceeds bounded text scan")
    return file.read_text(encoding="utf-8",errors="replace")


def inspect_original_case(folder: Path, *, case: dict) -> dict:
    """Validate original Vitis receipt and list actual HDL, never execute it.

    No case evidence is modified; any missing RTL or absent receipt is
    represented as explicit unknown, not zero mapped RAM resources.
    """
    idx = case.get("case")
    if (not isinstance(idx,str)
            or not re.fullmatch(r"case_[0-9]{3}",idx)
            or folder.name != idx
            or folder.is_symlink()
            or not folder.is_dir()):
        raise ValueError("R2w original case identity/path mismatch")
    receipt_path=folder/"vitis_invocation.json"
    if receipt_path.is_symlink():
        raise ValueError("R2w refuses symlinked original Vitis invocation")
    if not receipt_path.exists():
        return {
            "case":idx,"rtl_sha_attested":False,
            "rtl_file_count":None,"rom_file_count":None,
            "scratch_name_rtl_file_count":None,
            "memory_name_rtl_file_count":None,
            "primitive_token_occurrences":None,
            "report_scratch_line_count":None,
            "rtl_observation_status":"UNKNOWN_NO_NATIVE_INVOCATION",
            "original_rtl_sha256":{},"original_rom_sha256":{},
            "scratch_name_rtl_files":[],
            "memory_name_rtl_files":[],
            "report_scratch_snippets":[],
            "raw_csynth_rpt_sha256":None,
        }
    receipt=json.loads(_limited_text(receipt_path))
    if (receipt.get("schema")!="fpgai-research/vendor-invocation/v1"
            or receipt.get("tool")!="vitis_hls"
            or receipt.get("source_sha256")!=case["source_cpp_sha256"]
            or not receipt.get("inputs_stable_during_execution")):
        raise ValueError("R2w invalid original Vitis invocation provenance")
    root=folder/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
    if root.is_symlink():
        raise ValueError("R2w refuses symlinked HLS Verilog source directory")
    if not root.exists():
        return {
            "case":idx,"rtl_sha_attested":False,
            "rtl_file_count":None,"rom_file_count":None,
            "scratch_name_rtl_file_count":None,
            "memory_name_rtl_file_count":None,
            "primitive_token_occurrences":None,
            "report_scratch_line_count":None,
            "rtl_observation_status":"UNKNOWN_NO_GENERATED_RTL",
            "original_rtl_sha256":{},"original_rom_sha256":{},
            "scratch_name_rtl_files":[],
            "memory_name_rtl_files":[],
            "report_scratch_snippets":[],
            "raw_csynth_rpt_sha256":None,
        }
    inventory=inventory_vitis_rtl(folder)
    found_rtl=inventory["rtl_sha256"]
    found_rom=inventory["rom_sha256"]
    attested=(receipt.get("rtl_output_sha256")==found_rtl
              and receipt.get("rom_output_sha256")==found_rom)
    if not attested:
        # A newly changed or extra HDL file means original-source
        # attribution cannot be made. Do not use its lexical counters.
        raise ValueError("R2w original RTL/ROM SHA differs from Vitis receipt")
    memory_files=[]
    scratch_files=[]
    primitive_count=0
    for name in sorted(found_rtl):
        file=root/name
        source=_limited_text(file)
        if _MEMORY_NAME.search(name):
            memory_files.append(name)
        if "scratch" in name.lower() or re.search(
                r"\bscratch(?:_|\b)",source,re.I):
            scratch_files.append(name)
        primitive_count+=len(_PRIMITIVE.findall(source))
    rpt=folder/"fpgai_hls_project"/"solution1"/"syn"/"report"/"fpgai_model_csynth.rpt"
    report_sha=None
    matches=[]
    if rpt.exists():
        report_sha=_digest(rpt)
        report_rec=(receipt.get("fresh_report_sha256") or {}).get("csynth_rpt")
        if report_rec!=report_sha:
            raise ValueError("R2w original CSynth text report SHA not attested")
        for line in _limited_text(rpt).splitlines():
            if _REPORT_SCRATCH.search(line):
                matches.append(line.strip()[:300])
    return {
        "case":idx,"rtl_sha_attested":True,
        "rtl_file_count":len(found_rtl),"rom_file_count":len(found_rom),
        "scratch_name_rtl_file_count":len(scratch_files),
        "memory_name_rtl_file_count":len(memory_files),
        "primitive_token_occurrences":primitive_count,
        "report_scratch_line_count":len(matches) if report_sha else None,
        "rtl_observation_status":"LEXICAL_HDLSOURCE_ONLY",
        "original_rtl_sha256":found_rtl,"original_rom_sha256":found_rom,
        "scratch_name_rtl_files":scratch_files,
        "memory_name_rtl_files":memory_files,
        "report_scratch_snippets":matches[:_MAX_SNIPPETS],
        "raw_csynth_rpt_sha256":report_sha,
    }


def analyze(*, study: str | Path) -> dict:
    root=Path(study)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("R2w frozen original R2u directory missing or symlinked")
    original=audit_native(study=root,require_current_source=False)
    factor=review(original,expected_source_revision=ORIGINAL_R2U_SOURCE_REVISION)
    if (original["original_vitis_invocation_receipts_present"]!=32
            or original["native_CSim_E2_qualified_cases"]!=32
            or original["native_CSynth_E3_qualified_cases"]!=30
            or factor["E3_qualified_conditional_factor_pairs"]!=57
            or factor["E3_unknown_conditional_factor_pairs"]!=7):
        raise ValueError("R2w original source-pinned E2/E3 factorial not recognized")
    cases=[]
    by_id={}
    for record in original["cases"]:
        observed=inspect_original_case(root/record["case"],case=record)
        row={
            **{k:record[k] for k in (
                "case","graph_index","storage","address","binding","padding"
            )},
            "E2_qualified":record["original_E2_qualified"],
            "E3_qualified":record["original_E3_qualified"],
            "HLS_estimated_BRAM_18K":record["estimated_resources"]["bram_18k"],
            **{k:v for k,v in observed.items() if k!="case"},
        }
        cases.append(row)
        by_id[record["case"]]=row
    contrasts=[]
    for c in factor["contrasts"]:
        a=by_id[c["arm_a"]];b=by_id[c["arm_b"]]
        both=bool(a["rtl_sha_attested"] and b["rtl_sha_attested"])
        status=("ORIGINAL_RTL_TEXT_ATTESTED" if both
                else "UNKNOWN_MISSING_OR_UNATTESTED_RTL")
        deltas={}
        for key,out in (
            ("rtl_file_count","rtl_file_count_delta"),
            ("scratch_name_rtl_file_count","scratch_name_file_count_delta"),
            ("primitive_token_occurrences","primitive_token_occurrences_delta"),
            ("report_scratch_line_count","report_scratch_line_count_delta"),
        ):
            va=a[key];vb=b[key]
            deltas[out]=(vb-va if both and type(va) is int
                         and type(vb) is int else None)
        contrasts.append({
            "factor":c["factor"],"graph_index":c["graph_index"],
            "arm_a":a["case"],"arm_b":b["case"],
            "original_E3_both_qualified":c["e3_both_qualified"],
            "both_RTL_sha_attested":both,"observation_status":status,
            "estimated_BRAM_18K_delta":c["bram_18k_b_minus_a"]
                if c["e3_both_qualified"] else None,
            **deltas,
        })
    if len(cases)!=32 or len(contrasts)!=64:
        raise AssertionError("R2w original full study population incomplete")
    return {
        "schema":SCHEMA,"original_source_revision":ORIGINAL_R2U_SOURCE_REVISION,
        "original_R2u_cases":32,"original_R2v_factor_pairs":64,
        "original_E3_qualified_cases":30,
        "original_E3_qualified_factor_pairs":57,
        "original_E3_UNKNOWN_pairs":7,
        "source_attested_RTL_case_count":sum(
            r["rtl_sha_attested"] for r in cases),
        "conditional_pairs_with_both_RTL_sha_attested":sum(
            c["both_RTL_sha_attested"] for c in contrasts),
        "case_evidence":cases,"factor_evidence":contrasts,
        "original_native_vendor_calls_during_review":0,
        "physical_BRAM":None,"routed_Fmax":None,
        "limits":[
            "Raw source Verilog lexical inventory, not netlist inferred RAM or primitive binding",
            "Existing Vitis RTL and ROM SHA from original invocation, no new tool calls",
            "Identifier and primitive-token counts may include module names or dead logic",
            "No claim that unused source padding survives CSynth or RTL logic optimization",
            "No Vivado place/route, RAMB allocation, achieved timing or power",
            "E3 ineligible cases remain in population but have null HLS estimate deltas",
        ],
    }


def run(*,study:str|Path,output:str|Path)->dict:
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("R2w read-only HLS forensics output strictly create-only")
    report=analyze(study=study)
    dest.mkdir(parents=True)
    with (dest/"summary.json").open("x",encoding="utf-8") as out:
        json.dump(report,out,sort_keys=True,indent=2);out.write("\n")
    for filename,fields,key in (
        ("cases.csv",CASE_FIELDS,"case_evidence"),
        ("contrasts.csv",PAIR_FIELDS,"factor_evidence"),
    ):
        with (dest/filename).open("x",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report[key])
    return {k:v for k,v in report.items()
            if k not in ("case_evidence","factor_evidence")}


def main(argv:list[str]|None=None)->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(run(study=args.study,output=args.output),
                     sort_keys=True,indent=2))


if __name__=="__main__":
    main()
