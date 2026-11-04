"""R2x: read-only STATIC HLS Verilog hierarchy and scratch memory candidates.

Uses only the frozen original R2u source-pinned, SHA-attested RTL from
32 Vitis HLS cases. Starts with R2w's original invocation/source/ROM/
Verilog evidence check; never calls Vitis, Vivado, a simulator or RL.

The finite scanner recognizes basic Verilog module definitions and
type-named module instantiations, including parameter overrides. It is
NOT a Verilog preprocessor, elaborator, netlist parser or placement
report. Missing/unsupported syntax produces explicit UNKNOWN status.
Static reachability over module TYPES is provisional and does not
prove that RTL is instantiated after synthesis or mapped to RAMB cells.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
import json
from pathlib import Path
import re

from fpgai_research.experiments import cst_r2w_rtl_memory_forensics as r2w

SCHEMA = "fpgai-research/cst-R2x-read-only-RTL-static-module-hierarchy/v1"
ORIGINAL_SHA = "0c6d85f189062df8d5157237f3c8ac74ac8da4b8"
_TOP = "fpgai_model"
_IDENTIFIER = r"[A-Za-z_$][A-Za-z0-9_$]*"
_TOKENS = re.compile(rf"{_IDENTIFIER}|[#().,;:\[\]{{}}]")
_COMMENT_STRING = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"',re.S)
_ATTRIBUTES = re.compile(r"\(\*.*?\*\)",re.S)
_MODULE = re.compile(
    rf"\bmodule\s+(?:automatic\s+)?(?P<name>{_IDENTIFIER})\b"
    rf"(?P<body>.*?)\bendmodule\b",re.S
)
_REG_ARRAY = re.compile(
    rf"\breg\s+(?:signed\s+)?(?:\[[^\]\n]+\]\s*)?"
    rf"(?P<name>{_IDENTIFIER})\s*"
    rf"\[\s*(?P<low>[0-9]+)\s*:\s*(?P<high>[0-9]+)\s*\]\s*;",
    re.I,
)
_MEM_NAMES = re.compile(r"scratch|_ram|_rom|_mem|_fifo|bram|uram",re.I)
_PRIMITIVE = re.compile(
    r"(?:RAMB18E[12]|RAMB36E[12]|URAM288|"
    r"RAM32M|RAM64M|xpm_memory_[A-Za-z0-9_]+)\Z",re.I
)
CASE_FIELDS = (
    "case","graph_index","address","storage","binding","padding",
    "original_E3_qualified","original_RTL_SHA_attested",
    "parse_status","declared_module_type_count",
    "static_local_instance_count","top_status","reachable_module_type_count",
    "reachable_scratch_module_type_count",
    "reachable_memory_module_type_count",
    "reachable_memory_module_type_instances",
    "candidate_reachable_primitive_instantiations",
    "reachable_reg_memory_candidate_count",
)
PAIR_FIELDS = (
    "factor","graph_index","arm_a","arm_b",
    "original_E3_both_qualified","original_RTL_both_SHA_attested",
    "both_top_static_reachability_known","estimated_BRAM_delta_or_null",
    "reachable_memory_module_type_count_delta",
    "reachable_memory_instance_count_delta",
    "reachable_scratch_module_type_count_delta",
    "candidate_reachable_primitive_instance_count_delta",
    "structural_pair_status",
)


def strip_noncode(source:str)->str:
    """Replace comments, attributes and quoted strings with spaces."""
    cleaned=_COMMENT_STRING.sub(lambda m:" "*len(m.group()),source)
    return _ATTRIBUTES.sub(lambda m:" "*len(m.group()),cleaned)


def _skip_balanced(tokens:list[str],start:int)->int|None:
    if start>=len(tokens) or tokens[start]!="(":
        return None
    level=0
    for i in range(start,len(tokens)):
        if tokens[i]=="(":
            level+=1
        elif tokens[i]==")":
            level-=1
            if level==0:
                return i+1
            if level<0:
                return None
    return None


def _header_end(body:str)->int|None:
    """Find first non-parenthesized semicolon in a module header."""
    nest=0
    for i,c in enumerate(body):
        if c=="(":
            nest+=1
        elif c==")":
            nest-=1
            if nest<0:
                return None
        elif c==";" and nest==0:
            return i+1
    return None


def _candidate_instantiations(body:str, recognized:set[str])->list[dict]:
    """Conservative recognizer: type [#(params)] instance (connections);"""
    tokens=[m.group() for m in _TOKENS.finditer(body)]
    result=[]
    for i,token in enumerate(tokens):
        if token not in recognized:
            continue
        # Exclude named port connections and "parameter type" context.
        if i and tokens[i-1] in (".","parameter","localparam","typedef","function"):
            continue
        j=i+1
        if j<len(tokens) and tokens[j]=="#":
            j=_skip_balanced(tokens,j+1)
            if j is None:
                continue
        if j is None or j+1>=len(tokens):
            continue
        instance=tokens[j]
        if re.fullmatch(_IDENTIFIER,instance) is None:
            continue
        end=_skip_balanced(tokens,j+1)
        if end is None or end>=len(tokens) or tokens[end]!=";":
            continue
        result.append({"type":token,"instance":instance})
    return result


def parse_static_modules(source_files:dict[str,str], *,
                         top:str=_TOP)->dict:
    if (not isinstance(source_files,dict) or not source_files
            or len(source_files)>4096
            or re.fullmatch(_IDENTIFIER,top) is None):
        raise ValueError("R2x requires bounded named Verilog source and top")
    definitions={}
    unsupported=[]
    for filename,source in sorted(source_files.items()):
        if (not isinstance(filename,str) or not re.fullmatch(
                r"[A-Za-z0-9_][A-Za-z0-9_.+-]*\.(?:v|sv)",filename)
                or not isinstance(source,str)):
            raise ValueError("R2x invalid source filename or text")
        cleaned=strip_noncode(source)
        blocks=list(_MODULE.finditer(cleaned))
        # Unknown syntax or unresolved module fragments are not
        # silently declared safe/unused simply because regex skipped them.
        declared=len(re.findall(r"\bmodule\b",cleaned))
        if declared!=len(blocks):
            unsupported.append({"file":filename,
                                "reason":"UNKNOWN_MODULE_DECLARATION_SYNTAX"})
        for match in blocks:
            name=match.group("name")
            if name in definitions:
                unsupported.append({"file":filename,
                                    "reason":"DUPLICATE_MODULE_TYPE",
                                    "name":name})
                continue
            body=match.group("body")
            head=_header_end(body)
            if head is None:
                unsupported.append({"file":filename,
                                    "reason":"UNKNOWN_MODULE_HEADER",
                                    "name":name})
                continue
            executable=body[head:]
            regs=[]
            for a in _REG_ARRAY.finditer(executable):
                first=int(a.group("low"))
                last=int(a.group("high"))
                regs.append({
                    "name":a.group("name"),
                    "literal_unpacked_elements":abs(last-first)+1,
                })
            definitions[name]={
                "name":name,"filename":filename,
                "body":executable,
                "reg_array_declarations":regs,
            }
    recognized=set(definitions)
    for special in (
        "RAMB18E1","RAMB18E2","RAMB36E1","RAMB36E2","URAM288",
        "RAM32M","RAM64M",
    ):
        recognized.add(special)
    edges=[]
    for name,info in sorted(definitions.items()):
        # All unknown external module types are intentionally NOT
        # guessed as instances. The scanner proves only a structural
        # subset; no full elaboration claim is made.
        named_types=recognized | set(
            re.findall(r"\bxpm_memory_[A-Za-z0-9_]+\b",info["body"])
        )
        for inst in _candidate_instantiations(info["body"],named_types):
            edges.append({"parent_type":name,
                          "child_type":inst["type"],
                          "instance":inst["instance"]})
    if len(definitions)>4096 or len(edges)>65536:
        raise ValueError("R2x unbounded static RTL source hierarchy")
    adjacency=defaultdict(list)
    for edge in edges:
        adjacency[edge["parent_type"]].append(edge)
    known_top=top in definitions
    reached=set()
    work=deque([top] if known_top else [])
    while work:
        name=work.popleft()
        if name in reached:
            continue
        reached.add(name)
        for edge in adjacency[name]:
            if edge["child_type"] in definitions:
                work.append(edge["child_type"])
    reachable_edges=[
        e for e in edges if e["parent_type"] in reached
    ] if known_top else []
    mem_types={name for name in reached if _MEM_NAMES.search(name)}
    scratch_types={name for name in reached if "scratch" in name.lower()}
    mem_instances=sum(e["child_type"] in mem_types for e in reachable_edges)
    primitive=sum(bool(_PRIMITIVE.fullmatch(e["child_type"]))
                  for e in reachable_edges)
    reg_arrays=sum(len(definitions[name]["reg_array_declarations"])
                   for name in reached)
    if unsupported:
        status="UNKNOWN_UNSUPPORTED_OR_AMBIGUOUS_VERILOG"
    elif not known_top:
        status="UNKNOWN_TOP_MODULE_NOT_DEFINED"
    else:
        status="STATIC_TYPE_GRAPH_ONLY"
    valid=(status=="STATIC_TYPE_GRAPH_ONLY")
    return {
        "parse_status":status,
        "top_status":("TOP_DEFINED_STATIC_ONLY" if known_top
                      else "UNKNOWN_TOP_MODULE_NOT_DEFINED"),
        "declared_module_type_count":len(definitions),
        "static_local_instance_count":len(edges),
        "reachable_module_type_count":len(reached) if valid else None,
        "reachable_scratch_module_type_count":len(scratch_types) if valid else None,
        "reachable_memory_module_type_count":len(mem_types) if valid else None,
        "reachable_memory_module_type_instances":mem_instances if valid else None,
        "candidate_reachable_primitive_instantiations":primitive if valid else None,
        "reachable_reg_memory_candidate_count":reg_arrays if valid else None,
        "all_local_module_types":sorted(definitions),
        "provisionally_reachable_module_types":sorted(reached) if valid else [],
        "provisionally_reachable_scratch_module_types":sorted(scratch_types)
            if valid else [],
        "provisionally_reachable_memory_module_types":sorted(mem_types)
            if valid else [],
        "static_instance_edges":edges,
        "unparsed_module_diagnostics":unsupported,
        "limits":[
            "Subset recognizer of source Verilog, not preprocessed/elaborated RTL",
            "A reachable module type is not proof of an instantiated hardware cell",
            "Declared reg arrays and RAMB names are textual candidates only",
            "Unrecognized external modules and generate conditions remain unresolved",
        ],
    }


def analyze(*,study:str|Path)->dict:
    root=Path(study)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("R2x original source-pinned R2u folder missing")
    baseline=r2w.analyze(study=root)
    if (baseline["original_source_revision"]!=ORIGINAL_SHA
            or baseline["original_R2u_cases"]!=32
            or baseline["original_R2v_factor_pairs"]!=64
            or baseline["source_attested_RTL_case_count"]!=32
            or baseline["conditional_pairs_with_both_RTL_sha_attested"]!=64
            or baseline["original_E3_qualified_cases"]!=30):
        raise ValueError("R2x original R2w 32/64 attested cohort not verified")
    original_cases=[]
    indexed={}
    for row in baseline["case_evidence"]:
        if row["rtl_sha_attested"] is not True:
            raise ValueError("R2x cannot consume unattested original RTL case")
        case_folder=root/row["case"]
        rtl_folder=case_folder/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
        if rtl_folder.is_symlink() or not rtl_folder.is_dir():
            raise ValueError("R2x original HLS RTL directory absent or symlinked")
        source_files={}
        for name,digest in sorted(row["original_rtl_sha256"].items()):
            path=rtl_folder/name
            if path.is_symlink() or not path.is_file():
                raise ValueError("R2x original generated RTL file unavailable")
            data=path.read_bytes()
            from hashlib import sha256
            if sha256(data).hexdigest()!=digest:
                raise ValueError("R2x originally attested RTL changed after R2w audit")
            source_files[name]=data.decode("utf-8",errors="replace")
        decoded=parse_static_modules(source_files)
        rec={
            **{key:row[key] for key in (
                "case","graph_index","address","storage","binding","padding"
            )},
            "original_E3_qualified":row["E3_qualified"],
            "original_RTL_SHA_attested":True,
            "original_RTL_file_SHA256":row["original_rtl_sha256"],
            **decoded,
        }
        original_cases.append(rec)
        indexed[row["case"]]=rec
    contrasts=[]
    for old in baseline["factor_evidence"]:
        a=indexed[old["arm_a"]];b=indexed[old["arm_b"]]
        eligible=(a["parse_status"]=="STATIC_TYPE_GRAPH_ONLY"
                  and b["parse_status"]=="STATIC_TYPE_GRAPH_ONLY")
        delta={}
        for key,name in (
            ("reachable_memory_module_type_count",
             "reachable_memory_module_type_count_delta"),
            ("reachable_memory_module_type_instances",
             "reachable_memory_instance_count_delta"),
            ("reachable_scratch_module_type_count",
             "reachable_scratch_module_type_count_delta"),
            ("candidate_reachable_primitive_instantiations",
             "candidate_reachable_primitive_instance_count_delta"),
        ):
            left=a[key];right=b[key]
            delta[name]=right-left if eligible and type(left) is int \
                and type(right) is int else None
        contrasts.append({
            "factor":old["factor"],"graph_index":old["graph_index"],
            "arm_a":a["case"],"arm_b":b["case"],
            "original_E3_both_qualified":old["original_E3_both_qualified"],
            "original_RTL_both_SHA_attested":old["both_RTL_sha_attested"],
            "both_top_static_reachability_known":eligible,
            "estimated_BRAM_delta_or_null":old["estimated_BRAM_18K_delta"],
            "structural_pair_status":(
                "STATIC_TYPE_GRAPH_COMPARISON_ONLY" if eligible else
                "UNKNOWN_UNSUPPORTED_STATIC_RTL"),
            **delta,
        })
    if len(original_cases)!=32 or len(contrasts)!=64:
        raise AssertionError("R2x source case/conditional pair population incomplete")
    return {
        "schema":SCHEMA,"original_source_revision":ORIGINAL_SHA,
        "original_R2u_cases":32,"original_R2v_conditional_pairs":64,
        "original_R2u_E3_qualified_cases":30,
        "original_R2v_E3_qualified_pairs":57,
        "original_R2v_E3_unknown_pairs":7,
        "original_RTL_SHA_attested_cases":32,
        "static_top_reachable_case_count":sum(
            item["parse_status"]=="STATIC_TYPE_GRAPH_ONLY"
            for item in original_cases),
        "static_top_reachable_pair_count":sum(
            item["both_top_static_reachability_known"] for item in contrasts),
        "cases":original_cases,"contrasts":contrasts,
        "original_vendor_calls_during_analysis":0,
        "physical_BRAM":None,"achieved_routed_Fmax":None,
        "limits":[
            "Static source-level type graph, not Verilog preprocessing/elaboration",
            "Source module definitions and candidate instances do NOT equal FPGA RAM cells",
            "Unknown module syntax/top or external modules cannot establish absence",
            "HLS estimated BRAM not physical BRAM; original E3 7 unknown pairs",
            "No RTL synthesis, place/route, simulation or vendor calls",
        ],
    }


def run(*,study:str|Path,output:str|Path)->dict:
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("R2x structural RTL result folder strictly create-only")
    result=analyze(study=study)
    dest.mkdir(parents=True)
    with (dest/"summary.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    for filename,key,fields in (
        ("cases.csv","cases",CASE_FIELDS),
        ("contrasts.csv","contrasts",PAIR_FIELDS),
    ):
        with (dest/filename).open("x",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
            writer.writeheader()
            writer.writerows(result[key])
    return {k:v for k,v in result.items() if k not in ("cases","contrasts")}


def main(argv:list[str]|None=None)->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    print(json.dumps(run(study=args.study,output=args.output),
                     indent=2,sort_keys=True))


if __name__=="__main__":
    main()
