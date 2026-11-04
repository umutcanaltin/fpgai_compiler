"""Read-only fresh-RTL file-set diagnostics for equal native E3 estimates.

Only byte-level structural evidence, not routed E4, logic equivalence,
synthesis schedule or per-loop achieved II. Never export RTL names, hashes,
sources, trained weights or private project paths.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from fpgai_research.experiments.isolated_hls_e3_ablation import collect_isolated_hls_e3_ablation
from fpgai_research.experiments.isolated_hls_e3_source_diagnostics import diagnose_equal_e3_sources
from fpgai_research.experiments.isolated_hls_replay import audit_isolated_hls_replay, _regular_bytes
from fpgai_research.experiments.native_hls_rerun_plan import _read_json
from fpgai_research.experiments.native_schedule_reports import inspect_native_schedule_case
from fpgai_research.experiments.results import _attested, _native_tool_identity

SCHEMA="fpgai-research/isolated-e3-paired-rtl-inventory/v1"
_RTL_REL=Path("fpgai_hls_project/solution1/syn/verilog")
_REPORT_LIMIT=12*1024*1024
_RTL_LIMIT=24*1024*1024
_MAX_FILES=4096
_FILENAME=re.compile(r"[A-Za-z0-9_.-]+\.(?:v|sv)\Z")
_SHA=re.compile(r"[0-9a-f]{64}\Z")


# Verilog/SV lexical comparison is intentionally weaker than parsing/elaboration:
# it discards whitespace and comments while retaining every non-comment
# token (including strings, preprocessor directives, literals and identifiers).
# No token or source text is exported.
_TOKEN=re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*|[0-9]+|[^\s]")

def _rtl_lexical_digest(raw:bytes)->str|None:
    """Private lexical fingerprint; None for malformed/unsupported text.

    Handle comments ONLY outside strings and escaped identifiers. Token
    equality is an observation about bytes of generated text, not about
    semantic/logic equivalence; macros are NOT evaluated.
    """
    try:
        code=raw.decode("utf-8","strict")
    except UnicodeDecodeError:
        return None
    tokens=[]
    i=0
    n=len(code)
    while i<n:
        c=code[i]
        if c.isspace():
            i+=1
            continue
        if c=="/" and i+1<n and code[i+1]=="/":
            end=code.find("\n",i+2)
            i=n if end<0 else end+1
            continue
        if c=="/" and i+1<n and code[i+1]=="*":
            end=code.find("*/",i+2)
            if end<0:
                return None
            i=end+2
            continue
        if c=='"':
            start=i
            i+=1
            while i<n:
                if code[i]=="\\":
                    i+=2
                    if i>n:
                        return None
                    continue
                if code[i]=='"':
                    i+=1
                    break
                if code[i] in "\r\n":
                    return None
                i+=1
            else:
                return None
            tokens.append(code[start:i])
            continue
        if c=="\\":
            # An escaped Verilog identifier runs until whitespace.
            start=i
            i+=1
            while i<n and not code[i].isspace():
                i+=1
            if i==start+1 or i==n:
                return None
            tokens.append(code[start:i])
            continue
        match=_TOKEN.match(code,i)
        if match is None:
            return None
        tokens.append(match.group(0))
        i=match.end()
    return sha256(json.dumps(tokens,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()


def _pairwise_byte_and_lexical_summary(first:dict,second:dict)->dict|None:
    """Only local-in-memory verified maps enter; expose integer counts."""
    amap,bmap=first.get("_digest_map"),second.get("_digest_map")
    if not isinstance(amap,dict) or not isinstance(bmap,dict):
        return None
    alex,blex=first.get("_lexical_map"),second.get("_lexical_map")
    if not isinstance(alex,dict) or not isinstance(blex,dict):
        return None
    names_a,names_b=set(amap),set(bmap)
    shared=names_a&names_b
    same=sum(amap[name]==bmap[name] for name in shared)
    changed=[name for name in shared if amap[name]!=bmap[name]]
    lex_equal=sum(alex.get(name) is not None and
                  alex.get(name)==blex.get(name) for name in changed)
    lex_diff=sum(alex.get(name) is not None and blex.get(name) is not None
                 and alex[name]!=blex[name] for name in changed)
    return {
        "common_file_count":len(shared),
        "unchanged_byte_file_count":same,
        "changed_byte_file_count":len(changed),
        "only_in_case_a_file_count":len(names_a-names_b),
        "only_in_case_b_file_count":len(names_b-names_a),
        "changed_files_lexically_equal_count":lex_equal,
        "changed_files_lexically_different_count":lex_diff,
        "changed_files_lexically_unclassified_count":
            len(changed)-lex_equal-lex_diff,
        "lexical_comparison_is_logic_equivalence":False,
    }

def _case_rtl_inventory(case:Path, audit_row:dict)->dict:
    """Verify actual file bytes vs native runner's full and fresh RTL maps."""
    source_sha=audit_row.get("source_sha256")
    if (audit_row.get("evidence")!="E3-HLS-synthesis-report"
            or audit_row.get("fixed_point_csim_passed") is not True
            or audit_row.get("fresh_hls_report_attested") is not True
            or not isinstance(source_sha,str) or not _SHA.fullmatch(source_sha)):
        raise ValueError("RTL comparison requires independently source-qualified E3")
    verified=inspect_native_schedule_case(case,source_sha,audit_row["evidence"])
    if verified["status"]!="ATTESTED_RAW_REPORT_AVAILABLE":
        raise ValueError("Original native E3 text report provenance no longer valid")
    receipt_path=case/"vitis_invocation.json"
    if receipt_path.is_symlink():
        raise ValueError("Linked native Vitis receipt")
    receipt=_attested(receipt_path,"vitis_hls",case/"run_hls.tcl",
                      case/"model.cpp",testbench=case/"tb.cpp")
    if not _native_tool_identity(receipt,"vitis_hls"):
        raise ValueError("Native Vitis invocation no longer source-attested")
    rtl_root=case
    for part in _RTL_REL.parts:
        rtl_root=rtl_root/part
        if rtl_root.is_symlink():
            raise ValueError("Linked generated RTL directory component")
    if not rtl_root.is_dir():
        return {"status":"RTL_DIRECTORY_ABSENT","file_count":None,"_digest_map":None}
    expected=receipt.get("rtl_output_sha256")
    changed=receipt.get("rtl_new_or_changed_files")
    # A receipt without a complete captured RTL inventory cannot prove
    # anything about equality of generated files, even if RTL is on disk.
    if not isinstance(expected,dict) or not isinstance(changed,list):
        return {"status":"RTL_NOT_CAPTURED_IN_RECEIPT","file_count":None,"_digest_map":None}
    if (len(expected)>_MAX_FILES or len(changed)>_MAX_FILES
            or not all(isinstance(k,str) and _FILENAME.fullmatch(k)
                       and isinstance(v,str) and _SHA.fullmatch(v)
                       for k,v in expected.items())
            or not all(isinstance(k,str) and _FILENAME.fullmatch(k)
                       for k in changed)
            or len(changed)!=len(set(changed))):
        raise ValueError("Malformed native RTL file inventory")
    candidates=sorted((*rtl_root.glob("*.v"),*rtl_root.glob("*.sv")))
    if len(candidates)>_MAX_FILES:
        raise ValueError("Generated RTL file count exceeds safe bound")
    if any(not name.is_file() or name.is_symlink() for name in candidates):
        raise ValueError("Linked or nonregular generated RTL file")
    current_names={path.name for path in candidates}
    if (current_names!=set(expected) or not set(changed)<=set(expected)):
        return {"status":"RTL_FILE_SET_CHANGED_SINCE_INVOCATION",
                "file_count":None,"_digest_map":None}
    if not expected:
        return {"status":"NO_GENERATED_RTL_FILES",
                "file_count":0,"_digest_map":None}
    actual={}
    lexical={}
    for path in candidates:
        raw=_regular_bytes(path,_RTL_LIMIT)
        actual[path.name]=sha256(raw).hexdigest()
        lexical[path.name]=_rtl_lexical_digest(raw)
    if actual!=expected:
        return {"status":"RTL_BYTES_CHANGED_SINCE_INVOCATION",
                "file_count":None,"_digest_map":None}
    if set(changed)!=set(expected):
        return {"status":"HASH_MATCH_BUT_NOT_ALL_RTL_FRESH",
                "file_count":len(expected),"_digest_map":None}
    # Digests and file names are deliberately kept in memory for pairwise
    # equality only; never include them in serializable result dictionaries.
    return {"status":"FRESH_SOURCE_ATTESTED_RTL",
            "file_count":len(expected),"_digest_map":actual,
            "_lexical_map":lexical}

def inspect_tied_e3_rtl(directory,comparison:dict,audit:dict)->dict:
    """Check each E3-tied pair against the invocation's generated RTL map."""
    root=Path(directory)
    # The tied-source analysis independently validates source/manifest pair
    # provenance and defines the pair inventory. Never infer new pairs from
    # raw vendor files.
    tied=diagnose_equal_e3_sources(root,comparison,audit)
    inventory={}
    for row in audit["cases"]:
        inventory[(row["model"],row["case"])]=row
    results=[]
    for pair in tied["pairs"]:
        model,a,b=pair["model"],pair["case_a"],pair["case_b"]
        first=_case_rtl_inventory(root/model/a,inventory[(model,a)])
        second=_case_rtl_inventory(root/model/b,inventory[(model,b)])
        comparable=(first["status"]=="FRESH_SOURCE_ATTESTED_RTL"
                    and second["status"]=="FRESH_SOURCE_ATTESTED_RTL")
        identical=(first["_digest_map"]==second["_digest_map"]) if comparable else None
        private_delta=_pairwise_byte_and_lexical_summary(first,second) if comparable else None
        results.append({
            "model":model,"case_a":a,"case_b":b,
            "source_cpp_identical":pair["source_cpp_sha256_equal"],
            "hls_estimate_vector_equal":True,
            "case_a_rtl_status":first["status"],
            "case_b_rtl_status":second["status"],
            "case_a_rtl_file_count":first["file_count"],
            "case_b_rtl_file_count":second["file_count"],
            "rtl_comparison_status":(
                "IDENTICAL_GENERATED_RTL_FILE_BYTES" if identical is True else
                "DIFFERENT_GENERATED_RTL_FILE_BYTES" if identical is False else
                "INCONCLUSIVE_RTL_PROVENANCE_OR_AVAILABILITY"),
            "rtl_file_set_byte_identical":identical,
            "rtl_file_difference_counts":private_delta,
            "logic_equivalence_verified":False,
            "routed_e4_evidence_verified":False,
            "operator_loop_ii_verified":False,
        })
    counts=Counter(x["rtl_comparison_status"] for x in results)
    return {
        "schema":SCHEMA,
        "source_suite_sha256":tied["source_suite_sha256"],
        "equal_e3_estimate_pair_count":len(results),
        "fresh_comparable_pair_count":len(results)-counts["INCONCLUSIVE_RTL_PROVENANCE_OR_AVAILABILITY"],
        "identical_rtl_file_set_pair_count":counts["IDENTICAL_GENERATED_RTL_FILE_BYTES"],
        "different_rtl_file_set_pair_count":counts["DIFFERENT_GENERATED_RTL_FILE_BYTES"],
        "inconclusive_rtl_pair_count":counts["INCONCLUSIVE_RTL_PROVENANCE_OR_AVAILABILITY"],
        "pairs":results,
        "native_tools_executed":False,
        "private_rtl_exported":False,
        "rtl_filenames_exported":False,
        "rtl_digests_exported":False,
        "weight_values_exported":False,
        "logic_equivalence_verified":False,
        "operator_schedule_attribution_verified":False,
        "e4_physical_comparison_performed":False,
        "limitations":(
            "File-set byte equality covers only Vitis-generated Verilog "
            "files whose entire set and per-file content hashes match the "
            "completed invocation AND all were recorded as new or changed. "
            "Different bytes do not prove different implemented hardware; "
            "equal bytes do not prove formal circuit equivalence. No "
            "post-route timing, power, on-device runtime, achieved operator "
            "II or causal HLS optimization explanation is established. "
            "Lexical equality discards comments/whitespace but retains RTL "
            "identifiers, strings, literals and macro directives without "
            "evaluating them; it is NOT logic equivalence. Lexically "
            "unclassifiable files are explicitly counted, not assumed "
            "equivalent. File counts are only descriptive, not resource "
            "or module utilization. A missing or incomplete receipt is "
            "inconclusive, never equal."
        ),
    }

def export_tied_e3_rtl_inventory(directory,comparison_path,output)->dict:
    root=Path(directory)
    saved=_read_json(Path(comparison_path))
    if saved!=collect_isolated_hls_e3_ablation(root):
        raise ValueError("E3 summary no longer matches independent native replay audit")
    audit=audit_isolated_hls_replay(root)
    result=inspect_tied_e3_rtl(root,saved,audit)
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("RTL inventory output must be a new directory")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-tied-e3-rtl-",dir=target.parent) as temp:
        stage=Path(temp)
        (stage/"tied_e3_rtl_inventory.json").write_text(
            json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",
            encoding="utf-8")
        lines=[
            "# Matched E3 equal-estimate pairs: generated Verilog file-set evidence","",
            "**Only source-attested fresh Vitis RTL file bytes. No logic "
            "equivalence or routed E4 measurements.**","",
            f"Pairs: {result['equal_e3_estimate_pair_count']}; fresh comparable: "
            f"{result['fresh_comparable_pair_count']}; identical RTL byte "
            f"sets: {result['identical_rtl_file_set_pair_count']}; "
            f"different RTL byte sets: {result['different_rtl_file_set_pair_count']}; "
            f"inconclusive: {result['inconclusive_rtl_pair_count']}.","",
            "| Model | Cases | RTL A status | RTL B status | Generated Verilog file counts A/B | Byte comparison |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in result["pairs"]:
            counts=(
                f"{row['case_a_rtl_file_count'] if row['case_a_rtl_file_count'] is not None else 'TBD'} / "
                f"{row['case_b_rtl_file_count'] if row['case_b_rtl_file_count'] is not None else 'TBD'}"
            )
            lines.append("| "+" | ".join((
                row["model"],row["case_a"]+" / "+row["case_b"],
                row["case_a_rtl_status"],row["case_b_rtl_status"],
                counts,row["rtl_comparison_status"],
            ))+" |")
        lines.extend(["",
            "## Per-pair file differences (no filenames or private RTL)","",
            "Counts compare same-named Verilog files. Lexical comparison "
            "removes only whitespace/comments, not identifiers or RTL values.","",
            "| Model | Cases | Byte-changed named files | Only in A/B | "
            "Changed files lexically equal / different / unclassified |",
            "| --- | --- | ---: | --- | --- |",
        ])
        for row in result["pairs"]:
            diff=row["rtl_file_difference_counts"]
            if diff is None:
                values=("TBD","TBD","TBD")
            else:
                values=(
                    str(diff["changed_byte_file_count"]),
                    str(diff["only_in_case_a_file_count"])+"/"+
                        str(diff["only_in_case_b_file_count"]),
                    str(diff["changed_files_lexically_equal_count"])+"/"+
                        str(diff["changed_files_lexically_different_count"])+"/"+
                        str(diff["changed_files_lexically_unclassified_count"]),
                )
            lines.append("| "+" | ".join((
                row["model"],row["case_a"]+" / "+row["case_b"],*values
            ))+" |")
        lines.extend(["","## Scientific boundary","",result["limitations"],""])
        (stage/"tied_e3_rtl_inventory.md").write_text("\n".join(lines),encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("RTL inventory destination appeared")
        stage.rename(target)
    return result
