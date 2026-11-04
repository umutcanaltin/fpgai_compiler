"""Read-only generated hls4ml Softmax source inventory after native RF replay audit.

The original E1 model-owned firmware files (including headers) are compared
byte-for-byte with their Vitis CSim clones before inspection. Do NOT report
trained tensor names, weights or raw generated code: only exact file hashes,
line numbers, syntax categories and bounded numeric precision declarations.
No vendor execution, source mutation, root-cause proof or E2 promotion.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from fpgai_research.experiments.hls4ml_native_pair_reaudit import audit_native_pair

SCHEMA="fpgai-research/hls4ml-native-softmax-source-inventory/v1"
EXTENSIONS=frozenset({".h",".hpp",".cpp",".cc",".cxx"})
SOFTMAX_RE=re.compile(r"softmax",re.IGNORECASE)
TABLE_RE=re.compile(r"\b(?:exp_table|inv_table|table_size|softmax_implementation)\b",re.IGNORECASE)
FIXED_RE=re.compile(r"\b(ap_u?fixed)\s*<\s*(\d+)\s*,\s*(\d+)([^>]*)>")
MODE_RE=re.compile(r"\bAP_(?:RND(?:_CONV|_ZERO|_INF|_MIN_INF)?|TRN(?:_ZERO)?|SAT(?:_ZERO|_SYM)?|WRAP(?:_SM)?)\b")


def _firmware_files(project):
    root=Path(project)
    folder=root/"firmware"
    if (root.is_symlink() or folder.is_symlink() or not folder.is_dir()):
        raise ValueError("Missing regular model-owned firmware directory")
    entries={}
    total=0
    for file in sorted(folder.iterdir()):
        if file.is_symlink():
            raise ValueError("Generated firmware contains linked file")
        if not file.is_file() or file.suffix.lower() not in EXTENSIONS:
            continue
        if (len(entries)>=48 or not 0<file.stat().st_size<=16*1024*1024):
            raise ValueError("Unbounded or empty generated source file")
        raw=file.read_bytes()
        total+=len(raw)
        if total>64*1024*1024:
            raise ValueError("Oversized direct model source inventory")
        entries[file.name]={"sha256":sha256(raw).hexdigest(),
                            "content":raw.decode("utf-8")}
    if not entries:
        raise ValueError("Missing own-model C++ and header inventory")
    return entries


def _extract(entries):
    softmax=[]
    types=[]
    table=[]
    for filename,item in sorted(entries.items()):
        for number,line in enumerate(item["content"].splitlines(),start=1):
            if SOFTMAX_RE.search(line):
                kind=("softmax_call" if ("nnet::softmax" in line or
                        "softmax<" in line) else "softmax_declaration")
                softmax.append({"file":filename,"line":number,"kind":kind})
            if TABLE_RE.search(line):
                for keyword in ("exp_table","inv_table","table_size",
                                "softmax_implementation"):
                    if re.search(r"\b"+keyword+r"\b",line,re.I):
                        found=re.search(
                            r"\btable_size\b\s*=\s*(\d+)\b",
                            line) if keyword=="table_size" else None
                        table.append({
                            "file":filename,"line":number,"keyword":keyword,
                            "declared_table_entries":(
                                int(found.group(1)) if found else None),
                        })
            for match in FIXED_RE.finditer(line):
                width=int(match.group(2))
                integer=int(match.group(3))
                if not (1<=width<=128 and 0<=integer<=width):
                    raise ValueError("Invalid generated fixed-point precision")
                types.append({
                    "file":filename,"line":number,
                    "signed":match.group(1)=="ap_fixed",
                    "width_bits":width,"integer_bits":integer,
                    "fraction_bits":width-integer,
                    "explicit_modes":MODE_RE.findall(match.group(4)),
                    "appears_on_softmax_line":SOFTMAX_RE.search(line) is not None,
                })
    if (not softmax or len(softmax)>256 or len(types)>256 or len(table)>512):
        raise ValueError("No bounded generated Softmax evidence")
    return {
        "softmax_source_locations":softmax,
        "fixed_point_declarations":types,
        "table_and_implementation_locations":table,
    }


def inspect_native_softmax_source(stage_root,global_default,native_run):
    original=audit_native_pair(stage_root,global_default,native_run)
    if (original.get("real_native_csim_logs_fingerprint_rechecked") is not True
            or original.get("real_native_csim_log_count")!=2
            or original.get("native_csim_observed_in_source_attributed_logs") is not True):
        raise ValueError("Original native paired Vitis log/source audit failed")
    root=Path(stage_root)
    variants={}
    for rf in (1,2):
        label="rf"+str(rf)
        e1=_firmware_files(root/("e1_"+label))
        csim=_firmware_files(root/("csim_"+label)/"project")
        if (set(e1)!=set(csim) or
                any(e1[k]["sha256"]!=csim[k]["sha256"] for k in e1)):
            raise ValueError("Original generated firmware headers/source differ from CSim clone")
        matches=_extract(csim)
        variants[label]={
            "original_e1_and_native_clone_firmware_identical":True,
            "model_owned_source_sha256":{
                name:value["sha256"] for name,value in e1.items()},
            "source_file_count":len(e1),
            **matches,
        }
    one,two=(variants["rf1"],variants["rf2"])
    return {
        "schema":SCHEMA,
        "native_pair_log_source_reattested":True,
        "native_pair_replay_receipt_sha256":original["native_run_receipt_sha256"],
        "original_e1_and_native_clone_headers_identical":True,
        "rf1_rf2_model_firmware_file_count_equal":
            one["source_file_count"]==two["source_file_count"],
        "rf1_rf2_identical_source_file_count":sum(
            one["model_owned_source_sha256"].get(name)==digest
            for name,digest in two["model_owned_source_sha256"].items()),
        "rf1_rf2_source_file_union_count":len(
            set(one["model_owned_source_sha256"])|
            set(two["model_owned_source_sha256"])),
        "variants":variants,
        "vendor_tools_executed_by_analysis":False,
        "source_files_or_weights_mutated":False,
        "operator_root_cause_proven":False,
        "original_raw_numeric_E2_qualified":False,
        "hardware_qor_comparison_eligible":False,
        "limitations":(
            "Only direct model-owned generated firmware C++/headers are scanned; "
            "hls4ml library template implementations and native arithmetic "
            "internals are NOT proved by this inventory. No private code or "
            "trained weights are exported. A matching textual Softmax call "
            "cannot establish table accuracy, output row-sum behavior, "
            "causal failing layer, numerical equivalence, or routed QoR."),
    }


def export_native_softmax_source(stage_root,global_default,native_run,output):
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Use a fresh private source-inspection directory")
    report=inspect_native_softmax_source(stage_root,global_default,native_run)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".hls-softmax-source-",dir=dest.parent) as tmp:
        folder=Path(tmp)/"report"
        folder.mkdir()
        (folder/"private_softmax_source.json").write_text(
            json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
        lines=[
            "# Source-linked hls4ml generated Softmax configuration census","",
            "Actual RF1/RF2 cloned and original generated C++/headers were "
            "verified byte-identical, after existing native Vitis CSim "
            "replay audit. Original model source and weights were not changed.",
            "This is not an E2 pass or a root-cause determination.","",
            "| RF | Owned files | Softmax mentions | Table/config mentions | Fixed types |",
            "|---|---:|---:|---:|---:|"]
        for rf,one in report["variants"].items():
            lines.append(
                f"| {rf} | {one['source_file_count']} | "
                f"{len(one['softmax_source_locations'])} | "
                f"{len(one['table_and_implementation_locations'])} | "
                f"{len(one['fixed_point_declarations'])} |")
        for rf,one in report["variants"].items():
            lines.extend(["","## "+rf,"","### Generated Softmax locations","",
                          "| File | Line | Classification |",
                          "|---|---:|---|"])
            for row in one["softmax_source_locations"]:
                lines.append(f"| {row['file']} | {row['line']} | {row['kind']} |")
            lines.extend(["","### Table and strategy references","",
                          "| File | Line | Kind | Declared table size |",
                          "|---|---:|---|---:|"])
            for row in one["table_and_implementation_locations"]:
                lines.append(
                    f"| {row['file']} | {row['line']} | {row['keyword']} | "
                    + (str(row["declared_table_entries"])
                       if row["declared_table_entries"] is not None
                       else "—") + " |")
            lines.extend(["","### Fixed-point declarations (model-owned)","",
                          "| File | Line | Signed | W | I | F | Explicit modes |",
                          "|---|---:|---|---:|---:|---:|---|"])
            for row in one["fixed_point_declarations"]:
                lines.append(
                    f"| {row['file']} | {row['line']} | {row['signed']} | "
                    f"{row['width_bits']} | {row['integer_bits']} | "
                    f"{row['fraction_bits']} | "
                    f"{','.join(row['explicit_modes']) or 'not explicit'} |")
        lines.extend(["","No captured layer outputs or independent template "
                      "implementation audit is provided. Before changing "
                      "Softmax configuration, inspect the generated "
                      "model file/line and installed hls4ml library "
                      "implementation, then use a NEW source revision "
                      "and CSim run while preserving frozen 0.05 policy.",""])
        (folder/"summary.md").write_text("\n".join(lines))
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Destination appeared during report export")
        folder.rename(dest)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-root",required=True)
    p.add_argument("--global-default",required=True)
    p.add_argument("--native-run",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args(argv)
    r=export_native_softmax_source(
        a.stage_root,a.global_default,a.native_run,a.output)
    print(json.dumps({
        "schema":SCHEMA,
        "source_and_log_reattested":True,
        "rf_model_owned_source_files":{
            k:v["source_file_count"] for k,v in r["variants"].items()},
        "softmax_location_counts":{
            k:len(v["softmax_source_locations"])
            for k,v in r["variants"].items()},
        "vendor_tools_executed_by_analysis":False,
    },indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
