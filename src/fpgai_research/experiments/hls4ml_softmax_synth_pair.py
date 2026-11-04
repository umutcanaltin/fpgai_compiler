"""Source-matched exploratory HLS synthesis of 1024 vs 4096 reciprocal LUT.

Both native-selected numerical receipts are independently replay-attested
before staging two *new* synthesis-only Vitis projects. The computational
source stays byte-identical to the corresponding previously observed CSim
variant. build_opt.tcl switches ONLY csim 1->0 and synth 0->1, and the
hls4ml output path is relocated in each new clone.

This is a *within-hls4ml* exploratory HLS E3 cost comparison, NOT
FPGAI-vs-hls4ml QoR, routed E4, full-model E2, or hardware power.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

from fpgai_research.experiments.hls4ml_native_4096_replay import audit_native_4096
from fpgai_research.experiments.hls4ml_softmax_inverse_table_size import _check as audit_table_stage
from fpgai_research.experiments.hls4ml_softmax_intervention import (
    _regular,_static_source_inventory,
)
from fpgai_research.experiments.hls4ml_csim_stage import csim_only_options

SCHEMA="fpgai-research/paired-4096-softmax-hls-synth-stage/v1"
RUN_SCHEMA="fpgai-research/paired-4096-softmax-hls-synth-native/v1"
SYNTH_ONLY={
    "reset":0,"csim":0,"synth":1,"cosim":0,
    "validation":0,"export":0,"vsynth":0,"fifo_opt":0,
}
AREA_KEYS=("BRAM_18K","DSP","FF","LUT","URAM","SRL")
TIMING_KEYS=("EstimatedClockPeriod","TargetClockPeriod")
LATENCY_KEYS=("Best-caseLatency","Worst-caseLatency","Interval-min","Interval-max")


def _hash(path):
    return sha256(_regular(path,16*1024*1024).read_bytes()).hexdigest()


def synth_only_options(csim_content):
    if csim_content!=csim_only_options(csim_content):
        raise ValueError("Original must be Vitis CSim-only source")
    return "array set opt {\n"+"".join(
        f"    {name:<10} {value}\n" for name,value in SYNTH_ONLY.items())+"}\n"


def _sources(stage_1024,stage_4096):
    roots={"1024":Path(stage_1024),"4096":Path(stage_4096)}
    inventory={}
    for label,root in roots.items():
        files,hashes=_static_source_inventory(root/"project")
        options=_regular(files["build_opt.tcl"]).read_text()
        synth_only_options(options)
        yaml=_regular(files["hls4ml_config.yml"]).read_text()
        expected=str((root/"project").absolute())
        if yaml.count(expected)!=1:
            raise ValueError("Project YAML path not unique or not source-bound")
        inventory[label]={"files":files,"sha":hashes,
                          "build_options":synth_only_options(options),
                          "yaml":yaml,"old_path":expected}
    one=inventory["1024"]["sha"]
    two=inventory["4096"]["sha"]
    if (set(one)!=set(two)
            or [k for k in sorted(one) if one[k]!=two[k]]
                !=["firmware/parameters.h","hls4ml_config.yml"]):
        raise ValueError("1024 and 4096 model sources differ outside reciprocal table size")
    return inventory


def _parent(stage_4096,native_4096):
    qualified=audit_native_4096(stage_4096,native_4096)
    if (qualified.get("qualified_selected_16_image_native_numeric_acceptance")
            is not True
            or qualified.get("qualified_full_model_external_numeric_e2")
                is not False
            or qualified.get("qualified_hardware_qor_comparison")
                is not False):
        raise ValueError("4096 selected-native evidence is not scoped correctly")
    linked=audit_table_stage(stage_4096)
    original=linked.get("parent_stage")
    if not isinstance(original,str):
        raise ValueError("Missing parent 1024 reciprocal table source")
    return qualified,linked,original


def _preflight(folder, *, published_root=None):
    root=Path(folder)
    doc=json.loads(_regular(root/"experiment.json").read_text())
    if (doc.get("schema")!=SCHEMA
            or doc.get("source_changed_only_synthesis_controls") is not True
            or doc.get("native_synthesis_executed_by_staging") is not False
            or doc.get("qualifies_full_model_e2") is not False
            or doc.get("qualifies_routed_e4") is not False):
        raise ValueError("Unrecognized synthesis-only experiment receipt")
    original=doc.get("4096_source_stage")
    native=doc.get("4096_native_run")
    if not isinstance(original,str) or not isinstance(native,str):
        raise ValueError("Missing original CSim provenance")
    qualified,linked,parent=_parent(original,native)
    if (doc.get("native_replay_output_sha256")!=qualified[
            "actual_160_native_output_sha256"]
            or doc.get("native_replay_log_sha256")!=qualified[
                "native_vitis_log_sha256"]
            or doc.get("parent_18_6_stage")!=parent):
        raise ValueError("Native numerical evidence or parent source differs")
    inv=_sources(parent,original)
    final=Path(published_root) if published_root is not None else root
    for label in ("1024","4096"):
        base=inv[label]
        previous=base["sha"]
        expected=dict(previous)
        expected["build_opt.tcl"]=sha256(
            base["build_options"].encode()).hexdigest()
        expected["hls4ml_config.yml"]=sha256(
            base["yaml"].replace(
                base["old_path"],
                str((final/label/"project").absolute())).encode()).hexdigest()
        recorded=doc.get("synthesis_project_sha256",{}).get(label)
        if recorded!=expected:
            raise ValueError("Source manifest does not describe exact synthesis-only clone")
        for relative,digest in expected.items():
            file=_regular(root/label/"project"/relative,16*1024*1024)
            if _hash(file)!=digest:
                raise ValueError("Native synthesis project source changed")
        if _regular(root/label/"project"/"build_opt.tcl").read_text()!=(
                base["build_options"]):
            raise ValueError("Vitis options could execute unwanted stages")
    return doc


def stage_synth_pair(stage_4096,native_4096,output):
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Choose new immutable synthesis study directory")
    qualified,linked,parent=_parent(stage_4096,native_4096)
    inventory=_sources(parent,stage_4096)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent,prefix=".softmax-hls-pair-") as temp:
        stage=Path(temp)/"report"
        stage.mkdir()
        hashes={}
        for label in ("1024","4096"):
            base=inventory[label]
            clone=stage/label/"project"
            clone.mkdir(parents=True)
            mapping={}
            for relative,source in base["files"].items():
                target=clone/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                if relative=="build_opt.tcl":
                    target.write_text(base["build_options"])
                elif relative=="hls4ml_config.yml":
                    target.write_text(base["yaml"].replace(
                        base["old_path"],
                        str((dest/label/"project").absolute())))
                else:
                    shutil.copyfile(source,target)
                mapping[relative]=_hash(target)
            hashes[label]=mapping
        doc={
            "schema":SCHEMA,
            "4096_source_stage":str(stage_4096),
            "4096_native_run":str(native_4096),
            "parent_18_6_stage":str(parent),
            "native_replay_output_sha256":qualified[
                "actual_160_native_output_sha256"],
            "native_replay_log_sha256":qualified["native_vitis_log_sha256"],
            "synthesis_project_sha256":hashes,
            "source_changed_only_synthesis_controls":True,
            "native_synthesis_executed_by_staging":False,
            "qualifies_full_model_e2":False,
            "qualifies_routed_e4":False,
            "publication_fair_fpgaivs_hls4ml":False,
            "vendor_tool_executed":False,
        }
        (stage/"experiment.json").write_text(
            json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+"\n")
        _preflight(stage,published_root=dest)
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Study output appeared before atomic publication")
        stage.rename(dest)
    return doc


def _report(project):
    project=Path(project)
    reports=[]
    for file in project.rglob("*_csynth.xml"):
        if (file.is_symlink() or not file.is_file()
                or "syn" not in file.parts or "report" not in file.parts):
            continue
        if file.stat().st_size>4*1024*1024:
            raise ValueError("Unbounded HLS report")
        reports.append(file)
    if len(reports)!=1:
        raise ValueError("Require exactly one native HLS csynth XML, got "+
                         str(len(reports)))
    file=reports[0]
    content=file.read_bytes()
    root=ET.fromstring(content)
    metrics={}
    for group,keys in (
            ("AreaEstimates/Resources",AREA_KEYS),
            ("PerformanceEstimates/SummaryOfTimingAnalysis",TIMING_KEYS),
            ("PerformanceEstimates/SummaryOfOverallLatency",LATENCY_KEYS)):
        elem=root.find(".//"+group)
        if elem is None:
            continue
        for key in keys:
            child=elem.find(key)
            if child is not None and child.text and child.text.strip():
                metrics[key]=child.text.strip()
    if not any(k in metrics for k in AREA_KEYS):
        raise ValueError("No native HLS resource estimates in report")
    return {
        "relative_csynth_report":str(file.relative_to(project)),
        "sha256":sha256(content).hexdigest(),
        "metrics":metrics,
    }


def run_synth_pair(stage,output,*,execute=False,vitis="vitis_hls",timeout=3600):
    root=Path(stage)
    prepared=_preflight(root)
    if not execute:
        return {"schema":RUN_SCHEMA,"ready":True,
                "vendor_tools_executed":False,
                "synthesis_only":True,
                "comparison_scope":"within-hls4ml exploratory E3"}
    if not isinstance(vitis,str) or not vitis or "/" in vitis:
        raise ValueError("Configured Vitis executable must be a name")
    if type(timeout) is not int or not 60<=timeout<=14400:
        raise ValueError("Native synthesis timeout outside bounded limits")
    out=Path(output)
    if out.exists() or out.is_symlink():
        raise FileExistsError("Native synthesis results already exist")
    for label in ("1024","4096"):
        if any((root/label/"project").rglob("*_csynth.xml")):
            raise ValueError("Staged project already has HLS synthesis results")
    out.mkdir(parents=True)
    report={
        "schema":RUN_SCHEMA,
        "stage_receipt_sha256":_hash(root/"experiment.json"),
        "native_tool_execution_requested":True,
        "variant_native_results":{},
        "full_model_e2_qualified":False,
        "external_cross_compiler_qor_eligible":False,
        "routed_e4_qualified":False,
        "vendor_vivado_executed":False,
        "exploratory_within_hls4ml_only":True,
    }
    for label in ("1024","4096"):
        project=root/label/"project"
        try:
            process=subprocess.run(
                [vitis,"-f","build_prj.tcl"],cwd=project,
                capture_output=True,text=True,check=False,
                timeout=timeout)
            output_text=process.stdout+"\n"+process.stderr
            return_code=process.returncode
        except (OSError,subprocess.TimeoutExpired) as exc:
            output_text="Native Vitis HLS failed: "+type(exc).__name__+"\n"
            return_code=-1
        (out/(label+"_vitis_hls.log")).write_text(output_text)
        one={
            "native_attempted":True,
            "native_returncode":return_code,
            "native_log_sha256":sha256(output_text.encode()).hexdigest(),
            "report":None,
        }
        try:
            _preflight(root)
            one["source_rechecked_after_native"]=True
            if return_code==0:
                one["report"]=_report(project)
            else:
                one["negative_reason"]="Vitis HLS exited nonzero"
        except (OSError,ValueError,ET.ParseError) as exc:
            one["negative_reason"]="Report/source audit: "+type(exc).__name__
            one["source_rechecked_after_native"]=False
        report["variant_native_results"][label]=one
        # Don't compare missing, failed or corrupted reports.
        if one["report"] is None:
            break
    both=report["variant_native_results"]
    if len(both)==2 and all(both[k]["report"] is not None for k in ("1024","4096")):
        a=both["1024"]["report"]["metrics"]
        b=both["4096"]["report"]["metrics"]
        common=set(a).intersection(b)
        delta={}
        for k in sorted(common):
            try:
                av,bv=float(a[k]),float(b[k])
            except ValueError:
                continue
            if np_isfinite(av) and np_isfinite(bv):
                delta[k]={"1024":av,"4096":bv,"difference":bv-av}
        report["matched_tool_report_metrics"]=delta
        report["native_exploratory_hls_pair_complete"]=True
    else:
        report["native_exploratory_hls_pair_complete"]=False
    (out/"synthesis_results.json").write_text(
        json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return report


def np_isfinite(value):
    from math import isfinite
    return isfinite(value)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("stage")
    a.add_argument("--source-stage",required=True)
    a.add_argument("--native-run",required=True)
    a.add_argument("--output",required=True)
    b=sub.add_parser("run")
    b.add_argument("--stage",required=True)
    b.add_argument("--output",required=True)
    b.add_argument("--execute",action="store_true")
    b.add_argument("--vitis-executable",default="vitis_hls")
    b.add_argument("--timeout",type=int,default=3600)
    args=p.parse_args(argv)
    if args.command=="stage":
        r=stage_synth_pair(args.source_stage,args.native_run,args.output)
        print(json.dumps({
            "schema":r["schema"],
            "variants":["1024","4096"],
            "only_synthesis_controls_changed":True,
            "vendor_tool_executed":False,
        },indent=2,sort_keys=True))
    else:
        r=run_synth_pair(args.stage,args.output,execute=args.execute,
                         vitis=args.vitis_executable,timeout=args.timeout)
        print(json.dumps(r,indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
