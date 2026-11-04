"""Controlled single-parameter hls4ml Softmax intervention (CSim only).

Stage from source-replayed, hash-verified original RF1 project. Replace
ONLY the selected generated Softmax implementation from stable to latency
in a fresh clone; never modify the original E1/CSim projects. The
separate --execute stage runs only real Vitis CSim and reports original
raw 16x10 probabilities against the original ONNX reference and old RF1.
No automatic E2 upgrade, routed hardware claim or posthoc normalization.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.experiments.hls4ml_csim_numeric import (
    compare_numeric_outputs,numeric_matrix,
)
from fpgai_research.experiments.hls4ml_csim_native_observation import (
    discover_selected_csim_results,
)
from fpgai_research.experiments.hls4ml_csim_stage import csim_only_options
from fpgai_research.experiments.hls4ml_softmax_semantics import (
    inspect_native_softmax_semantics,
)
from fpgai_research.experiments.hls4ml_native_pair_reaudit import audit_native_pair

SCHEMA="fpgai-research/hls4ml-controlled-softmax-strategy-csim/v1"
RUN_SCHEMA="fpgai-research/hls4ml-controlled-softmax-native-result/v1"
MUTATION_FROM="stable"
MUTATION_TO="latency"
MAX_BYTES=64*1024*1024
ROOT_EXT=frozenset({".tcl",".cpp",".cc",".cxx",".h",".hpp",".yaml",".yml"})
TARGET=re.compile(
    r"\bimplementation\s*=\s*(?:nnet::)?softmax_implementation::stable\b"
)
DESTINATION_TOKEN="softmax_implementation::latency"


def _regular(file,limit=16*1024*1024):
    p=Path(file)
    if p.is_symlink() or not p.is_file() or not 0<p.stat().st_size<=limit:
        raise ValueError("Missing bounded regular HLS source")
    return p


def _static_source_inventory(project):
    """Allow only project scripts, model firmware and selected testbench."""
    source=Path(project)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Original native project not an ordinary directory")
    files={}
    count=0
    total=0
    for item in sorted(source.iterdir()):
        if item.is_symlink():
            raise ValueError("Original project has linked top-level file")
        if item.is_file() and item.suffix.lower() in ROOT_EXT:
            files[item.name]=item
        elif item.is_dir() and item.name=="firmware":
            for p in sorted(item.rglob("*")):
                if p.is_symlink() or not (p.is_dir() or p.is_file()):
                    raise ValueError("Original firmware includes unsafe linked entry")
                if p.is_file():
                    files[str(p.relative_to(source))]=p
        elif item.is_dir() and item.name=="tb_data":
            for name in ("tb_input_features.dat","tb_output_predictions.dat"):
                files["tb_data/"+name]=_regular(item/name)
    for mandatory in (
        "build_prj.tcl","project.tcl","build_opt.tcl",
        "hls4ml_config.yml","fpgairesearch_baseline_test.cpp",
        "firmware/parameters.h","firmware/nnet_utils/nnet_activation.h",
        "tb_data/tb_input_features.dat","tb_data/tb_output_predictions.dat",
    ):
        if mandatory not in files:
            raise ValueError("Missing essential original E1/CSim source: "+mandatory)
    hashes={}
    for rel,p in files.items():
        p=_regular(p)
        count+=1
        total+=p.stat().st_size
        if count>3000 or total>MAX_BYTES:
            raise ValueError("Unbounded original static source inventory")
        hashes[rel]=sha256(p.read_bytes()).hexdigest()
    return files,hashes


def _prepare_target_header(source):
    original=source.decode("utf-8")
    matches=list(TARGET.finditer(original))
    if len(matches)!=1:
        raise ValueError("Exactly one selected stable Softmax implementation required")
    match=matches[0]
    before=original[match.start():match.end()]
    replacement=before.replace(
        "softmax_implementation::stable",DESTINATION_TOKEN)
    if replacement==before or replacement.count(DESTINATION_TOKEN)!=1:
        raise ValueError("Cannot isolate one Softmax strategy setting")
    modified=original[:match.start()]+replacement+original[match.end():]
    if (modified.replace(replacement,before,1)!=original
            or original.count("softmax_implementation::stable")!=1):
        raise ValueError("Source edit must be exactly one strategy token")
    return modified.encode("utf-8")


def _read_receipt(stage, *, published_root=None):
    stage=Path(stage)
    file=_regular(stage/"experiment.json",1024*1024)
    manifest=json.loads(file.read_text())
    if (not isinstance(manifest,dict) or manifest.get("schema")!=SCHEMA
            or manifest.get("source_changed_only_parameter") is not True
            or manifest.get("change")!={
                "parameter":"softmax_implementation",
                "from":"stable","to":"latency"}):
        raise ValueError("Unknown controlled native stage")
    expected=manifest.get("prepared_files_sha256")
    original=manifest.get("source_files_sha256")
    baseline=manifest.get("source_stage_dir")
    if (not isinstance(expected,dict) or not expected
            or not isinstance(original,dict) or not original
            or not isinstance(baseline,str) or not baseline):
        raise ValueError("Missing original and generated source fingerprints")
    original_project=Path(baseline)/"csim_rf1"/"project"
    files,current_original=_static_source_inventory(original_project)
    if current_original!=original or set(expected)!=set(original):
        raise ValueError("Original native source or source manifest changed")
    old_result=discover_selected_csim_results(Path(baseline)/"csim_rf1")
    if sha256(old_result.read_bytes()).hexdigest()!=manifest.get(
            "original_rf1_csim_output_sha256"):
        raise ValueError("Original native RF1 numerical output changed")
    # Verify the entire actual intervention against the genuine baseline,
    # not against an attacker-edited replacement receipt.
    if (expected["firmware/parameters.h"]!=sha256(
            _prepare_target_header(
                _regular(files["firmware/parameters.h"]).read_bytes())).hexdigest()
            or expected["hls4ml_config.yml"]!=sha256(
                _regular(files["hls4ml_config.yml"]).read_text().replace(
                    str(original_project.absolute()),
                    str(((Path(published_root) if published_root is not None
                           else stage)/"project").absolute())).encode()).hexdigest()
            or any(expected[key]!=original[key] for key in original
                   if key not in ("firmware/parameters.h",
                                  "hls4ml_config.yml"))):
        raise ValueError("Manifest does not describe exactly one source intervention")
    for relative,digest in expected.items():
        rel=Path(relative)
        if (not isinstance(relative,str) or rel.is_absolute()
                or ".." in rel.parts or len(digest)!=64):
            raise ValueError("Invalid source manifest path or SHA")
        p=_regular(stage/"project"/rel)
        if sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError("Staged native HLS source mutated")
    opt=_regular(stage/"project"/"build_opt.tcl").read_text()
    if opt!=csim_only_options(opt):
        raise ValueError("Staged source not limited to Vitis CSim")
    return manifest


def stage_softmax_intervention(stage_root,global_default,native_run,output):
    root=Path(stage_root)
    destination=Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Controlled CSim stage directory already exists")
    evidence=inspect_native_softmax_semantics(
        stage_root,global_default,native_run)
    if (evidence.get("native_pair_source_and_log_reattested") is not True
            or evidence["variants"]["rf1"]["selected_implementation"]!="stable"
            or evidence["variants"]["rf2"]["selected_implementation"]!="stable"
            or not evidence["same_bundled_implementation_sha256"]):
        raise ValueError("Original native Softmax source replay or stable strategy missing")
    # Original CSim generated C++ must match source and native logs.
    project=root/"csim_rf1"/"project"
    files,prior_hashes=_static_source_inventory(project)
    original_rf1_result=discover_selected_csim_results(root/"csim_rf1")
    original_rf1_sha256=sha256(original_rf1_result.read_bytes()).hexdigest()
    parameter=_regular(files["firmware/parameters.h"]).read_bytes()
    replacement=_prepare_target_header(parameter)
    old_config=_regular(files["hls4ml_config.yml"]).read_text()
    if (not 1<=old_config.count(str(project.absolute()))<=8):
        raise ValueError("Cannot uniquely relocate generated hls4ml project path")
    old_opt=_regular(files["build_opt.tcl"]).read_text()
    if old_opt!=csim_only_options(old_opt):
        raise ValueError("Prior native stage must have only CSim tools enabled")
    new_config=old_config.replace(
        str(project.absolute()),str((destination/"project").absolute()))
    destination.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=destination.parent,prefix=".hls-softmax-change-") as tmp:
        work=Path(tmp)/"stage"
        output_project=work/"project"
        output_project.mkdir(parents=True)
        hashes={}
        for rel,source in files.items():
            copied=output_project/rel
            copied.parent.mkdir(parents=True,exist_ok=True)
            if rel=="firmware/parameters.h":
                copied.write_bytes(replacement)
            elif rel=="hls4ml_config.yml":
                copied.write_text(new_config)
            else:
                shutil.copyfile(source,copied)
            hashes[rel]=sha256(copied.read_bytes()).hexdigest()
        different=[
            path for path in hashes if hashes[path]!=prior_hashes[path]]
        if sorted(different)!=[
            "firmware/parameters.h","hls4ml_config.yml"]:
            raise ValueError("Unexpected static generated source difference")
        # No direct vendor build output is copied. No Python or HLS
        # training weights have been edited or regenerated.
        manifest={
            "schema":SCHEMA,
            "change":{"parameter":"softmax_implementation",
                      "from":"stable","to":"latency"},
            "source_changed_only_parameter":True,
            "source_stage_dir":str(root),
            "source_stage_parameters_sha256":prior_hashes[
                "firmware/parameters.h"],
            "source_stage_softmax_library_sha256":prior_hashes[
                "firmware/nnet_utils/nnet_activation.h"],
            "previous_native_receipt_sha256":evidence[
                "source_replay_receipt_sha256"],
            "original_rf1_csim_output_sha256":original_rf1_sha256,
            "reference_tb_sha256":prior_hashes[
                "tb_data/tb_output_predictions.dat"],
            "inputs_tb_sha256":prior_hashes["tb_data/tb_input_features.dat"],
            "source_files_sha256":prior_hashes,
            "prepared_files_sha256":hashes,
            "changed_project_files":sorted(different),
            "native_vitis_csim_executed":False,
            "modified_native_source_e2_qualified":False,
            "original_preexisting_numerical_observation_preserved":True,
            "hls_synthesis_executed":False,
            "vivado_executed":False,
            "raw_numeric_max_error_limit":0.05,
            "row_sum_semantics_limit":0.01,
        }
        (work/"experiment.json").write_text(json.dumps(
            manifest,sort_keys=True,indent=2,allow_nan=False)+"\n")
        _read_receipt(work,published_root=destination)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Controlled stage appeared during immutable preparation")
        work.rename(destination)
    return manifest


def run_softmax_intervention(stage,output,*,execute=False,
                             vitis="vitis_hls",timeout=1800):
    src=Path(stage)
    receipt=_read_receipt(src)
    if not execute:
        return {"schema":RUN_SCHEMA,"ready":True,
                "native_tools_executed":False,
                "project":str(src/"project"),
                "raw_numeric_policy_limit":0.05}
    if type(timeout) is not int or not 10<=timeout<=7200:
        raise ValueError("Native CSim timeout must be 10..7200 seconds")
    if not isinstance(vitis,str) or not vitis or "/" in vitis:
        raise ValueError("Vitis executable must be a configured executable name")
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Native result directory already exists")
    dest.mkdir(parents=True)
    project=src/"project"
    try:
        done=subprocess.run(
            [vitis,"-f","build_prj.tcl"],cwd=project,
            capture_output=True,text=True,
            timeout=timeout,check=False)
        logs=done.stdout+"\n"+done.stderr
        code=done.returncode
    except (OSError,subprocess.TimeoutExpired) as exc:
        logs="Native Vitis CSim failed: "+type(exc).__name__+"\n"
        code=-1
    logfile=dest/"native_vitis_csim.log"
    logfile.write_text(logs)
    result={
        "schema":RUN_SCHEMA,
        "original_stage_receipt_sha256":sha256(
            _regular(src/"experiment.json").read_bytes()).hexdigest(),
        "native_tools_executed":True,
        "native_csim_attempted":True,
        "native_csim_exited_zero":code==0,
        "native_csim_output_log_sha256":sha256(logs.encode()).hexdigest(),
        "native_csim_completion_logged":bool(
            code==0 and "CSim done with 0 errors" in logs),
        "raw_numeric_policy_limit":0.05,
        "row_sum_semantics_limit":0.01,
        "numerical_result":None,
        "native_e2_qualified_by_this_driver":False,
        "external_qor_comparison_eligible":False,
        "vendor_hls_synthesis_executed":False,
        "vivado_executed":False,
    }
    try:
        _read_receipt(src)
        if not result["native_csim_completion_logged"]:
            result["negative_reason"]="Native Vitis CSim did not complete"
        else:
            results=discover_selected_csim_results(src)
            values=numeric_matrix(results.read_text())
            reference=numeric_matrix((project/"tb_data"/"tb_output_predictions.dat").read_text())
            raw=compare_numeric_outputs(reference,values,absolute_limit=0.05)
            original_path=discover_selected_csim_results(
                Path(receipt["source_stage_dir"])/"csim_rf1")
            original_values=numeric_matrix(original_path.read_text())
            original_raw=compare_numeric_outputs(
                reference,original_values,absolute_limit=0.05)
            difference=np.abs(values-original_values)
            sums=np.sum(values,axis=1)
            row_sum_max=float(np.max(np.abs(sums-1)))
            range_bad=int(np.count_nonzero((values<0)|(values>1)))
            result["numerical_result"]={
                **raw,
                "max_probability_row_sum_error":row_sum_max,
                "out_of_range_probabilities":range_bad,
                "probability_semantics_valid":bool(
                    row_sum_max<=0.01 and range_bad==0),
                "raw_numeric_and_semantics_passed":bool(
                    raw["selected_numeric_policy_passed"]
                    and row_sum_max<=0.01 and range_bad==0),
                "native_output_sha256":sha256(results.read_bytes()).hexdigest(),
                "original_onnx_probability_reference_sha256":
                    receipt["reference_tb_sha256"],
                "original_rf1_raw_max_error":
                    original_raw["selected_probability_max_absolute_difference"],
                "new_minus_original_rf1_max_abs_output_difference":
                    float(np.max(difference)),
                "new_minus_original_rf1_mean_abs_output_difference":
                    float(np.mean(difference)),
                "raw_max_error_change_vs_original_rf1":
                    (raw["selected_probability_max_absolute_difference"]
                     - original_raw["selected_probability_max_absolute_difference"]),
                "no_posthoc_normalization_used":True,
            }
            if (not result["numerical_result"][
                    "raw_numeric_and_semantics_passed"]):
                result["negative_reason"]="Raw numerical or probability semantics failed"
    except (OSError,ValueError) as exc:
        result["negative_reason"]="Source/output audit failed: "+type(exc).__name__
    (dest/"native_result.json").write_text(json.dumps(
        result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("stage")
    a.add_argument("--stage-root",required=True)
    a.add_argument("--global-default",required=True)
    a.add_argument("--native-run",required=True)
    a.add_argument("--output",required=True)
    b=sub.add_parser("run")
    b.add_argument("--stage",required=True)
    b.add_argument("--output",required=True)
    b.add_argument("--execute",action="store_true")
    b.add_argument("--vitis-executable",default="vitis_hls")
    args=p.parse_args(argv)
    if args.command=="stage":
        r=stage_softmax_intervention(
            args.stage_root,args.global_default,args.native_run,args.output)
        print(json.dumps({
            "schema":r["schema"],
            "changed_project_files":r["changed_project_files"],
            "source_changed_only_parameter":r["source_changed_only_parameter"],
            "native_tools_executed":False,
        },sort_keys=True,indent=2))
    else:
        r=run_softmax_intervention(args.stage,args.output,
                                  execute=args.execute,
                                  vitis=args.vitis_executable)
        print(json.dumps(r,sort_keys=True,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
