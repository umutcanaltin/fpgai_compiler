"""Controlled hls4ml stable Softmax reciprocal address-resolution experiment.

Change only the generated Softmax config's inv_inp_t typedef from the
original independently attested signed 18/8 fixed-point type to 18/6, preserving
signedness and all explicit rounding/saturation modes. No reciprocal-value
precision, exp-table,
accumulator, external input/output precision, selected strategy, weights,
sample cohort or acceptance criteria change. Clone new static CSim project
with a relocated OutputDir, then require explicit --execute for real Vitis.
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

from fpgai_research.experiments.hls4ml_softmax_semantics import (
    inspect_native_softmax_semantics,_parse_config,
)
from fpgai_research.experiments.hls4ml_softmax_intervention import (
    _regular,_static_source_inventory,
)
from fpgai_research.experiments.hls4ml_csim_stage import csim_only_options
from fpgai_research.experiments.hls4ml_csim_native_observation import (
    discover_selected_csim_results,
)
from fpgai_research.experiments.hls4ml_csim_numeric import (
    compare_numeric_outputs,numeric_matrix,
)

SCHEMA="fpgai-research/controlled-softmax-inverse-address-stage/v1"
RUN_SCHEMA="fpgai-research/controlled-softmax-inverse-address-native-csim/v1"
WIDTH=18
TARGET_INTEGER_BITS=6
ADDRESS_BITS=10
FROZEN_MAX_ERROR=0.05
FROZEN_ROW_SUM_ERROR=0.01
ROLE_RE=re.compile(
    r"(?m)^(?P<indent>[ \t]*)typedef[ \t]+"
    r"(?P<source>[A-Za-z_]\w*)[ \t]+inv_inp_t[ \t]*;"
)


def _prepare(firmware):
    meta=_parse_config(firmware)
    if meta["selected_implementation"]!="stable":
        raise ValueError("Source baseline is not stable Softmax")
    original=meta["config_type_roles"]["inv_inp_t"]
    precision=original["precision"]
    if (original["declared_in_config"] is not True
            or not isinstance(precision,dict)
            or precision["width_bits"]!=18
            or precision["integer_bits"]!=8
            or precision["fraction_bits"]!=10):
        raise ValueError("Original inverse lookup-address type is not independently attested 18/8")
    modes=precision["explicit_modes"]
    if any(mode not in (
            "AP_RND","AP_RND_ZERO","AP_RND_INF","AP_RND_MIN_INF",
            "AP_RND_CONV","AP_TRN","AP_TRN_ZERO",
            "AP_SAT","AP_SAT_ZERO","AP_SAT_SYM","AP_WRAP","AP_WRAP_SM")
           for mode in modes):
        raise ValueError("Unsupported original rounding/overflow mode")
    if len(modes)>2:
        raise ValueError("Ambiguous original numerical modes")
    src=firmware["parameters.h"]["content"]
    match=list(ROLE_RE.finditer(src))
    if len(match)!=1:
        raise ValueError("Expected exactly one simple inv_inp_t type alias")
    old=match[0].group(0)
    representation=("ap_fixed" if precision["signed"] else "ap_ufixed")
    signature=f"{representation}<{WIDTH},{TARGET_INTEGER_BITS}"
    if modes:
        signature+=","+",".join(modes)
    signature+=">"
    new=match[0].group("indent")+"typedef "+signature+" inv_inp_t;"
    changed=src[:match[0].start()]+new+src[match[0].end():]
    if (changed.count(new)!=1 or
            changed.replace(new,old,1)!=src or
            "stable" not in changed):
        raise ValueError("More than one generated type role changed")
    if _parse_config({**firmware,
                      "parameters.h":{"content":changed}})[
                          "selected_implementation"]!="stable":
        raise ValueError("Generated Softmax implementation changed unexpectedly")
    return changed.encode("utf-8"),{
        "role":"inv_inp_t",
        "original_width_bits":precision["width_bits"],
        "new_width_bits":WIDTH,
        "original_integer_bits":precision["integer_bits"],
        "new_integer_bits":TARGET_INTEGER_BITS,
        "lookup_address_bits":ADDRESS_BITS,
        "original_address_step":2.0**(precision["integer_bits"]-ADDRESS_BITS),
        "new_address_step":2.0**(TARGET_INTEGER_BITS-ADDRESS_BITS),
        "original_fraction_bits":precision["fraction_bits"],
        "new_fraction_bits":WIDTH-TARGET_INTEGER_BITS,
        "signed":precision["signed"],
        "explicit_modes":modes,
    }


def _original(root):
    project=Path(root)/"csim_rf1"/"project"
    files,hashes=_static_source_inventory(project)
    firmware={}
    for rel,p in files.items():
        if rel.startswith("firmware/") and Path(rel).parent==Path("firmware"):
            firmware[Path(rel).name]={"content":_regular(p).read_text()}
    if "parameters.h" not in firmware or "defines.h" not in firmware:
        raise ValueError("Original generated model headers unavailable")
    changed,control=_prepare(firmware)
    old_results=discover_selected_csim_results(Path(root)/"csim_rf1")
    result_sha=sha256(_regular(old_results,4*1024*1024).read_bytes()).hexdigest()
    old_yaml=_regular(files["hls4ml_config.yml"]).read_text()
    if old_yaml.count(str(project.absolute()))<1:
        raise ValueError("Cannot rebase original hls4ml source path")
    opt=_regular(files["build_opt.tcl"]).read_text()
    if opt!=csim_only_options(opt):
        raise ValueError("Original Vitis configuration is not CSim-only")
    return project,files,hashes,changed,control,result_sha,old_yaml


def _stage_audit(root,*,published_root=None):
    dest=Path(root)
    doc=json.loads(_regular(dest/"experiment.json",2*1024*1024).read_text())
    if (doc.get("schema")!=SCHEMA
            or doc.get("stage_executed_native") is not False
            or doc.get("scientific_e2_qualified") is not False
            or doc.get("frozen_max_error")!=FROZEN_MAX_ERROR
            or doc.get("frozen_row_sum_error")!=FROZEN_ROW_SUM_ERROR):
        raise ValueError("Unrecognized native inverse-address experiment")
    baseline=doc.get("baseline_root")
    if not isinstance(baseline,str) or not baseline:
        raise ValueError("Missing original native source anchor")
    project,files,hashes,changed,control,old_hash,old_yaml=_original(baseline)
    if (doc.get("original_source_sha256")!=hashes
            or doc.get("original_native_output_sha256")!=old_hash
            or doc.get("precision_control")!=control):
        raise ValueError("Original native generated source/outputs changed")
    recorded=doc.get("staged_source_sha256")
    if not isinstance(recorded,dict) or set(recorded)!=set(hashes):
        raise ValueError("Missing complete staged source inventory")
    final=Path(published_root) if published_root is not None else dest
    expected=dict(hashes)
    expected["firmware/parameters.h"]=sha256(changed).hexdigest()
    expected["hls4ml_config.yml"]=sha256(
        old_yaml.replace(str(project.absolute()),str(
            (final/"project").absolute())).encode()).hexdigest()
    if recorded!=expected:
        raise ValueError("Experiment manifest claims additional source changes")
    source_dir=dest/"project"
    for rel,digest in expected.items():
        file=_regular(source_dir/rel)
        if sha256(file.read_bytes()).hexdigest()!=digest:
            raise ValueError("Staged HLS source differs from independently reconstructed edit")
    if _regular(source_dir/"build_opt.tcl").read_text()!=csim_only_options(
            _regular(source_dir/"build_opt.tcl").read_text()):
        raise ValueError("Staged project could run non-CSim vendor stages")
    return doc


def stage_inverse_address(stage_root,global_default,native_run,output):
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("New immutable inverse-address stage required")
    verified=inspect_native_softmax_semantics(
        stage_root,global_default,native_run)
    if (verified.get("native_pair_source_and_log_reattested") is not True
            or not verified.get("same_bundled_implementation_sha256")
            or any(verified["variants"][label]["declared_dimensions_and_tables"].get("inv_table_size")!=1024
                   or verified["variants"][label]["template_features"].get("index_helper_mentions_msb_slicing") is not True
                   or verified["variants"][label]["template_features"].get("reciprocal_table_lookup_present") is not True
                   for label in ("rf1","rf2"))
            or any(verified["variants"][key]["selected_implementation"]!="stable"
                   for key in ("rf1","rf2"))):
        raise ValueError("Original native source/strategy not independently verified")
    project,files,hashes,changed,control,old_hash,yaml=_original(stage_root)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent,prefix=".softmax-inverse-address-") as temp:
        stage=Path(temp)/"stage"
        clone=stage/"project"
        clone.mkdir(parents=True)
        stage_hash={}
        for rel,source in files.items():
            out=clone/rel
            out.parent.mkdir(parents=True,exist_ok=True)
            if rel=="firmware/parameters.h":
                out.write_bytes(changed)
            elif rel=="hls4ml_config.yml":
                out.write_text(yaml.replace(
                    str(project.absolute()),str((dest/"project").absolute())))
            else:
                shutil.copyfile(source,out)
            stage_hash[rel]=sha256(out.read_bytes()).hexdigest()
        diff=sorted(k for k in hashes if hashes[k]!=stage_hash[k])
        if diff!=["firmware/parameters.h","hls4ml_config.yml"]:
            raise ValueError("Unexpected generated source changes")
        doc={
            "schema":SCHEMA,
            "baseline_root":str(stage_root),
            "original_native_replay_receipt_sha256":
                verified["source_replay_receipt_sha256"],
            "original_source_sha256":hashes,
            "original_native_output_sha256":old_hash,
            "staged_source_sha256":stage_hash,
            "precision_control":control,
            "changed_source_files":diff,
            "source_isolated":True,
            "frozen_max_error":FROZEN_MAX_ERROR,
            "frozen_row_sum_error":FROZEN_ROW_SUM_ERROR,
            "stage_executed_native":False,
            "scientific_e2_qualified":False,
            "vendor_hls_synthesis_executed":False,
            "vendor_vivado_executed":False,
        }
        (stage/"experiment.json").write_text(json.dumps(
            doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
        _stage_audit(stage,published_root=dest)
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Destination appeared during immutable publication")
        stage.rename(dest)
    return doc


def run_inverse_address(stage,output,*,execute=False,
                          vitis="vitis_hls",timeout=1800):
    src=Path(stage)
    doc=_stage_audit(src)
    if not execute:
        return {
            "schema":RUN_SCHEMA,"ready":True,
            "vendor_tools_executed":False,
            "changed_source_files":doc["changed_source_files"],
            "precision_control":doc["precision_control"],
            "frozen_max_error":FROZEN_MAX_ERROR,
        }
    if type(timeout) is not int or not 10<=timeout<=7200:
        raise ValueError("Vitis timeout outside safety bounds")
    if not isinstance(vitis,str) or not vitis or "/" in vitis:
        raise ValueError("Vitis must be a configured tool executable name")
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Use new native result directory")
    dest.mkdir(parents=True)
    try:
        run=subprocess.run(
            [vitis,"-f","build_prj.tcl"],cwd=src/"project",
            capture_output=True,text=True,timeout=timeout,check=False)
        log=run.stdout+"\n"+run.stderr
        code=run.returncode
    except (OSError,subprocess.TimeoutExpired) as err:
        log="Vitis CSim did not finish: "+type(err).__name__+"\n"
        code=-1
    (dest/"native_vitis_csim.log").write_text(log)
    finished=code==0 and "CSim done with 0 errors" in log
    report={
        "schema":RUN_SCHEMA,
        "native_vendor_csim_attempted":True,
        "native_vendor_csim_complete":finished,
        "native_log_sha256":sha256(log.encode()).hexdigest(),
        "stage_receipt_sha256":sha256(
            _regular(src/"experiment.json").read_bytes()).hexdigest(),
        "precision_control":doc["precision_control"],
        "frozen_max_error":FROZEN_MAX_ERROR,
        "frozen_row_sum_error":FROZEN_ROW_SUM_ERROR,
        "numerical_observation":None,
        "source_replay_after_native":False,
        "native_e2_qualified_by_analysis":False,
        "cross_compiler_hardware_qor_eligible":False,
        "vendor_hls_synthesis_executed":False,
        "vendor_vivado_executed":False,
    }
    try:
        _stage_audit(src)
        report["source_replay_after_native"]=True
        if finished:
            result_file=discover_selected_csim_results(src)
            result=numeric_matrix(_regular(result_file,4*1024*1024).read_text())
            reference=numeric_matrix(_regular(
                src/"project"/"tb_data"/"tb_output_predictions.dat").read_text())
            original_file=discover_selected_csim_results(
                Path(doc["baseline_root"])/"csim_rf1")
            old=numeric_matrix(_regular(original_file,4*1024*1024).read_text())
            raw=compare_numeric_outputs(reference,result,
                                        absolute_limit=FROZEN_MAX_ERROR)
            baseline=compare_numeric_outputs(reference,old,
                                             absolute_limit=FROZEN_MAX_ERROR)
            sums=result.sum(axis=1)
            sum_error=float(np.max(np.abs(sums-1)))
            out_of_range=int(np.count_nonzero((result<0)|(result>1)))
            report["numerical_observation"]={
                **raw,
                "max_row_sum_error":sum_error,
                "out_of_range_probability_count":out_of_range,
                "original_rf1_raw_max_error":
                    baseline["selected_probability_max_absolute_difference"],
                "raw_max_error_delta_vs_original":
                    raw["selected_probability_max_absolute_difference"]-
                    baseline["selected_probability_max_absolute_difference"],
                "new_vs_original_max_abs_output_delta":
                    float(np.max(np.abs(result-old))),
                "raw_policy_and_probability_semantics_pass":bool(
                    raw["selected_numeric_policy_passed"]
                    and sum_error<=FROZEN_ROW_SUM_ERROR
                    and out_of_range==0),
                "actual_native_output_sha256":sha256(result_file.read_bytes()).hexdigest(),
                "posthoc_normalization_performed":False,
            }
            if not report["numerical_observation"][
                    "raw_policy_and_probability_semantics_pass"]:
                report["negative_reason"]="Frozen raw E2 numerical policy failed"
        else:
            report["negative_reason"]="Vitis CSim exited without native completion"
    except (OSError,ValueError) as err:
        report["negative_reason"]="Native source/output replay failed: "+type(err).__name__
    (dest/"native_result.json").write_text(json.dumps(
        report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    prep=commands.add_parser("stage")
    prep.add_argument("--stage-root",required=True)
    prep.add_argument("--global-default",required=True)
    prep.add_argument("--native-run",required=True)
    prep.add_argument("--output",required=True)
    run=commands.add_parser("run")
    run.add_argument("--stage",required=True)
    run.add_argument("--output",required=True)
    run.add_argument("--execute",action="store_true")
    run.add_argument("--vitis-executable",default="vitis_hls")
    args=parser.parse_args(argv)
    if args.command=="stage":
        result=stage_inverse_address(
            args.stage_root,args.global_default,args.native_run,args.output)
        print(json.dumps({
            "schema":SCHEMA,
            "source_isolated":result["source_isolated"],
            "precision_control":result["precision_control"],
            "changed_source_files":result["changed_source_files"],
            "vendor_tools_executed":False,
        },sort_keys=True,indent=2))
    else:
        result=run_inverse_address(
            args.stage,args.output,execute=args.execute,
            vitis=args.vitis_executable)
        print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
