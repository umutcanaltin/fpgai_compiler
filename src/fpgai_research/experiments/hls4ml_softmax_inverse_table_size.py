"""CSim-only reciprocal-table resolution continuation of audited 18/6 Softmax.

New source change: inv_table_size 1024 -> 4096. The parent MUST be the
previously executed, independently source-qualified inv_inp_t=18/6
experiment; original RF1 native results and the 18/6 result are rechecked.
No original source changes, posthoc normalization or paper-facing E2 claim.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from fpgai_research.experiments.hls4ml_softmax_inverse_address import (
    _stage_audit as audit_address_stage, RUN_SCHEMA as ADDRESS_RUN_SCHEMA,
)
from fpgai_research.experiments.hls4ml_softmax_intervention import (
    _regular, _static_source_inventory,
)
from fpgai_research.experiments.hls4ml_softmax_semantics import _parse_config
from fpgai_research.experiments.hls4ml_csim_stage import csim_only_options
from fpgai_research.experiments.hls4ml_csim_native_observation import (
    discover_selected_csim_results,
)
from fpgai_research.experiments.hls4ml_csim_numeric import (
    compare_numeric_outputs, numeric_matrix,
)

SCHEMA = "fpgai-research/controlled-softmax-inverse-table-size-stage/v1"
RUN_SCHEMA = "fpgai-research/controlled-softmax-inverse-table-size-native/v1"
OLD_SIZE = 1024
NEW_SIZE = 4096
FROZEN_ABS_LIMIT = 0.05
FROZEN_SUM_LIMIT = 0.01
SIZE_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)static[ \t]+const[ \t]+"
    r"(?P<kind>unsigned|int)[ \t]+inv_table_size[ \t]*="
    r"[ \t]*1024[ \t]*;"
)


def _hash(path, limit=4*1024*1024):
    return sha256(_regular(path,limit).read_bytes()).hexdigest()


def _firmware(files):
    return {
        Path(rel).name: {"content": _regular(path).read_text()}
        for rel,path in files.items()
        if rel.startswith("firmware/")
        and Path(rel).parent==Path("firmware")
    }


def _change_table(files):
    prior=_firmware(files)
    old=_parse_config(prior)
    dims=old["declared_dimensions_and_tables"]
    address=old["config_type_roles"]["inv_inp_t"]["precision"]
    if (old["selected_implementation"]!="stable"
            or dims.get("inv_table_size")!=OLD_SIZE
            or dims.get("exp_table_size")!=1024
            or not isinstance(address,dict)
            or address["width_bits"]!=18
            or address["integer_bits"]!=6
            or address["fraction_bits"]!=12):
        raise ValueError("Unexpected stable 18/6 parent or lookup configuration")
    raw=prior["parameters.h"]["content"]
    matches=list(SIZE_RE.finditer(raw))
    if len(matches)!=1:
        raise ValueError("Exactly one literal original inverse lookup size required")
    m=matches[0]
    original=m.group(0)
    modified=(
        m.group("indent")+f"static const {m.group('kind')} "
        f"inv_table_size = {NEW_SIZE};"
    )
    patched=raw[:m.start()]+modified+raw[m.end():]
    if patched.replace(modified,original,1)!=raw:
        raise ValueError("Inverse-table size edit cannot be isolated")
    new=_parse_config({**prior,"parameters.h":{"content":patched}})
    if (new["declared_dimensions_and_tables"]!={**dims,
                                                  "inv_table_size":NEW_SIZE}
            or new["selected_implementation"]!="stable"
            or new["config_type_roles"]!=old["config_type_roles"]
            or new["call_input_precision"]!=old["call_input_precision"]
            or new["call_output_precision"]!=old["call_output_precision"]):
        raise ValueError("New config changes unrelated Softmax controls")
    return patched.encode("utf-8")


def _parent(parent,prior_run):
    parent=Path(parent)
    prior_run=Path(prior_run)
    record=audit_address_stage(parent)
    control=record["precision_control"]
    if (control.get("role")!="inv_inp_t"
            or control.get("new_width_bits")!=18
            or control.get("new_integer_bits")!=6
            or record.get("frozen_max_error")!=FROZEN_ABS_LIMIT
            or record.get("frozen_row_sum_error")!=FROZEN_SUM_LIMIT):
        raise ValueError("Parent is not the source-verified 18/6 experiment")
    native_file=_regular(prior_run/"native_result.json",2*1024*1024)
    native_raw=native_file.read_bytes()
    native=json.loads(native_raw)
    measurement=native.get("numerical_observation")
    log_path=_regular(prior_run/"native_vitis_csim.log",16*1024*1024)
    log_sha=sha256(log_path.read_bytes()).hexdigest()
    if (native.get("schema")!=ADDRESS_RUN_SCHEMA
            or native.get("native_vendor_csim_attempted") is not True
            or native.get("native_log_sha256")!=log_sha
            or "CSim done with 0 errors" not in log_path.read_text()
            or native.get("native_vendor_csim_complete") is not True
            or native.get("source_replay_after_native") is not True
            or native.get("native_e2_qualified_by_analysis") is not False
            or native.get("cross_compiler_hardware_qor_eligible") is not False
            or not isinstance(measurement,dict)
            or measurement.get("raw_policy_and_probability_semantics_pass") is not False
            or measurement.get("selected_numeric_policy_passed") is not True
            or measurement.get("posthoc_normalization_performed") is not False
            or native.get("stage_receipt_sha256")!=_hash(
                parent/"experiment.json",2*1024*1024)):
        raise ValueError("Parent native result is not qualified as a genuine partial observation")
    output=discover_selected_csim_results(parent)
    if _hash(output)!=measurement.get("actual_native_output_sha256"):
        raise ValueError("Parent native output changed after original native receipt")
    prior=numeric_matrix(_regular(output).read_text())
    ref=numeric_matrix(_regular(
        parent/"project"/"tb_data"/"tb_output_predictions.dat").read_text())
    check=compare_numeric_outputs(ref,prior,absolute_limit=FROZEN_ABS_LIMIT)
    row_sum=float(np.max(np.abs(prior.sum(axis=1)-1)))
    if (not isclose(check["selected_probability_max_absolute_difference"],
                    measurement.get("selected_probability_max_absolute_difference",-1),
                    abs_tol=1e-9,rel_tol=0)
            or not isclose(row_sum,measurement.get("max_row_sum_error",-1),
                           abs_tol=1e-9,rel_tol=0)
            or check["selected_top1_disagreements"]!=measurement.get(
                "selected_top1_disagreements")):
        raise ValueError("Parent numerical receipt contradicts its original outputs")
    files,hashes=_static_source_inventory(parent/"project")
    changed=_change_table(files)
    config=_regular(files["hls4ml_config.yml"]).read_text()
    project=parent/"project"
    if config.count(str(project.absolute()))<1:
        raise ValueError("Parent project configuration cannot be relocated")
    script=_regular(files["build_opt.tcl"]).read_text()
    if script!=csim_only_options(script):
        raise ValueError("Parent project has non-CSim vendor actions")
    return record,native,sha256(native_raw).hexdigest(),files,hashes,changed,config


def _check(stage,*,published_root=None):
    stage=Path(stage)
    manifest=json.loads(_regular(stage/"experiment.json",2*1024*1024).read_text())
    if (manifest.get("schema")!=SCHEMA
            or manifest.get("source_isolated") is not True
            or manifest.get("e2_qualified") is not False
            or manifest.get("frozen_max_error")!=FROZEN_ABS_LIMIT
            or manifest.get("frozen_row_sum_error")!=FROZEN_SUM_LIMIT
            or manifest.get("change")!={"parameter":"inv_table_size",
                                       "old":OLD_SIZE,"new":NEW_SIZE}):
        raise ValueError("Unknown reciprocal lookup-size stage")
    parent=manifest.get("parent_stage")
    run=manifest.get("parent_native_run")
    if not isinstance(parent,str) or not isinstance(run,str):
        raise ValueError("Missing parent source and original native run")
    _,_,native_hash,files,hashes,changed,yaml=_parent(parent,run)
    if (manifest.get("parent_native_receipt_sha256")!=native_hash
            or manifest.get("parent_source_sha256")!=hashes):
        raise ValueError("Original source or native lineage no longer matches")
    expected=dict(hashes)
    expected["firmware/parameters.h"]=sha256(changed).hexdigest()
    final=Path(published_root) if published_root is not None else stage
    expected["hls4ml_config.yml"]=sha256(yaml.replace(
        str((Path(parent)/"project").absolute()),
        str((final/"project").absolute())).encode()).hexdigest()
    if (manifest.get("staged_source_sha256")!=expected
            or manifest.get("changed_files")!=[
                "firmware/parameters.h","hls4ml_config.yml"]):
        raise ValueError("Source manifest permits unexpected changes")
    for rel,digest in expected.items():
        item=_regular(stage/"project"/rel,16*1024*1024)
        if _hash(item,16*1024*1024)!=digest:
            raise ValueError("Staged CSim project differs from independently rebuilt source")
    opts=_regular(stage/"project"/"build_opt.tcl").read_text()
    if opts!=csim_only_options(opts):
        raise ValueError("CSim-only configuration was modified")
    return manifest


def stage_inverse_table_size(parent_stage,parent_native_run,output):
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Use a new immutable CSim stage")
    previous,old_native,native_hash,files,hashes,changed,yaml=_parent(
        parent_stage,parent_native_run)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent,prefix=".softmax-inv-size-") as tmp:
        staged=Path(tmp)/"stage"
        clone=staged/"project"
        clone.mkdir(parents=True)
        newhash={}
        for rel,source in files.items():
            target=clone/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            if rel=="firmware/parameters.h":
                target.write_bytes(changed)
            elif rel=="hls4ml_config.yml":
                target.write_text(yaml.replace(
                    str((Path(parent_stage)/"project").absolute()),
                    str((dest/"project").absolute())))
            else:
                shutil.copyfile(source,target)
            newhash[rel]=_hash(target,16*1024*1024)
        if sorted(x for x in hashes if hashes[x]!=newhash[x])!=[
                "firmware/parameters.h","hls4ml_config.yml"]:
            raise ValueError("New source differs beyond the single lookup-size setting")
        doc={
            "schema":SCHEMA,
            "change":{"parameter":"inv_table_size",
                      "old":OLD_SIZE,"new":NEW_SIZE},
            "parent_stage":str(parent_stage),
            "parent_native_run":str(parent_native_run),
            "parent_native_receipt_sha256":native_hash,
            "parent_source_sha256":hashes,
            "staged_source_sha256":newhash,
            "changed_files":["firmware/parameters.h","hls4ml_config.yml"],
            "source_isolated":True,
            "nominal_original_index_step":0.0625,
            "nominal_new_index_step":0.015625,
            "reciprocal_table_entries_multiplier":4,
            "frozen_max_error":FROZEN_ABS_LIMIT,
            "frozen_row_sum_error":FROZEN_SUM_LIMIT,
            "e2_qualified":False,
            "native_vendor_tools_executed":False,
            "hls_synthesis_executed":False,
            "vivado_executed":False,
        }
        (staged/"experiment.json").write_text(json.dumps(
            doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
        _check(staged,published_root=dest)
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("New CSim stage appeared during atomic publication")
        staged.rename(dest)
    return doc


def run_inverse_table_size(stage,output,*,execute=False,
                           vitis="vitis_hls",timeout=1800):
    staged=Path(stage)
    source=_check(staged)
    if not execute:
        return {
            "schema":RUN_SCHEMA,"ready":True,
            "native_vendor_tools_executed":False,
            "change":source["change"],
            "nominal_new_index_step":source["nominal_new_index_step"],
        }
    if not isinstance(vitis,str) or not vitis or "/" in vitis:
        raise ValueError("Vitis executable must be a configured name")
    if type(timeout) is not int or not 10<=timeout<=7200:
        raise ValueError("Native timeout outside declared bounds")
    output=Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("A native output directory already exists")
    if any((staged/"project").rglob("csim_results.log")):
        raise ValueError("Staged project already has a native CSim output")
    output.mkdir(parents=True)
    try:
        r=subprocess.run(
            [vitis,"-f","build_prj.tcl"],cwd=staged/"project",
            capture_output=True,text=True,timeout=timeout,check=False)
        log=r.stdout+"\n"+r.stderr
        code=r.returncode
    except (OSError,subprocess.TimeoutExpired) as exc:
        log="Native Vitis failed: "+type(exc).__name__+"\n"
        code=-1
    (output/"native_vitis_csim.log").write_text(log)
    completed=code==0 and "CSim done with 0 errors" in log
    result={
        "schema":RUN_SCHEMA,
        "native_vitis_csim_attempted":True,
        "native_vitis_csim_completed":completed,
        "native_log_sha256":sha256(log.encode()).hexdigest(),
        "stage_receipt_sha256":_hash(
            staged/"experiment.json",2*1024*1024),
        "source_rechecked_after_vitis":False,
        "numerical_observation":None,
        "native_e2_qualified_by_driver":False,
        "external_hardware_qor_eligible":False,
        "vendor_hls_synthesis_executed":False,
        "vivado_executed":False,
    }
    try:
        doc=_check(staged)
        result["source_rechecked_after_vitis"]=True
        if not completed:
            result["negative_reason"]="Vitis CSim did not complete"
        else:
            file=discover_selected_csim_results(staged)
            observed=numeric_matrix(_regular(file).read_text())
            reference=numeric_matrix(_regular(
                staged/"project"/"tb_data"/"tb_output_predictions.dat").read_text())
            previous=discover_selected_csim_results(Path(doc["parent_stage"]))
            prior=numeric_matrix(_regular(previous).read_text())
            comparison=compare_numeric_outputs(
                reference,observed,absolute_limit=FROZEN_ABS_LIMIT)
            sums=observed.sum(axis=1)
            max_abs_sum=float(np.max(np.abs(sums-1)))
            min_sum=float(np.min(sums))
            max_sum=float(np.max(sums))
            invalid=int(np.count_nonzero((observed<0)|(observed>1)))
            acceptable=bool(
                comparison["selected_numeric_policy_passed"]
                and max_abs_sum<=FROZEN_SUM_LIMIT and invalid==0)
            result["numerical_observation"]={
                **comparison,
                "max_abs_probability_row_sum_error":max_abs_sum,
                "minimum_probability_row_sum":min_sum,
                "maximum_probability_row_sum":max_sum,
                "out_of_range_probability_count":invalid,
                "new_vs_18_6_max_abs_output_delta":float(
                    np.max(np.abs(observed-prior))),
                "native_output_sha256":_hash(file),
                "selected_raw_and_probability_semantics_passed":acceptable,
                "posthoc_normalization_performed":False,
            }
            if not acceptable:
                result["negative_reason"]="Frozen raw probability acceptance failed"
    except (OSError,ValueError,KeyError) as exc:
        result["negative_reason"]="Native source or numeric verification failed: "+type(exc).__name__
    (output/"native_result.json").write_text(json.dumps(
        result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    stage=sub.add_parser("stage")
    stage.add_argument("--parent-stage",required=True)
    stage.add_argument("--parent-native-run",required=True)
    stage.add_argument("--output",required=True)
    run=sub.add_parser("run")
    run.add_argument("--stage",required=True)
    run.add_argument("--output",required=True)
    run.add_argument("--execute",action="store_true")
    run.add_argument("--vitis-executable",default="vitis_hls")
    args=p.parse_args(argv)
    if args.command=="stage":
        doc=stage_inverse_table_size(
            args.parent_stage,args.parent_native_run,args.output)
        print(json.dumps({
            "schema":SCHEMA,
            "changed_files":doc["changed_files"],
            "change":doc["change"],
            "nominal_index_step":doc["nominal_new_index_step"],
            "native_vendor_tools_executed":False,
        },indent=2,sort_keys=True))
    else:
        print(json.dumps(run_inverse_table_size(
            args.stage,args.output,execute=args.execute,
            vitis=args.vitis_executable),
            indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
