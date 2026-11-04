"""Explicit native Vitis CSim-only paired runner for selected hls4ml MLP RF.

Default is READ-ONLY and strict. The diagnostic backend-effects flag
never qualifies an isolated reuse comparison; combined with --execute it
only produces a numerical observation of the already-attributed backend.
The --execute flag authorizes CSim only,
never HLS synthesis, Vivado, export, board deployment or publication.
Native project clones are intentionally modified by Vitis when executing.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
from hashlib import sha256
import sys

from fpgai_research.experiments.hls4ml_csim_pair import collect_csim_stage_pair
from fpgai_research.experiments.hls4ml_name_config_delta import compare_e1_configuration
from fpgai_research.experiments.hls4ml_csim_native_observation import audit_native_csim_outputs

SCHEMA="fpgai-research/hls4ml-selected-paired-native-csim/v1"


def preflight(stage_root,global_default,*,diagnostic_backend_effects=False):
    root=Path(stage_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Selected E1/CSim-only pair is missing")
    a=root/"csim_rf1"
    b=root/"csim_rf2"
    comparison=compare_e1_configuration(global_default,root/"e1_rf1",root/"e1_rf2")
    isolated = comparison["name_rf1_vs_rf2_only_explicit_reuse_changed"]
    request_isolated = comparison.get("preconversion_only_explicit_reuse_changed") is True
    # Never qualify a hardware reuse comparison from effective settings
    # alone. A source receipt must prove that the actual user request
    # itself had exactly one controlled parameter difference.
    if not request_isolated:
        raise ValueError(
            "Missing or unisolated pre-conversion reuse request; "
            "cannot qualify CSim comparison")
    if not isolated and not diagnostic_backend_effects:
        raise ValueError("Unexpected hls4ml effective config differences; CSim BLOCKED")
    if (comparison["name_rf1_vs_rf2_selected_requested_layer_count"]!=1
            or comparison["name_rf1_vs_rf2_changed_non_reuse_key_count"]!=0):
        raise ValueError("RF1/RF2 input configurations not matched")
    pair=collect_csim_stage_pair(a,b)
    if pair["both_csim_only"] is not True or pair["matched_selected_testbench_data"] is not True:
        raise ValueError("Pair fails matched CSim-only source checks")
    for stage in (a,b):
        project=stage/"project"
        script=project/"build_prj.tcl"
        if (project.is_symlink() or not project.is_dir()
                or script.is_symlink() or not script.is_file()):
            raise ValueError("Source-attested generated CSim project missing")
    return {
        "schema":SCHEMA,
        "config_isolated":isolated,
        "preconversion_request_isolated":
            comparison.get("preconversion_only_explicit_reuse_changed",False),
        "postconversion_additional_changed_key_count":
            comparison.get("postconversion_additional_changed_key_count"),
        "diagnostic_backend_effects_only":bool(diagnostic_backend_effects),
        "hardware_reuse_only_comparison_eligible":
            bool(isolated and request_isolated and not diagnostic_backend_effects),
        "matched_selected_inputs":True,
        "csim_only":True,"stages":[str(a),str(b)],
        "numeric_e2_qualified":False,"native_executed":False,
    }


def run_pair(stage_root,global_default,output,*,execute=False,
             vitis="vitis_hls",timeout=1800,
             diagnostic_backend_effects=False):
    pre=preflight(stage_root,global_default,
                  diagnostic_backend_effects=diagnostic_backend_effects)
    if not execute:
        return pre
    if type(timeout) is not int or not 10<=timeout<=7200:
        raise ValueError("Native CSim timeout must be 10..7200 seconds")
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Native observation receipt destination must be new")
    dest.mkdir(parents=True)
    attempts=[]
    for rf in (1,2):
        stage=Path(stage_root)/("csim_rf"+str(rf))
        project=stage/"project"
        try:
            done=subprocess.run([vitis,"-f","build_prj.tcl"],cwd=project,
                                text=True,capture_output=True,timeout=timeout,
                                check=False)
            raw=done.stdout+"\n"+done.stderr
            code=done.returncode
        except (OSError,subprocess.TimeoutExpired) as exc:
            raw="Native CSim failed before completion: "+type(exc).__name__
            code=-1
        logfile=dest/("rf"+str(rf)+"_vitis_csim.log")
        logfile.write_text(raw,encoding="utf-8")
        good=(code==0 and "CSim done with 0 errors" in raw)
        attempt={
            "rf":rf,"native_vitis_csim_executed":good,
            "returncode":code,"log_sha256":sha256(raw.encode()).hexdigest(),
            "numerical_observation_16_rows":False,
            "exploratory_numeric_policy_passed":False,
            "native_synthesis_or_vivado_executed":False,
        }
        if good:
            try:
                numeric=audit_native_csim_outputs(stage,dest/("rf"+str(rf)+"_numeric"))
                attempt["numerical_observation_16_rows"]=numeric["observed_output_rows"]==16
                attempt["exploratory_numeric_policy_passed"]=numeric["selected_numeric_policy_passed"]
                attempt["max_absolute_probability_difference"]=numeric["max_absolute_probability_difference"]
                attempt["selected_top1_disagreements"]=numeric["selected_top1_disagreements"]
            except (ValueError,FileNotFoundError) as exc:
                attempt["numeric_audit_error_type"]=type(exc).__name__
        attempts.append(attempt)
        if not good:
            break
    result={
        **pre,"native_executed":True,"attempts":attempts,
        "both_native_csim_completed":len(attempts)==2 and all(
            row["native_vitis_csim_executed"] for row in attempts),
        "pair_selected_policy_passed":len(attempts)==2 and all(
            row["exploratory_numeric_policy_passed"] for row in attempts),
        "pair_source_config_reuse_only_qualified":
            pre["hardware_reuse_only_comparison_eligible"],
        "diagnostic_only_not_fair_reuse_comparison":
            bool(diagnostic_backend_effects),
        # Native CLI log is independently archived but not automatically
        # elevated to a separately source-reaudited E2 acceptance receipt.
        "qualified_e2":False,
        "hardware_qor_comparison_eligible":False,
        "native_e3_e4_executed":False,
    }
    (dest/"paired_native_csim.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",
        encoding="utf-8")
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-root",required=True)
    p.add_argument("--global-default",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--execute",action="store_true",
                   help="Explicitly invoke licensed Vitis CSim in both staged clones")
    p.add_argument("--diagnostic-backend-effects",action="store_true",
                   help="Allow CSim diagnosis of backend-added RF settings; "
                        "NEVER qualifies an isolated reuse benchmark")
    p.add_argument("--vitis-executable",default="vitis_hls")
    a=p.parse_args(argv)
    result=run_pair(a.stage_root,a.global_default,a.output,
                    execute=a.execute,vitis=a.vitis_executable,
                    diagnostic_backend_effects=a.diagnostic_backend_effects)
    print(json.dumps(result,indent=2,sort_keys=True))
    return int(a.execute and not result["pair_selected_policy_passed"])


if __name__=="__main__":
    raise SystemExit(main())
