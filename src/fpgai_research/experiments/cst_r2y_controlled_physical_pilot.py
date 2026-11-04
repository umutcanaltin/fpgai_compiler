"""R2y: source-pinned, create-only Vivado OOC pilot on frozen R2u HLS RTL.

Nine E3-qualified source cases, seven controlled pairs. Unlike earlier
R2j/R2m (32 ns, 0.1 CSim), these original HLS cases used 8 ns,
1.5 ns uncertainty and 0.01 CSim tolerance.

Preparation does NOT call Vivado; execution requires BOTH explicit
--execute-native and --confirm-isolated-native. Original R2u, R2v,
R2w, R2x studies and all vendor receipts remain strictly read-only.
No bitstream, board-I/O/power, trained RL, or automatic DRC waivers.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import shutil

from fpgai_research.experiments.cst_r2u_hls_mechanism_execute import audit_native
from fpgai_research.experiments.cst_r2v_hls_factorial_review import (
    ORIGINAL_R2U_SOURCE_REVISION, review as review_factorial,
)
from fpgai_research.toolchains.vivado.project import make_vivado_tcl
from fpgai_research.toolchains.vivado.rtl_inputs import (
    inventory_vitis_rtl, compare_staged_roms,
)
from fpgai_research.toolchains.vivado.rom_read_attestation import (
    verify_post_run_readmem_absence,
)
from fpgai_research.toolchains.vivado.report import parse_vivado_reports
from fpgai_research.toolchains.vivado.runner import run_vivado

SCHEMA="fpgai-research/cst-R2y-original-R2u-Vivado-OOC-pilot/v1"
EXPORT_SCHEMA="fpgai-research/cst-R2y-original-Vivado-OOC-pilot-report/v1"
ORIGINAL_CASE_INDICES=(0,1,2,8,16,17,18,20,24)
# Control is always the first arm; intervention differs in only ONE
# factor within exactly one original 2^4 graph factorial block.
PAIRS=(
    ("storage",0,8),("binding",0,2),("padding",0,1),
    ("storage",16,24),("address",16,20),
    ("binding",16,18),("padding",16,17),
)
EXPECTED_CLOCK_NS=8.0
ROM_POLICY="ATTESTED_VENDOR_READ_POST_RUN_ABSENCE_V1"
REPORTS=("route_status.rpt","timing_summary.rpt","utilization.rpt","drc.rpt")
RESOURCES=("bram","lut","ff","dsp")


def _sha(file: Path)->str:
    if file.is_symlink() or not file.is_file():
        raise ValueError("R2y missing or symlinked source file: "+str(file))
    return sha256(file.read_bytes()).hexdigest()


def _original_inputs(origin: Path):
    if origin.is_symlink() or not origin.is_dir():
        raise ValueError("R2y original frozen R2u source directory unavailable")
    old=audit_native(study=origin,require_current_source=False)
    original=review_factorial(old)
    if (old["original_source_revision"]!=ORIGINAL_R2U_SOURCE_REVISION
            or old["observed_source_cases"]!=32
            or old["native_CSim_E2_qualified_cases"]!=32
            or old["native_CSynth_E3_qualified_cases"]!=30
            or original["E3_qualified_conditional_factor_pairs"]!=57):
        raise ValueError("R2y source original Vitis/R2v cohort not qualified")
    cohort=origin/"factorial_source_manifest.json"
    if cohort.is_symlink() or not cohort.is_file():
        raise ValueError("R2y original Vitis source cohort manifest missing")
    source=json.loads(cohort.read_text())
    if (source.get("source_revision")!=ORIGINAL_R2U_SOURCE_REVISION
            or source.get("clock_ns")!=EXPECTED_CLOCK_NS
            or source.get("clock_uncertainty_ns")!=1.5
            or len(source.get("rows",[]))!=32):
        raise ValueError("R2y original HLS source revision/clock not matched")
    observed={int(row["case"].split("_")[1]):row for row in old["cases"]}
    first={int(row["case"].split("_")[1]):row for row in source["rows"]}
    if set(observed)!=set(range(32)) or set(first)!=set(range(32)):
        raise ValueError("R2y incomplete original Vitis case identities")
    for i in ORIGINAL_CASE_INDICES:
        if observed[i]["original_E3_qualified"] is not True:
            raise ValueError(f"R2y prespecified original E3 case {i} unqualified")
    for factor,a,b in PAIRS:
        x,y=first[a],first[b]
        if (x["graph_index"]!=y["graph_index"]
                or x["functional_sha256"]!=y["functional_sha256"]
                or x["architecture_sha256"]!=y["architecture_sha256"]
                or x["numeric_input_sha256"]!=y["numeric_input_sha256"]
                or x["testbench_sha256"]!=y["testbench_sha256"]
                or x["tcl_sha256"]!=y["tcl_sha256"]):
            raise ValueError("R2y original HLS matched graph/source controls differ")
        for knob in ("storage","address","binding","padding"):
            expected_same = knob != factor
            if (x[knob] == y[knob]) is not expected_same:
                raise ValueError("R2y source pair must change only its named factor")
    return source,observed,first


def prepare(*,study:str|Path,output:str|Path)->dict:
    """Create a NEW standalone 9-case Vivado directory from original HLS RTL."""
    original_root=Path(study)
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("R2y original physical package strictly create-only")
    if dest.resolve().is_relative_to(original_root.resolve()):
        raise ValueError("R2y physical package must be OUTSIDE frozen original HLS study")
    source,observed,source_rows=_original_inputs(original_root)
    part=source["part"]
    script=make_vivado_tcl(part=part,clock_ns=EXPECTED_CLOCK_NS)
    original_payload=[]
    records=[]
    for i in ORIGINAL_CASE_INDICES:
        name=f"case_{i:03d}"
        src=original_root/name
        row=source_rows[i]
        observed_case=observed[i]
        inv=src/"vitis_invocation.json"
        if inv.is_symlink() or not inv.is_file():
            raise ValueError("R2y original vendor invocation receipt missing")
        receipt=json.loads(inv.read_text())
        inventory=inventory_vitis_rtl(src)
        if (receipt.get("schema")!="fpgai-research/vendor-invocation/v1"
                or receipt.get("tool")!="vitis_hls"
                or receipt.get("completed_successfully") is not True
                or receipt.get("source_sha256")!=row["source_cpp_sha256"]
                or receipt.get("rtl_output_sha256")!=inventory["rtl_sha256"]
                or receipt.get("rom_output_sha256")!=inventory["rom_sha256"]
                or receipt.get("inputs_stable_during_execution") is not True):
            raise ValueError("R2y original HLS source/RTL/ROM invocation SHA invalid")
        if not inventory["rtl_sha256"]:
            raise ValueError("R2y original HLS generated no attested Verilog")
        for filename, field in (
            ("model.cpp","source_cpp_sha256"),
            ("tb.cpp","testbench_sha256"),
            ("manifest.json","case_manifest_sha256"),
        ):
            if _sha(src/filename)!=row[field]:
                raise ValueError("R2y original HLS prepared source hash invalid")
        original_payload.append((name,src,inventory))
        records.append({
            "case":name,"original_hls_source_revision":ORIGINAL_R2U_SOURCE_REVISION,
            "graph_index":row["graph_index"],
            "storage":row["storage"],"address":row["address"],
            "binding":row["binding"],"padding":row["padding"],
            "original_E3_qualified":True,
            "original_HLS_estimated_BRAM_18K":
                observed_case["estimated_resources"]["bram_18k"],
            "model_cpp_sha256":row["source_cpp_sha256"],
            "tb_sha256":row["testbench_sha256"],
            "manifest_sha256":row["case_manifest_sha256"],
            "original_vitis_invocation_sha256":_sha(inv),
            "original_rtl_sha256":inventory["rtl_sha256"],
            "original_rom_sha256":inventory["rom_sha256"],
            "original_literal_rom_references":inventory[
                "literal_readmem_references"],
            "vivado_tcl_sha256":sha256(script.encode()).hexdigest(),
        })
    # Preflight the entire source/RTL/ROM population BEFORE creating
    # the immutable physical-study output directory.
    dest.mkdir(parents=True)
    for name,src,inventory in original_payload:
        target=dest/name
        target.mkdir()
        for filename in ("model.cpp","tb.cpp","manifest.json"):
            shutil.copyfile(src/filename,target/filename)
        rtl=target/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
        rtl.mkdir(parents=True)
        upstream=src/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
        for filename in (*inventory["rtl_sha256"],*inventory["rom_sha256"]):
            shutil.copyfile(upstream/filename,rtl/filename)
        vd=target/"vivado"
        vd.mkdir()
        for filename in inventory["rom_sha256"]:
            shutil.copyfile(upstream/filename,vd/filename)
        (vd/"run_vivado.tcl").write_text(script)
    result={
        "schema":SCHEMA,"original_source_revision":ORIGINAL_R2U_SOURCE_REVISION,
        "original_hls_part":part,"requested_clock_ns":EXPECTED_CLOCK_NS,
        "original_hls_clock_uncertainty_ns":1.5,
        "vivado_OOC_clock_uncertainty_ns":None,
        "original_source_study":str(original_root),
        "prepared_physical_cases":len(records),"planned_matched_pairs":len(PAIRS),
        "preselected_original_case_indices":list(ORIGINAL_CASE_INDICES),
        "pairs":[{"factor":f,"control":f"case_{a:03d}",
                  "intervention":f"case_{b:03d}"} for f,a,b in PAIRS],
        "cases":records,
        "original_HLS_invocations_rerun":0,
        "original_vivado_calls":0,"strict_E4_qualified_cases":0,
        "strict_E4_qualified_pairs":0,
        "rom_policy":ROM_POLICY,
        "limitations":[
            "Nine preselected original E3 qualified HLS arms, not exhaustive 32",
            "8 ns Vivado OOC period; original HLS 1.5 ns uncertainty is NOT mirrored into Vivado Tcl",
            "Original CSim 0.01 tolerance not bit-exact",
            "Separate Vivado OOC research only, not a KV260 board bitstream",
            "No DRC waivers. Routed warning observations are not clean E4",
            "Preparation is source-only; physical resources remain UNKNOWN",
        ],
    }
    with (dest/"physical_source_manifest.json").open("x") as f:
        json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
    return {k:v for k,v in result.items() if k!="cases"}


def audit(*,study:str|Path)->dict:
    """Read-only source, Vivado receipt, route/timing/DRC qualification."""
    root=Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R2y original physical study directory missing")
    path=root/"physical_source_manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("R2y immutable source study manifest missing")
    payload=json.loads(path.read_text())
    expected=[{"factor":f,"control":f"case_{a:03d}",
               "intervention":f"case_{b:03d}"} for f,a,b in PAIRS]
    if (payload.get("schema")!=SCHEMA
            or payload.get("original_source_revision")!=ORIGINAL_R2U_SOURCE_REVISION
            or payload.get("preselected_original_case_indices")!=list(ORIGINAL_CASE_INDICES)
            or payload.get("prepared_physical_cases")!=len(ORIGINAL_CASE_INDICES)
            or payload.get("planned_matched_pairs")!=len(PAIRS)
            or payload.get("pairs")!=expected
            or payload.get("requested_clock_ns")!=EXPECTED_CLOCK_NS
            or payload.get("rom_policy")!=ROM_POLICY
            or len(payload.get("cases",[]))!=len(ORIGINAL_CASE_INDICES)):
        raise ValueError("R2y immutable source cohort/protocol changed")
    result=[]
    for index,case_index in enumerate(ORIGINAL_CASE_INDICES):
        row=payload["cases"][index]
        case=f"case_{case_index:03d}"
        folder=root/case
        if (folder.is_symlink() or not folder.is_dir()
                or row.get("case")!=case
                or row.get("original_hls_source_revision")!=
                    ORIGINAL_R2U_SOURCE_REVISION):
            # Original HLS SHA is already pinned in top-level manifest.
            raise ValueError("R2y source case or physical original folder changed")
        for name,field in (
            ("model.cpp","model_cpp_sha256"),
            ("tb.cpp","tb_sha256"),
            ("manifest.json","manifest_sha256"),
        ):
            if _sha(folder/name)!=row[field]:
                raise ValueError("R2y immutable original C++/testbench package altered")
        inv=inventory_vitis_rtl(folder)
        if (inv["rtl_sha256"]!=row["original_rtl_sha256"]
                or inv["rom_sha256"]!=row["original_rom_sha256"]
                or inv["literal_readmem_references"]!=row["original_literal_rom_references"]
                or _sha(folder/"vivado"/"run_vivado.tcl")!=row["vivado_tcl_sha256"]):
            raise ValueError("R2y original generated RTL/ROM/Vivado Tcl altered")
        vd=folder/"vivado"
        staged=compare_staged_roms(vd,inv["rom_sha256"])
        rc=vd/"vivado_invocation.json"
        if rc.is_symlink():
            raise ValueError("R2y original Vivado receipt symlink refused")
        state={
            "case":case,"graph_index":row["graph_index"],
            "storage":row["storage"],"address":row["address"],
            "binding":row["binding"],"padding":row["padding"],
            "original_HLS_estimated_BRAM_18K":row[
                "original_HLS_estimated_BRAM_18K"],
            "original_vivado_invocation_present":rc.is_file(),
            "original_source_SHA_attested":True,
            "routed_observation_valid":False,"strict_clean_DRC_E4":False,
            "routed_timing_met":None,"routed_WNS_ns":None,
            "physical_resources":None,"drc":None,
            "routed_report_sha256":None,
            "status":"PREPARED_NOT_RUN",
            "ROM_input_attestation":"SOURCE_AND_STAGED_ROM_STABLE"
                if staged["matches_original_roms"] else "ROM_WORKING_COPY_NOT_STABLE",
        }
        if not rc.is_file():
            if not staged["matches_original_roms"]:
                state["status"]="REJECTED_PRE_RUN_ROM_MISSING_OR_ALTERED"
            result.append(state)
            continue
        receipt=json.loads(rc.read_text())
        witness=None
        allowed_read_absence=False
        if not staged["matches_original_roms"]:
            if receipt.get("allow_verified_readmem_post_absence") is True:
                witness=verify_post_run_readmem_absence(
                    expected=inv["rom_sha256"],
                    literal_refs=inv["literal_readmem_references"],
                    before=receipt.get("staged_rom_input_integrity_before"),
                    after=staged,
                    stdout_file=vd/"logs"/"vivado.stdout.log",
                    expected_stdout_sha256=receipt.get("stdout_sha256"))
                allowed_read_absence=(
                    witness["qualified_postrun_absence"] is True
                    and witness==receipt.get(
                        "verified_readmem_post_absence_witness")
                    and receipt.get("staged_rom_input_integrity_after")==staged)
        if not (staged["matches_original_roms"] or allowed_read_absence):
            state["status"]="REJECTED_ORIGINAL_ROM_COPY_ALTERED"
            result.append(state)
            continue
        state["ROM_input_attestation"]=(
            "HASH_ATTESTED_VENDOR_READ_POST_RUN_ABSENCE"
            if allowed_read_absence else "ORIGINAL_SOURCE_AND_WORKING_ROM_STABLE"
        )
        if (receipt.get("tool")!="vivado"
                or receipt.get("completed_successfully") is not True
                or receipt.get("inputs_stable_during_execution") is not True
                or receipt.get("source_sha256")!=row["model_cpp_sha256"]
                or receipt.get("testbench_sha256")!=row["tb_sha256"]
                or receipt.get("script_sha256")!=row["vivado_tcl_sha256"]
                or receipt.get("rtl_input_sha256")!=inv["rtl_sha256"]
                or receipt.get("rom_input_sha256")!=inv["rom_sha256"]
                or receipt.get("rom_inputs_stable_during_execution") is not True
                or (receipt.get("staged_rom_input_integrity_before") or {}).get(
                    "matches_original_roms") is not True):
            state["status"]="REJECTED_VIVADO_INVOCATION_PROVENANCE"
            result.append(state)
            continue
        if not (allowed_read_absence or (
                receipt.get("staged_rom_inputs_stable_during_execution") is True
                and (receipt.get("staged_rom_input_integrity_after") or {}).get(
                    "matches_original_roms") is True)):
            state["status"]="REJECTED_VIVADO_WORKING_ROM_PROVENANCE"
            result.append(state)
            continue
        try:
            routed=parse_vivado_reports(vd)
        except (ValueError,FileNotFoundError) as err:
            state["status"]="REJECTED_INCOMPLETE_VIVADO_ROUTED_REPORTS"
            state["report_error"]=str(err)[:250]
            result.append(state)
            continue
        reports=routed.get("raw_report_sha256") or {}
        if (routed.get("routed_nets_complete") is not True
                or any(
                    reports.get(name)!=(receipt.get("fresh_report_sha256") or {}).get(name)
                    or not isinstance(reports.get(name),str)
                    for name in REPORTS)):
            state["status"]="REJECTED_NONFRESH_OR_INCOMPLETE_ROUTED_REPORT"
            result.append(state)
            continue
        timing=routed["timing"];util=routed["utilization"];drc=routed["drc"]
        if (any(type(util.get(key)) not in (int,float)
                or not isfinite(util[key]) or util[key]<0 for key in RESOURCES)
                or type(timing.get("wns_ns")) not in (int,float)
                or not isfinite(timing["wns_ns"])):
            state["status"]="REJECTED_INCOMPLETE_RESOURCE_OR_WNS"
            result.append(state)
            continue
        drc_verified=(drc.get("summary_verified") is True)
        good=(
            timing.get("timing_met") is True
            and drc_verified
            and drc.get("clean") is True
            and drc.get("critical_warnings")==0
            and drc.get("violations")==0
        )
        # Valid original routing and timing can be reported with DRC
        # warnings for descriptive science, but is NOT strict E4.
        observation=(drc_verified and timing.get("timing_met") is True
                     and drc.get("critical_warnings")==0)
        state.update({
            "routed_observation_valid":observation,
            "strict_clean_DRC_E4":good,
            "routed_timing_met":timing["timing_met"],
            "routed_WNS_ns":timing["wns_ns"],
            "physical_resources":util,
            "drc":drc,"routed_report_sha256":reports,
            "status":("STRICT_CLEAN_DRC_VIVADO_OOC_E4" if good else
                      "ROUTED_WITH_DISCLOSED_DRC_NOT_E4" if observation else
                      "REJECTED_TIMING_OR_DRC_NOT_E4"),
        })
        result.append(state)
    return {
        "schema":SCHEMA,"original_source_revision":ORIGINAL_R2U_SOURCE_REVISION,
        "requested_clock_ns":payload["requested_clock_ns"],
        "original_physical_cases":len(result),"original_planned_pairs":len(PAIRS),
        "original_vivado_invocation_receipts":sum(
            x["original_vivado_invocation_present"] for x in result),
        "strict_clean_DRC_E4_cases":sum(x["strict_clean_DRC_E4"] for x in result),
        "valid_routed_warning_or_clean_observation_cases":sum(
            x["routed_observation_valid"] for x in result),
        "cases":result,
        "original_vendor_calls_during_audit":0,
        "board_bitstream_qualified_cases":0,
    }


def summarize_pairs(study:dict)->dict:
    cases={row["case"]:row for row in study["cases"]}
    if (study.get("schema")!=SCHEMA
            or len(cases)!=len(ORIGINAL_CASE_INDICES)
            or set(cases)!={f"case_{i:03d}" for i in ORIGINAL_CASE_INDICES}):
        raise ValueError("R2y pair report requires full 9-case audit")
    contrasts=[]
    for factor,a,b in PAIRS:
        x=cases[f"case_{a:03d}"];y=cases[f"case_{b:03d}"]
        eligible=x["strict_clean_DRC_E4"] and y["strict_clean_DRC_E4"]
        descriptor=x["routed_observation_valid"] and y[
            "routed_observation_valid"]
        def diff(key):
            return y["physical_resources"][key]-x["physical_resources"][key] \
                if eligible else None
        contrasts.append({
            "factor":factor,"graph_index":x["graph_index"],
            "control":x["case"],"intervention":y["case"],
            "HLS_estimated_BRAM_18K_delta":
                y["original_HLS_estimated_BRAM_18K"] -
                x["original_HLS_estimated_BRAM_18K"],
            "strict_clean_DRC_E4_pair":eligible,
            "routed_descriptive_pair_with_disclosed_DRC":descriptor,
            "physical_BRAM_delta_E4":diff("bram"),
            "physical_LUT_delta_E4":diff("lut"),
            "physical_FF_delta_E4":diff("ff"),
            "physical_DSP_delta_E4":diff("dsp"),
            "routed_WNS_delta_E4_ns":(
                y["routed_WNS_ns"]-x["routed_WNS_ns"]
                if eligible else None),
            "warning_only_physical_BRAM_delta":(
                y["physical_resources"]["bram"] -
                x["physical_resources"]["bram"]
                if descriptor and not eligible else None),
            "pair_status":(
                "QUALIFIED_CLEAN_DRC_OOC_E4" if eligible else
                "DESCRIPTIVE_ONLY_DRC_DISCLOSED" if descriptor else
                "UNKNOWN_NOT_BOTH_STRICT_E4"),
        })
    return {
        "schema":EXPORT_SCHEMA,
        "original_source_revision":ORIGINAL_R2U_SOURCE_REVISION,
        "prespecified_original_cases":len(ORIGINAL_CASE_INDICES),
        "prespecified_matched_pairs":len(PAIRS),
        "strict_clean_DRC_E4_pairs":sum(
            x["strict_clean_DRC_E4_pair"] for x in contrasts),
        "warning_only_descriptive_pairs":sum(
            x["pair_status"]=="DESCRIPTIVE_ONLY_DRC_DISCLOSED"
            for x in contrasts),
        "unknown_or_rejected_pairs":sum(
            x["pair_status"]=="UNKNOWN_NOT_BOTH_STRICT_E4"
            for x in contrasts),
        "cases":study["cases"],"pairs":contrasts,
        "new_original_vendor_calls_during_export":0,
        "physical_board_bitstream_qualified":False,
        "limits":[
            "Vivado research OOC only, not physical KV260 board deployment",
            "HLS 8 ns source-clock request, no achieved deployed Fmax/power",
            "Only original preselected 9/32 HLS arms, 7 conditional contrasts",
            "Historical prior Vivado evidence not overwritten or pooled",
            "Only BOTH strictly clean DRC and timed/fully routed qualify E4",
            "DRC warning-only resource deltas reported separately, never E4",
        ],
    }


def execute(*,study:str|Path,start_case:int=0,max_cases:int=1,
            execute_native:bool=False,confirm_isolated_native:bool=False,
            vivado_executable:str="vivado",timeout_seconds:int=14400)->dict:
    report=audit(study=study)
    if (type(start_case) is not int or type(max_cases) is not int
            or start_case<0 or not 1<=max_cases<=len(ORIGINAL_CASE_INDICES)
            or start_case+max_cases>len(ORIGINAL_CASE_INDICES)):
        raise ValueError("R2y physical case selection must be bounded to 9")
    selected=report["cases"][start_case:start_case+max_cases]
    if not execute_native:
        return {"schema":SCHEMA,"status":"DRY_RUN_NO_VIVADO",
                "cases":[x["case"] for x in selected],"new_vivado_calls":0}
    if confirm_isolated_native is not True:
        raise ValueError("R2y original native needs explicit confirmation")
    if (type(timeout_seconds) is not int
            or not 1<=timeout_seconds<=172800
            or shutil.which(vivado_executable) is None):
        raise ValueError("R2y native Vivado executable/timeout unavailable")
    root=Path(study)
    for row in selected:
        vd=root/row["case"]/"vivado"
        if row["status"]!="PREPARED_NOT_RUN" or any(
            (vd/name).exists() or (vd/name).is_symlink()
            for name in (
                "logs","reports","research_implementation","vivado_invocation.json"
            )
        ):
            raise FileExistsError("R2y original physical case already attempted")
    executed=[]
    for row in selected:
        case=root/row["case"]
        receipt=run_vivado(
            case,executable=vivado_executable,
            timeout_seconds=timeout_seconds,
            allow_verified_readmem_post_absence=True)
        observed={r["case"]:r for r in audit(study=root)["cases"]}[row["case"]]
        executed.append({
            "case":row["case"],
            "original_vivado_process_completed":receipt[
                "completed_successfully"],
            "status":observed["status"],
            "strict_clean_DRC_E4":observed["strict_clean_DRC_E4"],
            "routed_descriptive_with_warning":
                observed["routed_observation_valid"] and not
                observed["strict_clean_DRC_E4"],
        })
    return {"schema":SCHEMA,"status":"NATIVE_VIVADO_ATTEMPTED",
            "new_vivado_calls":len(executed),"cases":executed,
            "board_bitstream_qualified":False}


def export(*,study:str|Path,output:str|Path)->dict:
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("R2y original physical report strictly create-only")
    if dest.resolve().is_relative_to(Path(study).resolve()):
        raise ValueError("R2y report must not be written inside frozen physical study")
    computed=summarize_pairs(audit(study=study))
    dest.mkdir(parents=True)
    with (dest/"summary.json").open("x",encoding="utf-8") as f:
        json.dump(computed,f,sort_keys=True,indent=2);f.write("\n")
    return {k:v for k,v in computed.items() if k not in ("cases","pairs")}


def main(argv:list[str]|None=None)->None:
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare",action="store_true")
    mode.add_argument("--audit",action="store_true")
    mode.add_argument("--execute",action="store_true")
    mode.add_argument("--export",action="store_true")
    p.add_argument("--study",required=True)
    p.add_argument("--output")
    p.add_argument("--start-case",type=int,default=0)
    p.add_argument("--max-cases",type=int,default=1)
    p.add_argument("--execute-native",action="store_true")
    p.add_argument("--confirm-isolated-native",action="store_true")
    p.add_argument("--vivado-executable",default="vivado")
    p.add_argument("--timeout-seconds",type=int,default=14400)
    args=p.parse_args(argv)
    if args.prepare or args.export:
        if not args.output:
            p.error("--output required for --prepare or --export")
    elif args.output:
        p.error("--output only valid for --prepare or --export")
    if args.prepare:
        res=prepare(study=args.study,output=args.output)
    elif args.audit:
        res=audit(study=args.study)
        res={k:v for k,v in res.items() if k!="cases"}
    elif args.execute:
        res=execute(
            study=args.study,start_case=args.start_case,max_cases=args.max_cases,
            execute_native=args.execute_native,
            confirm_isolated_native=args.confirm_isolated_native,
            vivado_executable=args.vivado_executable,
            timeout_seconds=args.timeout_seconds)
    else:
        res=export(study=args.study,output=args.output)
    print(json.dumps(res,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
