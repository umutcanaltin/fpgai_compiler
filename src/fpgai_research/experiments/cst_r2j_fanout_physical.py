"""Source-pinned, create-only R2i fanout Vitis RTL -> Vivado OOC bridge.

Copies only source-attributed Vitis-generated Verilog from an original
HLS study into a NEW, separate physical-study tree. Never modifies
historical HLS case files. Native Vivado execution is opt-in, bounded,
foreground; read-only E4 qualification demands fresh routed reports,
stable RTL and Tcl hashes, source attribution, complete routing and
nonnegative WNS. No bitstream, board I/O or achieved Fmax claim.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from fpgai_research.experiments.cst_r2i_fanout_native_execute import audit_native
from fpgai_research.experiments.cst_r2i_fanout_native_prepare import audit_prepared
from fpgai_research.toolchains.runner import rtl_sha256
from fpgai_research.toolchains.vivado.project import make_vivado_tcl
from fpgai_research.toolchains.vivado.rtl_inputs import (
    inventory_vitis_rtl, compare_staged_roms,
)
from fpgai_research.toolchains.vivado.report import parse_vivado_reports
from fpgai_research.toolchains.vivado.rom_read_attestation import (
    verify_post_run_readmem_absence,
)
from fpgai_research.toolchains.vivado.runner import run_vivado

SCHEMA="fpgai-research/cst-R2j-original-source-pinned-fanout-Vivado-OOC/v1"
# Protocol cohort: all 36 original graph/Phi source pairs stay visible.
# Only E3-qualified original cases may enter Vivado packaging.
TOTAL_CASES=72
TOTAL_PAIRS=36
ROM_POST_POLICY="ATTESTED_VENDOR_READ_POST_RUN_ABSENCE_V1"
# Exploratory, explicitly prospective secondary endpoint; NOT strict E4.
DRC_OBSERVATION_POLICY="DISCLOSED_OOC_NONCRITICAL_WARNING_OBSERVATION_V1"
# Pilot-informed from original v5 case_000. Other warning classes stay UNKNOWN.
_ALLOWED_OOC_WARNING_RULES=frozenset(("DPIP-2","DPOP-3","DPOP-4","RTSTAT-10"))


def _file_sha(path):
    path=Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Missing or symlinked original research file")
    return sha256(path.read_bytes()).hexdigest()


def _validate_rtl(case):
    """Vitis pre-ROM historical receipts hash only Verilog HDL."""
    return inventory_vitis_rtl(case)["rtl_sha256"]


def prepare_routed_study(*,native_study:str|Path,output:str|Path):
    origin,dest=Path(native_study),Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Routed R2j fanout physical study is create-only")
    prepared=audit_prepared(origin,require_current_source=False)
    audit=audit_native(origin,require_current_source=False)
    original=json.loads((origin/"native_source_manifest.json").read_text())
    if (audit["prespecified_source_cases"]!=72 or
            audit["prespecified_matched_pairs"]!=36 or
            prepared["cases"]!=72 or prepared["matched_pairs"]!=36 or
            len(audit["per_case_status"])!=72):
        raise ValueError("Original R2i source denominator incomplete")
    entries=[]
    payload=[]
    for index,record in enumerate(original["cases"]):
        name=f"case_{index:03d}"
        if (record["case"]!=name or
                audit["per_case_status"][index]["case"]!=name or
                record["source_storage"] not in ("disjoint","per_element_optimal") or
                record["source_storage"]!=
                    ("disjoint" if index%2==0 else "per_element_optimal")):
            raise ValueError("Original case order differs from audited HLS evidence")
        source=origin/name
        qualified=(
            audit["per_case_status"][index][
                "original_CSynth_timing_resources_qualified"] is True)
        entry={
            "case":name,"group":index//2,
            "graph":record["graph"],"mapping":record["mapping"],
            "source_storage":record["source_storage"],
            "source_cpp_sha256":record["source_cpp_sha256"],
            "testbench_sha256":record["testbench_sha256"],
            "case_manifest_sha256":record["case_manifest_sha256"],
            "original_hls_qualified":qualified,
            "original_hls_csim_qualified":audit["per_case_status"][index][
                "original_CSim_qualified"],
            "original_hls_status":audit["per_case_status"][index][
                "qualification_status"],
            "original_hls_source_revision":original["source_revision"],
            "part":original["part"],
            "original_hls_target_clock_ns":original["requested_clock_ns"],
            "original_tcl_sha256":record["tcl_sha256"],
            "physical_package_status":"BLOCKED_NO_QUALIFIED_ORIGINAL_HLS",
            "copied_rtl_sha256":None,
            "vivado_tcl_sha256":None,
            "original_hls_invocation_sha256":None,
            "original_vivado_calls":0,
            "physical_qualified":False,
        }
        if qualified:
            inventory=inventory_vitis_rtl(source)
            source_rtl=inventory["rtl_sha256"]
            source_rom=inventory["rom_sha256"]
            receipt_path=source/"vitis_invocation.json"
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise ValueError("Qualified HLS case missing original Vitis receipt")
            receipt=json.loads(receipt_path.read_text())
            if (receipt.get("completed_successfully") is not True
                    or receipt.get("source_sha256")!=record["source_cpp_sha256"]
                    or receipt.get("rtl_output_sha256")!=source_rtl
                    or (receipt.get("rom_output_sha256") is not None
                        and receipt["rom_output_sha256"]!=source_rom)
                    or receipt.get("inputs_stable_during_execution") is not True):
                raise ValueError("Original HLS vendor output RTL lacks exact source attribution")
            if not source_rtl:
                raise ValueError("Qualified original HLS source has no attested Verilog RTL")
            script=make_vivado_tcl(
                part=original["part"],clock_ns=original["requested_clock_ns"])
            payload.append((name,source,source_rtl,source_rom,script))
            entry.update({
                "physical_package_status":"PREPARED_ORIGINAL_RTL_E3_INPUT",
                "copied_rtl_sha256":source_rtl,
                "copied_rom_sha256":source_rom,
                "literal_readmem_references":inventory["literal_readmem_references"],
                "rom_attestation":(
                    "ORIGINAL_HLS_VENDOR_RECEIPT_HASHED"
                    if receipt.get("rom_output_sha256") is not None
                    else "LEGACY_ROM_HASH_CAPTURED_DURING_PHYSICAL_PREPARATION"),
                "vivado_tcl_sha256":sha256(script.encode()).hexdigest(),
                "original_hls_invocation_sha256":_file_sha(receipt_path),
            })
        entries.append(entry)
    dest.mkdir(parents=True)
    for name,source,rtl,rom,script in payload:
        target=dest/name
        target.mkdir()
        for filename in ("model.cpp","tb.cpp","manifest.json"):
            if (source/filename).is_symlink():
                raise ValueError("Original HLS input may not be symlinked")
            shutil.copyfile(source/filename,target/filename)
        rtl_target=target/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
        rtl_target.mkdir(parents=True)
        rtl_source=source/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
        for filename in (*rtl,*rom):
            shutil.copyfile(rtl_source/filename,rtl_target/filename)
        vivado=target/"vivado"
        vivado.mkdir()
        # $readmemh/$readmemb local names resolve from vendor cwd.
        for filename in rom:
            shutil.copyfile(rtl_source/filename,vivado/filename)
        (vivado/"run_vivado.tcl").write_text(script)
    report={
        "schema":SCHEMA,"original_hls_source_revision":original["source_revision"],
        "original_hls_study_manifest_sha256":_file_sha(
            origin/"native_source_manifest.json"),
        "original_hls_storage_comparison":[
            "disjoint","per_element_optimal"],
        "part":original["part"],"requested_clock_ns":original["requested_clock_ns"],
        "prespecified_cases":72,"prespecified_pairs":36,
        "rom_postrun_policy":ROM_POST_POLICY,
        "drc_warning_observation_policy":DRC_OBSERVATION_POLICY,
        "drc_warning_allowlist":sorted(_ALLOWED_OOC_WARNING_RULES),
        "rtl_packaged_cases":len(payload),
        "not_yet_qualified_hls_cases":72-len(payload),
        "cases":entries,"original_vivado_calls":0,
        "routed_e4_qualified_cases":0,
        "limitations":[
            "36 matched pairs remain in denominator even if only 24 were HLS E3-qualified",
            "Separate Vivado OOC source package; historical HLS studies unchanged",
            "Original local .dat/.mem ROM assets preserved and hash-pinned",
            "Older Vitis receipts do not attest ROM payloads: explicitly labeled",
            "Original HLS CSim/CSynth and source-attributed RTL are prerequisites",
            "No original Vivado calls, route or physical QoR implied by packaging",
            "OOC implementation is not a deployable KV260 pin-constrained bitstream",
        ],
    }
    (dest/"physical_study_manifest.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report


def audit_routed_study(study:str|Path):
    root=Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Physical study root unavailable")
    p=root/"physical_study_manifest.json"
    if p.is_symlink() or not p.is_file():
        raise ValueError("Source-attributed physical study manifest missing")
    manifest=json.loads(p.read_text())
    if (manifest.get("schema")!=SCHEMA or manifest.get("prespecified_cases")!=72
            or len(manifest.get("cases",[]))!=72
            or manifest.get("rom_postrun_policy") not in (None,ROM_POST_POLICY)
            or manifest.get("drc_warning_observation_policy") not in
               (None,DRC_OBSERVATION_POLICY)):
        raise ValueError("Incomplete original physical study")
    policy=manifest.get("rom_postrun_policy")
    warning_policy=manifest.get("drc_warning_observation_policy")
    if (warning_policy==DRC_OBSERVATION_POLICY and
            manifest.get("drc_warning_allowlist")!=
                sorted(_ALLOWED_OOC_WARNING_RULES)):
        raise ValueError("Source-pinned DRC warning allowlist differs from study")
    results=[]
    for index,entry in enumerate(manifest["cases"]):
        name=f"case_{index:03d}"
        if entry["case"]!=name:
            raise ValueError("Unexpected physical study case order")
        state={
            "case":name,"group":index//2,
            "source_storage":entry["source_storage"],"graph":entry["graph"],
            "mapping":entry["mapping"],"source_cpp_sha256":entry["source_cpp_sha256"],
            "status":entry["physical_package_status"],
            "original_hls_qualified":entry["original_hls_qualified"],
            "rom_source_attestation":entry.get("rom_attestation"),
            "original_rom_sha256":entry.get("copied_rom_sha256"),
            "original_vivado_invocation_present":False,
            "routed_qualified":False,
            "routed_observation_valid":False,
            "routed_warning_observation":False,
            "drc_warning_observation_policy":warning_policy,
            "drc_warning_summary":None,
            "routed_timing_met":None,
            "routed_wns_ns":None,"physical_resources":None,
            "drc_verified_clean":None,
            "original_routed_report_sha256":None,
            "requested_clock_ns":entry["original_hls_target_clock_ns"],
            "board_bitstream_qualified":False,
        }
        if entry["physical_package_status"]!="PREPARED_ORIGINAL_RTL_E3_INPUT":
            results.append(state)
            continue
        case=root/name
        inventory=inventory_vitis_rtl(case)
        rtl=inventory["rtl_sha256"]
        rom=inventory["rom_sha256"]
        staged=case/"vivado"
        if (policy==ROM_POST_POLICY and
                "ROM_READ_ATTESTATION_PROTOCOL_V1" not in
                    (staged/"run_vivado.tcl").read_text()):
            raise ValueError("Post-run ROM study policy lacks source-Tcl protocol marker")
        if (warning_policy==DRC_OBSERVATION_POLICY and
                "DRC_WARNING_OBSERVATION_PROTOCOL_V1" not in
                    (staged/"run_vivado.tcl").read_text()):
            raise ValueError("Prospective DRC observation policy lacks source-Tcl marker")
        receipt=staged/"vivado_invocation.json"
        if receipt.is_symlink():
            raise ValueError("Original Vivado receipt symlink rejected")
        state["original_vivado_invocation_present"]=receipt.is_file()
        # Always check immutable source provenance BEFORE classifying vendor
        # working-directory side effects, even when a staged ROM vanished.
        if (rtl!=entry["copied_rtl_sha256"] or
                rom!=entry.get("copied_rom_sha256",{}) or
                inventory["literal_readmem_references"]!=entry.get("literal_readmem_references",[]) or
                _file_sha(case/"model.cpp")!=entry["source_cpp_sha256"] or
                _file_sha(case/"tb.cpp")!=entry["testbench_sha256"] or
                _file_sha(case/"manifest.json")!=entry["case_manifest_sha256"] or
                _file_sha(case/"vivado"/"run_vivado.tcl")!=entry["vivado_tcl_sha256"]):
            raise ValueError("Physical study immutable RTL/C++/Vivado source tampered")
        staged_check=compare_staged_roms(staged,rom)
        state["staged_rom_integrity"]=staged_check
        if not receipt.is_file() and staged_check["matches_original_roms"]:
            state["status"]="PREPARED_NOT_RUN"
            results.append(state)
            continue
        original=(json.loads(receipt.read_text()) if receipt.is_file()
                  else {})
        post_missing_witness=None
        verified_post_missing=False
        if (not staged_check["matches_original_roms"] and
                policy==ROM_POST_POLICY and receipt.is_file()
                and original.get("allow_verified_readmem_post_absence") is True
                and isinstance(original.get("stdout_sha256"),str)
                and len(original["stdout_sha256"])==64):
            post_missing_witness=verify_post_run_readmem_absence(
                expected=rom,
                literal_refs=inventory["literal_readmem_references"],
                before=original.get("staged_rom_input_integrity_before"),
                after=staged_check,
                stdout_file=staged/"logs"/"vivado.stdout.log",
                expected_stdout_sha256=original.get("stdout_sha256"))
            verified_post_missing=(
                post_missing_witness["qualified_postrun_absence"] is True
                and post_missing_witness==
                    original.get("verified_readmem_post_absence_witness")
                and original.get("staged_rom_input_integrity_after")==
                    staged_check
                and original.get("staged_rom_inputs_stable_during_execution")
                    is False)
        state["verified_vendor_read_post_absence"]=verified_post_missing
        state["post_run_readmem_witness"]=post_missing_witness
        state["rom_input_evidence_mode"]=(
            "PRE_RUN_HASH_AND_VENDOR_REPORTED_READ_POST_ABSENCE"
            if verified_post_missing else
            "ORIGINAL_SOURCE_AND_WORKING_ROM_STABLE"
            if staged_check["matches_original_roms"] else
            "UNQUALIFIED_ROM_INTEGRITY")
        if not (staged_check["matches_original_roms"] or verified_post_missing):
            state["status"]=(
                "REJECTED_ORIGINAL_STAGED_ROM_MODIFIED_OR_MISSING"
                if not staged_check["original_inputs_unchanged"]
                else "REJECTED_UNATTRIBUTED_VIVADO_ROM_ASSETS")
            results.append(state)
            continue
        if not receipt.is_file():
            state["status"]="PREPARED_NOT_RUN"
            results.append(state)
            continue
        if (original.get("tool")!="vivado" or
                original.get("completed_successfully") is not True or
                original.get("inputs_stable_during_execution") is not True or
                original.get("source_sha256")!=entry["source_cpp_sha256"] or
                original.get("script_sha256")!=entry["vivado_tcl_sha256"] or
                original.get("rtl_input_sha256")!=rtl or
                original.get("rom_input_sha256")!=rom or
                original.get("rom_inputs_stable_during_execution") is not True or
                (original.get("staged_rom_input_integrity_before") or {})
                    .get("matches_original_roms") is not True or
                not (verified_post_missing or (
                    original.get("staged_rom_inputs_stable_during_execution") is True
                    and (original.get("staged_rom_input_integrity_after") or {})
                        .get("matches_original_roms") is True))):
            state["status"]="REJECTED_VENDOR_PROVENANCE"
            results.append(state)
            continue
        try:
            routed=parse_vivado_reports(case/"vivado")
        except (ValueError,FileNotFoundError) as err:
            state["status"]="REJECTED_ROUTING_OR_REPORTS"
            state["report_error"]=str(err)[:250]
            results.append(state)
            continue
        if routed.get("routed_nets_complete") is not True:
            state["status"]="REJECTED_INCOMPLETE_OR_UNVERIFIED_ROUTE"
            results.append(state)
            continue
        digests=routed["raw_report_sha256"]
        required=("route_status.rpt","timing_summary.rpt",
                  "utilization.rpt","drc.rpt")
        if (any(name not in digests or
                (original.get("fresh_report_sha256") or {}).get(name)!=digests[name]
                for name in required)):
            state["status"]="REJECTED_STALE_OR_UNATTRIBUTED_PHYSICAL_REPORTS"
            results.append(state)
            continue
        timing=routed["timing"]
        util=routed["utilization"]
        if any(util.get(k) is None for k in ("lut","ff","dsp","bram")):
            state["status"]="UNKNOWN_INCOMPLETE_PHYSICAL_RESOURCES"
            results.append(state)
            continue
        drc=routed.get("drc") or {}
        drc_clean=(drc.get("summary_verified") is True
                   and drc.get("clean") is True
                   and drc.get("critical_warnings")==0)
        route_and_drc=(timing["timing_met"] is True and drc_clean)
        # Exploratory secondary: authentic, fully routed, timed and hash-
        # attested observations, with every warning disclosed. This does NOT
        # weaken strict DRC-clean E4 or establish board/functional readiness.
        rules=drc.get("rules")
        allowed_warnings=(
            warning_policy==DRC_OBSERVATION_POLICY
            and timing["timing_met"] is True
            and drc.get("summary_verified") is True
            and drc.get("clean") is False
            and drc.get("critical_warnings")==0
            and drc.get("unconstrained_io") is False
            and isinstance(drc.get("violations"),int)
            and drc["violations"]>0
            and isinstance(rules,dict) and bool(rules)
            and set(rules).issubset(_ALLOWED_OOC_WARNING_RULES)
            and all(isinstance(row,dict)
                    and row.get("severity")=="Warning"
                    and type(row.get("violations")) is int
                    and row["violations"]>0
                    for row in rules.values())
            and sum(row["violations"] for row in rules.values())
                    ==drc["violations"]
        )
        state.update({
            "routed_timing_met":timing["timing_met"],
            "routed_wns_ns":timing["wns_ns"],
            "physical_resources":util,
            "original_routed_report_sha256":digests,
            "drc_verified_clean":drc_clean,
            "drc_warning_summary":drc,
            "routed_qualified":route_and_drc,
            "routed_warning_observation":allowed_warnings,
            "routed_observation_valid":route_and_drc or allowed_warnings,
            "status":("QUALIFIED_VIVADO_OOC_ROUTE_TIMING_E4" if route_and_drc
                      else "FAILED_POST_ROUTE_TIMING" if timing["timing_met"] is not True
                      else "OBSERVED_OOC_ROUTE_TIMING_WITH_DISCLOSED_DRC_WARNINGS"
                      if allowed_warnings
                      else "UNKNOWN_OR_FAILED_VIVADO_DRC"),
        })
        results.append(state)
    return {
        "schema":SCHEMA,"prespecified_cases":72,
        "routed_qualified_cases":sum(r["routed_qualified"] for r in results),
        "routed_warning_observation_cases":sum(
            r["routed_warning_observation"] for r in results),
        "routed_observation_valid_cases":sum(
            r["routed_observation_valid"] for r in results),
        "vivado_invocation_receipts":sum(
            r["original_vivado_invocation_present"] for r in results),
        "original_hls_source_revision":manifest["original_hls_source_revision"],
        "cases":results,
        "evidence":"Vivado-OOC-only; no physical board I/O/bitstream qualification",
    }


def run_routed_study(study:str|Path,*,execute_native=False,
                     confirm_isolated_native=False,start_case=0,max_cases=1,
                     vivado_executable="vivado",timeout_seconds=14400):
    root=Path(study)
    audit=audit_routed_study(root)
    if (type(start_case) is not int or type(max_cases) is not int or
            not 0<=start_case<72 or not 1<=max_cases<=72 or
            start_case+max_cases>72):
        raise ValueError("Choose bounded original Vivado case range")
    selected=audit["cases"][start_case:start_case+max_cases]
    if not execute_native:
        return {"schema":SCHEMA,"status":"DRY_RUN_NO_VIVADO",
                "cases":[{"case":x["case"],"status":x["status"]} for x in selected],
                "new_vivado_calls":0}
    if not confirm_isolated_native:
        raise ValueError("Original physical run needs --confirm-isolated-native")
    if (type(timeout_seconds) is not int or
            not 1<=timeout_seconds<=172800 or not isinstance(vivado_executable,str)
            or not shutil.which(vivado_executable)):
        raise ValueError("Native Vivado executable/timeout unavailable")
    for row in selected:
        case=root/row["case"]
        if (row["status"]!="PREPARED_NOT_RUN" or
                any((case/"vivado"/name).exists() for name in
                    ("reports","logs","research_implementation","vivado_invocation.json"))):
            raise FileExistsError("Refusing to overwrite/skip unqualified physical case "+row["case"])
    ran=[]
    for row in selected:
        case=root/row["case"]
        policy=json.loads((root/"physical_study_manifest.json").read_text()
                          ).get("rom_postrun_policy")
        receipt=run_vivado(
            case,executable=vivado_executable,
            timeout_seconds=timeout_seconds,
            allow_verified_readmem_post_absence=(policy==ROM_POST_POLICY))
        measured=next(
            r for r in audit_routed_study(root)["cases"] if r["case"]==row["case"])
        ran.append({
            "case":row["case"],
            "original_vendor_process_success":receipt["completed_successfully"],
            "status":measured["status"],
            "routed_timing_met":measured["routed_timing_met"],
            "routed_wns_ns":measured["routed_wns_ns"],
            "routed_qualified":measured["routed_qualified"],
            "routed_warning_observation":measured["routed_warning_observation"],
            "routed_observation_valid":measured["routed_observation_valid"],
            "board_bitstream_qualified":False,
        })
    return {"schema":SCHEMA,"status":"FOREGROUND_VIVADO_EXECUTED",
            "new_vivado_calls":len(ran),"cases":ran}



def export_routed_evidence(*,study:str|Path,output:str|Path):
    """Create-only, all-case original Vivado physical study result export."""
    import csv
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Original routed physical report output is create-only")
    report=audit_routed_study(study)
    cases=report["cases"]
    if len(cases)!=72 or any(
            x["case"]!=f"case_{i:03d}" for i,x in enumerate(cases)):
        raise ValueError("Incomplete original Vivado physical case denominator")
    pairs=[]
    for i in range(0,72,2):
        a,b=cases[i:i+2]
        if (a["graph"]!=b["graph"] or a["mapping"]!=b["mapping"] or
                (a["source_storage"],b["source_storage"])!=(
                    "disjoint","per_element_optimal") or
                a["requested_clock_ns"]!=b["requested_clock_ns"]):
            raise ValueError("Original physical implementation pair is not matched")
        qualified=a["routed_qualified"] is True and b["routed_qualified"] is True
        observed=(a["routed_observation_valid"] is True and
                  b["routed_observation_valid"] is True)
        warned=(a["routed_warning_observation"] is True or
                b["routed_warning_observation"] is True)
        aa,bb=a["physical_resources"],b["physical_resources"]
        pairs.append({
            "pair":i//2,"cases":[a["case"],b["case"]],
            "graph":a["graph"],"mapping":a["mapping"],
            "both_source_attributed_routed_timing_drc_qualified":qualified,
            "both_routed_observation_valid":observed,
            "either_case_has_noncritical_drc_warnings":warned,
            "case_drc_warning_rules":[
                sorted((c.get("drc_warning_summary") or {}).get("rules",{}))
                for c in (a,b)],
            "descriptive_bram_tile_difference":(
                aa["bram"]-bb["bram"] if observed else None),
            "descriptive_lut_difference":(
                aa["lut"]-bb["lut"] if observed else None),
            "descriptive_ff_difference":(
                aa["ff"]-bb["ff"] if observed else None),
            "descriptive_dsp_difference":(
                aa["dsp"]-bb["dsp"] if observed else None),
            "descriptive_wns_difference_ns":(
                a["routed_wns_ns"]-b["routed_wns_ns"] if observed else None),
            "physical_bram_tile_difference":(
                aa["bram"]-bb["bram"] if qualified else None),
            "physical_lut_difference":(
                aa["lut"]-bb["lut"] if qualified else None),
            "physical_ff_difference":(
                aa["ff"]-bb["ff"] if qualified else None),
            "physical_dsp_difference":(
                aa["dsp"]-bb["dsp"] if qualified else None),
            "routed_wns_difference_ns":(
                a["routed_wns_ns"]-b["routed_wns_ns"] if qualified else None),
            "achieved_maximum_fmax":None,
            "board_bitstream_qualified":False,
            "status":"MATCHED_OOC_ROUTED_RESULTS" if qualified
                else "OBSERVED_MATCHED_OOC_ROUTE_WITH_DISCLOSED_WARNINGS"
                if observed else "UNKNOWN_OR_REJECTED_PHYSICAL_PAIR",
        })
    summary={
        "schema":"fpgai-research/cst-R2j-original-routed-fanout-pair-export/v1",
        "source_revision":report["original_hls_source_revision"],
        "prespecified_cases":72,"prespecified_pairs":36,
        "original_vivado_invocation_receipts":report["vivado_invocation_receipts"],
        "routed_qualified_cases":report["routed_qualified_cases"],
        "routed_qualified_pairs":sum(
            pair["both_source_attributed_routed_timing_drc_qualified"]
            for pair in pairs),
        "routed_warning_observation_cases":report["routed_warning_observation_cases"],
        "routed_observation_valid_cases":report["routed_observation_valid_cases"],
        "routed_observation_valid_pairs":sum(
            pair["both_routed_observation_valid"] for pair in pairs),
        "routed_observation_pairs_containing_warnings":sum(
            pair["both_routed_observation_valid"] and
            pair["either_case_has_noncritical_drc_warnings"]
            for pair in pairs),
        "unqualified_or_unattempted_cases":72-report["routed_qualified_cases"],
        "new_native_calls_during_export":0,
        "board_bitstream_qualified_cases":0,
        "limitations":[
            "12 R2i original pairs have E2-only undefined HLS latency and remain BLOCKED",
            "OOC routed resource and WNS comparison; not pin-constrained board validation",
            "Original HLS and physical reports remain source-pinned in separate studies",
            "Missing/failed cases retained as UNKNOWN, never replaced with zero",
            "Warning-only observations are distinct from strict clean-DRC E4",
            "The exploratory DRC warning allowlist was informed by the original v5 pilot",
        ],
    }
    root.mkdir(parents=True)
    (root/"summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    (root/"original_physical_audit.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    (root/"matched_pairs.json").write_text(
        json.dumps(pairs,sort_keys=True,indent=2)+"\n")
    with (root/"cases.csv").open("w",newline="") as stream:
        fields=("case","graph","mapping","source_storage","status",
                "routed_qualified","routed_observation_valid",
                "routed_warning_observation","rom_input_evidence_mode",
                "routed_wns_ns","drc_verified_clean",
                "drc_warning_count","drc_warning_rule_ids",
                "lut","ff","dsp","bram","uram")
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in cases:
            util=row.get("physical_resources") or {}
            drc=row.get("drc_warning_summary") or {}
            writer.writerow({
                **{k:row.get(k) for k in (
                    "case","graph","mapping","source_storage","status",
                    "routed_qualified","routed_observation_valid",
                    "routed_warning_observation","rom_input_evidence_mode",
                    "routed_wns_ns","drc_verified_clean")},
                "drc_warning_count":drc.get("violations"),
                "drc_warning_rule_ids":json.dumps(
                    sorted((drc.get("rules") or {}).keys())),
                **{name:util.get(name) for name in
                   ("lut","ff","dsp","bram","uram")},
            })
    with (root/"pairs.csv").open("w",newline="") as stream:
        fields=("pair","graph","mapping","status",
                "physical_bram_tile_difference","physical_lut_difference",
                "physical_ff_difference","physical_dsp_difference",
                "routed_wns_difference_ns",
                "both_source_attributed_routed_timing_drc_qualified",
                "both_routed_observation_valid",
                "either_case_has_noncritical_drc_warnings",
                "descriptive_bram_tile_difference",
                "descriptive_lut_difference","descriptive_ff_difference",
                "descriptive_dsp_difference","descriptive_wns_difference_ns")
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(pairs)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-output")
    group.add_argument("--study")
    parser.add_argument("--native-study")
    parser.add_argument("--export-output")
    parser.add_argument("--audit",action="store_true")
    parser.add_argument("--execute-native",action="store_true")
    parser.add_argument("--confirm-isolated-native",action="store_true")
    parser.add_argument("--start-case",type=int,default=0)
    parser.add_argument("--max-cases",type=int,default=1)
    parser.add_argument("--timeout-seconds",type=int,default=14400)
    args=parser.parse_args(argv)
    if args.prepare_output:
        if not args.native_study:
            parser.error("--prepare-output also requires --native-study")
        result=prepare_routed_study(
            native_study=args.native_study,output=args.prepare_output)
    elif args.export_output:
        if args.audit or args.execute_native:
            parser.error("--export-output must not be combined with audit/native execution")
        result=export_routed_evidence(
            study=args.study,output=args.export_output)
    elif args.audit:
        result=audit_routed_study(args.study)
    else:
        result=run_routed_study(
            args.study,execute_native=args.execute_native,
            confirm_isolated_native=args.confirm_isolated_native,
            start_case=args.start_case,max_cases=args.max_cases,
            timeout_seconds=args.timeout_seconds)
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
