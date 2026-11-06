"""R4w NEW isolated Vivado feasibility sweep; never overwrite original R4p.

Paired candidate periods for unchanged, source-matched R4k/R4l RTL/ROM:
16, 20, 24 ns. The old 12 ns study failed WNS and DRC and is PRESERVED.
Clock relaxation is a *diagnostic*, NOT a source-memory/port DRC fix.
Only WNS>=0 AND routing complete AND DRC zero AND intact source/log/report
receipts can qualify a new OOC implementation. No board latency claim.

The default CLI only plans or audits. Vendor execution needs two explicit
flags and a ONE-TIME attempt lock; failures remain visible, not retried.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
import shutil

from fpgai_research.experiments import cst_r4p_matched_vivado_ooc_native as r4p
from fpgai_research.experiments import cst_r4o_matched_vivado_ooc_prepare as r4o
from fpgai_research.toolchains.vivado.report import parse_vivado_reports
from fpgai_research.toolchains.vivado.advisory_observation import (
    classify_ooc_advisory_observation,
)
from fpgai_research.toolchains.vivado.runner import run_vivado

SCHEMA = "fpgai-research/R4w-isolated-matched-routed-feasible-clock-probe/v1"
ARMS = ("full_storage", "row_storage")
PERIODS_NS = (16.0, 20.0, 24.0)
SOURCE_UNCERTAINTY_NS = 1.5
RESOURCE = ("bram", "dsp", "lut", "ff", "uram")
REQUIRED_REPORTS = ("route_status.rpt", "timing_summary.rpt",
                    "utilization.rpt", "drc.rpt")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*\Z",re.ASCII)


def _sha(p: Path) -> str:
    if p.is_symlink() or not p.is_file():
        raise ValueError(f"R4w input/report missing or symlinked: {p}")
    return sha256(p.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>16_000_000:
        raise ValueError("R4w original report is missing or oversized")
    obj=json.loads(path.read_text())
    if not isinstance(obj,dict):
        raise ValueError("R4w study report must be an object")
    return obj


def _case(arm: str, period_ns: float) -> str:
    if arm not in ARMS or period_ns not in PERIODS_NS:
        raise ValueError("R4w unregistered arm or predeclared clock")
    return f"{arm}_period_{int(period_ns):02d}ns"


def _tcl(arm: str, period_ns: float, filenames: list[str]) -> str:
    if arm not in ARMS or period_ns not in PERIODS_NS:
        raise ValueError("R4w unscheduled test")
    if not filenames or any(not _ALLOWED.fullmatch(f) or
         not f.endswith((".v",".sv")) for f in filenames):
        raise ValueError("R4w unsafe HDL filenames")
    lines="\n".join(f"read_verilog {'-sv ' if f.endswith('.sv') else ''}{{{f}}}"
                    for f in filenames)
    return (
        "# NEW one-time R4w OOC clock sweep. NOT board deployment or signoff.\n"
        "# Source RTL and ROM are exact SHA-attested copies from original R4o.\n"
        "set_param general.maxThreads 4\n"
        "set case_dir [file normalize [file dirname [info script]]]\n"
        "cd [file join $case_dir rtl]\n"
        "file mkdir [file join $case_dir reports]\n"
        "file mkdir [file join $case_dir outputs]\n"
        f"{lines}\n"
        f"synth_design -top {r4o.TOP} -part {r4o.PART} -mode out_of_context\n"
        f"create_clock -name ap_clk -period {period_ns:.2f} [get_ports ap_clk]\n"
        f"set_clock_uncertainty {SOURCE_UNCERTAINTY_NS:.2f} [get_clocks ap_clk]\n"
        "opt_design\n"
        "place_design\n"
        "route_design\n"
        "report_route_status -file [file join $case_dir reports/route_status.rpt]\n"
        "report_timing_summary -delay_type max -file [file join $case_dir reports/timing_summary.rpt]\n"
        "report_utilization -file [file join $case_dir reports/utilization.rpt]\n"
        "report_drc -file [file join $case_dir reports/drc.rpt]\n"
        "report_timing -delay_type max -max_paths 10 -file [file join $case_dir reports/critical_paths.rpt]\n"
        "write_checkpoint -force [file join $case_dir outputs/routed.dcp]\n"
        "puts {R4W_COMPLETE_REQUIRES_STRICT_SHA_ROUTE_WNS_DRC_AUDIT}\n"
    )


def _origins(*, r4o_study, r4k_study, original_r4d, original_r4a):
    return r4p._original_r4o(
        r4o_study=r4o_study,r4k_study=r4k_study,
        original_r4d=original_r4d,original_r4a=original_r4a)


def _original_roots(opts: dict) -> tuple[Path,...]:
    return tuple(Path(opts[k]).resolve() for k in
        ("r4o_study","r4k_study","original_r4d","original_r4a"))


def _validate_dest(dest: Path, opts: dict):
    out=dest.resolve()
    for original in _original_roots(opts):
        if out==original or original in out.parents or out in original.parents:
            raise ValueError("R4w new run must be outside all original evidence roots")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("R4w original new output must be create-only")


def prepare(*, r4o_study: str|Path, r4k_study: str|Path,
            original_r4d: str|Path, original_r4a: str|Path,
            output: str|Path) -> dict:
    opts=locals().copy()
    dest=Path(output)
    _validate_dest(dest,opts)
    originals=_origins(**{k:opts[k] for k in
        ("r4o_study","r4k_study","original_r4d","original_r4a")})
    if originals.get("target_part")!=r4o.PART:
        raise ValueError("R4w original part changed")
    dest.mkdir(parents=True,exist_ok=False)
    try:
        cases={}
        for period in PERIODS_NS:
            for arm in ARMS:
                cid=_case(arm,period)
                group=originals["arms"][arm]
                signatures={**group["rtl_sha256"],**group["rom_sha256"]}
                folder=dest/cid
                canonical=folder/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
                native=folder/"vivado"
                consumed=native/"rtl"
                canonical.mkdir(parents=True)
                consumed.mkdir(parents=True)
                origin=Path(r4o_study)/arm/"rtl"
                for name,digest in sorted(signatures.items()):
                    if not _ALLOWED.fullmatch(name) or not _SHA.fullmatch(digest):
                        raise ValueError("R4w original RTL/ROM source name/digest invalid")
                    for location in (canonical/name, consumed/name):
                        shutil.copyfile(origin/name, location, follow_symlinks=False)
                        if _sha(location)!=digest:
                            raise ValueError("R4w copied original RTL/ROM SHA mismatch")
                    if name in group["rom_sha256"]:
                        shutil.copyfile(origin/name,native/name,follow_symlinks=False)
                        if _sha(native/name)!=digest:
                            raise ValueError("R4w runner-local ROM SHA mismatch")
                script=_tcl(arm,period,sorted(group["rtl_sha256"]))
                (native/"run_vivado.tcl").write_text(script)
                cases[cid]={
                    "arm":arm,"period_ns":period,
                    "top":r4o.TOP,"part":r4o.PART,
                    "source_cpp_sha256":group["source_cpp_sha256"],
                    "rtl_sha256":group["rtl_sha256"],
                    "rom_sha256":group["rom_sha256"],
                    "tcl_sha256":_sha(native/"run_vivado.tcl"),
                }
        receipt={
            "schema":SCHEMA,"status":"PREPARED_FOR_NEW_ISOLATED_FEASIBILITY_PROBES",
            "original_source_revision":originals["original_R4k_source_revision"],
            "original_r4o_study":str(r4o_study),
            "part":r4o.PART,"top":r4o.TOP,
            "periods_ns":list(PERIODS_NS),"paired_arms":list(ARMS),
            "predeclared_cases":cases,"new_vendor_calls":0,
            "original_evidence_modified":False,
            "strict_physical_E4":None,"board_qualified":False,
        }
        (dest/"summary.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
        manifest={p.relative_to(dest).as_posix():_sha(p)
                  for p in sorted(dest.rglob("*")) if p.is_file()}
        (dest/"artifact_sha256.json").write_text(
            json.dumps(manifest,indent=2,sort_keys=True)+"\n")
        return receipt
    except BaseException as err:
        (dest/"FAILED.json").write_text(json.dumps({
            "schema":SCHEMA+"/preserved-original-failure",
            "error":type(err).__name__+": "+str(err)[:2048],
            "original_evidence_modified":False,
            "vendor_calls":0,
        },indent=2,sort_keys=True)+"\n")
        raise


def audit_prepared(*, study: str|Path,r4o_study: str|Path,
                   r4k_study: str|Path,original_r4d: str|Path,
                   original_r4a: str|Path) -> dict:
    opts=locals().copy()
    root=Path(study)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("R4w prepared study root missing/symlinked")
    if any(root.resolve()==x or root.resolve() in x.parents
           or x in root.resolve().parents for x in _original_roots(opts)):
        raise ValueError("R4w prepared study overlaps an original")
    originals=_origins(**{k:opts[k] for k in
        ("r4o_study","r4k_study","original_r4d","original_r4a")})
    receipt=_read(root/"summary.json")
    if (receipt.get("schema")!=SCHEMA or
            receipt.get("status")!="PREPARED_FOR_NEW_ISOLATED_FEASIBILITY_PROBES" or
            receipt.get("original_source_revision")!=originals["original_R4k_source_revision"] or
            receipt.get("periods_ns")!=list(PERIODS_NS) or
            receipt.get("paired_arms")!=list(ARMS) or
            receipt.get("original_r4o_study")!=str(r4o_study) or
            receipt.get("part")!=r4o.PART or receipt.get("top")!=r4o.TOP or
            receipt.get("original_evidence_modified") is not False or
            receipt.get("new_vendor_calls")!=0):
        raise ValueError("R4w immutable source/root/target/clock plan changed")
    cases=receipt.get("predeclared_cases")
    expected={_case(arm,p) for p in PERIODS_NS for arm in ARMS}
    if not isinstance(cases,dict) or set(cases)!=expected:
        raise ValueError("R4w candidate arm/clock cohort incomplete")
    manifest=_read(root/"artifact_sha256.json")
    expected_manifest={"summary.json"}
    for cid,case in cases.items():
        arm=case["arm"]
        original=originals["arms"][arm]
        if case!={
            "arm":arm,"period_ns":case["period_ns"],"top":r4o.TOP,
            "part":r4o.PART,"source_cpp_sha256":original["source_cpp_sha256"],
            "rtl_sha256":original["rtl_sha256"],
            "rom_sha256":original["rom_sha256"],
            "tcl_sha256":case["tcl_sha256"],
        } or _case(arm,case["period_ns"])!=cid:
            raise ValueError("R4w original case source identities changed")
        rtl={**original["rtl_sha256"],**original["rom_sha256"]}
        for n,d in rtl.items():
            for path in (
                f"{cid}/fpgai_hls_project/solution1/syn/verilog/{n}",
                f"{cid}/vivado/rtl/{n}",
            ):
                expected_manifest.add(path)
                if _sha(root/path)!=d:
                    raise ValueError("R4w staged source HDL/ROM SHA changed")
            if n in original["rom_sha256"]:
                path=f"{cid}/vivado/{n}"
                expected_manifest.add(path)
                if _sha(root/path)!=d:
                    raise ValueError("R4w runner ROM source SHA changed")
        tcl=f"{cid}/vivado/run_vivado.tcl"
        expected_manifest.add(tcl)
        target=root/tcl
        if (target.read_text()!=_tcl(arm,case["period_ns"],
                                    sorted(original["rtl_sha256"]))
                or _sha(target)!=case["tcl_sha256"]):
            raise ValueError("R4w Vivado script or clock constraint changed")
        for folder in (
            root/cid/"fpgai_hls_project"/"solution1"/"syn"/"verilog",
            root/cid/"vivado"/"rtl"):
            if folder.is_symlink() or not folder.is_dir():
                raise ValueError("R4w staged original source folder missing")
            present={p.name for p in folder.iterdir()}
            if present!=set(rtl) or any(p.is_symlink() or not p.is_file()
                                      for p in folder.iterdir()):
                raise ValueError("R4w unexpected RTL/ROM files under source root")
    if set(manifest)!=expected_manifest or any(
            not isinstance(v,str) or _SHA.fullmatch(v) is None or _sha(root/n)!=v
            for n,v in manifest.items()):
        raise ValueError("R4w pre-tool exact staged artifact SHA inventory mismatch")
    return receipt


def _case_native(case_root: Path, cfg: dict) -> dict:
    native=case_root/"vivado"
    answer={
        "arm":cfg["arm"],"period_ns":cfg["period_ns"],
        "status":"UNKNOWN_NO_SOURCE_ATTESTED_VIVADO_NATIVE_ATTEMPT",
        "tool_invocation_attested":False,"routed_complete":False,
        "timing_met":False,"drc_clean":False,
        "strict_clean_OOC_qualified":False,
        "advisory_only_timing_met_not_strict":False,
        "advisory_observation_reason":None,
        "wns_ns":None,"drc_violations":None,
        "resources":{k:None for k in RESOURCE},
        "physical_core_latency_ns":None,
        "board_inference_latency_ns":None,"board_qualified":False,
    }
    receiptpath=native/"vivado_invocation.json"
    if not receiptpath.is_file() or receiptpath.is_symlink():
        return answer
    receipt=_read(receiptpath)
    expected=[
        "-mode","batch","-source","run_vivado.tcl","-notrace"]
    exe=receipt.get("executable")
    argv=receipt.get("argv")
    reportdir=native/"reports"
    fresh=receipt.get("fresh_report_sha256")
    logs=native/"logs"
    if (receipt.get("schema")!="fpgai-research/vendor-invocation/v1"
            or receipt.get("tool")!="vivado"
            or not isinstance(receipt.get("tool_version_output"),str)
            or "2023.2" not in receipt["tool_version_output"]
            or type(exe) is not str or not exe
            or type(argv) is not list or len(argv)!=6
            or argv[1:]!=expected
            or not isinstance(argv[0],str)
            or Path(argv[0]).resolve()!=Path(exe).resolve()
            or receipt.get("cwd")!=str(native.resolve())
            or receipt.get("completed_successfully") is not True
            or receipt.get("returncode")!=0
            or receipt.get("timed_out") is not False
            or receipt.get("inputs_stable_during_execution") is not True
            or receipt.get("rom_inputs_stable_during_execution") is not True
            or receipt.get("staged_rom_inputs_stable_during_execution") is not True
            or receipt.get("rtl_input_sha256")!=cfg["rtl_sha256"]
            or receipt.get("rom_input_sha256")!=cfg["rom_sha256"]
            or receipt.get("script_sha256")!=cfg["tcl_sha256"]
            or receipt.get("input_hashes_before_execution")!={
                "tcl":cfg["tcl_sha256"]}
            or receipt.get("input_hashes_after_execution")!={
                "tcl":cfg["tcl_sha256"]}
            or receipt.get("staged_rom_input_integrity_before",{}).get(
                "matches_original_roms") is not True
            or receipt.get("staged_rom_input_integrity_after",{}).get(
                "matches_original_roms") is not True
            or receipt.get("allow_verified_readmem_post_absence") is not False
            or type(receipt.get("started_utc")) is not str
            or not receipt["started_utc"]
            or type(receipt.get("duration_seconds")) not in (int,float)
            or not isfinite(receipt["duration_seconds"])
            or receipt["duration_seconds"]<0
            or any(
                not isinstance(receipt.get(key),str)
                or _SHA.fullmatch(receipt[key]) is None
                or not (logs/log).is_file()
                or (logs/log).is_symlink()
                or _sha(logs/log)!=receipt[key]
                for key,log in (("stdout_sha256","vivado.stdout.log"),
                                ("stderr_sha256","vivado.stderr.log")))
            or not isinstance(fresh,dict)
            or not set(REQUIRED_REPORTS).issubset(fresh)
            or any(not isinstance(fresh.get(name),str)
                   or _SHA.fullmatch(fresh[name]) is None
                   or not (reportdir/name).is_file()
                   or (reportdir/name).is_symlink()
                   or _sha(reportdir/name)!=fresh[name]
                   for name in REQUIRED_REPORTS)):
        answer["status"]="UNQUALIFIED_UNATTESTED_VENDOR_RECEIPT"
        return answer
    # Confirm consumed source copies are unchanged AFTER native tool.
    for name,digest in {**cfg["rtl_sha256"],**cfg["rom_sha256"]}.items():
        for p in (case_root/"fpgai_hls_project"/"solution1"/"syn"/"verilog"/name,
                  native/"rtl"/name):
            if _sha(p)!=digest:
                answer["status"]="UNQUALIFIED_NATIVE_SOURCE_DRIFT"
                return answer
        if name in cfg["rom_sha256"] and _sha(native/name)!=digest:
            answer["status"]="UNQUALIFIED_NATIVE_ROM_DRIFT"
            return answer
    answer["tool_invocation_attested"]=True
    try:
        observed=parse_vivado_reports(native)
    except (FileNotFoundError,ValueError) as e:
        answer["status"]="UNQUALIFIED_INCOMPLETE_PHYSICAL_REPORTS"
        answer["parser_issue"]=type(e).__name__+": "+str(e)[:150]
        return answer
    if observed.get("raw_report_sha256")!={n:fresh[n] for n in REQUIRED_REPORTS}:
        answer["status"]="UNQUALIFIED_ROUTED_REPORT_SHA_MISMATCH"
        return answer
    resources=observed["utilization"]
    drc=observed["drc"]
    wns=observed["timing"]["wns_ns"]
    answer["routed_complete"]=observed["routed_nets_complete"] is True
    answer["wns_ns"]=wns
    answer["timing_met"]=wns>=0
    answer["drc_violations"]=drc.get("violations")
    answer["drc_clean"]=drc.get("summary_verified") is True and drc.get("clean") is True
    answer["resources"]={k:resources.get(k) for k in RESOURCE}
    valid=(
        answer["routed_complete"] and answer["timing_met"]
        and answer["drc_clean"]
        and all(type(resources.get(k)) in (int,float)
                and isfinite(resources[k]) and resources[k]>=0
                for k in ("bram","dsp","lut","ff"))
        and drc.get("board_deployment_qualified") is False
    )
    answer["strict_clean_OOC_qualified"]=bool(valid)
    # Reuse the repo's EXISTING explicit DSP-advisory-only classifier.
    # It never grants a clean DRC or board qualification. This explains
    # the original DPIP/DPOP performance warnings without treating them
    # as memory port safety violations or waiving any original evidence.
    warning = classify_ooc_advisory_observation(
        {"evidence":"E4-Vivado-implementation",
         "csim_passed":True,  # Original R4l E2 qualified in _origins preflight.
         "vivado_metrics":observed},
        receipt)
    answer["advisory_only_timing_met_not_strict"]=(
        warning.get("eligible") is True)
    answer["advisory_observation_reason"]=warning.get("reason")
    answer["status"]=(
        "QUALIFIED_TIMING_CLEAN_DRC_ROUTED_OOC_NOT_BOARD"
        if valid else
        "TIMING_MET_RECOGNIZED_DSP_ADVISORIES_NOT_STRICT_OR_BOARD"
        if answer["advisory_only_timing_met_not_strict"] else
        "NONQUALIFIED_TIMING_DRC_OR_RESOURCES")
    return answer


def audit_native(*,study: str|Path,r4o_study: str|Path,
                 r4k_study: str|Path,original_r4d: str|Path,
                 original_r4a: str|Path) -> dict:
    opts=locals().copy()
    meta=audit_prepared(**opts)
    root=Path(study)
    lock=root/"r4w_one_time_native_attempt_lock.json"
    locked=False
    if lock.exists() and not lock.is_symlink():
        l=_read(lock)
        locked=(l.get("schema")==SCHEMA+"/one-time-vendor-attempt"
                and l.get("source_revision")==meta["original_source_revision"]
                and l.get("cases")==sorted(meta["predeclared_cases"])
                and l.get("never_retry") is True)
    cases={
        cid:_case_native(root/cid,cfg)
        for cid,cfg in meta["predeclared_cases"].items()
    }
    if not locked:
        for item in cases.values():
            item["status"]="UNQUALIFIED_MISSING_OR_INVALID_NATIVE_ATTEMPT_LOCK"
            item["strict_clean_OOC_qualified"]=False
    paired=[]
    for period in PERIODS_NS:
        a,b=(cases[_case(arm,period)] for arm in ARMS)
        qualified=(a["strict_clean_OOC_qualified"] and
                   b["strict_clean_OOC_qualified"])
        paired.append({
            "clock_period_ns":period,
            "two_arms_same_period":True,
            "both_timing_met":a["timing_met"] and b["timing_met"],
            "both_clean_drc":a["drc_clean"] and b["drc_clean"],
            "both_timing_met_with_narrow_known_DSP_advisories_not_strict":(
                a["advisory_only_timing_met_not_strict"] and
                b["advisory_only_timing_met_not_strict"]),
            "strict_physical_pair_qualified":bool(qualified),
            "source_mapped_bram_delta_row_minus_full": (
                b["resources"]["bram"]-a["resources"]["bram"]
                if qualified else None),
            "source_routed_wns_delta_row_minus_full_ns":(
                b["wns_ns"]-a["wns_ns"] if qualified else None),
            "physical_latency_ns":None,
            "board_latency_ns":None,
        })
    return {
        "schema":SCHEMA,"status":"ORIGINAL_NEW_VIVADO_FEASIBILITY_SWEEP_AUDITED_NOT_BOARD",
        "original_R4k_source_revision":meta["original_source_revision"],
        "candidate_cases":len(cases),
        "native_attempt_lock_verified":locked,
        "cases":cases,"paired_periods":paired,
        "qualified_pair_count":sum(p["strict_physical_pair_qualified"]
                                   for p in paired),
        "all_original_probes_attempted":locked and
            all((root/k/"vivado"/"vivado_invocation.json").is_file()
                for k in cases),
        "actual_new_vivado_calls_during_audit":0,
        "original_vendor_evidence_modified":False,
        "board_qualified":False,
        "board_inference_latency_ns":None,
    }


def execute(*,study: str|Path,r4o_study: str|Path,r4k_study: str|Path,
            original_r4d: str|Path,original_r4a: str|Path,
            execute_native: bool=False,
            confirm_isolated_new_native: bool=False,
            vivado_executable: str="vivado",
            timeout_seconds: int=14400) -> dict:
    opts=dict(study=study,r4o_study=r4o_study,r4k_study=r4k_study,
              original_r4d=original_r4d,original_r4a=original_r4a)
    meta=audit_prepared(**opts)
    if not execute_native:
        return {
            "schema":SCHEMA,"status":"R4W_DRY_RUN_NO_VENDOR_CALLS",
            "planned_new_vivado_calls":len(meta["predeclared_cases"]),
            "actual_new_vivado_calls":0,
        }
    if not confirm_isolated_new_native:
        raise ValueError("R4w native requires two explicit opt-in flags")
    if type(timeout_seconds) is not int or not 1<=timeout_seconds<=172800:
        raise ValueError("R4w bounded native timeout required")
    binary=shutil.which(vivado_executable)
    if binary is None:
        raise FileNotFoundError("R4w Vivado 2023.2 binary unavailable")
    root=Path(study)
    lock=root/"r4w_one_time_native_attempt_lock.json"
    if lock.exists() or lock.is_symlink():
        raise FileExistsError("R4w one-time native attempts already locked")
    for case in meta["predeclared_cases"]:
        native=root/case/"vivado"
        for name in ("logs","reports","outputs","vivado_invocation.json",
                     "vivado.jou","vivado.log"):
            p=native/name
            if p.exists() or p.is_symlink():
                raise FileExistsError("R4w original new native case was already attempted")
    # Atomic exclusive creation; preserve this lock even if first case fails.
    with lock.open("x") as f:
        f.write(json.dumps({
            "schema":SCHEMA+"/one-time-vendor-attempt",
            "source_revision":meta["original_source_revision"],
            "cases":sorted(meta["predeclared_cases"]),
            "never_retry":True,
        },indent=2,sort_keys=True)+"\n")
    attempts=[]
    for period in PERIODS_NS:
        for arm in ARMS:
            case=_case(arm,period)
            result=run_vivado(
                root/case/"vivado",executable=binary,
                timeout_seconds=timeout_seconds,
                allow_verified_readmem_post_absence=False)
            attempts.append({
                "case":case,
                "process_completed":result.get("completed_successfully") is True,
            })
            if result.get("completed_successfully") is not True:
                break
        if attempts and attempts[-1]["process_completed"] is not True:
            break
    return {
        "schema":SCHEMA,
        "status":"NEW_ISOLATED_VENDOR_ATTEMPTS_RECORDED_AUDIT_REQUIRED",
        "original_source_revision":meta["original_source_revision"],
        "actual_new_vivado_calls":len(attempts),
        "planned_new_vivado_calls":len(meta["predeclared_cases"]),
        "attempts":attempts,
        "original_vendor_evidence_modified":False,
        "board_qualified":False,
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study")
    p.add_argument("--r4o-study",required=True)
    p.add_argument("--r4k-study",required=True)
    p.add_argument("--original-r4d",required=True)
    p.add_argument("--original-r4a",required=True)
    p.add_argument("--prepare-output")
    p.add_argument("--execute-native",action="store_true")
    p.add_argument("--confirm-isolated-new-native",action="store_true")
    p.add_argument("--audit-native",action="store_true")
    p.add_argument("--vivado",default="vivado")
    p.add_argument("--timeout-seconds",type=int,default=14400)
    args=p.parse_args(argv)
    opts=dict(r4o_study=args.r4o_study,r4k_study=args.r4k_study,
              original_r4d=args.original_r4d,original_r4a=args.original_r4a)
    if args.prepare_output:
        if args.study or args.audit_native or args.execute_native:
            p.error("--prepare-output is source-only and must be separate")
        out=prepare(**opts,output=args.prepare_output)
    elif args.study:
        out=(audit_native(**opts,study=args.study) if args.audit_native
             else execute(**opts,study=args.study,
                execute_native=args.execute_native,
                confirm_isolated_new_native=args.confirm_isolated_new_native,
                vivado_executable=args.vivado,
                timeout_seconds=args.timeout_seconds))
    else:
        p.error("Provide --prepare-output OR --study")
    if args.prepare_output:
        out={k:v for k,v in out.items() if k!="predeclared_cases"}
    print(json.dumps(out,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
