"""Run every *legal* prepared architecture experiment on a real workstation.

This is an opt-in, synchronous, resumable native execution driver; it never
fabricates results. Each case is independently attested before/after invoking
the existing Vitis/Vivado runners. Infeasible source-level architecture
requests remain explicitly excluded, never silently counted as wins.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from fpgai_research.experiments.full_suite import audit_suite
from fpgai_research.experiments.execution import run_campaign
from fpgai_research.experiments.results import summarize_case
from fpgai_research.experiments.suite_native_batch import _qualify, _qualify_hls

SCHEMA="fpgai-research/exhaustive-eligible-native-workstation-run/v1"
CASE_ID=re.compile(r"case_[0-9]{4,8}\Z")
STUDY_ID=re.compile(r"[a-z][a-z0-9_]{0,40}\Z")


def _file_hash(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Required immutable suite manifest is missing")
    return sha256(path.read_bytes()).hexdigest()


def _write_atomic(path, document):
    """Durable incremental journal, never a vendor qualification receipt."""
    dest=Path(path)
    if dest.is_symlink():
        raise ValueError("Journal must not be a symbolic link")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(mode="w",encoding="utf-8",dir=dest.parent,
                            prefix=".workstation-campaign-",delete=False) as handle:
        temp=Path(handle.name)
        try:
            json.dump(document,handle,sort_keys=True,indent=2,allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp,dest)


def _summarize(case):
    result=summarize_case(str(case))
    # The existing HLS/implementation gates require native, source-attributed
    # evidence; command exit status cannot replace these independent checks.
    return {
        "native_evidence":result.get("evidence"),
        "csim_numerical_pass":result.get("csim_passed") is True,
        "csim_sample_count":(result.get("csim_receipt") or {}).get("samples"),
        "csim_failures":(result.get("csim_receipt") or {}).get("failures"),
        "csim_rmse":(result.get("csim_receipt") or {}).get("rmse"),
        "csim_max_absolute_error":(result.get("csim_receipt") or {}).get(
            "max_absolute_error"),
        "numerical_tolerance":result.get("numerical_tolerance"),
        "csim_diagnostics_source":(result.get("csim_receipt") or {}).get("source"),
        "hls_synthesis_qualified":not _qualify_hls(result),
        "routed_implementation_qualified":not _qualify(result),
        "hls_blockers":_qualify_hls(result),
        "physical_blockers":_qualify(result),
        "hls_estimated_latency":(result.get("metrics") or {}).get("latency_best"),
        "postroute_wns_ns":((result.get("vivado_metrics") or {}).get("timing") or {}).get("wns_ns"),
    }


def _new_report(root, suite_sha):
    return {
        "schema":SCHEMA,
        "prepared_suite_sha256":suite_sha,
        "execution_explicitly_requested":False,
        "qualified_native_cases_cannot_be_inferred_from_exit_code":True,
        "vendor_tools_run_by_this_driver":False,
        "cases":[],
        "excluded_infeasible_architectures":[],
        "blocked_source_studies":[],
    }


def execute_comprehensive_suite(directory, *,
                                execute=False, resume=False, journal=None,
                                start=0, max_cases=0,
                                continue_on_error=True,
                                vitis_executable="vitis_hls",
                                vivado_executable="vivado",
                                progress=True):
    if (type(execute) is not bool or type(resume) is not bool
            or type(continue_on_error) is not bool
            or type(progress) is not bool):
        raise ValueError("Execution switches must be explicit booleans")
    if type(start) is not int or type(max_cases) is not int or start<0 or max_cases<0:
        raise ValueError("Invalid selected architecture interval")
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing real prepared native suite")
    suite=root/"suite_results.json"
    sha=_file_hash(suite)
    report=audit_suite(root,require_host=execute,require_all_axes=False)
    if not report.get("all_prepared_models_have_legal_executable_subsets",
                      report.get("all_models_source_ready", False)):
        raise ValueError("One or more prepared studies have unaudited legal cases")
    manifest=json.loads(suite.read_text(encoding="utf-8"))
    source_studies=manifest.get("studies",[])
    # Infeasible points inside a *prepared* study are legitimate negative
    # feasibility observations; unavailable or unsupported entire models
    # are reported separately and make completion claims impossible.
    if type(source_studies) is not list or not source_studies:
        raise ValueError("Empty or malformed experiment suite")
    if not any(item.get("status")=="E1-PREPARED" for item in source_studies):
        raise ValueError("No legal architecture studies prepared for native execution")
    output=Path(journal) if journal else None
    if execute and output is None:
        raise ValueError("Explicit --journal required for native execution")
    if resume and (output is None or output.is_symlink() or not output.is_file()):
        raise ValueError("Resumption requires the existing unlinked journal")
    if output and output.is_symlink():
        raise ValueError("Symbolic-link journal is prohibited")
    if execute and output.exists() and not resume:
        raise FileExistsError("Use a fresh journal or explicit --resume")
    previous=None
    if resume:
        previous=json.loads(output.read_text(encoding="utf-8"))
        if (previous.get("schema")!=SCHEMA
                or previous.get("prepared_suite_sha256")!=sha):
            raise ValueError("Resume journal cannot be transferred between suites")

    selected=[]
    skipped=[]
    blocked=[]
    for item in source_studies:
        name=item.get("id")
        if not isinstance(name,str) or not STUDY_ID.fullmatch(name):
            raise ValueError("Unsafe study identity")
        if item.get("status")!="E1-PREPARED":
            skipped.append({"study":name,"reason":item.get("reason"),
                            "status":item.get("status")})
            continue
        folder=root/name
        original=folder/"campaign.json"
        _file_hash(original)
        experiment=json.loads(original.read_text(encoding="utf-8"))
        if experiment.get("schema")!="fpgai-research/paper-campaign/v1":
            raise ValueError("Unrecognized frozen architecture inventory")
        for case in experiment.get("cases",[]):
            case_id=case.get("case")
            if not isinstance(case_id,str) or not CASE_ID.fullmatch(case_id):
                raise ValueError("Unsafe architecture case identity")
            status=case.get("status")
            identity={"study":name,"case":case_id,"axis":case.get("dimension"),
                      "requested":case.get("requested")}
            if status=="BLOCKED":
                blocked.append({**identity,"status":"INFEASIBLE_ARCHITECTURE",
                                "reason":case.get("reason")})
            elif status=="READY":
                selected.append((identity,folder,case))
            else:
                raise ValueError("Invalid source experiment status")
    if not selected:
        raise ValueError("No legal architecture cases available")
    window=selected[start:start+max_cases] if max_cases else selected[start:]
    if not window:
        raise ValueError("Selected architecture window has no cases")
    if resume:
        # Never copy a claimed physical success from the prior journal.
        # Every previously selected case is independently reaudited below.
        previous_keys={(x.get("study"),x.get("case")) for x in previous.get("cases",[])}
        if len(previous_keys)!=len(previous.get("cases",[])):
            raise ValueError("Resume journal duplicates case identities")
    result=_new_report(root,sha)
    result["execution_explicitly_requested"]=execute
    result["excluded_infeasible_architectures"]=blocked
    result["blocked_source_studies"]=skipped
    result["total_legal_architecture_cases"]=len(selected)
    result["selected_legal_cases"]=len(window)
    result["selected_window"]={"start":start,"max_cases":max_cases}
    result["scientific_limitations"]=(
        "Native C simulation, HLS synthesis and routed implementation must "
        "independently pass source-attributed audits. Existing data is never "
        "represented as newly executed. No measured board latency, power, "
        "full-dataset fixed-point accuracy or external compiler superiority "
        "is implied. A blocked architecture is not a passing circuit."
    )
    if execute:
        _write_atomic(output,result)
    if progress:
        print(f"[FPGAI] Native campaign: {len(window)} selected of "
              f"{len(selected)} legal architecture cases; "
              f"{len(blocked)} infeasible requests excluded", flush=True)
    for position,(identity,folder,case) in enumerate(window, start=1):
        project=folder/identity["case"]
        if progress:
            print(f"[FPGAI] case {position}/{len(window)}: "
                  f"{identity['study']}/{identity['case']} "
                  f"axis={identity['axis']}", flush=True)
        if execute:
            result["active_case"]={
                "study":identity["study"],"case":identity["case"],
                "axis":identity["axis"],"state":"RUNNING_OR_AUDITING",
                "started_utc":__import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc).isoformat(),
            }
            _write_atomic(output,result)
        # Keep a private detailed vendor run receipt in the already
        # prepared case folder, not in the paper-facing journal.
        original=_summarize(project)
        if original["routed_implementation_qualified"]:
            record={**identity,"action":"ALREADY_QUALIFIED","before":original,
                    "after":original,"vendor_tools_executed_for_case":False}
        elif not execute:
            record={**identity,"action":"DRY_RUN_ONLY","before":original,
                    "vendor_tools_executed_for_case":False,
                    "needed_stages":(
                        ["vitis","vivado"] if not original["hls_synthesis_qualified"]
                        else ["vivado"]
                    )}
        else:
            stages=(("vivado",) if original["hls_synthesis_qualified"]
                    else ("vitis","vivado"))
            error=None
            try:
                outcome=run_campaign(folder,stages=stages,execute=True,
                    first=sum(x.get("status")=="READY" for x in
                              json.loads((folder/"campaign.json").read_text())["cases"]
                              if int(x["case"].split("_")[1])<int(identity["case"].split("_")[1])),
                    limit=1,continue_on_error=False,require_csim_pass=True,
                    vitis_executable=vitis_executable,
                    vivado_executable=vivado_executable)
                commands=outcome["cases"][0]["stages"]
                if any(x.get("status")=="FAILED" for x in commands.values()):
                    error="One or more native commands failed; consult case receipt"
            except (OSError,RuntimeError,ValueError) as exc:
                commands={}
                error=f"{type(exc).__name__}: {exc}"
            # Even a successful tool invocation does not establish physical
            # qualification; recompute the actual source-linked vendor report.
            try:
                after=_summarize(project)
            except (OSError,RuntimeError,ValueError) as exc:
                after=None
                error=(error+"; " if error else "")+f"Audit: {type(exc).__name__}: {exc}"
            record={**identity,"action":"EXECUTED","requested_stages":list(stages),
                    "vendor_stage_status":commands,"before":original,
                    "after":after,"error":error,
                    "vendor_tools_executed_for_case":True}
        result["cases"].append(record)
        result["vendor_tools_run_by_this_driver"] |= record["vendor_tools_executed_for_case"]
        if execute:
            result.pop("active_case",None)
            _write_atomic(output,result)
        if progress:
            print(f"[FPGAI] case {position}/{len(window)}: "
                  f"{record['action']} / "
                  f"{(record.get('after') or record.get('before') or {}).get('native_evidence')}; "
                  f"routed-qualified={bool((record.get('after') or record.get('before') or {}).get('routed_implementation_qualified'))}",
                  flush=True)
        if execute and not continue_on_error and (
                record.get("error") or
                (record.get("after") or {}).get("routed_implementation_qualified") is not True):
            break
    result["native_hls_qualified_selected"]=sum(
        bool((x.get("after") or x.get("before") or {}).get("hls_synthesis_qualified"))
        for x in result["cases"])
    result["native_routed_qualified_selected"]=sum(
        bool((x.get("after") or x.get("before") or {}).get("routed_implementation_qualified"))
        for x in result["cases"])
    result["every_selected_case_routed_qualified"]=(
        len(result["cases"])==len(window)
        and result["native_routed_qualified_selected"]==len(window))
    result["all_eligible_suite_cases_routed_qualified"]=(
        start==0 and max_cases==0
        and not skipped
        and result["every_selected_case_routed_qualified"]
    )
    result["all_requested_architecture_points_routed_qualified"]=(
        result["all_eligible_suite_cases_routed_qualified"] and not blocked
    )
    if execute:
        _write_atomic(output,result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",required=True)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--journal",help="Fresh immutable-source-bound native execution journal")
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--max-cases",type=int,default=0,
                        help="0 executes all prepared legal cases")
    parser.add_argument("--fail-fast",action="store_true")
    parser.add_argument("--vitis-executable",default="vitis_hls")
    parser.add_argument("--vivado-executable",default="vivado")
    parser.add_argument("--verbose-cases",action="store_true",
                        help="Also print per-case progress for dry runs")
    parser.add_argument("--full-json",action="store_true",
                        help="Print the complete, potentially huge case inventory")
    parser.add_argument("--report-file",
                        help="Save the complete case inventory to a NEW JSON file")
    args=parser.parse_args(argv)
    result=execute_comprehensive_suite(args.directory,
        execute=args.execute,resume=args.resume,journal=args.journal,
        start=args.start,max_cases=args.max_cases,
        continue_on_error=not args.fail_fast,
        vitis_executable=args.vitis_executable,
        vivado_executable=args.vivado_executable,
        progress=args.execute or args.verbose_cases)
    if args.report_file:
        destination=Path(args.report_file)
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open("x",encoding="utf-8") as handle:
            json.dump(result,handle,sort_keys=True,indent=2,allow_nan=False)
            handle.write("\n")
    if args.full_json:
        print(json.dumps(result,sort_keys=True,indent=2))
    else:
        summary={
            "schema":result["schema"],
            "execution_explicitly_requested":result["execution_explicitly_requested"],
            "selected_legal_cases":result["selected_legal_cases"],
            "total_legal_architecture_cases":result["total_legal_architecture_cases"],
            "cases_inspected":len(result["cases"]),
            "excluded_infeasible_architectures":len(
                result["excluded_infeasible_architectures"]),
            "blocked_source_studies":len(result["blocked_source_studies"]),
            "native_hls_qualified_selected":result["native_hls_qualified_selected"],
            "native_routed_qualified_selected":result["native_routed_qualified_selected"],
            "every_selected_case_routed_qualified":
                result["every_selected_case_routed_qualified"],
            "vendor_tools_run_by_this_driver":result["vendor_tools_run_by_this_driver"],
            "detailed_report_file":args.report_file,
            "native_journal":args.journal if args.execute else None,
        }
        print(json.dumps(summary,sort_keys=True,indent=2))
    return int(args.execute and not result["every_selected_case_routed_qualified"])


if __name__=="__main__":
    raise SystemExit(main())
