"""Independently replay immutable 1024/4096 Vitis HLS synthesis report pair.

Rechecks source, 4096 selected-native numerical evidence, both original
Vitis logs and corresponding vendor csynth XML resources. This does NOT
qualify trained-model hls4ml vs FPGAI E3, routed E4 or power.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.hls4ml_softmax_synth_pair import (
    _preflight,_hash,_regular,_report,RUN_SCHEMA,
)

SCHEMA="fpgai-research/paired-softmax-hls-resource-replay/v1"


def audit_hls_synth_pair(stage,native_run):
    source=Path(stage)
    root=Path(native_run)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing safe native HLS synthesis result directory")
    prepared=_preflight(source)
    receipt_path=_regular(root/"synthesis_results.json")
    recorded=json.loads(receipt_path.read_text())
    if (not isinstance(recorded,dict)
            or recorded.get("schema")!=RUN_SCHEMA
            or recorded.get("stage_receipt_sha256")!=_hash(
                source/"experiment.json")
            or recorded.get("native_tool_execution_requested") is not True
            or recorded.get("native_exploratory_hls_pair_complete") is not True
            or recorded.get("full_model_e2_qualified") is not False
            or recorded.get("external_cross_compiler_qor_eligible") is not False
            or recorded.get("routed_e4_qualified") is not False
            or recorded.get("vendor_vivado_executed") is not False
            or recorded.get("exploratory_within_hls4ml_only") is not True):
        raise ValueError("HLS synthesis native receipt is not complete/scoped")
    runs=recorded.get("variant_native_results")
    if not isinstance(runs,dict) or set(runs)!={"1024","4096"}:
        raise ValueError("Native HLS paired receipt must contain both variants")
    verified={}
    for label in ("1024","4096"):
        item=runs[label]
        if (not isinstance(item,dict)
                or item.get("native_attempted") is not True
                or item.get("native_returncode")!=0
                or item.get("source_rechecked_after_native") is not True
                or not isinstance(item.get("report"),dict)):
            raise ValueError("One HLS synthesis is not independently attested")
        log_path=_regular(root/(label+"_vitis_hls.log"),16*1024*1024)
        log_hash=sha256(log_path.read_bytes()).hexdigest()
        if item.get("native_log_sha256")!=log_hash:
            raise ValueError("Original Vitis HLS log fingerprint changed")
        actual=_report(source/label/"project")
        if item["report"]!=actual:
            raise ValueError("Native HLS csynth report content/hash changed")
        verified[label]={
            "native_log_sha256":log_hash,
            "csynth_report":actual,
        }
    a=verified["1024"]["csynth_report"]["metrics"]
    b=verified["4096"]["csynth_report"]["metrics"]
    delta={}
    for name in sorted(set(a)&set(b)):
        try:
            first,second=float(a[name]),float(b[name])
        except ValueError:
            continue
        if isfinite(first) and isfinite(second):
            delta[name]={
                "1024":first,"4096":second,"difference":second-first}
    if recorded.get("matched_tool_report_metrics")!=delta or not delta:
        raise ValueError("Recorded HLS synthesis metric deltas cannot be reproduced")
    return {
        "schema":SCHEMA,
        "source_manifest_sha256":_hash(source/"experiment.json"),
        "original_native_synthesis_receipt_sha256":
            sha256(receipt_path.read_bytes()).hexdigest(),
        "exact_both_hls_logs_and_xml_reports_rechecked":True,
        "source_and_selected_native_4096_rechecked":True,
        "variants":verified,
        "matched_hls_report_metrics":delta,
        "qualified_exploratory_within_hls4ml_synthesis_receipts":True,
        "qualified_fair_cross_compiler_e3":False,
        "qualified_routed_e4":False,
        "full_10000_fixed_point_measured":False,
        "new_vendor_tools_executed_by_replay":False,
        "limitations":(
            "Native HLS synthesis report comparison within one hls4ml "
            "model only; the 1024 variant previously failed probability "
            "semantics. Neither external fair compiler comparison nor "
            "routed hardware, board power or held-out 10k accuracy "
            "is established by this result."),
    }


def export_hls_synth_pair_replay(stage,native_run,output):
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("New immutable HLS synthesis replay directory required")
    doc=audit_hls_synth_pair(stage,native_run)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".hls-softmax-resource-replay-",dir=dest.parent) as tmp:
        written=Path(tmp)/"report"
        written.mkdir()
        (written/"resource_replay.json").write_text(json.dumps(
            doc,indent=2,sort_keys=True,allow_nan=False)+"\n")
        lines=[
            "# hls4ml reciprocal lookup size: replayed HLS resource estimates",
            "",
            "1024-entry vs 4096-entry source-matched project, "
            "native Vitis logs and csynth XML fingerprints rechecked.",
            "",
            "| Native metric | 1024 entries | 4096 entries | Difference |",
            "|---|---:|---:|---:|",
        ]
        for key,values in doc["matched_hls_report_metrics"].items():
            lines.append(
                f"| {key} | {values['1024']:.8g} | "
                f"{values['4096']:.8g} | {values['difference']:+.8g} |")
        lines.extend(["","This is exploratory within-hls4ml HLS resource "
                      "sensitivity only. The 1024 CSim variant fails "
                      "probability semantics. No fair cross-compiler E3, "
                      "route E4, board power or full-model E2 is qualified.",""])
        (written/"summary.md").write_text("\n".join(lines))
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("HLS resource replay output appeared")
        written.rename(dest)
    return doc


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",required=True)
    p.add_argument("--native-run",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args(argv)
    doc=export_hls_synth_pair_replay(a.stage,a.native_run,a.output)
    print(json.dumps({
        "schema":doc["schema"],
        "source_attested":doc[
            "source_and_selected_native_4096_rechecked"],
        "native_reports_attested":doc[
            "exact_both_hls_logs_and_xml_reports_rechecked"],
        "cross_compiler_e3_qualified":False,
        "new_vendor_execution":False,
        "matched_resource_metrics":doc["matched_hls_report_metrics"],
    },indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
