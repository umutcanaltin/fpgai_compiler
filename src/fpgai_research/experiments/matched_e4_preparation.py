"""Read-only first; optional canonical Vivado Tcl staging, never vendor execution."""
from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from fpgai_research.experiments.isolated_hls_e3_ablation import collect_isolated_hls_e3_ablation
from fpgai_research.experiments.isolated_hls_replay import (
    audit_isolated_hls_replay, _COPY_FILES, _regular_bytes,
)
from fpgai_research.experiments.isolated_hls_e3_rtl_inventory import _case_rtl_inventory
from fpgai_research.experiments.native_hls_rerun_plan import _read_json
from fpgai_research.experiments.results import summarize_case
from fpgai_research.toolchains.vivado.project import make_vivado_tcl,write_vivado_package

SCHEMA="fpgai-research/matched-native-e4-preparation/v1"
AXES={"case_0000":"baseline","case_0004":"parallelism",
      "case_0005":"pes","case_0006":"simd"}
MODELS=("mnist_cnn","mnist_mlp")

def preflight_matched_e4(root, comparison, audit):
    """Require all eight native E3 cases, source hashes, fixed-point and RTL."""
    root=Path(root)
    if (comparison.get("schema")!="fpgai-research/isolated-native-hls-e3-ablation/v1"
            or audit.get("schema")!="fpgai-research/isolated-native-hls-replay-audit/v1"
            or comparison.get("source_suite_sha256")!=audit.get("source_suite_sha256")
            or comparison.get("native_tools_executed") is not False
            or audit.get("native_tools_executed") is not False
            or audit.get("per_loop_schedule_evidence_available") is not False):
        raise ValueError("Expected fresh isolated E3 lineage")
    expected={(m,c) for m in MODELS for c in AXES}
    rows=comparison.get("comparison_rows")
    native=audit.get("cases")
    if (not isinstance(rows,list) or not isinstance(native,list)
            or len(rows)!=8 or len(native)!=8):
        raise ValueError("Expected eight E3 and source audit cases")
    source_map={}
    native_map={}
    for row in rows:
        key=(row.get("model"),row.get("case"))
        if key not in expected or key in source_map or row.get("axis")!=AXES[key[1]]:
            raise ValueError("Unknown/duplicate architecture case")
        if (row.get("comparison_status")!="QUALIFIED_E3_HLS_ESTIMATE"
                or row.get("numerical_csim_passed") is not True
                or row.get("fresh_report_attested") is not True
                or row.get("evidence")!="E3-HLS-synthesis-report"):
            raise ValueError("Missing native E3 or fixed-point qualification")
        source_map[key]=row
    for row in native:
        key=(row.get("model"),row.get("case"))
        if key not in expected or key in native_map:
            raise ValueError("Unknown/duplicate E3 audit identity")
        native_map[key]=row
    if set(source_map)!=expected or set(native_map)!=expected:
        raise ValueError("Incomplete eight-case matched cohort")
    output=[]
    for model in MODELS:
        baseline=None
        for index,(case,axis) in enumerate(AXES.items()):
            row=source_map[(model,case)]
            n=native_map[(model,case)]
            folder=root/model/case
            if (folder.is_symlink() or not folder.is_dir()
                    or n.get("source_sha256")!=row.get("source_sha256")
                    or n.get("axis")!=axis
                    or n.get("evidence")!="E3-HLS-synthesis-report"
                    or n.get("fixed_point_csim_passed") is not True
                    or n.get("fresh_hls_report_attested") is not True
                    or n.get("operator_loop_ii_verified") is not False):
                raise ValueError("Incomplete attested E3 case provenance")
            manifest=_read_json(folder/"manifest.json")
            observed=summarize_case(str(folder))
            if (observed.get("source_sha256")!=row["source_sha256"]
                    or observed.get("evidence")!="E3-HLS-synthesis-report"
                    or observed.get("csim_passed") is not True
                    or _case_rtl_inventory(folder,n)["status"]!="FRESH_SOURCE_ATTESTED_RTL"):
                raise ValueError("Source/CSim/RTL changed or is not fresh")
            part=manifest.get("part"); clock=manifest.get("clock_ns")
            bits=observed.get("precision_bits")
            integer=observed.get("integer_bits")
            tolerance=observed.get("numerical_tolerance")
            if (not isinstance(part,str) or not part
                    or type(clock) not in (int,float) or clock<=0
                    or type(bits) is not int or type(integer) is not int
                    or type(tolerance) not in (float,int)
                    or observed.get("target_part")!=part
                    or observed.get("clock_ns")!=clock):
                raise ValueError("Native E3 target/clock/number contract invalid")
            fingerprint=(manifest.get("functional_sha256"),part,clock,bits,integer,tolerance)
            if baseline is None:
                baseline=fingerprint
            elif fingerprint!=baseline:
                raise ValueError("Numerical contract/part/clock changed between variants")
            canonical=make_vivado_tcl(part=part,clock_ns=clock)
            output_dir=folder/"vivado"
            if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
                raise ValueError("Unsafe existing Vivado destination")
            existing=list(output_dir.iterdir()) if output_dir.exists() else []
            if existing and (len(existing)!=1 or existing[0].name!="run_vivado.tcl"
                             or existing[0].is_symlink() or not existing[0].is_file()
                             or existing[0].read_text(encoding="utf-8")!=canonical):
                raise ValueError("Existing physical experiment would be overwritten")
            output.append({"model":model,"case":case,"axis":axis,
                           "ready_index":index,
                           "tcl_status":"STAGED_CANONICAL_TCL" if existing else "NEEDS_CANONICAL_TCL",
                           "_folder":folder,"_part":part,"_clock":clock})
    return output

def stage_independent_e4_e1_workspace(root,checked,destination):
    """Create a new PRIVATE E1-only replay; no native receipts may be copied."""
    target=Path(destination)
    if target.exists() or target.is_symlink() or target.resolve()==Path(root).resolve():
        raise FileExistsError("Use a fresh independent private E4 workspace")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-clean-e4-",dir=target.parent) as tmp:
        staged=Path(tmp)
        (staged/"replay_provenance.json").write_bytes(
            _regular_bytes(Path(root)/"replay_provenance.json",4*1024*1024))
        for model in MODELS:
            dest=staged/model
            dest.mkdir()
            (dest/"campaign.json").write_bytes(
                _regular_bytes(Path(root)/model/"campaign.json",8*1024*1024))
        for row in checked:
            case=staged/row["model"]/row["case"]
            case.mkdir()
            original=row["_folder"]
            for name,limit in _COPY_FILES.items():
                inp=original/name
                if name=="progressive_ir.json" and not inp.exists() and not inp.is_symlink():
                    continue
                (case/name).write_bytes(_regular_bytes(inp,limit))
            write_vivado_package(case,part=row["_part"],clock_ns=row["_clock"])
        independently_audited=audit_isolated_hls_replay(staged)
        if (independently_audited["case_count"]!=8
                or independently_audited["e3_source_attested_case_count"]!=0
                or independently_audited["csim_passed_case_count"]!=0
                or independently_audited["historic_native_receipts_inherited"] is not False):
            raise ValueError("Fresh E4 workspace inherited native vendor evidence")
        if target.exists() or target.is_symlink():
            raise FileExistsError("E4 stage destination appeared")
        staged.rename(target)
    return {"source_cases":8,"canonical_vivado_scripts":8,
            "native_receipts_inherited":False,"vendor_tools_executed":False}


def export_matched_e4_preparation(directory,comparison_path,output,*,stage_directory=None):
    """Do not touch the E3 source; optionally stage a clean independent E1 suite."""
    if stage_directory is not None and (
            not isinstance(stage_directory,str) or not stage_directory):
        raise ValueError("Stage directory must be an explicit nonempty path")
    root=Path(directory)
    saved=_read_json(Path(comparison_path))
    if saved!=collect_isolated_hls_e3_ablation(root):
        raise ValueError("Snapshot mismatches source-attested native E3")
    cases=preflight_matched_e4(root,saved,audit_isolated_hls_replay(root))
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Immutable E4 plan already exists")
    clean=None
    if stage_directory is not None:
        if Path(stage_directory).resolve()==target.resolve():
            raise ValueError("Stage directory and output report must be different")
        clean=stage_independent_e4_e1_workspace(root,cases,stage_directory)
    public=[{k:v for k,v in case.items() if not k.startswith("_")} for case in cases]
    result={
        "schema":SCHEMA,"source_suite_sha256":saved["source_suite_sha256"],
        "case_count":len(public),"model_count":len(MODELS),
        "clean_independent_e1_workspace_created":clean is not None,
        "staged_tcl_count":8 if clean else 0,
        "cases":public,"native_vendor_tools_executed":False,
        "native_e4_qualified":False,"historical_e3_receipts_overwritten":False,
        "original_e3_workspace_untouched":True,"native_receipts_inherited":False,
        "private_paths_exported":False,"trained_tensor_values_exported":False,
        "limitations":(
            "A separately staged E1 workspace reruns Vitis and obtains new E2/E3 "
            "source-attested results BEFORE running Vivado. Canonical Tcl is not E4. "
            "No native tools run during preparation. Post-route timing, "
            "resource, route status, DRC and numerical provenance are required."
        ),
    }
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-native-e4-",dir=target.parent) as tmp:
        folder=Path(tmp)
        (folder/"matched_e4_preparation.json").write_text(
            json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
        md=["# Matched OOC Vivado E4 preparation","",
            "Original source-attested E3 replay stays frozen. "
            "A new E1 workspace is optional and inherits NO native receipts.","",
            "| Model | Case | Axis | Ready index | Original E3 Tcl |",
            "| --- | --- | --- | ---: | --- |"]
        for row in public:
            md.append("| "+" | ".join(str(row[k]) for k in (
                "model","case","axis","ready_index","tcl_status"))+" |")
        md+=["","## Explicit native execution on NEW E1 workspace","",
             "Only after --stage-directory, use E4_DIR as the new source-only private "
             "workspace. Run new Vitis first, verify E2/E3, then Vivado.","",
             "~~~bash",
             "export E4_DIR=build/mnist_e4_independent_20261025",
             'fpgai-research-tools run-campaign --directory "$E4_DIR/mnist_cnn" '
             '--stage vitis --first 0 --limit 1 --execute',
             'fpgai-research-tools run-campaign --directory "$E4_DIR/mnist_cnn" '
             '--stage vivado --first 0 --limit 1 --require-csim-pass --execute',
             "~~~","",
             "Ready indexes 0=baseline,1=parallelism,2=PEs,3=SIMD. "
             "Do not run Vivado in the original E3 publication replay. "
             "Tool exit success is not an E4 qualification.","",
             "## Evidence boundary","",result["limitations"],""]
        (folder/"matched_e4_preparation.md").write_text("\n".join(md),encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("E4 output appeared")
        folder.rename(target)
    return result
