"""Read-only hls4ml comparison preflight from independently qualified evidence.

Never claim that identical ONNX input, target part and nominal bit width
prove bit-exact numerical equivalence, source lineage into E4, or fair
routed performance. hls4ml/QONNX preprocessing and tool execution are
explicit separate steps on the user's licensed workstation.
"""
from __future__ import annotations

from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import re

from fpgai_research.experiments.independent_e4_audit import collect_independent_e4
from fpgai_research.experiments.mnist_heldout_paper import (
    collect_heldout_paper, DEFAULT_SUITE, _DEFAULT_TRAINED,
)
from fpgai_research.experiments.results import summarize_case

SCHEMA="fpgai-research/hls4ml-comparison-preflight/v1"
PART=re.compile(r"[A-Za-z0-9_.+\-]+\Z")
_MODELS=("mlp","cnn")
_BASELINE_CASE="case_0000"
_NATIVE_ROOT="build/mnist_e4_independent_20261025"


def _as_verified_positive(value)->bool:
    return type(value) in (int,float) and isfinite(value) and value>0


def collect_hls4ml_comparison_preflight(repository)->dict:
    """Verify pinned E0 inputs + baseline E4 contracts without tool execution.

    Returning an eligible *source-preparation* contract never upgrades
    an external framework to E2/E3/E4, nor proves the native source
    is from the same ONNX weights.
    """
    root=Path(repository)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Expected a regular private research repository")
    source=collect_heldout_paper(root)
    if (source.get("model_count")!=2
            or source.get("both_source_identical_to_assembled_paper_suite") is not True
            or source.get("both_original_training_accuracies_reverified") is not True
            or source.get("native_e3_e4_weight_identity_verified") is not False):
        raise ValueError("Trained models lack independently source-attested E0 evidence")
    native=root/_NATIVE_ROOT
    if native.is_symlink() or not native.is_dir():
        raise ValueError("Independent eight-case private physical replay missing")
    audit=collect_independent_e4(native)
    if (audit.get("e3_qualified_case_count")!=8
            or audit.get("e4_qualified_case_count")!=8
            or audit.get("qualified_matched_e4_pair_count")!=4
            or audit.get("vendor_tools_executed_by_audit") is not False):
        raise ValueError("Original native E4 physical suite incomplete")
    audit_cases=audit.get("cases")
    if (not isinstance(audit_cases,list) or len(audit_cases)!=8
            or any(not isinstance(r,dict) for r in audit_cases)):
        raise ValueError("Original native E4 inventory must contain eight cases")
    indexed={(r.get("model"),r.get("case")):r for r in audit_cases}
    expected={(f"mnist_{kind}",case)
              for kind in _MODELS
              for case in ("case_0000","case_0004","case_0005","case_0006")}
    if len(indexed)!=len(audit_cases) or set(indexed)!=expected:
        raise ValueError("Original native E4 case inventory duplicated or mismatched")
    rows=[]
    for kind in _MODELS:
        e4_model="mnist_"+kind
        case=native/e4_model/_BASELINE_CASE
        if case.is_symlink() or not case.is_dir():
            raise ValueError("Unsafe independent native baseline case")
        original=root/_DEFAULT_TRAINED[kind]/"model.onnx"
        assembled=root/DEFAULT_SUITE/"assets"/e4_model/"model.onnx"
        if (original.is_symlink() or assembled.is_symlink()
                or not original.is_file() or not assembled.is_file()):
            raise ValueError("Missing original trained ONNX and paper-suite copy")
        # collect_heldout_paper already checks these original files and hashes.
        import hashlib
        if hashlib.sha256(original.read_bytes()).digest()!=hashlib.sha256(assembled.read_bytes()).digest():
            raise ValueError("Trained ONNX and assembled paper suite differ")
        evidence=indexed.get((e4_model,_BASELINE_CASE))
        if (not isinstance(evidence,dict)
                or evidence.get("qualified_e4") is not True
                or evidence.get("qualification_status")!="E4_QUALIFIED_MATCHED_OOC"
                or evidence.get("axis")!="baseline"
                or evidence.get("numerical_csim_passed") is not True
                or evidence.get("implementation_mode")!="out_of_context"):
            raise ValueError("Requested native baseline is not independently qualified")
        summary=summarize_case(str(case))
        if (summary.get("evidence")!="E4-Vivado-implementation"
                or summary.get("csim_passed") is not True):
            raise ValueError("Native baseline source/numerical reports not reattested")
        width=summary.get("precision_bits")
        integer=summary.get("integer_bits")
        part=summary.get("target_part")
        clock=summary.get("clock_ns")
        if (type(width) is not int or type(integer) is not int
                or not 2<=width<=32 or not 1<=integer<=width
                or not isinstance(part,str) or not PART.fullmatch(part)
                or not _as_verified_positive(clock)):
            raise ValueError("Native numeric, part or clock contract incomplete")
        rows.append({
            "model":kind,
            "trained_onnx_identical_to_assembled_suite":True,
            "native_baseline_case":_BASELINE_CASE,
            "native_source_attested_e3_e4":True,
            "native_fixed_point_selected_csim_passed":True,
            "native_target_part":part,
            "native_target_clock_ns":clock,
            "native_precision_total_bits":width,
            "native_precision_integer_bits":integer,
            "candidate_hls4ml_io_mode":"io_parallel",
            "native_arithmetic_rounding_overflow_accumulation_matched":False,
            "hls4ml_input_layout_and_onnx_ops_verified":False,
            "hls4ml_converted_e1_verified":False,
            "hls4ml_fixed_point_numerical_e2_verified":False,
            "hls4ml_native_synthesis_e3_verified":False,
            "hls4ml_routed_e4_verified":False,
            "trained_onnx_to_native_hls_weight_identity_verified":False,
        })
    return {
        "schema":SCHEMA,"model_count":2,"cases":rows,
        "native_fpga_tools_executed_by_preflight":False,
        "hls4ml_invoked_by_preflight":False,
        "all_native_e4_baseline_cases_qualified":True,
        "both_trained_models_source_qualified":True,
        "external_qor_comparison_eligible":False,
        "external_latency_or_resource_advantage_claimed":False,
        "no_private_model_paths_hashes_or_weights_exported":True,
        "status":"E1_EXTERNAL_PREPARATION_REQUIRES_NUMERIC_AND_FRONTEND_REVIEW",
        "next_checks":(
            "Pin hls4ml, QONNX and Vitis 2023.2-compatible tooling. Verify "
            "raw ONNX graph operators, Gemm decomposition and CNN channels-last "
            "preprocessing. Match per-layer weight/accumulator precisions, "
            "rounding, saturation/wrap, I/O layout, reuse and clock/FPGA "
            "target to the *native* model. Establish independently verified "
            "ONNX-to-generated-native-source weight lineage, then execute "
            "independent numerical CSim and source/RTL-attested HLS and "
            "routed physical reports. Until all these checks pass, no "
            "head-to-head latency/resource/accuracy claim is supported."
        ),
    }


def export_hls4ml_comparison_preflight(repository,output)->dict:
    result=collect_hls4ml_comparison_preflight(repository)
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Immutable external comparison preflight already exists")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-hls4ml-preflight-",dir=target.parent) as temp:
        stage=Path(temp)
        (stage/"preflight.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",
            encoding="utf-8")
        lines=[
            "# hls4ml baseline comparison — read-only preflight","",
            "**This is not a measured external benchmark.**","",
            "| Model | Native baseline E4 | FPGA part | Clock (ns) | Native bits | Source ONNX matches suite |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
        for row in result["cases"]:
            lines.append(
                f'| {row["model"].upper()} | qualified | '
                f'{row["native_target_part"]} | '
                f'{row["native_target_clock_ns"]} | '
                f'{row["native_precision_total_bits"]}/'
                f'{row["native_precision_integer_bits"]} | yes |'
            )
        lines.extend([
            "","## Unresolved controls","",
            result["next_checks"],"",
            "No hls4ml native synthesis, CSim, routing or performance "
            "comparison was run by this audit.",
        ])
        (stage/"preflight.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("External preflight output appeared during export")
        stage.rename(target)
    return result
