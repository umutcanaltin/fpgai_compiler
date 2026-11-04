"""Reaudit a generated hls4ml source-only E1 project against immutable MNIST E0.

This deliberately does NOT run CSim, compile, synthesize or route, and never
upgrades native original E4/source-weight lineage or numerical equivalence.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.hls4ml_baseline import _attest_derived_e0
from fpgai_research.experiments.hls4ml_frontend_readiness import PINNED_HLS4ML_VERSION

SCHEMA="fpgai-research/hls4ml-e1-source-reaudit/v1"


def _json(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>4*1024*1024:
        raise ValueError("Missing/unsafe hls4ml private E1 source receipt")
    result=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result,dict):
        raise ValueError("Expected E1 receipt object")
    return result


def collect_e1_source_reaudit(repository, candidate_dir, full_e0_dir,
                              model_kind, project_dir):
    root=Path(project_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing ordinary E1 project directory")
    receipt=_json(root/"research_baseline.json")
    original=Path(candidate_dir)/"candidate.onnx"
    if original.is_symlink() or not original.is_file():
        raise ValueError("Missing immutable E0 candidate ONNX")
    source_hash=sha256(original.read_bytes()).hexdigest()
    binding=_attest_derived_e0(
        repository=repository, candidate_dir=candidate_dir,
        full_e0_dir=full_e0_dir, kind=model_kind, model_sha256=source_hash)
    if (receipt.get("schema")!="fpgai-research/hls4ml-baseline/v2"
            or receipt.get("evidence")!="E1-source-only"
            or receipt.get("model_sha256")!=source_hash
            or receipt.get("derived_candidate_full_e0_lineage")!=binding
            or receipt.get("hls4ml_version")!=PINNED_HLS4ML_VERSION
            or receipt.get("numerical_equivalence")!="NOT_VERIFIED"
            or receipt.get("architecture_equivalence")!="NOT_ASSERTED"
            or receipt.get("native_synthesis")!="NOT_EXECUTED"
            or receipt.get("native_e3_e4_weight_identity_verified") is not False
            or receipt.get("backend")!="Vitis"
            or receipt.get("io_type")!="io_parallel"):
        raise ValueError("E1 project lacks exact full-E0-attested source provenance")
    inventory=receipt.get("generated_cpp_source_inventory")
    if not isinstance(inventory,list) or not inventory:
        raise ValueError("Empty E1 generated C++ inventory")
    seen=set()
    for item in inventory:
        if not isinstance(item,dict) or set(item)!={"relative_path","sha256"}:
            raise ValueError("Malformed E1 C++ source entry")
        rel=item["relative_path"]
        if (not isinstance(rel,str) or not rel
                or Path(rel).is_absolute()
                or ".." in Path(rel).parts or rel in seen
                or Path(rel).suffix.lower() not in (".cpp",".cc",".cxx")):
            raise ValueError("Invalid or duplicated HLS source path")
        seen.add(rel)
        file=root/rel
        if (file.is_symlink() or not file.is_file()
                or file.stat().st_size>16*1024*1024
                or sha256(file.read_bytes()).hexdigest()!=item["sha256"]):
            raise ValueError("Generated HLS C++ source mutated since E1")
    config=root/"hls4ml_config.yml"
    if config.is_symlink() or not config.is_file():
        raise ValueError("hls4ml project configuration missing or linked")
    text=config.read_text(encoding="utf-8")
    if "OutputDir" not in text or str(root.absolute()) not in text:
        raise ValueError("hls4ml project configuration does not point at final E1 location")
    return {
        "schema":SCHEMA,
        "model_kind":model_kind,
        "candidate_onnx_sha256":source_hash,
        "full_e0_receipt_sha256":binding["full_e0_receipt_sha256"],
        "generated_cpp_file_count":len(inventory),
        "generated_cpp_sha256":{r["relative_path"]:r["sha256"] for r in inventory},
        "hls4ml_version":receipt["hls4ml_version"],
        "original_full_heldout_e0_reattested":True,
        "hls4ml_e1_generated_source_reattested":True,
        "external_numerical_e2_verified":False,
        "external_hls_synthesis_e3_verified":False,
        "external_physical_e4_verified":False,
        "trained_to_original_native_e4_weight_identity_verified":False,
        "external_qor_comparison_eligible":False,
        "vendor_tools_executed":False,
    }


def export_e1_source_reaudit(repository, candidate_dir, full_e0_dir,
                             model_kind, project_dir, output):
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Immutable E1 audit output already exists")
    result=collect_e1_source_reaudit(
        repository,candidate_dir,full_e0_dir,model_kind,project_dir)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-e1-reaudit-",dir=dest.parent) as temp:
        stage=Path(temp)/"audit"
        stage.mkdir()
        (stage/"reaudit.json").write_text(
            json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
        (stage/"reaudit.md").write_text(
            "# hls4ml original-trained-model E1 source audit\n\n"
            f'Model: {model_kind.upper()}; generated C++ files: {result["generated_cpp_file_count"]}.\n\n'
            "Source input, selected/full E0 lineage, pinned hls4ml release "
            "and source SHA inventory rechecked.\n\n"
            "**No E2 numerical equivalence, E3 synthesis, E4 routed "
            "performance or FPGAI advantage has been established.**\n")
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("E1 audit output appeared concurrently")
        stage.rename(dest)
    return result
