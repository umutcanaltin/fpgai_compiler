"""Verify the two FPGAI and hls4ml candidates share original trained networks.

Source-level only. Cross-compiler source hashes cannot prove identical
native fixed-point weights, numerical semantics or HLS/route execution.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

SCHEMA="fpgai-research/same-network-two-compiler-source-audit/v1"


def _read_json(file):
    p=Path(file)
    if p.is_symlink() or not p.is_file() or not 0<p.stat().st_size<=2*1024*1024:
        raise ValueError("Missing/unsafe source receipt JSON")
    return json.loads(p.read_text(encoding="utf-8"))


def _actual_digest(path,*,max_bytes=128*1024*1024):
    p=Path(path)
    if p.is_symlink() or not p.is_file() or not 0<p.stat().st_size<=max_bytes:
        raise ValueError("Missing/unsafe original model or cohort bytes")
    return sha256(p.read_bytes()).hexdigest()


def attest_same_networks(fpgai_source_bundle, mlp_hls_receipt, cnn_hls_receipt):
    root=Path(fpgai_source_bundle)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing FPGAI original-model private bundle")
    source=_read_json(root/"bundle_receipt.json")
    if (not isinstance(source,dict)
            or source.get("schema")!="fpgai-research/private-seven-axis-suite/v1"
            or source.get("models")!=2
            or source.get("native_e3_e4_executed") is not False):
        raise ValueError("Unqualified FPGAI source bundle")
    assets=source.get("source_asset_sha256")
    if not isinstance(assets,dict) or set(assets)!={"mlp_onnx","mlp_input","cnn_onnx","cnn_input"}:
        raise ValueError("Missing original FPGAI source assets")
    for key,rel in (("mlp_onnx","mlp.onnx"),("mlp_input","mlp.npy"),
                    ("cnn_onnx","cnn.onnx"),("cnn_input","cnn.npy")):
        if _actual_digest(root/"inputs"/rel)!=assets[key]:
            raise ValueError("FPGAI original source asset digest mismatch")
    rows=[]
    receipts={"mlp":mlp_hls_receipt,"cnn":cnn_hls_receipt}
    for kind in ("mlp","cnn"):
        p=Path(receipts[kind])
        h=_read_json(p)
        if (not isinstance(h,dict) or h.get("schema")!="fpgai-research/hls4ml-baseline/v2"
                or h.get("evidence")!="E1-source-only"
                or h.get("native_synthesis")!="NOT_EXECUTED"
                or h.get("part")!="xck26-sfvc784-2LV-c"
                or h.get("clock_ns")!=5
                or h.get("backend")!="Vitis"
                or h.get("hls4ml_version")!="1.3.0"):
            raise ValueError("Unmatched hls4ml part, tool or E1 receipt")
        lineage=h.get("derived_candidate_full_e0_lineage")
        if (not isinstance(lineage,dict)
                or lineage.get("model_kind")!=kind
                or lineage.get("full_e0_selected_candidate_qualified") is not True
                or lineage.get("original_to_native_e3_e4_weight_identity_verified") is not False
                or lineage.get("derived_candidate_onnx_sha256")!=h.get("model_sha256")
                or lineage.get("original_source_onnx_sha256")!=assets[kind+"_onnx"]):
            raise ValueError("Not the same original-trained model or E0 qualified candidate")
        if _actual_digest(h.get("model_path",""))!=h["model_sha256"]:
            raise ValueError("Actual derived hls4ml ONNX source bytes do not match")
        rows.append({
            "model_kind":kind,
            "original_trained_onnx_sha256":assets[kind+"_onnx"],
            "ordered_source_input_sha256":assets[kind+"_input"],
            "derived_compatible_onnx_sha256":h["model_sha256"],
            "hls4ml_e1_receipt_sha256":_actual_digest(p,max_bytes=2*1024*1024),
            "original_to_derived_full_software_e0_receipt_qualified":True,
            "original_network_source_identity":True,
            "same_native_weights_and_fixed_point_numerics_proven":False,
        })
    return {
        "schema":SCHEMA,
        "source_bundle_receipt_sha256":_actual_digest(
            root/"bundle_receipt.json",max_bytes=2*1024*1024),
        "models_compared":2,
        "all_original_source_models_pinned":True,
        "records":rows,
        "same_ordered_native_sample_cohort_qualified":False,
        "original_trained_weights_in_generated_hls_verified":False,
        "vendor_tools_executed_by_audit":False,
        "head_to_head_e2_e3_e4_qualified":False,
        "cross_compiler_hardware_superiority_proven":False,
    }


def export_same_networks(bundle,mlp_hls,cnn_hls,output):
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Two-compiler source audit must use a new output path")
    result=attest_same_networks(bundle,mlp_hls,cnn_hls)
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".same-network-",dir=target.parent) as tmp:
        folder=Path(tmp)/"report"
        folder.mkdir()
        (folder/"same_networks.json").write_text(
            json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Output appeared during source-audit export")
        folder.rename(target)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fpgai-source-bundle",required=True)
    parser.add_argument("--mlp-hls4ml-receipt",required=True)
    parser.add_argument("--cnn-hls4ml-receipt",required=True)
    parser.add_argument("--output",required=True)
    a=parser.parse_args(argv)
    r=export_same_networks(a.fpgai_source_bundle,a.mlp_hls4ml_receipt,
                           a.cnn_hls4ml_receipt,a.output)
    print(json.dumps(r,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
