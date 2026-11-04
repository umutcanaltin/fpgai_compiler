"""Expand two original-trained FPGAI networks into every compute-operator study.

Creates separate comprehensive seven-axis plans for each actual MatMul,
Gemm or Conv operator without changing original model/weight/input bytes.
This does not execute native vendor synthesis or count blocked points as wins.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from fpgai_research.experiments.full_suite import SCHEMA as SUITE_SCHEMA, validate_suite
from fpgai_research.experiments.private_seven_axis_suite import SCHEMA as SOURCE_SCHEMA
from fpgai_research.experiments.plan_factory import make_comprehensive_seven_axis_plan
from fpgai_research.frontend.onnx import load_onnx

SCHEMA="fpgai-research/all-compute-operator-comprehensive-suite/v1"


def _load_regular(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file() or not 0<p.stat().st_size<=128*1024*1024:
        raise ValueError("Missing or unsafe original-trained source asset")
    return p.read_bytes()


def assemble_all_operator_suite(two_model_bundle, output):
    source=Path(two_model_bundle)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Missing private original-network source bundle")
    receipt=json.loads(_load_regular(source/"bundle_receipt.json"))
    if (not isinstance(receipt,dict) or receipt.get("schema")!=SOURCE_SCHEMA
            or receipt.get("models")!=2 or receipt.get("native_e3_e4_executed") is not False):
        raise ValueError("Expected two original-trained, not-native source inputs")
    digests=receipt.get("source_asset_sha256")
    if not isinstance(digests,dict) or set(digests)!={"mlp_onnx","mlp_input","cnn_onnx","cnn_input"}:
        raise ValueError("Incomplete original-model input fingerprint inventory")
    source_bytes={}
    models={}
    for kind in ("mlp","cnn"):
        for suffix in ("onnx","input"):
            key=kind+"_"+suffix
            ext=".onnx" if suffix=="onnx" else ".npy"
            path=source/"inputs"/(kind+ext)
            data=_load_regular(path)
            if sha256(data).hexdigest()!=digests[key]:
                raise ValueError("Original trained model/input bytes changed")
            source_bytes[key]=data
        model=load_onnx(str(source/"inputs"/(kind+".onnx")))
        model.validate()
        models[kind]=model
    output=Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Every operator-level campaign uses a new immutable directory")

    entries=[]
    plans={}
    operator_counts={}
    for kind in ("mlp","cnn"):
        selected=[
            operator.name for operator in models[kind].operators
            if operator.kind in ("MatMul","Gemm","Conv")
        ]
        if not selected or len(set(selected))!=len(selected):
            raise ValueError("Expected unique named hardware compute operators")
        operator_counts[kind]=len(selected)
        for index,operator in enumerate(selected):
            ident=f"{kind}_operator_{index:02d}"
            spec=make_comprehensive_seven_axis_plan(models[kind],operator=operator)
            rel=f"plans/{ident}.json"
            plans[rel]=json.dumps(spec,indent=2,sort_keys=True,allow_nan=False)+"\n"
            entries.append({"id":ident,"family":kind,
                "onnx":f"inputs/{kind}.onnx",
                "input":f"inputs/{kind}.npy",
                "plan":rel,
                "operator":operator})
    if len(entries)>16:
        raise ValueError("Operator inventory exceeds supported bounded suite capacity")
    suite={
        "schema":SUITE_SCHEMA,
        "notes":"All actual MLP/CNN compute operators; seven main effects and explicit interactions; no native execution",
        "experiments":entries
    }
    validate_suite(suite)
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".all-operator-suite-",dir=output.parent) as tmp:
        root=Path(tmp)/"suite"
        (root/"inputs").mkdir(parents=True)
        (root/"plans").mkdir()
        for kind in ("mlp","cnn"):
            for suffix in ("onnx","input"):
                key=kind+"_"+suffix
                ext=".onnx" if suffix=="onnx" else ".npy"
                (root/"inputs"/(kind+ext)).write_bytes(source_bytes[key])
        for path,text in plans.items():
            (root/path).write_text(text,encoding="utf-8")
        (root/"suite.json").write_text(json.dumps(suite,sort_keys=True,indent=2)+"\n")
        result={
            "schema":SCHEMA,
            "source_bundle_receipt_sha256":sha256(_load_regular(
                source/"bundle_receipt.json")).hexdigest(),
            "original_source_asset_sha256":digests,
            "operator_counts":operator_counts,
            "total_operator_experiments":len(entries),
            "all_eligible_compute_operators_declared":True,
            "profile":"seven_axes_all_pairwise_and_three_higher_order_interactions",
            "model_weights_changed":False,
            "native_vendor_tools_executed":False,
            "native_numerical_or_physical_results_qualified":False
        }
        (root/"source_audit.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Operator suite destination appeared")
        root.rename(output)
    return result
