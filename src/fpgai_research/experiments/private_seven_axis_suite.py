"""Assemble private MLP/CNN seven-axis suite for the existing full-suite engine.

Copies exactly selected input/model bytes into a fresh, private build
workspace with SHA-256 hashes; never downloads, trains or runs vendor tools.
The downstream suite validates actual Model IR and computes legal plans.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import shutil

SCHEMA="fpgai-research/private-seven-axis-suite/v1"
MAX_ONNX=128*1024*1024
MAX_NUMPY=128*1024*1024


def _check_file(value,suffix,limit):
    source=Path(value)
    if (source.is_symlink() or not source.is_file()
            or source.suffix.lower()!=suffix or source.stat().st_size>limit
            or source.stat().st_size==0):
        raise ValueError("Expected a regular bounded private research asset")
    return source


def assemble_seven_axis_suite(mlp_onnx,mlp_input,cnn_onnx,cnn_input,output):
    from fpgai_research.experiments.full_suite import (
        SCHEMA as SUITE_SCHEMA, validate_suite,
    )
    dst=Path(output)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError("Seven-axis suite destination must be NEW")
    originals={
        "mlp_onnx":_check_file(mlp_onnx,".onnx",MAX_ONNX),
        "mlp_input":_check_file(mlp_input,".npy",MAX_NUMPY),
        "cnn_onnx":_check_file(cnn_onnx,".onnx",MAX_ONNX),
        "cnn_input":_check_file(cnn_input,".npy",MAX_NUMPY),
    }
    # Do not infer data/weight equivalence from same filenames.
    source_digests={k:sha256(v.read_bytes()).hexdigest() for k,v in originals.items()}
    spec={
        "schema":SUITE_SCHEMA,
        "notes":"Private 2-model seven-axis E0/E1 preparation; no native results",
        "experiments":[
            {"id":"mnist_mlp_axes","family":"mlp",
             "onnx":"inputs/mlp.onnx","input":"inputs/mlp.npy"},
            {"id":"mnist_cnn_axes","family":"cnn",
             "onnx":"inputs/cnn.onnx","input":"inputs/cnn.npy"},
        ],
    }
    validate_suite(spec)
    dst.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-private-axis-",dir=dst.parent) as tmp:
        bundle=Path(tmp)/"bundle"
        inputs=bundle/"inputs"
        inputs.mkdir(parents=True)
        names={"mlp_onnx":"mlp.onnx","mlp_input":"mlp.npy",
               "cnn_onnx":"cnn.onnx","cnn_input":"cnn.npy"}
        for name,src in originals.items():
            target=inputs/names[name]
            shutil.copyfile(src,target,follow_symlinks=False)
            if sha256(target.read_bytes()).hexdigest()!=source_digests[name]:
                raise ValueError("Seven-axis private source bytes changed during staging")
        (bundle/"suite.json").write_text(
            json.dumps(spec,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        receipt={
            "schema":SCHEMA,"source_asset_sha256":source_digests,
            "models":2,"stage":"E0/E1-input-bundle-only",
            "full_seven_axis_evidence_qualified":False,
            "numerical_e2_qualified":False,
            "native_e3_e4_executed":False,
            "weights_retrained":False,
            "vendor_tools_executed":False,
        }
        (bundle/"bundle_receipt.json").write_text(
            json.dumps(receipt,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        if dst.exists() or dst.is_symlink():
            raise FileExistsError("Seven-axis suite appeared concurrently")
        bundle.rename(dst)
    return receipt
