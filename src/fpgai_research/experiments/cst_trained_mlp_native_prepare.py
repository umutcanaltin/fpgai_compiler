"""Create-only original-Vitis HLS package for actual trained MNIST CST MLP.

Requires previously qualified, checksum-verified original trained weights and
inputs. Never trains/synthesizes anything or mixes historical paper results.
The source package may subsequently be executed on a distinct original
Vitis-equipped checkout; actual numeric failure is a scientific result.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.row_chain import compile_row_chain
from fpgai_research.experiments.mnist_assets import _sha, _validate
from fpgai_research.frontend.onnx import load_onnx
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.toolchains.vitis_hls.package import write_vitis_package


def prepare(*, trained_directory: str | Path, output: str | Path,
            sample_count: int = 1,
            part: str = "xck26-sfvc784-2LV-c",
            clock_ns: float = 32.0,
            clock_uncertainty_ns: float = 8.64,
            numeric_tolerance: float = 0.05) -> dict:
    if type(sample_count) is not int or not 1 <= sample_count <= 4:
        raise ValueError("sample_count must be 1..4")
    if not (np.isfinite(clock_ns) and np.isfinite(clock_uncertainty_ns)
            and 0 <= clock_uncertainty_ns < clock_ns):
        raise ValueError("Invalid original Vitis clock/uncertainty")
    if not (np.isfinite(numeric_tolerance) and 0 < numeric_tolerance <= 1):
        raise ValueError("Invalid bounded fixed-point CSim tolerance")
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Create-only original trained-model native package")
    folder = Path(trained_directory)
    checked = _validate(folder, "mlp", check_onnx=True)
    if checked["sample_count"] < sample_count:
        raise ValueError("Not enough original provenance-qualified samples")
    model = load_onnx(str(folder/"model.onnx"))
    if [op.kind for op in model.operators] != [
            "Flatten","Gemm","Relu","Gemm"]:
        raise ValueError("Original trained model has unsupported CST operator sequence")
    if (model.tensor(model.inputs[0]).shape != (1,1,28,28)
            or model.tensor(model.outputs[0]).shape != (1,10)):
        raise ValueError("Expected the static batch-one trained MNIST graph")
    compiled = compile_row_chain(resolve(model, ArchitectureRequest()),row_block=1)
    images = np.load(folder/"inputs.npy",allow_pickle=False)[:sample_count]
    labels = np.load(folder/"labels.npy",allow_pickle=False)[:sample_count]
    pkg = write_vitis_package(
        str(root/"native_case"),model,compiled.emission,images,
        part=part, clock_ns=clock_ns, clock_uncertainty_ns=clock_uncertainty_ns,
        tolerance=numeric_tolerance,labels=labels)
    # Create a machine-readable immutable output root before native execution.
    sha = subprocess.run(["git","rev-parse","HEAD"],
                         capture_output=True,text=True,timeout=5)
    report = {
        "schema":"fpgai-research/trained-cst-original-vitis-source-package/v1",
        "source_revision":sha.stdout.strip() if sha.returncode==0 else None,
        "trained_model_sha256":checked["files_sha256"]["model.onnx"],
        "trained_inputs_sha256":checked["files_sha256"]["inputs.npy"],
        "trained_labels_sha256":checked["files_sha256"]["labels.npy"],
        "generated_cpp_sha256":pkg["cpp_sha256"],
        "testbench_sha256":pkg["testbench_sha256"],
        "tcl_sha256":sha256((root/"native_case"/"run_hls.tcl").read_bytes()).hexdigest(),
        "host_training_examples_used":False,
        "original_trained_weights_preserved":True,
        "prepared_input_samples":sample_count,
        "static_inference_batch_size":1,
        "source_scratch_ratio":1,
        "requested_clock_ns":clock_ns,
        "requested_clock_uncertainty_ns":clock_uncertainty_ns,
        "fixed_point_tolerance":numeric_tolerance,
        "source_package_directory":"native_case",
        "original_vitis_calls":0,
        "csim_executed":False,
        "csynth_executed":False,
        "original_vitis_measurements":None,
    }
    if (_sha(folder/"model.onnx") != checked["files_sha256"]["model.onnx"]
            or _sha(folder/"inputs.npy") != checked["files_sha256"]["inputs.npy"]
            or _sha(folder/"labels.npy") != checked["files_sha256"]["labels.npy"]):
        raise ValueError("Original trained assets changed during native preparation")
    (root/"source_manifest.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trained-directory",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--sample-count",type=int,default=1)
    p.add_argument("--part",default="xck26-sfvc784-2LV-c")
    p.add_argument("--clock-ns",type=float,default=32.0)
    p.add_argument("--clock-uncertainty-ns",type=float,default=8.64)
    p.add_argument("--numeric-tolerance",type=float,default=.05)
    a=p.parse_args(argv)
    print(json.dumps(prepare(
        trained_directory=a.trained_directory,output=a.output,
        sample_count=a.sample_count,part=a.part,clock_ns=a.clock_ns,
        clock_uncertainty_ns=a.clock_uncertainty_ns,
        numeric_tolerance=a.numeric_tolerance),indent=2))


if __name__=="__main__":
    main()
