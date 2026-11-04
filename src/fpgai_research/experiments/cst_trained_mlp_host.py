"""Run E1 CST compilation on *actual*, checksum-verified trained MNIST MLP.

This does not create model weights. It requires the existing trained research
suite's original ONNX, samples, labels and immutable metadata; it compiles
the complete ONNX Functional IR (including Flatten, Gemm bias and Relu).
It uses genuine ONNX Runtime as an external E0 oracle, while GCC-host-float
source is compared to FPGAI's independent NumPy reference.

This is not original Vitis fixed-point CSim/CSynth, classification accuracy
on 10,000 MNIST examples, or routed physical FPGA performance.
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
from fpgai_research.model_ir.semantics import functional_fingerprint
from fpgai_research.validation.host import validate_host
from fpgai_research.validation.reference import evaluate


def run(*, trained_directory: str | Path, output: str | Path,
        host_samples: int = 2, atol: float = 0.001,
        rtol: float = 0.001) -> dict:
    if type(host_samples) is not int or not 1 <= host_samples <= 8:
        raise ValueError("host_samples must be 1..8")
    if not (np.isfinite(atol) and np.isfinite(rtol) and
            0 < atol <= 0.01 and 0 < rtol <= 0.01):
        raise ValueError("Explicit numerical tolerance must be finite and positive")
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Create-only trained CST study already exists")
    source = Path(trained_directory)
    attested = _validate(source, "mlp", check_onnx=True)
    if attested["sample_count"] < host_samples:
        raise ValueError("Original trained sample count is too small")
    model_sha = attested["files_sha256"]["model.onnx"]
    graph = load_onnx(str(source / "model.onnx"))
    arch = resolve(graph, ArchitectureRequest())
    # Important: the compiler must preserve every layer. A trained model
    # with a different ONNX operator chain is an unsupported workload.
    compiled = compile_row_chain(arch, row_block=1)
    if [op.kind for op in graph.operators] != [
            "Flatten", "Gemm", "Relu", "Gemm"]:
        raise ValueError(
            "Trained MLP must have exactly Flatten/Gemm/Relu/Gemm for this study")
    if (len(graph.inputs) != 1 or len(graph.outputs) != 1
            or graph.tensor(graph.inputs[0]).shape != (1, 1, 28, 28)
            or graph.tensor(graph.outputs[0]).shape != (1, 10)):
        raise ValueError("Expected exact fixed-batch one-image trained MNIST MLP")
    import onnxruntime as ort
    session = ort.InferenceSession(
        str(source / "model.onnx"), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Original trained ONNX must have one runtime input/output")
    samples = np.load(source / "inputs.npy", allow_pickle=False)
    labels = np.load(source / "labels.npy", allow_pickle=False)
    rows = []
    for index in range(host_samples):
        inp = samples[index:index+1]
        if inp.shape != (1,1,28,28):
            raise ValueError("Original MNIST input shape differs")
        oracle = session.run(None, {session.get_inputs()[0].name: inp})[0]
        independent = evaluate(graph, {graph.inputs[0]: inp})[graph.outputs[0]]
        if oracle.shape != independent.shape:
            raise ValueError("Original ONNX Runtime and Model IR output shapes differ")
        reference_error = np.max(np.abs(oracle.astype(np.float64)
                                        - independent.astype(np.float64)))
        if not np.allclose(oracle, independent, atol=atol, rtol=rtol):
            raise AssertionError("Actual trained ONNX semantics differ from Model IR")
        host = validate_host(graph, compiled.emission, inp,
                             atol=atol, rtol=rtol)
        if not host["accepted"]:
            raise AssertionError("Generated CST host C++ disagrees with trained MNIST model")
        rows.append({
            "sample_index": index,
            "sample_sha256": sha256(inp.tobytes()).hexdigest(),
            "original_label": int(labels[index]),
            "onnx_top1": int(np.argmax(oracle)),
            "model_ir_top1": int(np.argmax(independent)),
            "onnx_vs_model_ir_max_abs_error": float(reference_error),
            "host_compiler_accepted": host["accepted"],
            "host_vs_model_ir_max_abs_error": host["max_absolute_error"],
            "host_cpp_sha256": host["cpp_sha256"],
            "native_vitis_executed": False,
        })
    if (_sha(source/"model.onnx") != model_sha
            or _sha(source/"inputs.npy") != attested["files_sha256"]["inputs.npy"]
            or _sha(source/"labels.npy") != attested["files_sha256"]["labels.npy"]):
        raise ValueError("Original model or evaluation asset changed during study")
    revision = subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=5)
    evidence = {
        "schema": "fpgai-research/trained-mnist-cst-host-experiment/v1",
        "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "original_trained_onnx_sha256": model_sha,
        "original_inputs_sha256": attested["files_sha256"]["inputs.npy"],
        "original_labels_sha256": attested["files_sha256"]["labels.npy"],
        "functional_graph_sha256": functional_fingerprint(graph),
        "generated_cpp_sha256": compiled.emission.manifest["cpp_sha256"],
        "operators": [op.kind for op in graph.operators],
        "trained_samples_checked": host_samples,
        "onnx_vs_model_ir_qualified": host_samples,
        "gcc_host_vs_model_ir_qualified": sum(x["host_compiler_accepted"] for x in rows),
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "row_count": 1,
        "actual_intermediate_scratch_ratio": 1,
        "source_scratch_is_not_physical_bram": True,
        "original_vitis_calls": 0,
        "evidence": "E0-original-trained-ONNX-plus-E1-host-GCC-only",
        "limitations": [
            "Only original supplied trained MNIST MLP samples, not full official test accuracy",
            "Batch size one means no source-buffer reduction by B-row scheduling",
            "Host float does not establish fixed-point Vitis CSim equivalence",
            "No physical latency/Fmax/area inference",
        ],
    }
    destination.mkdir(parents=True)
    (destination/"evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True)+"\n")
    (destination/"cases.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True)+"\n")
    return evidence


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trained-directory", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--host-samples", type=int, default=2)
    p.add_argument("--atol", type=float, default=.001)
    p.add_argument("--rtol", type=float, default=.001)
    a = p.parse_args(argv)
    print(json.dumps(run(trained_directory=a.trained_directory,
                         output=a.output, host_samples=a.host_samples,
                         atol=a.atol, rtol=a.rtol), indent=2))


if __name__ == "__main__":
    main()
