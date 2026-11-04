"""Source-provenance-checked *real trained model* CST coverage preflight.

This prepares the scientific workload gap analysis. It does not silently
strip Flatten/Relu/Gemm/Conv, approximate a trained network with MatMuls,
train new weights, or claim a CST native/Vitis result. The already-validated
trained-MNIST asset contract and strict ONNX importer are reused.

The canonical row-chain research path supports binary constant-weight
MatMul chains only. A full trained MLP with Flatten/Gemm/Relu must report
UNSUPPORTED rather than returning a fake positive CST result.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.contracts import ArchitectureRequest, FunctionalModel
from fpgai_research.model_ir.row_chain import resolve_row_chain
from fpgai_research.model_ir.semantics import functional_fingerprint


def inspect_model(model: FunctionalModel) -> dict:
    """E0 coverage analysis, never interpreted as lowered native compiler."""
    if not isinstance(model, FunctionalModel):
        raise TypeError("Expected typed FunctionalModel")
    model.validate()
    arch = resolve(model, ArchitectureRequest())
    operations = [op.kind for op in model.operators]
    edges = {}
    for op in model.operators:
        for name in op.inputs:
            edges.setdefault(name, []).append(op.name)
    shared = sorted(
        name for name, consumers in edges.items()
        if len(consumers) > 1 and name not in model.inputs
    )
    try:
        checked = resolve_row_chain(arch)
        state = ("SUPPORTED_STATIC_MATMUL_CHAIN"
                 if all(op.kind == "MatMul" for op in model.operators)
                 else "SUPPORTED_SEQUENTIAL_CST_ROW_CHAIN")
        blocker = None
        exposed = len(checked.composed.boundaries)
    except ValueError as exc:
        state = "UNSUPPORTED_BY_CURRENT_ROW_LIFETIME_LOWERING"
        blocker = str(exc)
        exposed = 0
    return {
        "schema": "fpgai-research/cst-real-graph-capability-audit/v1",
        "functional_sha256": functional_fingerprint(model),
        "operators": operations,
        "operator_count": len(operations),
        "model_input_shapes": [list(model.tensor(x).shape) for x in model.inputs],
        "model_output_shapes": [list(model.tensor(x).shape) for x in model.outputs],
        "non_matmul_operators": sorted(
            {kind for kind in operations if kind != "MatMul"}),
        "multireader_tensor_names": shared,
        "architecture_resolved": True,
        "row_lifetime_status": state,
        "first_blocker": blocker,
        "supported_compositional_boundaries": exposed,
        "cst_hls_source_emitted": False,
        "independent_numerical_reference_executed": False,
        "original_vitis_calls": 0,
        "note": "E0 workload admission only; not source, numerical or native HLS qualification",
    }


def audit_trained_mnist(*, trained_directory: str | Path, kind: str,
                        output: str | Path) -> dict:
    """Read a *real* training suite only with exact existing source hashes."""
    if kind not in ("mlp", "cnn"):
        raise ValueError("kind must be mlp or cnn")
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Create-only trained-model audit already exists")
    folder = Path(trained_directory)
    from fpgai_research.experiments.mnist_assets import _sha, _validate
    source = _validate(folder, kind, check_onnx=True)
    model_sha = source["files_sha256"]["model.onnx"]
    from fpgai_research.frontend.onnx import load_onnx
    model = load_onnx(str(folder/"model.onnx"))
    report = inspect_model(model)
    if _sha(folder/"model.onnx") != model_sha:
        raise ValueError("Trained ONNX model was modified during capability audit")
    report.update({
        "training_asset_kind": kind,
        "trained_model_onnx_sha256": model_sha,
        "trained_input_sha256": source["files_sha256"]["inputs.npy"],
        "trained_labels_sha256": source["files_sha256"]["labels.npy"],
        "trained_e0_conformance_recorded": source["onnx_e0_conformance_recorded"],
        "onnx_checker_passed": source["onnx_checker_passed"],
        "evaluated_samples_in_asset": source["sample_count"],
        "evaluation_of_trained_model_performed_by_this_tool": False,
        "historical_original_weights": source["historical_original_weights"],
    })
    # No raw samples, ONNX payload, labels or trained weights are exported.
    target.mkdir(parents=True)
    (target/"capability.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)+"\n")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trained-directory", required=True)
    p.add_argument("--kind", choices=("mlp", "cnn"), required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    print(json.dumps(audit_trained_mnist(
        trained_directory=a.trained_directory, kind=a.kind,
        output=a.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
