"""Compile a synthetic convolution/dense ONNX example with the real FPGAI pipeline.

This is a source-only example. The optional --verify-host mode compiles and
executes the generated C++ with GNU g++ and checks every intermediate tensor
against the independent NumPy reference. Neither mode invokes Vitis/Vivado.

Examples:
  python examples/compile_neural_network.py build/gallery/pool.onnx --output build/gallery/pool-scalar
  python examples/compile_neural_network.py build/gallery/pool.onnx --output build/gallery/pool-mapped --mapping independent --verify-host
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from math import prod
from pathlib import Path

from fpgai_research.architecture.resolver import resolve
from fpgai_research.compiler.neural_chain import compile_neural_chain
from fpgai_research.frontend.onnx import load_onnx
from fpgai_research.model_ir.contracts import ArchitectureRequest


def operator_mapping(model, *, mapping: str) -> dict:
    """Specify bounded operator-local Phi mapping, not a new emitter.

    Matches the legal small Conv/MatMul architecture shapes exercised by the
    existing mixed-network source campaign. All source legality and actual
    lowering remain owned by the real compiler.
    """
    if mapping not in ("scalar", "independent"):
        raise ValueError("Example mapping must be scalar or independent")
    if mapping == "scalar":
        return {}
    base = ArchitectureRequest()
    overrides = {}
    for op in model.operators:
        if op.kind == "Conv":
            shape = model.tensor(op.outputs[0]).shape
            _, channels, height, width = shape
            kernel = model.tensor(op.inputs[1]).shape
            overrides[op.name] = replace(
                base, pes=2, simd=2, serialization=2,
                tile=(shape[0], min(2, channels), min(2, height),
                      min(3, width), kernel[1], kernel[2], kernel[3]),
            )
        elif op.kind in ("MatMul", "Gemm"):
            output = model.tensor(op.outputs[0]).shape
            reduction = prod(model.tensor(op.inputs[0]).shape[1:])
            overrides[op.name] = replace(
                base, pes=2, simd=3, serialization=2,
                tile=(output[0], min(3, output[1]), reduction),
            )
    return overrides


def compile_example(
    model_path: str | Path, output: str | Path, *,
    mapping: str = "scalar", verify_host: bool = False,
) -> dict:
    """Write only new generated-source artifacts after successful validation."""
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite example output: {destination}")
    if type(verify_host) is not bool:
        raise TypeError("verify_host must be bool")
    model = load_onnx(str(model_path))
    arch = resolve(
        model, ArchitectureRequest(),
        operator_mapping(model, mapping=mapping),
    )
    compilation = compile_neural_chain(arch)
    emission = compilation.emission
    ops = [op.kind for op in model.operators]
    if ops != [x["kind"] for x in emission.manifest["operators"]]:
        raise AssertionError("Canonical compilation changed the original NN order")

    report = {
        "network": Path(model_path).stem,
        "mapping": mapping,
        "source_generated": True,
        "operator_kinds": ops,
        "input_shape": list(model.tensor(model.inputs[0]).shape),
        "output_shape": list(model.tensor(model.outputs[0]).shape),
        "canonical_cpp_sha256": emission.manifest["cpp_sha256"],
        "all_stages_lowered": len(compilation.ir.stages) == len(ops),
        "source_tensor_boundaries": len(compilation.ir.boundaries),
        "native_vitis_or_vivado_calls": 0,
        "physical_timing_or_resource_claim": False,
        "host_verification": "NOT_REQUESTED",
    }
    if not report["all_stages_lowered"]:
        raise AssertionError("Incomplete NN compilation")

    host_report = None
    if verify_host:
        import numpy as np
        from fpgai_research.validation.neural_chain_host import verify_neural_chain_gcc

        shape = model.tensor(model.inputs[0]).shape
        x = np.random.default_rng(197).uniform(
            -0.4, 0.4, size=shape).astype(np.float32)
        host_report = verify_neural_chain_gcc(
            model, emission, x, atol=1e-4, rtol=1e-3)
        if host_report.get("all_stages_accepted") is not True or (
                host_report.get("final_output_accepted") is not True):
            raise ValueError("Generated HLS C++ failed intermediate/reference checks")
        report["host_verification"] = "ALL_INTERMEDIATE_TENSORS_AGREED"
        report["verified_intermediate_stages"] = host_report["stage_count"]
        report["host_gcc_executions"] = host_report["actual_generated_GCC_executions"]

    destination.mkdir(parents=True, exist_ok=False)
    (destination / "model.cpp").write_text(emission.cpp, encoding="utf-8")
    (destination / "manifest.json").write_text(
        json.dumps(emission.manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    (destination / "network_contract.json").write_text(
        json.dumps(compilation.evidence, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    (destination / "example_summary.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    if host_report is not None:
        (destination / "host_reference_verification.json").write_text(
            json.dumps(host_report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Existing static ONNX model")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument(
        "--mapping", choices=("scalar", "independent"), default="scalar",
        help="Scalar control or separate Conv/Dense architecture mapping",
    )
    parser.add_argument(
        "--verify-host", action="store_true",
        help="Compile generated HLS C++ with g++ and check every intermediate "
             "tensor against an independent NumPy reference (no Vitis/Vivado)",
    )
    args = parser.parse_args(argv)
    report = compile_example(
        args.model, args.output,
        mapping=args.mapping, verify_host=args.verify_host,
    )
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
