from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.make_quantized_residual_cnn_example import write_model
from scripts.run_quantized_residual_cnn_ptq import _emit_testbench, _get, _load_config
from fpgai.backends.hls.codegen import emit_hls_stub
from fpgai.backends.hls.emit.params_cpp import emit_params_cpp
from fpgai.backends.hls.emit.params_h import emit_params_h
from fpgai.backends.hls.emit.types_h import emit_types_h
from fpgai.benchmark.training_qat_reference import (
    execute_frozen_qat_reference,
    run_qat_training_dataset_reference,
)
from fpgai.quantization import (
    apply_model_qat_to_hls_graph,
    execute_quantized_hls_reference,
    partition_residual_add_and_terminal_relu,
    partition_terminal_relu,
)
from fpgai.quantization.hardware import quantization_parameters_from_tensor
from fpgai.quantization.ptq import dequantize, quantize


def _training_dataset(sample_count: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if sample_count <= 0:
        raise ValueError("QAT residual CNN sample_count must be positive")
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(-1.5, 1.5, size=(sample_count, 1, 1, 4, 4)).astype(np.float32)
    # Train toward a stable residual-ReLU identity target.  The skip path already
    # carries the input, so QAT learns to reduce unnecessary residual correction
    # while exercising both convolutions and the residual merge.
    targets = np.maximum(inputs, np.float32(0.0)).astype(np.float32)
    return inputs, targets


def _numeric_comparison(*, fake_output: np.ndarray, integer_output: np.ndarray, output_qparams) -> dict:
    fake_integer = quantize(np.asarray(fake_output, dtype=np.float32), output_qparams).astype(np.int64)
    integer_output = np.asarray(integer_output, dtype=np.int64)
    dequantized_integer = dequantize(integer_output, output_qparams).astype(np.float32)
    difference = np.asarray(fake_output, dtype=np.float32) - dequantized_integer
    integer_difference = fake_integer - integer_output
    max_integer_abs_error = int(np.max(np.abs(integer_difference))) if integer_difference.size else 0
    mismatch_count = int(np.count_nonzero(integer_difference))
    return {
        "schema": "fpgai.qat-integer-numeric-comparison/v1",
        "integer_exact_after_output_quantization": bool(np.array_equal(fake_integer, integer_output)),
        "integer_mismatch_count": mismatch_count,
        "max_integer_abs_error_lsb": max_integer_abs_error,
        "within_one_lsb": bool(max_integer_abs_error <= 1),
        "max_abs_error": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "mae": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "fake_quant_output_integer": fake_integer.reshape(-1).astype(int).tolist(),
        "integer_reference_output": integer_output.reshape(-1).astype(int).tolist(),
        "fake_quant_output_float": np.asarray(fake_output).reshape(-1).astype(float).tolist(),
        "integer_reference_dequantized": dequantized_integer.reshape(-1).astype(float).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and export the maintained quantized residual CNN through QAT, then validate common integer/HLS lowering.")
    parser.add_argument("--config", default="configs/examples/quantized_residual_cnn_qat.yml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--part", default=None)
    parser.add_argument("--clock-mhz", type=int, default=None)
    parser.add_argument("--skip-hls", action="store_true")
    args = parser.parse_args()

    from fpgai.frontend.onnx import import_onnx

    raw = _load_config(Path(args.config))
    if str(_get(raw, "numerics.quantization.mode", "none")) != "qat":
        raise ValueError("QAT residual CNN runner requires numerics.quantization.mode=qat")
    out_dir = Path(args.out or _get(raw, "project.out_dir", "build/quantized_residual_cnn_qat")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = out_dir / "reports"
    reports.mkdir(exist_ok=True)

    model_path = write_model(out_dir / "quantized_residual_cnn.onnx")
    graph = import_onnx(str(model_path))
    inputs, targets = _training_dataset(int(args.samples), seed=int(args.seed))
    training = run_qat_training_dataset_reference(
        graph=graph,
        raw_cfg=raw,
        out_dir=out_dir,
        inputs=inputs,
        targets=targets,
    )

    hardware_graph = copy.deepcopy(training.trained_graph)
    lowering = apply_model_qat_to_hls_graph(hardware_graph, training.qat_result)
    (reports / "quantized_hls_lowering.json").write_text(
        json.dumps(lowering.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    validation_input = np.linspace(-1.5, 1.5, 16, dtype=np.float32).reshape(1, 1, 4, 4)
    fake_output = execute_frozen_qat_reference(
        graph=training.trained_graph,
        qat_result=training.qat_result,
        raw_cfg=raw,
        out_dir=out_dir / "qat_validation_reference",
        x_input=validation_input,
    )
    input_qparams = quantization_parameters_from_tensor(hardware_graph.get_tensor(hardware_graph.inputs[0]))
    output_qparams = quantization_parameters_from_tensor(hardware_graph.get_tensor(hardware_graph.outputs[0]))
    input_q = quantize(validation_input, input_qparams)
    full_expected_q = execute_quantized_hls_reference(hardware_graph, input_q)
    comparison = _numeric_comparison(
        fake_output=fake_output,
        integer_output=full_expected_q,
        output_qparams=output_qparams,
    )
    comparison_path = reports / "qat_integer_numeric_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    final_relu_backend = str(_get(raw, "targets.mixed_backend.final_relu_backend", "hls")).strip().lower()
    residual_add_backend = str(_get(raw, "targets.mixed_backend.residual_add_backend", "hls")).strip().lower()
    partition = None
    sum_expected_q = None
    if residual_add_backend == "vhdl" and final_relu_backend in {"vhdl", "hls"}:
        relu_only_graph = copy.deepcopy(hardware_graph)
        partition_terminal_relu(relu_only_graph, backend=final_relu_backend)
        sum_expected_q = execute_quantized_hls_reference(relu_only_graph, input_q)
        partition = partition_residual_add_and_terminal_relu(
            hardware_graph, add_backend="vhdl", relu_backend=final_relu_backend
        )
        hls_expected_q = execute_quantized_hls_reference(hardware_graph, input_q)
    elif residual_add_backend == "hls" and final_relu_backend == "vhdl":
        partition = partition_terminal_relu(hardware_graph, backend="vhdl")
        hls_expected_q = execute_quantized_hls_reference(hardware_graph, input_q)
    elif residual_add_backend == "hls" and final_relu_backend == "hls":
        hls_expected_q = full_expected_q
    else:
        raise ValueError("mixed-backend residual_add_backend/final_relu_backend combination is unsupported")
    if partition is not None:
        (reports / "quantized_operator_partition.json").write_text(
            json.dumps(partition.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    numeric = {
        "schema": "fpgai.quantized-residual-cnn-numeric/v1",
        "quantization_source": "qat_frozen_export",
        "input_float": validation_input.reshape(-1).tolist(),
        "input_integer": input_q.reshape(-1).astype(int).tolist(),
        "expected_integer": full_expected_q.reshape(-1).astype(int).tolist(),
        "hls_expected_integer": hls_expected_q.reshape(-1).astype(int).tolist(),
        "sum_expected_integer": (sum_expected_q.reshape(-1).astype(int).tolist() if sum_expected_q is not None else None),
        "input_quantization": input_qparams.to_dict(),
        "output_quantization": output_qparams.to_dict(),
        "partition": partition.to_dict() if partition is not None else None,
        "qat_integer_comparison": comparison,
        "status": "reference_ready",
    }
    (reports / "quantized_numeric_validation.json").write_text(
        json.dumps(numeric, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    project = emit_hls_stub(
        graph=hardware_graph,
        out_dir=out_dir,
        top_name="deeplearn",
        hls_options={
            "weights_mode": "embedded",
            "part": str(args.part or _get(raw, "targets.platform.part", "xck26-sfvc784-2LV-c")),
            "clk_mhz": int(args.clock_mhz or _get(raw, "targets.platform.clocks.0.target_mhz", 200)),
            "pipeline_mode": "inference",
            "raw_cfg": raw,
            "run_csim": True,
            "run_csynth": True,
            "export_ip": False,
        },
    )
    (project.hls_dir / "include" / "fpgai_types.h").write_text(
        emit_types_h(hardware_graph, top_name="deeplearn", raw_cfg=raw), encoding="utf-8"
    )
    (project.hls_dir / "include" / "fpgai_params.h").write_text(
        emit_params_h(hardware_graph, weights_mode="embedded"), encoding="utf-8"
    )
    (project.hls_dir / "src" / "fpgai_params.cpp").write_text(
        emit_params_cpp(hardware_graph, weights_mode="embedded", storage_impl="bram"), encoding="utf-8"
    )
    _emit_testbench(project.hls_dir / "src" / "tb.cpp", input_q, hls_expected_q)

    if args.skip_hls:
        print("project_ok: True")
        print(json.dumps({
            "schema": "fpgai.quantized-residual-cnn-qat-result/v1",
            "status": "project_generated",
            "qat_training_summary": str(training.summary_json),
            "numeric_comparison": str(comparison_path),
            "hls_dir": str(project.hls_dir),
        }, indent=2))
        return 0

    executable = shutil.which("vitis_hls")
    if executable is None:
        raise SystemExit("vitis_hls not found in PATH; source Vitis HLS settings64.sh first")
    logs = project.hls_dir / "logs"
    logs.mkdir(exist_ok=True)
    completed = subprocess.run([executable, str(project.run_tcl)], cwd=project.hls_dir, capture_output=True, text=True)
    stdout_path = logs / "vitis_hls_stdout.log"
    stderr_path = logs / "vitis_hls_stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    csynth = project.hls_dir / "fpgai_hls_proj" / "sol1" / "syn" / "report" / "csynth.rpt"
    passed = completed.returncode == 0 and "FPGAI_QUANTIZED_RESIDUAL_CNN_PASS" in completed.stdout and csynth.exists()
    result = {
        "schema": "fpgai.quantized-residual-cnn-qat-result/v1",
        "status": "passed" if passed else "failed",
        "returncode": completed.returncode,
        "csim_passed": "FPGAI_QUANTIZED_RESIDUAL_CNN_PASS" in completed.stdout,
        "csynth_report": str(csynth) if csynth.exists() else None,
        "qat_training_summary": str(training.summary_json),
        "qat_export": str(training.qat_report_path),
        "numeric_comparison": str(comparison_path),
        "integer_exact_after_output_quantization": comparison["integer_exact_after_output_quantization"],
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "validation_level": "hls_synthesized" if passed else "qat_integer_reference_ready",
    }
    (reports / "qat_hardware_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    common_hls_result = {
        "schema": "fpgai.quantized-residual-cnn-hls-result/v1",
        "status": result["status"],
        "returncode": result["returncode"],
        "csim_passed": result["csim_passed"],
        "csynth_report": result["csynth_report"],
        "stdout_log": result["stdout_log"],
        "stderr_log": result["stderr_log"],
        "validation_level": result["validation_level"],
        "auxiliary_hls_relu": None,
        "quantization_source": "qat",
        "qat_hardware_result": str(reports / "qat_hardware_result.json"),
        "qat_training_summary": result["qat_training_summary"],
        "qat_export": result["qat_export"],
        "numeric_comparison": result["numeric_comparison"],
    }
    (reports / "quantized_hls_tool_result.json").write_text(
        json.dumps(common_hls_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("project_ok: True")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
