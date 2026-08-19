from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

# Support direct execution via ``python scripts/run_quantized_residual_cnn_ptq.py``.
# In that mode Python places ``scripts/`` rather than the repository root on
# sys.path, so both the ``scripts`` namespace and source-tree ``fpgai`` package
# need the project root made explicit.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.make_quantized_residual_cnn_example import write_model
from fpgai.quantization import (
    apply_model_ptq_to_hls_graph,
    calibrate_model_ptq,
    execute_quantized_hls_reference,
    partition_terminal_relu,
    partition_residual_add_and_terminal_relu,
)
from fpgai.quantization.hardware import quantization_parameters_from_tensor, derive_requantization_contract
from fpgai.quantization.contracts import quantization_spec_from_mapping
from fpgai.quantization.ptq import quantize
from fpgai.quantization.reports import write_model_ptq_report
from fpgai.validation.mixed_external_hls import execute_mixed_graph_trace
from fpgai.backends.hls.codegen import emit_hls_stub
from fpgai.backends.hls.emit.types_h import emit_types_h
from fpgai.backends.hls.emit.params_h import emit_params_h
from fpgai.backends.hls.emit.params_cpp import emit_params_cpp


def _load_config(path: Path) -> dict:
    import yaml
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("PTQ residual CNN config must be a mapping")
    return raw


def _get(raw: dict, path: str, default=None):
    current = raw
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index < 0 or index >= len(current):
                return default
            current = current[index]
        else:
            return default
    return current


def _emit_testbench(path: Path, input_q: np.ndarray, expected_q: np.ndarray) -> None:
    input_values = ", ".join(str(int(v)) for v in np.asarray(input_q).reshape(-1))
    expected_values = ", ".join(str(int(v)) for v in np.asarray(expected_q).reshape(-1))
    n_in = int(np.asarray(input_q).size)
    n_out = int(np.asarray(expected_q).size)
    text = f'''#include <cstdio>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

typedef ap_axis<32,0,0,0> axis_t;
extern "C" void deeplearn(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream);

int main() {{
    const int INPUT_N = {n_in};
    const int OUTPUT_N = {n_out};
    const int input_values[INPUT_N] = {{ {input_values} }};
    const int expected_values[OUTPUT_N] = {{ {expected_values} }};
    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;
    for (int base = 0; base < INPUT_N; base += 4) {{
        axis_t packet; packet.data = 0; packet.keep = -1; packet.strb = -1; packet.last = 0;
        for (int lane = 0; lane < 4; ++lane) {{
            int index = base + lane;
            if (index < INPUT_N) {{
                ap_int<8> value = input_values[index];
                packet.data.range((lane + 1) * 8 - 1, lane * 8) = value.range(7, 0);
            }}
        }}
        packet.last = (base + 4 >= INPUT_N) ? 1 : 0;
        in_stream.write(packet);
    }}
    deeplearn(in_stream, out_stream);
    int index = 0;
    while (!out_stream.empty() && index < OUTPUT_N) {{
        axis_t packet = out_stream.read();
        for (int lane = 0; lane < 4 && index < OUTPUT_N; ++lane) {{
            ap_int<8> value; value.range(7, 0) = packet.data.range((lane + 1) * 8 - 1, lane * 8);
            const int actual = (int)value;
            if (actual != expected_values[index]) {{
                std::printf("FPGAI_QUANTIZED_RESIDUAL_CNN_FAIL index=%d expected=%d actual=%d\\n", index, expected_values[index], actual);
                return 2;
            }}
            ++index;
        }}
    }}
    if (index != OUTPUT_N) {{
        std::printf("FPGAI_QUANTIZED_RESIDUAL_CNN_FAIL output_count=%d expected_count=%d\\n", index, OUTPUT_N);
        return 3;
    }}
    std::printf("FPGAI_QUANTIZED_RESIDUAL_CNN_PASS outputs=%d\\n", OUTPUT_N);
    return 0;
}}
'''
    path.write_text(text, encoding="utf-8")


def _partitioned_relu_testbench_text(input_q: np.ndarray, expected_q: np.ndarray) -> str:
    input_values = ", ".join(str(int(v)) for v in np.asarray(input_q).reshape(-1))
    expected_values = ", ".join(str(int(v)) for v in np.asarray(expected_q).reshape(-1))
    n_in = int(np.asarray(input_q).size)
    n_out = int(np.asarray(expected_q).size)
    return f'''#include <cstdio>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

typedef ap_axis<32,0,0,0> axis_t;
extern "C" void quantized_relu_stage(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream);

int main() {{
    const int INPUT_N = {n_in};
    const int OUTPUT_N = {n_out};
    const int input_values[INPUT_N] = {{ {input_values} }};
    const int expected_values[OUTPUT_N] = {{ {expected_values} }};
    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;
    for (int base = 0; base < INPUT_N; base += 4) {{
        axis_t packet; packet.data = 0; packet.keep = -1; packet.strb = -1;
        packet.last = (base + 4 >= INPUT_N) ? 1 : 0;
        for (int lane = 0; lane < 4; ++lane) {{
            int index = base + lane;
            if (index < INPUT_N) {{
                ap_int<8> value = input_values[index];
                packet.data.range((lane + 1) * 8 - 1, lane * 8) = value.range(7, 0);
            }}
        }}
        in_stream.write(packet);
    }}
    quantized_relu_stage(in_stream, out_stream);
    int index = 0;
    while (!out_stream.empty() && index < OUTPUT_N) {{
        axis_t packet = out_stream.read();
        for (int lane = 0; lane < 4 && index < OUTPUT_N; ++lane) {{
            ap_int<8> value; value.range(7, 0) = packet.data.range((lane + 1) * 8 - 1, lane * 8);
            const int actual = (int)value;
            if (actual != expected_values[index]) {{
                std::printf("FPGAI_QUANTIZED_RELU_STAGE_FAIL index=%d expected=%d actual=%d\\n", index, expected_values[index], actual);
                return 2;
            }}
            ++index;
        }}
    }}
    if (index != OUTPUT_N) return 3;
    std::printf("FPGAI_QUANTIZED_RELU_STAGE_PASS outputs=%d\\n", OUTPUT_N);
    return 0;
}}
'''


def _emit_partitioned_hls_relu_project(
    *,
    out_dir: Path,
    partition: dict,
    raw_cfg: dict,
    input_q: np.ndarray,
    expected_q: np.ndarray,
    part: str,
    clock_mhz: int,
):
    """Emit the HLS terminal-ReLU control used by the VHDL-Add/HLS-ReLU matrix cell."""
    from fpgai.ir.graph import Graph

    relu = partition.get("relu") if isinstance(partition, dict) else None
    if not isinstance(relu, dict) or relu.get("backend") != "hls":
        raise ValueError("partition does not describe an HLS terminal ReLU")
    input_quant = relu.get("input_quantization")
    output_quant = relu.get("output_quantization")
    if not isinstance(input_quant, dict) or not isinstance(output_quant, dict):
        raise ValueError("HLS ReLU partition is missing quantization contracts")

    graph = Graph("quantized_terminal_relu_hls")
    graph.inputs = ["relu_input"]
    graph.outputs = ["relu_output"]
    shape = tuple(int(v) for v in np.asarray(input_q).shape)
    graph.add_tensor("relu_input", shape, "int8", quantization=input_quant)
    graph.add_tensor("relu_output", shape, "int8", quantization=output_quant)
    source_q = quantization_parameters_from_tensor(graph.get_tensor("relu_input"))
    destination_q = quantization_parameters_from_tensor(graph.get_tensor("relu_output"))
    contract = derive_requantization_contract(source_q, destination_q)
    rounding_code = {"nearest": 0, "floor": 1, "ceil": 2}.get(destination_q.spec.rounding)
    saturation_code = {"saturate": 0, "wrap": 1}.get(destination_q.spec.saturation)
    if rounding_code is None or saturation_code is None:
        raise ValueError("unsupported HLS ReLU rounding/saturation contract")

    op = graph.add_op("Relu", ["relu_input"], ["relu_output"], name="partitioned_terminal_relu")
    op.attrs["precision"] = {
        "activation": {"type": "ap_int", "bits": int(destination_q.spec.bits)},
        "weight": {"type": "ap_int", "bits": int(destination_q.spec.bits)},
        "bias": {"type": "ap_int", "bits": 32},
        "accum": {"type": "ap_int", "bits": 32},
    }
    op.attrs["quantized_relu"] = {
        "input_zero": int(source_q.zero_point),
        "multiplier": int(contract.multiplier),
        "shift": int(contract.shift),
        "output_zero": int(destination_q.zero_point),
        "qmin": int(destination_q.spec.qmin),
        "qmax": int(destination_q.spec.qmax),
        "rounding_mode": int(rounding_code),
        "saturation_mode": int(saturation_code),
    }

    project = emit_hls_stub(
        graph=graph,
        out_dir=out_dir,
        top_name="quantized_relu_stage",
        hls_options={
            "weights_mode": "embedded",
            "part": part,
            "clk_mhz": clock_mhz,
            "pipeline_mode": "inference",
            "raw_cfg": raw_cfg,
            "run_csim": True,
            "run_csynth": True,
            "export_ip": False,
        },
    )
    project.tb_cpp.write_text(_partitioned_relu_testbench_text(input_q, expected_q), encoding="utf-8")
    return project


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/examples/quantized_residual_cnn_ptq.yml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--part", default=None)
    parser.add_argument("--clock-mhz", type=int, default=None)
    parser.add_argument("--skip-hls", action="store_true")
    args = parser.parse_args()

    # Keep optional frontend dependencies lazy so ``--help`` and configuration
    # inspection do not require ONNX to be installed.
    from fpgai.frontend.onnx import import_onnx

    raw = _load_config(Path(args.config))
    out_dir = Path(args.out or _get(raw, "project.out_dir", "build/quantized_residual_cnn_ptq")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = write_model(out_dir / "quantized_residual_cnn.onnx")
    graph = import_onnx(str(model_path))

    rng = np.random.default_rng(17)
    calibration_count = int(_get(raw, "numerics.quantization.calibration.samples", 16))
    calibration_samples = [rng.uniform(-2.0, 2.0, size=(1, 1, 4, 4)).astype(np.float32) for _ in range(calibration_count)]
    activation_spec = quantization_spec_from_mapping(_get(raw, "numerics.quantization.activations", {}), path="numerics.quantization.activations")
    weight_spec = quantization_spec_from_mapping(_get(raw, "numerics.quantization.weights", {}), path="numerics.quantization.weights")
    ptq = calibrate_model_ptq(
        graph,
        calibration_samples,
        trace_fn=lambda g, x: execute_mixed_graph_trace(g, None, x),
        activation_spec=activation_spec,
        weight_spec=weight_spec,
        method=str(_get(raw, "numerics.quantization.calibration.method", "min_max")),
        percentile=float(_get(raw, "numerics.quantization.calibration.percentile", 99.9)),
    )
    reports = out_dir / "reports"
    reports.mkdir(exist_ok=True)
    write_model_ptq_report(ptq, reports / "ptq_calibration.json")
    lowering = apply_model_ptq_to_hls_graph(graph, ptq)
    (reports / "quantized_hls_lowering.json").write_text(json.dumps(lowering.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    validation_input = np.linspace(-1.5, 1.5, 16, dtype=np.float32).reshape(1, 1, 4, 4)
    input_qparams = quantization_parameters_from_tensor(graph.get_tensor(graph.inputs[0]))
    full_output_name = str(graph.outputs[0])
    output_qparams = quantization_parameters_from_tensor(graph.get_tensor(full_output_name))
    input_q = quantize(validation_input, input_qparams)
    full_expected_q = execute_quantized_hls_reference(graph, input_q)

    final_relu_backend = str(_get(raw, "targets.mixed_backend.final_relu_backend", "hls")).strip().lower()
    residual_add_backend = str(_get(raw, "targets.mixed_backend.residual_add_backend", "hls")).strip().lower()
    partition = None
    sum_expected_q = None
    if residual_add_backend == "vhdl" and final_relu_backend in {"vhdl", "hls"}:
        relu_only_graph = copy.deepcopy(graph)
        partition_terminal_relu(relu_only_graph, backend=final_relu_backend)
        sum_expected_q = execute_quantized_hls_reference(relu_only_graph, input_q)
        partition = partition_residual_add_and_terminal_relu(
            graph, add_backend="vhdl", relu_backend=final_relu_backend
        )
        hls_expected_q = execute_quantized_hls_reference(graph, input_q)
        (reports / "quantized_operator_partition.json").write_text(
            json.dumps(partition.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif residual_add_backend == "hls" and final_relu_backend == "vhdl":
        partition = partition_terminal_relu(graph, backend="vhdl")
        hls_expected_q = execute_quantized_hls_reference(graph, input_q)
        (reports / "quantized_operator_partition.json").write_text(
            json.dumps(partition.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif residual_add_backend == "hls" and final_relu_backend == "hls":
        hls_expected_q = full_expected_q
    else:
        raise ValueError("mixed-backend residual_add_backend/final_relu_backend combination is unsupported")

    numeric = {
        "schema": "fpgai.quantized-residual-cnn-numeric/v1",
        "input_float": validation_input.reshape(-1).tolist(),
        "input_integer": input_q.reshape(-1).astype(int).tolist(),
        "expected_integer": full_expected_q.reshape(-1).astype(int).tolist(),
        "hls_expected_integer": hls_expected_q.reshape(-1).astype(int).tolist(),
        "sum_expected_integer": (sum_expected_q.reshape(-1).astype(int).tolist() if sum_expected_q is not None else None),
        "input_quantization": input_qparams.to_dict(),
        "output_quantization": output_qparams.to_dict(),
        "partition": partition.to_dict() if partition is not None else None,
        "status": "reference_ready",
    }
    (reports / "quantized_numeric_validation.json").write_text(json.dumps(numeric, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    project = emit_hls_stub(
        graph=graph,
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
    (project.hls_dir / "include" / "fpgai_types.h").write_text(emit_types_h(graph, top_name="deeplearn", raw_cfg=raw), encoding="utf-8")
    (project.hls_dir / "include" / "fpgai_params.h").write_text(emit_params_h(graph, weights_mode="embedded"), encoding="utf-8")
    (project.hls_dir / "src" / "fpgai_params.cpp").write_text(emit_params_cpp(graph, weights_mode="embedded", storage_impl="bram"), encoding="utf-8")
    _emit_testbench(project.hls_dir / "src" / "tb.cpp", input_q, hls_expected_q)

    relu_project = None
    if residual_add_backend == "vhdl" and final_relu_backend == "hls":
        relu_project = _emit_partitioned_hls_relu_project(
            out_dir=out_dir / "hls_relu",
            partition=partition.to_dict(),
            raw_cfg=raw,
            input_q=sum_expected_q,
            expected_q=full_expected_q,
            part=str(args.part or _get(raw, "targets.platform.part", "xck26-sfvc784-2LV-c")),
            clock_mhz=int(args.clock_mhz or _get(raw, "targets.platform.clocks.0.target_mhz", 200)),
        )

    if args.skip_hls:
        print(f"project_ok: True\nproject: {project.hls_dir}")
        return 0

    executable = shutil.which("vitis_hls")
    if executable is None:
        raise SystemExit("vitis_hls not found in PATH; source Vitis HLS settings64.sh first")
    logs = project.hls_dir / "logs"
    logs.mkdir(exist_ok=True)
    completed = subprocess.run([executable, str(project.run_tcl)], cwd=project.hls_dir, capture_output=True, text=True)
    (logs / "vitis_hls_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (logs / "vitis_hls_stderr.log").write_text(completed.stderr, encoding="utf-8")
    csynth = project.hls_dir / "fpgai_hls_proj" / "sol1" / "syn" / "report" / "csynth.rpt"
    passed = completed.returncode == 0 and "FPGAI_QUANTIZED_RESIDUAL_CNN_PASS" in completed.stdout and csynth.exists()
    auxiliary_relu_result = None
    if passed and relu_project is not None:
        relu_logs = relu_project.hls_dir / "logs"
        relu_logs.mkdir(exist_ok=True)
        relu_completed = subprocess.run(
            [executable, str(relu_project.run_tcl)],
            cwd=relu_project.hls_dir,
            capture_output=True,
            text=True,
        )
        (relu_logs / "vitis_hls_stdout.log").write_text(relu_completed.stdout, encoding="utf-8")
        (relu_logs / "vitis_hls_stderr.log").write_text(relu_completed.stderr, encoding="utf-8")
        relu_csynth = relu_project.hls_dir / "fpgai_hls_proj" / "sol1" / "syn" / "report" / "csynth.rpt"
        relu_passed = (
            relu_completed.returncode == 0
            and "FPGAI_QUANTIZED_RELU_STAGE_PASS" in relu_completed.stdout
            and relu_csynth.exists()
        )
        auxiliary_relu_result = {
            "status": "passed" if relu_passed else "failed",
            "returncode": relu_completed.returncode,
            "csim_passed": "FPGAI_QUANTIZED_RELU_STAGE_PASS" in relu_completed.stdout,
            "csynth_report": str(relu_csynth) if relu_csynth.exists() else None,
            "rtl_dir": str(relu_project.hls_dir / "fpgai_hls_proj" / "sol1" / "syn" / "verilog"),
        }
        passed = passed and relu_passed
        (reports / "quantized_partitioned_hls_relu_tool_result.json").write_text(
            json.dumps(auxiliary_relu_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    tool_result = {
        "schema": "fpgai.quantized-residual-cnn-hls-result/v1",
        "status": "passed" if passed else "failed",
        "returncode": completed.returncode,
        "csim_passed": "FPGAI_QUANTIZED_RESIDUAL_CNN_PASS" in completed.stdout,
        "csynth_report": str(csynth) if csynth.exists() else None,
        "stdout_log": str(logs / "vitis_hls_stdout.log"),
        "stderr_log": str(logs / "vitis_hls_stderr.log"),
        "validation_level": "hls_synthesized" if passed else "quantized_hls_project_generated",
        "auxiliary_hls_relu": auxiliary_relu_result,
    }
    (reports / "quantized_hls_tool_result.json").write_text(json.dumps(tool_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("project_ok: True")
    print(json.dumps(tool_result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
