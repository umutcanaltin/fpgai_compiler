from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fpgai.analysis.mixed_backend_characterization import (
    characterize_mixed_backend_implementation,
    write_mixed_backend_characterization,
)
from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.mixed_backend import (
    DAGMixedBackendPhysicalRequest,
    HLSPhysicalBinding,
    VHDLPhysicalBinding,
    emit_dag_mixed_backend_physical_project,
    run_dag_mixed_backend_physical_project,
)
from fpgai.ir.graph import Graph
from fpgai.quantization import emit_quantized_add_int8x4_vhdl_package, emit_quantized_relu_int8x4_vhdl_package, emit_quantized_relu_int8x4_vhdl_package


def _pack_int8x4(values: list[int]) -> tuple[int, ...]:
    if len(values) % 4 != 0:
        raise ValueError("packed int8x4 validation requires a value count divisible by four")
    words: list[int] = []
    for base in range(0, len(values), 4):
        word = 0
        for lane, value in enumerate(values[base : base + 4]):
            word |= (int(value) & 0xFF) << (lane * 8)
        words.append(word)
    return tuple(words)


def _physical_graph(
    *,
    partition_type: str = "transport",
    vhdl_op_type: str = "QuantizedPacketBridge",
    residual_relu_backend: str = "vhdl",
) -> Graph:
    graph = Graph("quantized_residual_cnn_mixed_physical")
    graph.inputs = ["input_packet"]
    graph.outputs = ["output_packet"]
    if partition_type == "residual_add_relu":
        for name in ("input_packet", "model_input_packet", "skip_packet", "main_packet", "sum_packet", "output_packet"):
            graph.add_tensor(name, (1,), "uint32")
        graph.add_op("Split", ["input_packet"], ["model_input_packet", "skip_packet"], name="vhdl_input_split")
        graph.add_op("QuantizedResidualCNNBody", ["model_input_packet"], ["main_packet"], name="hls_quantized_residual_cnn")
        graph.add_op("Add", ["main_packet", "skip_packet"], ["sum_packet"], name="vhdl_quantized_add")
        if residual_relu_backend == "vhdl":
            graph.add_op("Relu", ["sum_packet"], ["output_packet"], name="vhdl_quantized_relu")
        elif residual_relu_backend == "hls":
            graph.add_op("Relu", ["sum_packet"], ["output_packet"], name="hls_quantized_relu")
        else:
            raise ValueError(f"unsupported residual ReLU backend: {residual_relu_backend!r}")
        return graph
    graph.add_tensor("input_packet", (1,), "uint32")
    graph.add_tensor("hls_packet", (1,), "uint32")
    graph.add_tensor("output_packet", (1,), "uint32")
    graph.add_op("QuantizedResidualCNN", ["input_packet"], ["hls_packet"], name="hls_quantized_residual_cnn")
    graph.add_op(vhdl_op_type, ["hls_packet"], ["output_packet"], name="vhdl_quantized_packet_bridge")
    return graph


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the PTQ residual CNN through a physical HLS-to-VHDL packed-int8 boundary"
    )
    parser.add_argument("--ptq-build", default="build/quantized_residual_cnn_ptq")
    parser.add_argument("--out", default="build/quantized_residual_cnn_mixed_backend")
    parser.add_argument("--part", default="xck26-sfvc784-2LV-c")
    parser.add_argument("--clock-mhz", type=float, default=200.0)
    parser.add_argument("--vivado", default="vivado")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--synthesis-only", action="store_true")
    args = parser.parse_args()

    ptq_build = Path(args.ptq_build).resolve()
    numeric_path = ptq_build / "reports" / "quantized_numeric_validation.json"
    hls_tool_path = ptq_build / "reports" / "quantized_hls_tool_result.json"
    rtl_dir = ptq_build / "hls" / "fpgai_hls_proj" / "sol1" / "syn" / "verilog"

    if not numeric_path.is_file():
        raise SystemExit(f"missing quantized numeric validation report: {numeric_path}")
    if not hls_tool_path.is_file():
        raise SystemExit(f"missing quantized HLS tool result: {hls_tool_path}")
    hls_tool = json.loads(hls_tool_path.read_text(encoding="utf-8"))
    if hls_tool.get("status") != "passed":
        raise SystemExit("quantized residual CNN HLS validation must pass before mixed-backend validation")
    if not rtl_dir.is_dir():
        raise SystemExit(f"missing synthesized HLS RTL directory: {rtl_dir}")

    numeric = json.loads(numeric_path.read_text(encoding="utf-8"))
    input_values = [int(value) for value in numeric["input_integer"]]
    expected_values = [int(value) for value in numeric["expected_integer"]]
    input_words = _pack_int8x4(input_values)
    expected_words = _pack_int8x4(expected_values)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    partition = numeric.get("partition")
    residual_partition = (
        isinstance(partition, dict)
        and partition.get("partition_type") == "residual_add_relu"
        and partition.get("schema") == "fpgai.quantized-residual-operator-partition/v1"
    )
    partitioned_relu = isinstance(partition, dict) and partition.get("op_type") == "Relu" and partition.get("backend") == "vhdl"

    if residual_partition:
        residual_relu_backend = str(partition.get("relu", {}).get("backend", "vhdl"))
        graph = _physical_graph(
            partition_type="residual_add_relu",
            residual_relu_backend=residual_relu_backend,
        )
        split_contract = implementation_contract_from_manifest(
            Path("examples/packages/quantized_int8x4_split_vhdl").resolve()
        )
        generated_add_root = emit_quantized_add_int8x4_vhdl_package(
            out_dir / "generated_packages" / "quantized_add_int8x4_vhdl", partition
        )
        add_contract = implementation_contract_from_manifest(generated_add_root)
        bindings = {
            "vhdl_input_split": VHDLPhysicalBinding("vhdl_input_split", split_contract),
            "hls_quantized_residual_cnn": HLSPhysicalBinding(
                "hls_quantized_residual_cnn", rtl_dir, "deeplearn",
                input_streams=("in_stream",), output_streams=("out_stream",),
                input_packet_words=(len(input_words),), output_packet_words=(len(expected_words),),
            ),
            "vhdl_quantized_add": VHDLPhysicalBinding("vhdl_quantized_add", add_contract),
        }
        if residual_relu_backend == "vhdl":
            generated_relu_root = emit_quantized_relu_int8x4_vhdl_package(
                out_dir / "generated_packages" / "quantized_relu_int8x4_vhdl", partition
            )
            relu_contract = implementation_contract_from_manifest(generated_relu_root)
            bindings["vhdl_quantized_relu"] = VHDLPhysicalBinding("vhdl_quantized_relu", relu_contract)
        elif residual_relu_backend == "hls":
            relu_tool_path = ptq_build / "reports" / "quantized_partitioned_hls_relu_tool_result.json"
            if not relu_tool_path.is_file():
                raise SystemExit(f"missing partitioned HLS ReLU tool result: {relu_tool_path}")
            relu_tool = json.loads(relu_tool_path.read_text(encoding="utf-8"))
            if relu_tool.get("status") != "passed":
                raise SystemExit("partitioned HLS ReLU validation must pass before mixed-backend validation")
            relu_rtl_dir = Path(str(relu_tool.get("rtl_dir", "")))
            if not relu_rtl_dir.is_dir():
                raise SystemExit(f"missing partitioned HLS ReLU RTL directory: {relu_rtl_dir}")
            bindings["hls_quantized_relu"] = HLSPhysicalBinding(
                "hls_quantized_relu",
                relu_rtl_dir,
                "quantized_relu_stage",
                input_streams=("in_stream",),
                output_streams=("out_stream",),
                input_packet_words=(len(expected_words),),
                output_packet_words=(len(expected_words),),
            )
        else:
            raise SystemExit(f"unsupported residual ReLU backend in partition: {residual_relu_backend!r}")
    else:
        if partitioned_relu:
            input_q = partition.get("input_quantization", {})
            output_q = partition.get("output_quantization", {})
            if input_q.get("scale") != output_q.get("scale") or input_q.get("zero_point") != output_q.get("zero_point"):
                raise SystemExit("quantized VHDL ReLU currently requires identical input/output scale and zero-point")
            if int(input_q.get("spec", {}).get("bits", 0)) != 8 or int(output_q.get("spec", {}).get("bits", 0)) != 8:
                raise SystemExit("quantized VHDL ReLU currently requires int8 tensor contracts")
            package_path = Path("examples/packages/quantized_relu_int8x4_vhdl").resolve()
            vhdl_op_type = "Relu"
        else:
            package_path = Path("examples/packages/quantized_int8_packet_bridge_vhdl").resolve()
            vhdl_op_type = "QuantizedPacketBridge"
        graph = _physical_graph(vhdl_op_type=vhdl_op_type)
        vhdl_contract = implementation_contract_from_manifest(package_path)
        bindings = {
            "hls_quantized_residual_cnn": HLSPhysicalBinding(
                "hls_quantized_residual_cnn", rtl_dir, "deeplearn",
                input_streams=("in_stream",), output_streams=("out_stream",),
                input_packet_words=(len(input_words),), output_packet_words=(len(expected_words),),
            ),
            "vhdl_quantized_packet_bridge": VHDLPhysicalBinding(
                "vhdl_quantized_packet_bridge", vhdl_contract,
            ),
        }

    project = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=out_dir,
            graph=graph,
            bindings=bindings,
            part=args.part,
            clock_period_ns=1000.0 / float(args.clock_mhz),
            input_values=input_words,
            expected_outputs=expected_words,
            fanout_buffer_depths=(
                {"skip_packet": len(input_words)} if residual_partition else None
            ),
            run_implementation=not args.synthesis_only,
        )
    )
    print("project_ok:", project.ok)
    if not project.ok:
        print("issues:", [issue.to_dict() for issue in project.issues])
        return 2

    physical_report = Path(project.report_path)
    payload = json.loads(physical_report.read_text(encoding="utf-8"))
    payload["quantized_model"] = {
        "source_build": str(ptq_build),
        "transport": {
            "axis_word_bits": 32,
            "activation_bits": 8,
            "values_per_word": 4,
            "lane_order": "least_significant_byte_first",
        },
        "numeric_reference": {
            "input_integer_count": len(input_values),
            "output_integer_count": len(expected_values),
            "input_words": list(input_words),
            "expected_words": list(expected_words),
        },
        "backend_partition": (
            {
                "model_compute": "vitis_hls",
                "explicit_input_split": "vhdl",
                "residual_add": "vhdl",
                "terminal_relu": residual_relu_backend,
                "vhdl_semantics": (
                    "quantized_residual_add_plus_relu_int8x4"
                    if residual_relu_backend == "vhdl"
                    else "quantized_residual_add_int8x4_with_hls_terminal_relu"
                ),
                "claim": (
                    "quantized neural-network residual partition: residual Add and terminal Relu implemented in VHDL"
                    if residual_relu_backend == "vhdl"
                    else "quantized neural-network residual partition: residual Add implemented in VHDL and terminal Relu implemented in HLS"
                ),
                "partition": partition,
            }
            if residual_partition
            else {
                "model_compute": "vitis_hls",
                "post_model_transport_bridge": "vhdl",
                "vhdl_semantics": (
                    "quantized_relu_int8x4"
                    if partitioned_relu
                    else "elastic_identity_on_packed_int8_transport"
                ),
                "claim": (
                    "quantized neural-network operator partition: terminal Relu implemented in VHDL"
                    if partitioned_relu
                    else "physical mixed-backend boundary validation; the VHDL stage is not yet a partitioned neural-network operator"
                ),
            }
        ),
    }
    physical_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tool = run_dag_mixed_backend_physical_project(
        project,
        vivado_executable=args.vivado,
        timeout=args.timeout,
    )
    print(json.dumps(tool, indent=2, sort_keys=True))

    if tool.get("implementation_reports_present"):
        characterization = characterize_mixed_backend_implementation(
            out_dir / "reports",
            target_clock_mhz=float(args.clock_mhz),
        )
        path = write_mixed_backend_characterization(
            characterization,
            out_dir / "reports" / "mixed_backend_implementation_characterization.json",
        )
        print("characterization:", path)
    return 0 if tool.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
