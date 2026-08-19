from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fpgai.analysis.mixed_backend_characterization import characterize_mixed_backend_implementation, write_mixed_backend_characterization
from fpgai.analysis.quantized_operator_backend_compare import build_quantized_add_backend_comparison, write_quantized_add_backend_comparison
from fpgai.implementations.mixed_backend import DAGMixedBackendPhysicalRequest, HLSPhysicalBinding, emit_dag_mixed_backend_physical_project, run_dag_mixed_backend_physical_project
from fpgai.ir.graph import Graph


def _pack_int8x4(values: list[int]) -> tuple[int, ...]:
    if len(values) % 4 != 0:
        raise ValueError("packed int8x4 validation requires a value count divisible by four")
    words: list[int] = []
    for base in range(0, len(values), 4):
        word = 0
        for lane, value in enumerate(values[base:base + 4]):
            word |= (int(value) & 0xFF) << (lane * 8)
        words.append(word)
    return tuple(words)


def _hls_only_graph() -> Graph:
    graph = Graph("quantized_residual_cnn_hls_only_physical")
    graph.inputs = ["input_packet"]
    graph.outputs = ["output_packet"]
    graph.add_tensor("input_packet", (1,), "uint32")
    graph.add_tensor("output_packet", (1,), "uint32")
    graph.add_op("QuantizedResidualCNN", ["input_packet"], ["output_packet"], name="hls_quantized_residual_cnn")
    return graph


def _require(path: Path, description: str) -> Path:
    if not path.is_file() and not path.is_dir():
        raise SystemExit(f"missing {description}: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare quantized residual Add implemented in HLS versus VHDL using routed Vivado whole-design measurements")
    parser.add_argument("--hls-ptq-build", default="build/quantized_residual_cnn_ptq")
    parser.add_argument("--vhdl-build", default="build/quantized_residual_cnn_vhdl_add")
    parser.add_argument("--vhdl-ptq-build", default="build/quantized_residual_cnn_ptq_add_partitioned")
    parser.add_argument("--out", default="build/quantized_residual_cnn_add_backend_comparison")
    parser.add_argument("--part", default="xck26-sfvc784-2LV-c")
    parser.add_argument("--clock-mhz", type=float, default=200.0)
    parser.add_argument("--vivado", default="vivado")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--skip-hls-vivado", action="store_true", help="Reuse an existing HLS-only physical implementation in --out/hls_baseline")
    args = parser.parse_args()

    hls_build = Path(args.hls_ptq_build).resolve()
    vhdl_build = Path(args.vhdl_build).resolve()
    vhdl_ptq_build = Path(args.vhdl_ptq_build).resolve()
    out_dir = Path(args.out).resolve()
    hls_out = out_dir / "hls_baseline"
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    hls_numeric_path = _require(hls_build / "reports" / "quantized_numeric_validation.json", "HLS numeric validation report")
    hls_tool_path = _require(hls_build / "reports" / "quantized_hls_tool_result.json", "HLS tool result")
    hls_rtl = _require(hls_build / "hls" / "fpgai_hls_proj" / "sol1" / "syn" / "verilog", "HLS RTL directory")
    hls_tool = json.loads(hls_tool_path.read_text(encoding="utf-8"))
    if hls_tool.get("status") != "passed":
        raise SystemExit("full-HLS PTQ build must pass HLS validation before routed comparison")
    hls_numeric = json.loads(hls_numeric_path.read_text(encoding="utf-8"))
    if isinstance(hls_numeric.get("partition"), dict):
        raise SystemExit("--hls-ptq-build must be the unpartitioned full-HLS residual CNN baseline")

    input_words = _pack_int8x4([int(v) for v in hls_numeric["input_integer"]])
    expected_words = _pack_int8x4([int(v) for v in hls_numeric["expected_integer"]])

    if not args.skip_hls_vivado:
        project = emit_dag_mixed_backend_physical_project(
            DAGMixedBackendPhysicalRequest(
                out_dir=hls_out,
                graph=_hls_only_graph(),
                bindings={
                    "hls_quantized_residual_cnn": HLSPhysicalBinding(
                        "hls_quantized_residual_cnn", hls_rtl, "deeplearn",
                        input_streams=("in_stream",), output_streams=("out_stream",),
                        input_packet_words=(len(input_words),), output_packet_words=(len(expected_words),),
                    )
                },
                part=args.part,
                clock_period_ns=1000.0 / float(args.clock_mhz),
                input_values=input_words,
                expected_outputs=expected_words,
                run_implementation=True,
            )
        )
        if not project.ok:
            raise SystemExit(json.dumps([issue.to_dict() for issue in project.issues], indent=2))
        hls_physical = json.loads(project.report_path.read_text(encoding="utf-8"))
        hls_physical["backend_partition"] = {
            "residual_add": "vitis_hls",
            "terminal_relu": "vitis_hls",
            "claim": "whole quantized residual CNN implemented by the HLS backend",
        }
        project.report_path.write_text(json.dumps(hls_physical, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tool = run_dag_mixed_backend_physical_project(project, vivado_executable=args.vivado, timeout=args.timeout)
        print("HLS baseline tool result:")
        print(json.dumps(tool, indent=2, sort_keys=True))
        if tool.get("status") != "passed":
            return 1
        char = characterize_mixed_backend_implementation(
            hls_out / "reports",
            target_clock_mhz=float(args.clock_mhz),
            scope="quantized_residual_cnn_all_hls",
        )
        write_mixed_backend_characterization(char, hls_out / "reports" / "mixed_backend_implementation_characterization.json")

    hls_char_path = _require(hls_out / "reports" / "mixed_backend_implementation_characterization.json", "HLS routed characterization")
    hls_phys_path = _require(hls_out / "reports" / "dag_mixed_backend_physical.json", "HLS physical report")
    hls_phys_tool = _require(hls_out / "reports" / "dag_mixed_backend_tool_result.json", "HLS physical tool result")

    vhdl_char_path = _require(vhdl_build / "reports" / "mixed_backend_implementation_characterization.json", "VHDL mixed routed characterization")
    vhdl_tool_path = _require(vhdl_build / "reports" / "dag_mixed_backend_tool_result.json", "VHDL mixed tool result")
    vhdl_phys_path = _require(vhdl_build / "reports" / "dag_mixed_backend_physical.json", "VHDL mixed physical report")
    vhdl_numeric_path = _require(vhdl_ptq_build / "reports" / "quantized_numeric_validation.json", "VHDL-partition PTQ numeric report")

    payload = build_quantized_add_backend_comparison(
        hls_characterization_path=hls_char_path,
        hls_tool_result_path=hls_phys_tool,
        hls_physical_path=hls_phys_path,
        hls_numeric_path=hls_numeric_path,
        vhdl_characterization_path=vhdl_char_path,
        vhdl_tool_result_path=vhdl_tool_path,
        vhdl_physical_path=vhdl_phys_path,
        vhdl_numeric_path=vhdl_numeric_path,
    )
    json_path, md_path = write_quantized_add_backend_comparison(payload, reports)
    print("comparison_json:", json_path)
    print("comparison_md:", md_path)
    print(json.dumps(payload["delta_vhdl_minus_hls"], indent=2, sort_keys=True))
    return 0 if payload.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
