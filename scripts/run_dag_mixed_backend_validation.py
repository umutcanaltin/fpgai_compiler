from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.mixed_backend import (
    DAGMixedBackendPhysicalRequest,
    HLSPhysicalBinding,
    RequantizationPhysicalBinding,
    VHDLPhysicalBinding,
    emit_dag_mixed_backend_physical_project,
    run_dag_mixed_backend_physical_project,
)
from fpgai.ir.graph import Graph
from fpgai.quantization import QuantizationParameters, QuantizationSpec


def _run_hls_stage(
    root: Path,
    *,
    name: str,
    source: Path,
    part: str,
    period_ns: float,
    vitis_hls: str,
) -> Path:
    hls_dir = root / f"hls_{name}"
    source_dir = hls_dir / "src"
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, source_dir / source.name)

    run_tcl = hls_dir / "run_hls.tcl"
    run_tcl.write_text(
        f'''open_project -reset fpgai_{name}
set_top {name}
add_files ./src/{source.name}
open_solution -reset sol1
set_part {part}
create_clock -period {period_ns}
csynth_design
exit
''',
        encoding="utf-8",
    )

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [vitis_hls, "-f", str(run_tcl)],
        cwd=hls_dir,
        text=True,
        capture_output=True,
    )
    (reports / f"{name}_vitis_hls_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (reports / f"{name}_vitis_hls_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"Vitis HLS stage {name} failed with return code {proc.returncode}; inspect {reports}")

    rtl_dir = hls_dir / f"fpgai_{name}" / "sol1" / "syn" / "verilog"
    if not rtl_dir.is_dir():
        raise SystemExit(f"HLS RTL directory not found for {name}: {rtl_dir}")
    return rtl_dir


def _vhdl_fork_join_graph() -> Graph:
    graph = Graph("mixed_backend_fork_join")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    for name in ("input", "pre", "left", "right", "left_done", "right_done", "output"):
        graph.add_tensor(name, (1,), "int16")

    graph.add_op("Scale2", ["input"], ["pre"], name="hls_pre")
    graph.add_op("Split", ["pre"], ["left", "right"], name="vhdl_split")
    graph.add_op("Add1", ["left"], ["left_done"], name="hls_left")
    graph.add_op("Scale2", ["right"], ["right_done"], name="hls_right")
    graph.add_op("Add", ["left_done", "right_done"], ["output"], name="vhdl_merge")
    return graph


def _multi_port_hls_graph() -> Graph:
    graph = Graph("multi_port_hls_vhdl_dag")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    for name in (
        "input", "left", "right", "left_done", "right_done", "merged",
        "vhdl_left", "vhdl_right", "output",
    ):
        graph.add_tensor(name, (1,), "int16")
    graph.add_op("Split2", ["input"], ["left", "right"], name="hls_split")
    graph.add_op("Add1", ["left"], ["left_done"], name="hls_left")
    graph.add_op("Scale2", ["right"], ["right_done"], name="hls_right")
    graph.add_op("Add2", ["left_done", "right_done"], ["merged"], name="hls_merge")
    graph.add_op("Split", ["merged"], ["vhdl_left", "vhdl_right"], name="vhdl_split")
    graph.add_op("Add", ["vhdl_left", "vhdl_right"], ["output"], name="vhdl_merge")
    return graph


def _quantized_bridge_graph() -> Graph:
    graph = Graph("quantized_requantization_bridge")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    graph.add_tensor("input", (1,), "int8")
    graph.add_tensor("wide_in", (1,), "int16")
    graph.add_tensor("wide_out", (1,), "int16")
    graph.add_tensor("output", (1,), "int8")

    int8_spec = QuantizationSpec(bits=8, scheme="symmetric", granularity="per_tensor")
    int16_spec = QuantizationSpec(bits=16, scheme="symmetric", granularity="per_tensor")
    q_input = QuantizationParameters(int8_spec, 0.5, 0, -64.0, 63.5)
    q_wide = QuantizationParameters(int16_spec, 0.25, 0, -8192.0, 8191.75)
    graph.set_tensor_quantization("input", q_input.to_dict())
    graph.set_tensor_quantization("wide_in", q_wide.to_dict())
    graph.set_tensor_quantization("wide_out", q_wide.to_dict())
    graph.set_tensor_quantization("output", q_input.to_dict())

    graph.add_op("Requantize", ["input"], ["wide_in"], name="requantize_up")
    graph.add_op("Add1", ["wide_in"], ["wide_out"], name="hls_add")
    graph.add_op("Requantize", ["wide_out"], ["output"], name="requantize_down")
    return graph


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate DAG mixed HLS/VHDL composition with grouped ready/valid multi-port VHDL boundaries"
    )
    parser.add_argument("--out", default="build/dag_mixed_backend_validation")
    parser.add_argument(
        "--profile",
        choices=("vhdl_fork_join", "multi_port_hls", "quantized_bridge"),
        default="vhdl_fork_join",
        help="Physical DAG validation profile to run",
    )
    parser.add_argument("--part", default="xck26-sfvc784-2LV-c")
    parser.add_argument("--clock-period-ns", type=float, default=5.0)
    parser.add_argument("--vitis-hls", default="vitis_hls")
    parser.add_argument("--vivado", default="vivado")
    args = parser.parse_args()

    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    scale2_rtl = _run_hls_stage(
        root,
        name="scale2_axis",
        source=Path("examples/mixed_backend/ready_valid/scale2_axis.cpp").resolve(),
        part=args.part,
        period_ns=args.clock_period_ns,
        vitis_hls=args.vitis_hls,
    )
    add1_rtl = _run_hls_stage(
        root,
        name="add1_axis",
        source=Path("examples/mixed_backend/ready_valid/add1_axis.cpp").resolve(),
        part=args.part,
        period_ns=args.clock_period_ns,
        vitis_hls=args.vitis_hls,
    )

    split_contract = implementation_contract_from_manifest(
        Path("examples/packages/split_grouped_ready_valid_vhdl")
    )
    add_contract = implementation_contract_from_manifest(
        Path("examples/packages/add_grouped_ready_valid_vhdl")
    )

    if args.profile == "quantized_bridge":
        graph = _quantized_bridge_graph()
        bindings = {
            "requantize_up": RequantizationPhysicalBinding("requantize_up"),
            "hls_add": HLSPhysicalBinding("hls_add", add1_rtl, "add1_axis"),
            "requantize_down": RequantizationPhysicalBinding("requantize_down"),
        }
        expected_output = 8
    elif args.profile == "multi_port_hls":
        split2_rtl = _run_hls_stage(
            root,
            name="split2_axis",
            source=Path("examples/mixed_backend/multi_port/split2_axis.cpp").resolve(),
            part=args.part,
            period_ns=args.clock_period_ns,
            vitis_hls=args.vitis_hls,
        )
        add2_rtl = _run_hls_stage(
            root,
            name="add2_axis",
            source=Path("examples/mixed_backend/multi_port/add2_axis.cpp").resolve(),
            part=args.part,
            period_ns=args.clock_period_ns,
            vitis_hls=args.vitis_hls,
        )
        graph = _multi_port_hls_graph()
        bindings = {
            "hls_split": HLSPhysicalBinding(
                "hls_split", split2_rtl, "split2_axis",
                input_streams=("input",), output_streams=("left", "right"),
            ),
            "hls_left": HLSPhysicalBinding("hls_left", add1_rtl, "add1_axis"),
            "hls_right": HLSPhysicalBinding("hls_right", scale2_rtl, "scale2_axis"),
            "hls_merge": HLSPhysicalBinding(
                "hls_merge", add2_rtl, "add2_axis",
                input_streams=("left_done", "right_done"), output_streams=("output",),
            ),
            "vhdl_split": VHDLPhysicalBinding("vhdl_split", split_contract),
            "vhdl_merge": VHDLPhysicalBinding("vhdl_merge", add_contract),
        }
        expected_output = 44
    else:
        graph = _vhdl_fork_join_graph()
        bindings = {
            "hls_pre": HLSPhysicalBinding("hls_pre", scale2_rtl, "scale2_axis"),
            "vhdl_split": VHDLPhysicalBinding("vhdl_split", split_contract),
            "hls_left": HLSPhysicalBinding("hls_left", add1_rtl, "add1_axis"),
            "hls_right": HLSPhysicalBinding("hls_right", scale2_rtl, "scale2_axis"),
            "vhdl_merge": VHDLPhysicalBinding("vhdl_merge", add_contract),
        }
        expected_output = 43

    project = emit_dag_mixed_backend_physical_project(
        DAGMixedBackendPhysicalRequest(
            out_dir=root,
            graph=graph,
            bindings=bindings,
            part=args.part,
            clock_period_ns=args.clock_period_ns,
            input_value=7,
            expected_output=expected_output,
        )
    )
    print("project_ok:", project.ok)
    if not project.ok:
        print("issues:", [issue.to_dict() for issue in project.issues])
        return 2

    print("physical_report:", project.report_path)
    result = run_dag_mixed_backend_physical_project(project, vivado_executable=args.vivado)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
