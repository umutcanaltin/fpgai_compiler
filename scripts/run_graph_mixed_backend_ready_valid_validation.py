from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from fpgai.ir.graph import Graph
from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.mixed_backend import (
    GraphMixedBackendPhysicalRequest,
    HLSPhysicalBinding,
    VHDLPhysicalBinding,
    emit_graph_mixed_backend_physical_project,
    run_graph_mixed_backend_physical_project,
)


def _run_hls_stage(root: Path, *, name: str, source: Path, part: str, period_ns: float, vitis_hls: str) -> Path:
    hls = root / f"hls_{name}"
    src = hls / "src"
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, src / source.name)
    tcl = hls / "run_hls.tcl"
    tcl.write_text(
        f'''open_project -reset fpgai_{name}\nset_top {name}\nadd_files ./src/{source.name}\nopen_solution -reset sol1\nset_part {part}\ncreate_clock -period {period_ns}\ncsynth_design\nexit\n''',
        encoding="utf-8",
    )
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([vitis_hls, "-f", str(tcl)], cwd=hls, text=True, capture_output=True)
    (reports / f"{name}_vitis_hls_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (reports / f"{name}_vitis_hls_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"Vitis HLS stage {name} failed with return code {proc.returncode}; inspect {reports}")
    rtl = hls / f"fpgai_{name}" / "sol1" / "syn" / "verilog"
    if not rtl.is_dir():
        raise SystemExit(f"HLS RTL directory not found for {name}: {rtl}")
    return rtl


def _graph() -> Graph:
    graph = Graph("mixed_hls_vhdl_hls_ready_valid")
    graph.inputs = ["input"]
    graph.outputs = ["output"]
    for name in ("input", "scaled", "vhdl_out", "output"):
        graph.add_tensor(name, (1,), "int16")
    graph.add_op("Scale2", ["input"], ["scaled"], name="hls_pre")
    graph.add_op("IdentityReadyValidVHDL", ["scaled"], ["vhdl_out"], name="vhdl_mid")
    graph.add_op("Add1", ["vhdl_out"], ["output"], name="hls_post")
    return graph


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate graph-driven HLS <-> VHDL ready/valid backpressure composition")
    ap.add_argument("--out", default="build/graph_mixed_backend_ready_valid")
    ap.add_argument("--part", default="xck26-sfvc784-2LV-c")
    ap.add_argument("--clock-period-ns", type=float, default=5.0)
    ap.add_argument("--vitis-hls", default="vitis_hls")
    ap.add_argument("--vivado", default="vivado")
    args = ap.parse_args()

    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    pre_rtl = _run_hls_stage(
        root,
        name="scale2_axis",
        source=Path("examples/mixed_backend/ready_valid/scale2_axis.cpp").resolve(),
        part=args.part,
        period_ns=args.clock_period_ns,
        vitis_hls=args.vitis_hls,
    )
    post_rtl = _run_hls_stage(
        root,
        name="add1_axis",
        source=Path("examples/mixed_backend/ready_valid/add1_axis.cpp").resolve(),
        part=args.part,
        period_ns=args.clock_period_ns,
        vitis_hls=args.vitis_hls,
    )
    vhdl = implementation_contract_from_manifest(Path("examples/packages/identity_ready_valid_vhdl"))
    project = emit_graph_mixed_backend_physical_project(
        GraphMixedBackendPhysicalRequest(
            out_dir=root,
            graph=_graph(),
            bindings={
                "hls_pre": HLSPhysicalBinding("hls_pre", pre_rtl, "scale2_axis"),
                "vhdl_mid": VHDLPhysicalBinding("vhdl_mid", vhdl),
                "hls_post": HLSPhysicalBinding("hls_post", post_rtl, "add1_axis"),
            },
            part=args.part,
            clock_period_ns=args.clock_period_ns,
            input_value=7,
            expected_output=15,
            physical_profile="linear_scalar_ready_valid_v1",
        )
    )
    print("project_ok:", project.ok)
    if not project.ok:
        print("issues:", [issue.to_dict() for issue in project.issues])
        return 2
    print("physical_report:", project.report_path)
    result = run_graph_mixed_backend_physical_project(project, vivado_executable=args.vivado)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
