"""Generic HLS project emission and Vitis HLS execution.

This module owns backend/toolchain concerns only. Model-family graph construction and
reference execution live outside the backend so attention, Transformer, YOLO, LLM,
and external models all use the same HLS project contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import shutil
import subprocess

import numpy as np

from fpgai.backends.hls.codegen import HLSProject, emit_hls_stub
from fpgai.ir import Graph


@dataclass(frozen=True)
class HLSBuild:
    project: HLSProject
    input_values: dict[str, np.ndarray]
    expected_output: np.ndarray
    reference_report: Path
    artifact_namespace: str = "hls"
    result_schema: str = "fpgai.hls-result/v1"
    reference_schema: str = "fpgai.hls-reference/v1"
    pass_token: str = "FPGAI_HLS_PASS"
    tolerance: float = 0.05


def _normalise_namespace(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip())
    return text or "hls"


def _emit_numeric_testbench(
    path: Path,
    graph: Graph,
    inputs: Mapping[str, np.ndarray],
    expected: np.ndarray,
    *,
    tolerance: float,
    pass_token: str,
) -> None:
    flattened_inputs: list[float] = []
    segment_comments: list[str] = []
    offset = 0
    for name in graph.inputs:
        flat = np.asarray(inputs[str(name)], dtype=np.float32).reshape(-1)
        segment_comments.append(f"// {name}: offset={offset} values={flat.size}")
        flattened_inputs.extend(float(x) for x in flat)
        offset += int(flat.size)

    input_values = ", ".join(f"{v:.9g}f" for v in flattened_inputs)
    expected_flat = np.asarray(expected, dtype=np.float32).reshape(-1)
    expected_values = ", ".join(f"{float(v):.9g}f" for v in expected_flat)
    fail_token = pass_token.replace("PASS", "FAIL") if "PASS" in pass_token else f"{pass_token}_FAIL"

    text = f'''#include <cstdio>\n#include <cmath>\n#include <hls_stream.h>\n#include <ap_axi_sdata.h>\n#include <ap_fixed.h>\n\ntypedef ap_axis<32,0,0,0> axis_t;\ntypedef ap_fixed<16,6> act_t;\nextern "C" void deeplearn(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream);\n\nint main() {{\n    const int INPUT_N = {len(flattened_inputs)};\n    const int OUTPUT_N = {int(expected_flat.size)};\n    const float input_values[INPUT_N] = {{ {input_values} }};\n    const float expected_values[OUTPUT_N] = {{ {expected_values} }};\n    {chr(10).join(segment_comments)}\n    hls::stream<axis_t> in_stream; hls::stream<axis_t> out_stream;\n    for (int base = 0; base < INPUT_N; base += 2) {{\n        axis_t packet; packet.data = 0; packet.keep = -1; packet.strb = -1; packet.last = 0;\n        for (int lane = 0; lane < 2; ++lane) {{\n            const int index = base + lane;\n            if (index < INPUT_N) {{ act_t value = (act_t)input_values[index]; packet.data.range((lane+1)*16-1, lane*16) = value.range(15,0); }}\n        }}\n        packet.last = (base + 2 >= INPUT_N) ? 1 : 0; in_stream.write(packet);\n    }}\n    deeplearn(in_stream, out_stream);\n    int index = 0; float max_abs_error = 0.0f;\n    while (!out_stream.empty() && index < OUTPUT_N) {{\n        axis_t packet = out_stream.read();\n        for (int lane = 0; lane < 2 && index < OUTPUT_N; ++lane) {{\n            act_t value; value.range(15,0) = packet.data.range((lane+1)*16-1, lane*16);\n            const float actual = (float)value; const float error = std::fabs(actual - expected_values[index]);\n            if (error > max_abs_error) max_abs_error = error;\n            if (error > {float(tolerance):.9g}f) {{ std::printf("{fail_token} index=%d expected=%f actual=%f error=%f\\n", index, expected_values[index], actual, error); return 2; }}\n            ++index;\n        }}\n    }}\n    if (index != OUTPUT_N) {{ std::printf("{fail_token} output_count=%d expected_count=%d\\n", index, OUTPUT_N); return 3; }}\n    std::printf("{pass_token} outputs=%d max_abs_error=%f\\n", OUTPUT_N, max_abs_error);\n    return 0;\n}}\n'''
    path.write_text(text, encoding="utf-8")


def emit_hls_project(
    graph: Graph,
    out_dir: Path,
    *,
    input_values: Mapping[str, np.ndarray],
    expected_output: np.ndarray,
    part: str = "xck26-sfvc784-2LV-c",
    clock_mhz: int = 200,
    artifact_namespace: str = "hls",
    result_schema: str = "fpgai.hls-result/v1",
    reference_schema: str = "fpgai.hls-reference/v1",
    pass_token: str = "FPGAI_HLS_PASS",
    tolerance: float = 0.05,
    raw_cfg: Mapping[str, Any] | None = None,
) -> HLSBuild:
    """Emit one generic inference HLS project with numeric C-sim validation."""

    out_dir = Path(out_dir)
    namespace = _normalise_namespace(artifact_namespace)
    inputs = {str(k): np.asarray(v, dtype=np.float32) for k, v in input_values.items()}
    expected = np.asarray(expected_output, dtype=np.float32)

    cfg = dict(raw_cfg or {})
    if not cfg:
        cfg = {
            "numerics": {"defaults": {
                "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
                "weight": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
                "bias": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
                "accum": {"type": "ap_fixed", "total_bits": 32, "int_bits": 12},
            }},
            "targets": {"hls": {"control_protocol": "s_axilite"}},
        }

    project = emit_hls_stub(
        graph=graph,
        out_dir=out_dir,
        top_name="deeplearn",
        hls_options={
            "weights_mode": "embedded",
            "part": part,
            "clk_mhz": int(clock_mhz),
            "pipeline_mode": "inference",
            "raw_cfg": cfg,
            "run_csim": True,
            "run_csynth": True,
            "export_ip": False,
        },
    )
    _emit_numeric_testbench(
        project.tb_cpp,
        graph,
        inputs,
        expected,
        tolerance=float(tolerance),
        pass_token=str(pass_token),
    )

    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    ref_path = reports / f"{namespace}_numeric_reference.json"
    ref_path.write_text(json.dumps({
        "schema": str(reference_schema),
        "graph": graph.name,
        "inputs": {k: np.asarray(v).reshape(-1).tolist() for k, v in inputs.items()},
        "expected_output": expected.reshape(-1).tolist(),
        "tolerance": float(tolerance),
        "measurement": "numpy_float_reference_vs_ap_fixed_hls_csim",
    }, indent=2) + "\n", encoding="utf-8")

    return HLSBuild(
        project=project,
        input_values=inputs,
        expected_output=expected,
        reference_report=ref_path,
        artifact_namespace=namespace,
        result_schema=str(result_schema),
        reference_schema=str(reference_schema),
        pass_token=str(pass_token),
        tolerance=float(tolerance),
    )


def run_vitis_hls(build: HLSBuild, *, timeout: int = 1800) -> dict[str, Any]:
    """Run Vitis HLS for a generic emitted project and collect common artifacts."""

    hls_dir = build.project.hls_dir
    reports = hls_dir.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    namespace = _normalise_namespace(build.artifact_namespace)
    stdout_log = reports / f"{namespace}_vitis_hls_stdout.log"
    stderr_log = reports / f"{namespace}_vitis_hls_stderr.log"
    result_path = reports / f"{namespace}_hls_tool_result.json"
    executable = shutil.which("vitis_hls")

    if executable is None:
        result: dict[str, Any] = {
            "schema": build.result_schema,
            "status": "tool_unavailable",
            "returncode": None,
            "csim_passed": False,
            "csynth_report": None,
            "validation_level": "hls_project_generated",
        }
    else:
        try:
            proc = subprocess.run(
                [executable, "-f", build.project.run_tcl.name],
                cwd=hls_dir,
                text=True,
                capture_output=True,
                timeout=int(timeout),
            )
            stdout_log.write_text(proc.stdout or "", encoding="utf-8")
            stderr_log.write_text(proc.stderr or "", encoding="utf-8")
            csynth = hls_dir / "fpgai_hls_proj" / "sol1" / "syn" / "report" / "csynth.rpt"
            csim_passed = build.pass_token in (proc.stdout or "")
            passed = proc.returncode == 0 and csim_passed and csynth.exists()
            result = {
                "schema": build.result_schema,
                "status": "passed" if passed else "failed",
                "returncode": int(proc.returncode),
                "csim_passed": bool(csim_passed),
                "csynth_report": str(csynth) if csynth.exists() else None,
                "validation_level": "hls_synthesized" if passed else "hls_project_generated",
            }
        except subprocess.TimeoutExpired as exc:
            stdout_log.write_text((exc.stdout or "") if isinstance(exc.stdout, str) else "", encoding="utf-8")
            stderr_log.write_text((exc.stderr or "") if isinstance(exc.stderr, str) else "", encoding="utf-8")
            result = {
                "schema": build.result_schema,
                "status": "timeout",
                "returncode": None,
                "csim_passed": False,
                "csynth_report": None,
                "validation_level": "hls_project_generated",
                "timeout_seconds": int(timeout),
            }

    result["stdout_log"] = str(stdout_log)
    result["stderr_log"] = str(stderr_log)
    result["reference_report"] = str(build.reference_report)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = ["HLSBuild", "emit_hls_project", "run_vitis_hls"]
