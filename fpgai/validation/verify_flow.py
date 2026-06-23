from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


def load_bin(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def run_cmd(cmd: str, cwd: Path | None = None, settings_path: str | None = None) -> None:
    """Run a shell command, optionally sourcing Xilinx settings for tool commands."""

    needs_xilinx = "vitis_hls" in cmd or "vivado" in cmd
    if needs_xilinx and settings_path and os.path.exists(settings_path):
        full_cmd = f"source {settings_path} && {cmd}"
        print(f"[CMD] {full_cmd}")
        subprocess.check_call(full_cmd, shell=True, cwd=cwd, executable="/bin/bash")
        return

    print(f"[CMD] {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def parse_hls_report(report_path: Path) -> dict[str, str] | None:
    if not report_path.exists():
        print(f"[WARN] Report not found: {report_path}")
        return None

    tree = ET.parse(report_path)
    root = tree.getroot()
    perf = root.find("PerformanceEstimates/SummaryOfOverallLatency/Average-case")
    area = root.find("AreaEstimates/Resources")

    return {
        "latency_cycles": perf.find("cycles").text if perf is not None and perf.find("cycles") is not None else "N/A",
        "dsp": area.find("DSP").text if area is not None and area.find("DSP") is not None else "0",
        "lut": area.find("LUT").text if area is not None and area.find("LUT") is not None else "0",
        "ff": area.find("FF").text if area is not None and area.find("FF") is not None else "0",
        "bram18": area.find("BRAM_18K").text if area is not None and area.find("BRAM_18K") is not None else "0",
    }


def ensure_input_bin(sess: object, input_bin_path: Path) -> np.ndarray:
    inp_obj = sess.get_inputs()[0]
    expected_shape = inp_obj.shape
    total_elements = 1
    concrete_shape: list[int] = []

    for dim in expected_shape:
        if isinstance(dim, int) and dim > 0:
            total_elements *= dim
            concrete_shape.append(dim)
        else:
            concrete_shape.append(1)

    if input_bin_path.exists():
        existing = np.fromfile(input_bin_path, dtype=np.float32)
        if existing.size == total_elements:
            return existing
        print("[INFO] Regenerating input.bin due to size mismatch")
    else:
        print(f"[INFO] Generating input.bin with shape {concrete_shape}")

    new_data = np.random.randn(total_elements).astype(np.float32) * 0.5
    new_data.tofile(input_bin_path)
    return new_data


def run_onnx_robust(sess: object, x_flat: np.ndarray) -> np.ndarray:
    inp_obj = sess.get_inputs()[0]
    inp_name = inp_obj.name
    concrete_shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp_obj.shape]

    candidates: list[np.ndarray] = []
    for shape in (concrete_shape, (-1,), (1, -1)):
        try:
            candidates.append(x_flat.reshape(shape))
        except Exception:
            pass

    for candidate in candidates:
        try:
            return sess.run(None, {inp_name: candidate})[0].flatten()
        except Exception:
            continue

    raise RuntimeError(f"shape mismatch for ONNX input {inp_name}")


def verify_flow(config_path: str | Path) -> int:
    import onnxruntime as ort

    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    toolchain = cfg.get("toolchain", {}) or {}
    vitis_settings = None
    if isinstance(toolchain.get("vitis_hls"), dict):
        vitis_settings = toolchain["vitis_hls"].get("settings64")
    if not vitis_settings:
        vitis_settings = toolchain.get("settings64") or toolchain.get("settings_path")

    print("\n[1/4] Compiling project...")
    run_cmd(f"PYTHONPATH=. python main.py --config {config_path}")

    out_dir = Path(cfg["project"]["out_dir"]).resolve()
    model_path = cfg["model"]["path"]
    input_bin = out_dir / "input.bin"

    print("\n[2/4] Running ONNX reference...")
    sess = ort.InferenceSession(model_path)
    x_flat = ensure_input_bin(sess, input_bin)
    y_onnx = run_onnx_robust(sess, x_flat)
    print(f"ONNX output: {y_onnx[:4]} ...")

    print("\n[3/4] Running HLS simulation and synthesis...")
    hls_dir = out_dir / "hls"
    try:
        run_cmd("vitis_hls -f run_hls.tcl", cwd=hls_dir, settings_path=vitis_settings)
    except subprocess.CalledProcessError:
        print("\n[ERROR] Vitis HLS failed.")
        if not vitis_settings:
            print("[TIP] No Vitis settings path found under toolchain.vitis_hls.settings64.")
        return 2

    report_path = hls_dir / "fpgai_hls_proj" / "sol1" / "syn" / "report" / "deeplearn_csynth.xml"
    report = parse_hls_report(report_path)
    if report:
        print("\n========================================")
        print(" HLS SYNTHESIS REPORT")
        print("========================================")
        print(f" Latency (cycles): {report['latency_cycles']}")
        print(f" DSP Usage       : {report['dsp']}")
        print(f" LUT Usage       : {report['lut']}")
        print(f" FF Usage        : {report['ff']}")
        print(f" BRAM18 Usage    : {report['bram18']}")
        print("========================================\n")

    hls_out_bin = hls_dir / "fpgai_hls_proj" / "sol1" / "csim" / "build" / "output.bin"
    if not hls_out_bin.exists():
        hls_out_bin = hls_dir / "output.bin"

    if not hls_out_bin.exists():
        print("Warning: HLS output.bin not found.")
        return 2

    y_hls = load_bin(hls_out_bin)
    print(f"HLS output: {y_hls[:4]} ...")

    print("\n[4/4] Final comparison results")
    print("-" * 40)
    n = min(len(y_onnx), len(y_hls))
    diff_hls = np.abs(y_hls[:n] - y_onnx[:n])
    print(f"{'Metric':<20} | {'HLS vs ONNX':<15}")
    print("-" * 40)
    print(f"{'Max Error':<20} | {diff_hls.max():<15.6f}")
    print(f"{'Mean Error':<20} | {diff_hls.mean():<15.6f}")
    print("-" * 40)

    if diff_hls.max() < 0.1:
        print("\nSUCCESS: HLS matches reference.")
        return 0

    print("\nWARNING: Large error detected.")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("Run FPGAI verification flow")
    parser.add_argument("--config", default="configs/examples/default_compile.yml")
    args = parser.parse_args(argv)
    return verify_flow(args.config)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
