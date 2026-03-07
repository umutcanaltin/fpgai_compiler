from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
import numpy as np

from fpgai.config.loader import load_config
from fpgai.engine.compiler import Compiler
from fpgai.util.binio import read_f32_bin


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def build_host_if_needed(out_dir: Path, top: str, *, verbose: bool) -> Path:
    host_dir = out_dir / "hostcpp"
    include = host_dir / "include"
    src = host_dir / "src"
    exe = host_dir / "host_ref.exe"

    if exe.exists():
        return exe

    if not host_dir.exists():
        raise FileNotFoundError(f"hostcpp dir not found: {host_dir}")

    compile_cmd = [
        "g++", "-O2", "-std=c++17",
        "-I", str(include),
        str(src / "run.cpp"),
        str(src / f"{top}_host.cpp"),
        str(src / f"{top}_params.cpp"),
        "-o", str(exe),
    ]

    if verbose:
        print("[FPGAI] Building host reference (missing exe)...")
        print("[CMD]", " ".join(compile_cmd))
        print("[CWD]", host_dir)

    subprocess.check_call(compile_cmd, cwd=str(host_dir))
    return exe


def run_host(exe: Path, input_bin: Path) -> np.ndarray:
    out = subprocess.check_output([str(exe), str(input_bin)], cwd=str(exe.parent))
    lines = out.decode("utf-8", errors="ignore").strip().splitlines()
    if not lines:
        raise RuntimeError("Host produced no output")

    # Find the last line that starts with "y["
    y_line = None
    for ln in reversed(lines):
        if ln.strip().startswith("y["):
            y_line = ln.strip()
            break
    if y_line is None:
        y_line = lines[-1].strip()

    # Only parse floats AFTER the "]:"
    cut = y_line
    marker = "]:"
    if marker in y_line:
        cut = y_line.split(marker, 1)[1]

    vals = _FLOAT_RE.findall(cut)
    if not vals:
        raise RuntimeError(f"Could not parse floats from host output line: {y_line!r}")

    return np.array([float(x) for x in vals], dtype=np.float32)


def run_onnxruntime(onnx_path: str, x_vec: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    inp_name = sess.get_inputs()[0].name

    tried = []
    for shaped in (
        x_vec.reshape(-1),
        x_vec.reshape(1, -1),
        x_vec.reshape(1, 1, 1, -1),
    ):
        try:
            outs = sess.run(None, {inp_name: shaped.astype(np.float32)})
            return outs[0].reshape(-1).astype(np.float32)
        except Exception as e:
            tried.append(str(e))

    raise RuntimeError("onnxruntime failed for all attempted input shapes:\n" + "\n---\n".join(tried))


def parse_args():
    p = argparse.ArgumentParser("Compare hostcpp vs onnxruntime")
    p.add_argument("--config", required=True)
    p.add_argument("--regen", action="store_true", help="Run compiler first (regenerate artifacts)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg.raw.get("project", {}).get("out_dir", "build/fpgai")).resolve()
    top = cfg.raw.get("pipeline", {}).get("outputs", {}).get("top_kernel_name", "deeplearn")
    onnx_path = cfg.model.path

    if args.regen:
        Compiler(cfg).compile()

    input_bin = out_dir / "input.bin"
    if not input_bin.exists():
        raise FileNotFoundError(f"input.bin not found: {input_bin}")

    host_exe = build_host_if_needed(out_dir, top, verbose=args.verbose)

    print("=== FPGAI Compare: hostcpp vs onnxruntime ===")
    print("out_dir   :", out_dir)
    print("input_bin :", input_bin)
    print("onnx      :", onnx_path)
    print("host_exe  :", host_exe)
    print("--------------------------------------------")

    x = read_f32_bin(input_bin)
    y_host = run_host(host_exe, input_bin)
    y_ort = run_onnxruntime(onnx_path, x)

    n = min(len(y_host), len(y_ort))
    diff = np.abs(y_host[:n] - y_ort[:n])

    print("outputs   :", n)
    print("max_abs   :", float(np.max(diff)))
    print("mean_abs  :", float(np.mean(diff)))
    print("y_host[0:8]:", " ".join(f"{v:.6f}" for v in y_host[:8]))
    print("y_ort [0:8]:", " ".join(f"{v:.6f}" for v in y_ort[:8]))
    print("============================================")


if __name__ == "__main__":
    main()
