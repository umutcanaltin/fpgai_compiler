from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import subprocess
import sys

from fpgai.engine import Compiler


def read_f32_bin(path: Path) -> np.ndarray:
    data = path.read_bytes()
    return np.frombuffer(data, dtype=np.float32)


def run_onnxruntime(onnx_path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: x.astype(np.float32)})[0]
    return np.array(out, dtype=np.float32).reshape(-1)


def run_host_exe(host_exe: Path, input_bin: Path) -> np.ndarray:
    out = subprocess.check_output([str(host_exe), str(input_bin)], cwd=str(host_exe.parent))
    # Expect machine-readable output.bin exists (preferred)
    out_bin = host_exe.parent / "output.bin"
    if out_bin.exists():
        return read_f32_bin(out_bin)
    # fallback: parse numbers from stdout
    txt = out.decode("utf-8", errors="ignore").strip().split()
    vals = []
    for t in txt:
        try:
            vals.append(float(t))
        except:
            pass
    return np.array(vals, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--regen", action="store_true", help="run compiler first")
    ap.add_argument("--mode", choices=["csim", "cosim"], default="csim")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()

    # 1) Compile (regen artifacts)
    compiler = Compiler.from_yaml(str(cfg_path))
    result = compiler.compile() if args.regen else compiler.compile()

    out_dir = Path(result.out_dir)
    onnx_path = Path(compiler.cfg.model.path)

    # Paths we expect compiler to have emitted
    input_bin = out_dir / "input.bin"
    host_dir = out_dir / "hostcpp"
    host_exe = host_dir / "host_ref.exe"
    hls_dir = out_dir / "hls"
    hls_out = hls_dir / "output.bin"

    if not input_bin.exists():
        print("ERROR missing input.bin:", input_bin)
        sys.exit(2)

    # 2) Build+run host (if missing)
    if not host_exe.exists():
        print("[FPGAI] host_ref.exe missing — run scripts/run_hostcpp.py first")
        sys.exit(2)

    y_host = run_host_exe(host_exe, input_bin)

    # 3) Run ONNXRuntime
    x = read_f32_bin(input_bin)
    y_ort = run_onnxruntime(onnx_path, x).reshape(-1)

    # 4) Run Vitis HLS (csim/cosim)
    from fpgai.backends.hls.vitis_runner import run_vitis_hls
    raw = compiler.cfg.raw
    top_name = str(raw.get("pipeline", {}).get("outputs", {}).get("top_kernel_name", "deeplearn"))
    part = str(raw.get("targets", {}).get("platform", {}).get("part", "xck26-sfvc784-2LV-c"))
    clock_mhz = int(raw.get("targets", {}).get("platform", {}).get("clocks", [{}])[0].get("target_mhz", 200))

    do_csim = args.mode == "csim"
    do_cosim = args.mode == "cosim"
    do_synth = do_cosim  # cosim typically needs synth

    rr = run_vitis_hls(hls_dir, top_name=top_name, part=part, clock_mhz=clock_mhz, do_csim=do_csim, do_cosim=do_cosim, do_synth=do_synth)

    if args.verbose:
        print("[FPGAI] vitis log:", rr.log_path)

    if not hls_out.exists():
        print("ERROR: Vitis did not produce output.bin at:", hls_out)
        print("Check:", rr.log_path)
        sys.exit(2)

    y_hls = read_f32_bin(hls_out)

    # 5) Compare
    def cmp(a, b):
        n = min(len(a), len(b))
        d = np.abs(a[:n] - b[:n])
        return float(np.max(d)), float(np.mean(d))

    m1, a1 = cmp(y_host, y_ort)
    m2, a2 = cmp(y_hls, y_ort)
    m3, a3 = cmp(y_hls, y_host)

    print("=== FPGAI Compare: vitis_hls vs hostcpp vs onnxruntime ===")
    print("out_dir:", out_dir)
    print("mode  :", args.mode)
    print("len(y): host", len(y_host), "hls", len(y_hls), "ort", len(y_ort))
    print("--------------------------------------------")
    print("host vs ort: max_abs", m1, "mean_abs", a1)
    print("hls  vs ort: max_abs", m2, "mean_abs", a2)
    print("hls  vs host: max_abs", m3, "mean_abs", a3)
    print("y_host[0:8]:", " ".join(f"{v:.6f}" for v in y_host[:8]))
    print("y_hls [0:8]:", " ".join(f"{v:.6f}" for v in y_hls[:8]))
    print("y_ort [0:8]:", " ".join(f"{v:.6f}" for v in y_ort[:8]))
    print("============================================")


if __name__ == "__main__":
    main()
