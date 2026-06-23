from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from fpgai.config.loader import load_config
from fpgai.engine.compiler import Compiler
from fpgai.util.binio import read_f32_bin

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class ComparisonStats:
    """Numerical comparison summary for two output vectors."""

    outputs: int
    max_abs: float
    mean_abs: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compare_arrays(a: np.ndarray, b: np.ndarray) -> ComparisonStats:
    """Return max/mean absolute error over the shared prefix of two arrays."""

    n = min(len(a), len(b))
    if n <= 0:
        raise ValueError("cannot compare empty arrays")
    diff = np.abs(a[:n] - b[:n])
    return ComparisonStats(outputs=n, max_abs=float(np.max(diff)), mean_abs=float(np.mean(diff)))


def build_host_if_needed(out_dir: Path, top: str, *, verbose: bool = False) -> Path:
    """Build the generated host C++ reference executable when it is missing."""

    host_dir = out_dir / "hostcpp"
    include = host_dir / "include"
    src = host_dir / "src"
    exe = host_dir / "host_ref.exe"

    if exe.exists():
        return exe
    if not host_dir.exists():
        raise FileNotFoundError(f"hostcpp dir not found: {host_dir}")

    compile_cmd = [
        "g++",
        "-O2",
        "-std=c++17",
        "-I",
        str(include),
        str(src / "run.cpp"),
        str(src / f"{top}_host.cpp"),
        str(src / f"{top}_params.cpp"),
        "-o",
        str(exe),
    ]

    if verbose:
        print("[FPGAI] Building host reference executable")
        print("[CMD]", " ".join(compile_cmd))
        print("[CWD]", host_dir)

    subprocess.check_call(compile_cmd, cwd=str(host_dir))
    return exe


def run_host_executable(exe: Path, input_bin: Path) -> np.ndarray:
    """Run a generated host executable and return the output vector."""

    out = subprocess.check_output([str(exe), str(input_bin)], cwd=str(exe.parent))

    output_bin = exe.parent / "output.bin"
    if output_bin.exists():
        return read_f32_bin(output_bin)

    lines = out.decode("utf-8", errors="ignore").strip().splitlines()
    if not lines:
        raise RuntimeError("host executable produced no output")

    y_line = None
    for line in reversed(lines):
        if line.strip().startswith("y["):
            y_line = line.strip()
            break
    if y_line is None:
        y_line = lines[-1].strip()

    cut = y_line.split("]:", 1)[1] if "]:" in y_line else y_line
    vals = _FLOAT_RE.findall(cut)
    if not vals:
        raise RuntimeError(f"could not parse floats from host output line: {y_line!r}")
    return np.array([float(x) for x in vals], dtype=np.float32)


def run_onnxruntime(onnx_path: str | Path, x_vec: np.ndarray) -> np.ndarray:
    """Run ONNX Runtime with a few common flattened/batched input shapes."""

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    inp_name = sess.get_inputs()[0].name
    tried: list[str] = []

    for shaped in (
        x_vec.reshape(-1),
        x_vec.reshape(1, -1),
        x_vec.reshape(1, 1, 1, -1),
    ):
        try:
            outs = sess.run(None, {inp_name: shaped.astype(np.float32)})
            return outs[0].reshape(-1).astype(np.float32)
        except Exception as exc:  # pragma: no cover - depends on model input shape
            tried.append(str(exc))

    raise RuntimeError("onnxruntime failed for all attempted input shapes:\n" + "\n---\n".join(tried))


def _config_output_context(config_path: str | Path, *, regen: bool) -> tuple[object, Path, str, str]:
    cfg = load_config(config_path)
    if regen:
        Compiler(cfg).compile()
    out_dir = Path(cfg.raw.get("project", {}).get("out_dir", "build/fpgai")).resolve()
    top = cfg.raw.get("pipeline", {}).get("outputs", {}).get("top_kernel_name", "deeplearn")
    return cfg, out_dir, str(top), str(cfg.model.path)


def compare_host_vs_onnx(config_path: str | Path, *, regen: bool = False, verbose: bool = False) -> dict[str, object]:
    """Compare generated host C++ output against ONNX Runtime output."""

    _cfg, out_dir, top, onnx_path = _config_output_context(config_path, regen=regen)
    input_bin = out_dir / "input.bin"
    if not input_bin.exists():
        raise FileNotFoundError(f"input.bin not found: {input_bin}")

    host_exe = build_host_if_needed(out_dir, top, verbose=verbose)
    x_vec = read_f32_bin(input_bin)
    y_host = run_host_executable(host_exe, input_bin)
    y_onnx = run_onnxruntime(onnx_path, x_vec)
    stats = compare_arrays(y_host, y_onnx)

    return {
        "out_dir": str(out_dir),
        "input_bin": str(input_bin),
        "onnx": str(onnx_path),
        "host_exe": str(host_exe),
        "host_vs_onnx": stats.to_dict(),
        "y_host_head": y_host[:8].tolist(),
        "y_onnx_head": y_onnx[:8].tolist(),
    }


def compare_vitis_vs_host_vs_onnx(
    config_path: str | Path,
    *,
    mode: str = "csim",
    regen: bool = False,
    verbose: bool = False,
) -> dict[str, object]:
    """Compare Vitis HLS output against host C++ and ONNX Runtime outputs."""

    if mode not in {"csim", "cosim"}:
        raise ValueError("mode must be 'csim' or 'cosim'")

    compiler = Compiler.from_yaml(str(Path(config_path).resolve()))
    result = compiler.compile() if regen else compiler.compile()

    out_dir = Path(result.out_dir)
    onnx_path = Path(compiler.cfg.model.path)
    input_bin = out_dir / "input.bin"
    host_dir = out_dir / "hostcpp"
    host_exe = host_dir / "host_ref.exe"
    hls_dir = out_dir / "hls"
    hls_out = hls_dir / "output.bin"

    if not input_bin.exists():
        raise FileNotFoundError(f"input.bin not found: {input_bin}")
    if not host_exe.exists():
        raise FileNotFoundError(f"host_ref.exe not found: {host_exe}")

    from fpgai.backends.hls.vitis_runner import run_vitis_hls

    raw = compiler.cfg.raw
    top_name = str(raw.get("pipeline", {}).get("outputs", {}).get("top_kernel_name", "deeplearn"))
    part = str(raw.get("targets", {}).get("platform", {}).get("part", "xck26-sfvc784-2LV-c"))
    clock_mhz = int(raw.get("targets", {}).get("platform", {}).get("clocks", [{}])[0].get("target_mhz", 200))

    rr = run_vitis_hls(
        hls_dir,
        top_name=top_name,
        part=part,
        clock_mhz=clock_mhz,
        do_csim=mode == "csim",
        do_cosim=mode == "cosim",
        do_synth=mode == "cosim",
    )
    if verbose:
        print("[FPGAI] Vitis log:", rr.log_path)

    if not hls_out.exists():
        raise FileNotFoundError(f"Vitis did not produce output.bin at: {hls_out}; check {rr.log_path}")

    x_vec = read_f32_bin(input_bin)
    y_host = run_host_executable(host_exe, input_bin)
    y_onnx = run_onnxruntime(onnx_path, x_vec)
    y_hls = read_f32_bin(hls_out)

    return {
        "out_dir": str(out_dir),
        "mode": mode,
        "lengths": {"host": len(y_host), "hls": len(y_hls), "onnx": len(y_onnx)},
        "host_vs_onnx": compare_arrays(y_host, y_onnx).to_dict(),
        "hls_vs_onnx": compare_arrays(y_hls, y_onnx).to_dict(),
        "hls_vs_host": compare_arrays(y_hls, y_host).to_dict(),
        "y_host_head": y_host[:8].tolist(),
        "y_hls_head": y_hls[:8].tolist(),
        "y_onnx_head": y_onnx[:8].tolist(),
    }


def _format_head(values: Sequence[float]) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def main_host_vs_onnx(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("Compare hostcpp vs onnxruntime")
    parser.add_argument("--config", required=True)
    parser.add_argument("--regen", action="store_true", help="Run compiler first")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    result = compare_host_vs_onnx(args.config, regen=args.regen, verbose=args.verbose)
    stats = result["host_vs_onnx"]

    print("=== FPGAI Compare: hostcpp vs onnxruntime ===")
    print("out_dir   :", result["out_dir"])
    print("input_bin :", result["input_bin"])
    print("onnx      :", result["onnx"])
    print("host_exe  :", result["host_exe"])
    print("--------------------------------------------")
    print("outputs  :", stats["outputs"])
    print("max_abs  :", stats["max_abs"])
    print("mean_abs :", stats["mean_abs"])
    print("y_host[0:8]:", _format_head(result["y_host_head"]))
    print("y_onnx[0:8]:", _format_head(result["y_onnx_head"]))
    print("============================================")
    return 0


def main_vitis_vs_host_vs_onnx(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser("Compare vitis_hls vs hostcpp vs onnxruntime")
    parser.add_argument("--config", required=True)
    parser.add_argument("--regen", action="store_true", help="Run compiler first")
    parser.add_argument("--mode", choices=["csim", "cosim"], default="csim")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    result = compare_vitis_vs_host_vs_onnx(
        args.config,
        mode=args.mode,
        regen=args.regen,
        verbose=args.verbose,
    )

    print("=== FPGAI Compare: vitis_hls vs hostcpp vs onnxruntime ===")
    print("out_dir:", result["out_dir"])
    print("mode   :", result["mode"])
    print("len(y) :", result["lengths"])
    print("--------------------------------------------")
    for key in ["host_vs_onnx", "hls_vs_onnx", "hls_vs_host"]:
        stats = result[key]
        print(f"{key}: max_abs {stats['max_abs']} mean_abs {stats['mean_abs']}")
    print("y_host[0:8]:", _format_head(result["y_host_head"]))
    print("y_hls [0:8]:", _format_head(result["y_hls_head"]))
    print("y_onnx[0:8]:", _format_head(result["y_onnx_head"]))
    print("============================================")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main_host_vs_onnx())
