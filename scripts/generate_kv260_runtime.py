#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


PY_TEMPLATE = """\
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

try:
    from pynq import Overlay, allocate
except Exception as e:
    raise RuntimeError(
        "PYNQ runtime is required on KV260 to run this script. "
        "Install or run on a PYNQ-enabled image."
    ) from e


THIS_DIR = Path(__file__).resolve().parent
META = json.loads((THIS_DIR / "model_meta.json").read_text(encoding="utf-8"))
REGMAP = json.loads((THIS_DIR / "register_map.json").read_text(encoding="utf-8"))

BITFILE = THIS_DIR / META["bitfile_name"]


def quantize_input(x: np.ndarray, scale: float, zero_point: int = 0) -> np.ndarray:
    q = np.round(x / scale).astype(np.int32) + int(zero_point)
    return q.astype(np.int16)


def dequantize_output(x: np.ndarray, scale: float, zero_point: int = 0) -> np.ndarray:
    return (x.astype(np.float32) - float(zero_point)) * float(scale)


def run_once(input_npy: str):
    overlay = Overlay(str(BITFILE))
    ip_name = META["ip_name"]
    dma_name = META["dma_name"]

    accel = getattr(overlay, ip_name)
    dma = getattr(overlay, dma_name)

    x = np.load(input_npy).astype(np.float32).reshape(-1)

    q_in = quantize_input(
        x,
        scale=META["input_quant"]["scale"],
        zero_point=META["input_quant"]["zero_point"],
    )

    in_buf = allocate(shape=(q_in.size,), dtype=np.int16)
    out_buf = allocate(shape=(META["output_length"],), dtype=np.int16)

    np.copyto(in_buf, q_in)
    out_buf[:] = 0

    dma.sendchannel.transfer(in_buf)
    dma.recvchannel.transfer(out_buf)

    # Optional control register kick if needed
    if REGMAP.get("ap_start_addr") is not None:
        accel.write(int(REGMAP["ap_start_addr"]), 0x1)

    dma.sendchannel.wait()
    dma.recvchannel.wait()

    y = dequantize_output(
        np.array(out_buf),
        scale=META["output_quant"]["scale"],
        zero_point=META["output_quant"]["zero_point"],
    )

    out_path = THIS_DIR / "last_output.npy"
    np.save(out_path, y)
    print(f"[OK] wrote {out_path}")
    print("output:", y)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input .npy")
    args = p.parse_args()
    run_once(args.input)
"""


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="fpgai.yml")
    parser.add_argument("--bitfile-name", default="design_1_wrapper.bit")
    parser.add_argument("--hwh-name", default="design_1.hwh")
    parser.add_argument("--ip-name", default="deeplearn_0")
    parser.add_argument("--dma-name", default="axi_dma_0")
    parser.add_argument("--output-length", type=int, default=10)
    args = parser.parse_args()

    cfg = load_cfg(Path(args.config).resolve())
    out_dir = Path(cfg["project"]["out_dir"]).resolve()
    runtime_dir = out_dir / "runtime" / "python"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    defaults = cfg.get("numerics", {}).get("defaults", {})
    act = defaults.get("activation", {})
    output_quant = {
        "scale": 1.0 / (2 ** max(int(act.get("total_bits", 16)) - int(act.get("int_bits", 6)), 1)),
        "zero_point": 0,
    }

    model_meta = {
        "project_name": cfg.get("project", {}).get("name", "fpgai_project"),
        "board": cfg.get("targets", {}).get("platform", {}).get("board", "kv260"),
        "part": cfg.get("targets", {}).get("platform", {}).get("part"),
        "bitfile_name": args.bitfile_name,
        "hwh_name": args.hwh_name,
        "ip_name": args.ip_name,
        "dma_name": args.dma_name,
        "output_length": args.output_length,
        "input_quant": {"scale": output_quant["scale"], "zero_point": 0},
        "output_quant": output_quant,
    }

    register_map = {
        "ap_start_addr": 0x00,
        "note": "Adjust this if your HLS AXI-Lite map differs.",
    }

    (runtime_dir / "run_inference.py").write_text(PY_TEMPLATE, encoding="utf-8")
    (runtime_dir / "model_meta.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")
    (runtime_dir / "register_map.json").write_text(json.dumps(register_map, indent=2), encoding="utf-8")

    readme = f"""\
FPGAI KV260 Python runtime
==========================

Files:
- run_inference.py
- model_meta.json
- register_map.json

Expected bitstream files in this folder:
- {args.bitfile_name}
- {args.hwh_name}

Example:
python3 run_inference.py --input sample_input.npy
"""
    (runtime_dir / "README.txt").write_text(readme, encoding="utf-8")

    print(f"[OK] Runtime package generated at: {runtime_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())