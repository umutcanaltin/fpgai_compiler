"""Create an independent ONNX/Model IR inference reference before board upload.

Run on the workstation. The PYNQ-Z2 image need not contain ONNX or the
compiler; only the numeric .npy input/reference and board bundle are copied.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


def generate_reference(onnx_path: str, input_path: str, output_path: str) -> dict:
    import numpy as np
    from fpgai_research.frontend.onnx import load_onnx
    from fpgai_research.validation.reference import evaluate

    model_file = Path(onnx_path).expanduser().resolve(strict=True)
    inp_file = Path(input_path).expanduser().resolve(strict=True)
    out_file = Path(output_path).expanduser()
    if out_file.exists() or out_file.is_symlink():
        raise FileExistsError("Refusing to replace a previous independent board reference")
    model = load_onnx(str(model_file))
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError("Board reference requires exactly one input and output tensor")
    data = np.load(inp_file, allow_pickle=False)
    inp = model.tensor(model.inputs[0])
    if data.dtype != np.float32 or tuple(data.shape) != inp.shape:
        raise ValueError(f"Expected exact float32 model input shape {inp.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("Reference input has nonfinite values")
    out_name = model.outputs[0]
    y = np.asarray(evaluate(model, {model.inputs[0]: data})[out_name], dtype=np.float32)
    if tuple(y.shape) != model.tensor(out_name).shape or not np.all(np.isfinite(y)):
        raise ValueError("Reference output is invalid")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("xb") as handle:
        np.save(handle, y, allow_pickle=False)
    return {
        "schema": "fpgai-research/board-software-reference/v1",
        "onnx_sha256": sha256(model_file.read_bytes()).hexdigest(),
        "input_npy_sha256": sha256(inp_file.read_bytes()).hexdigest(),
        "output_npy_sha256": sha256(out_file.read_bytes()).hexdigest(),
        "input_shape": list(inp.shape),
        "output_shape": list(y.shape),
        "output_file": str(out_file),
        "origin": "Independent float32 Model IR evaluation on workstation",
        "board_programmed": False,
        "on_device_comparison_performed": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = generate_reference(args.onnx, args.input, args.output)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"FPGAI board reference error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
