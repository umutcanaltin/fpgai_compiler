from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def write_float32_bin(values: Iterable[float], output_path: str | Path) -> Path:
    """Write values as a contiguous float32 binary file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(list(values), dtype=np.float32)
    arr.tofile(out)
    return out


def write_array_bin(array: np.ndarray, output_path: str | Path, dtype: str = "float32") -> Path:
    """Write a NumPy array to a raw binary file using the requested dtype."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(array, dtype=dtype).tofile(out)
    return out


def read_array_bin(input_path: str | Path, dtype: str = "float32") -> np.ndarray:
    """Read a raw binary file into a one-dimensional NumPy array."""
    return np.fromfile(Path(input_path), dtype=np.dtype(dtype))


def make_sequential_input(output_path: str | Path, count: int, scale: float = 1.0) -> Path:
    """Create a deterministic input vector useful for smoke testing."""
    if count < 0:
        raise ValueError("count must be non-negative")
    values = (np.arange(count, dtype=np.float32) * np.float32(scale)).astype(np.float32)
    return write_array_bin(values, output_path)


def make_constant_weights(output_path: str | Path, count: int, value: float = 0.01) -> Path:
    """Create a deterministic weights vector useful for smoke testing."""
    if count < 0:
        raise ValueError("count must be non-negative")
    values = np.full((count,), np.float32(value), dtype=np.float32)
    return write_array_bin(values, output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or inspect raw FPGAI binary arrays.")
    sub = parser.add_subparsers(dest="command", required=True)

    inp = sub.add_parser("make-input", help="Create a deterministic input .bin file")
    inp.add_argument("--out", required=True)
    inp.add_argument("--count", type=int, required=True)
    inp.add_argument("--scale", type=float, default=1.0)

    w = sub.add_parser("make-weights", help="Create a deterministic weights .bin file")
    w.add_argument("--out", required=True)
    w.add_argument("--count", type=int, required=True)
    w.add_argument("--value", type=float, default=0.01)

    r = sub.add_parser("read", help="Read a binary file and print basic metadata")
    r.add_argument("path")
    r.add_argument("--dtype", default="float32")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "make-input":
        out = make_sequential_input(args.out, args.count, args.scale)
        print(out)
        return 0

    if args.command == "make-weights":
        out = make_constant_weights(args.out, args.count, args.value)
        print(out)
        return 0

    if args.command == "read":
        arr = read_array_bin(args.path, args.dtype)
        print({"path": str(args.path), "dtype": str(arr.dtype), "shape": list(arr.shape), "count": int(arr.size)})
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
