"""Create one reproducible float32 NumPy input for an existing ONNX model.

Used by the explicit fpgai-research-tools prepare workflow.
No model training, network access, hardware invocation or vendor programs.

Usage:
    python examples/make_sample_input.py build/gallery/cnn_pool_dense.onnx build/gallery/sample.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path


def save_sample(model_path: str | Path, output: str | Path, *,
                seed: int = 197) -> Path:
    import numpy as np
    from fpgai_research.frontend.onnx import load_onnx

    dest = Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f"Refusing to overwrite sample input: {dest}")
    if type(seed) is not int or not 0 <= seed <= 2**32-1:
        raise ValueError("Seed must be an integer between 0 and 2**32-1")

    model = load_onnx(str(model_path))
    if len(model.inputs) != 1:
        raise ValueError("Example requires exactly one runtime input tensor")
    shape = model.tensor(model.inputs[0]).shape
    if not shape or any(type(d) is not int or d < 1 for d in shape):
        raise ValueError("Sample generation requires positive static dimensions")
    values = np.random.default_rng(seed).uniform(
        -0.25, 0.25, size=shape).astype(np.float32)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Existing ONNX model")
    parser.add_argument("output", help="New sample .npy path")
    parser.add_argument("--seed", type=int, default=197)
    args = parser.parse_args(argv)
    saved = save_sample(args.model, args.output, seed=args.seed)
    print(f"Wrote deterministic synthetic float32 sample: {saved}")


if __name__ == "__main__":
    main()
