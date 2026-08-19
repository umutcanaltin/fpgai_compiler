from __future__ import annotations

import argparse
from pathlib import Path

WEIGHT = (
    (0.10, 0.20, 0.30),
    (0.40, 0.50, 0.60),
    (0.70, 0.80, 0.90),
    (1.00, 1.10, 1.20),
)
BIAS = (0.01, 0.02, 0.03)


def build_model():
    """Build the equivalent PyTorch source model used by the FPGAI example."""
    import torch

    class LinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("weight", torch.tensor(WEIGHT, dtype=torch.float32))
            self.register_buffer("bias", torch.tensor(BIAS, dtype=torch.float32))

        def forward(self, x):
            return torch.matmul(x, self.weight) + self.bias

    return LinearModel().eval()


def export_onnx(out: Path) -> Path:
    import torch

    model = build_model()
    sample = torch.zeros((1, 4), dtype=torch.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        str(out),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        dynamo=False,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the PyTorch linear example to ONNX for FPGAI")
    parser.add_argument("--out", type=Path, default=Path("build/examples/frontends/pytorch_linear.onnx"))
    args = parser.parse_args()
    try:
        path = export_onnx(args.out)
    except Exception as exc:
        raise SystemExit(
            "PyTorch ONNX export requires PyTorch and the ONNX dependencies used by FPGAI."
        ) from exc
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
