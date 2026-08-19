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
    """Build the equivalent direct-ONNX graph used by the FPGAI example."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    weight = numpy_helper.from_array(np.asarray(WEIGHT, dtype=np.float32), name="weight")
    bias = numpy_helper.from_array(np.asarray(BIAS, dtype=np.float32), name="bias")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "weight"], ["matmul"], name="matmul"),
            helper.make_node("Add", ["matmul", "bias"], ["output"], name="bias_add"),
        ],
        "fpgai_frontend_linear",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
        initializer=[weight, bias],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 18)],
        producer_name="fpgai-example",
    )


def export_onnx(out: Path) -> Path:
    import onnx

    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(), str(out))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the direct ONNX linear example for FPGAI")
    parser.add_argument("--out", type=Path, default=Path("build/examples/frontends/onnx_linear.onnx"))
    args = parser.parse_args()
    try:
        path = export_onnx(args.out)
    except Exception as exc:
        raise SystemExit("Direct ONNX generation requires FPGAI's normal ONNX dependencies.") from exc
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
