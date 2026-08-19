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
    """Build the equivalent TensorFlow source model used by the FPGAI example."""
    import tensorflow as tf

    class LinearModel(tf.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = tf.constant(WEIGHT, dtype=tf.float32)
            self.bias = tf.constant(BIAS, dtype=tf.float32)

        @tf.function(input_signature=[tf.TensorSpec((1, 4), tf.float32, name="input")])
        def __call__(self, x):
            return tf.linalg.matmul(x, self.weight) + self.bias

    return LinearModel()


def export_onnx(out: Path) -> Path:
    import tensorflow as tf

    try:
        import tf2onnx
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow ONNX export uses the optional upstream tf2onnx converter."
        ) from exc

    model = build_model()
    signature = [tf.TensorSpec((1, 4), tf.float32, name="input")]
    out.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_function(
        model.__call__,
        input_signature=signature,
        opset=18,
        output_path=str(out),
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the TensorFlow linear example to ONNX for FPGAI")
    parser.add_argument("--out", type=Path, default=Path("build/examples/frontends/tensorflow_linear.onnx"))
    args = parser.parse_args()
    try:
        path = export_onnx(args.out)
    except Exception as exc:
        raise SystemExit(
            "TensorFlow export requires TensorFlow plus the optional upstream tf2onnx converter."
        ) from exc
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
