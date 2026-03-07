import json
from pathlib import Path

from fpgai.frontend.onnx import import_onnx
from fpgai.runtime.weights import build_weights_plan_from_ir, pack_weights_stream_float32


def main():
    g = import_onnx("models/mlp.onnx", canonicalize=True, infer_shapes=True, insert_missing_activations=False)
    plan = build_weights_plan_from_ir(g)

    payload = pack_weights_stream_float32(g, plan)
    print("Dense layers:", len(plan.dense_layers))
    print("Packed bytes :", len(payload))

    # very basic check: payload must be multiple of 4 (float32)
    assert len(payload) % 4 == 0

    # emit to build for inspection
    out = Path("build/tmp_weights_test")
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights_plan.json").write_text(json.dumps(plan.to_dict(), indent=2))
    (out / "weights.bin").write_bytes(payload)
    print("Wrote:", out)


if __name__ == "__main__":
    main()
