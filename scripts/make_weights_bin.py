import argparse
import json
from pathlib import Path

from fpgai.frontend.onnx import import_onnx
from fpgai.runtime.weights import build_weights_plan_from_ir, pack_weights_stream_float32


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to ONNX model")
    ap.add_argument("--out", default="build/weights.bin", help="Output binary path")
    ap.add_argument("--plan-out", default=None, help="Optional: also write weights_plan.json")
    return ap.parse_args()


def main():
    args = parse_args()

    g = import_onnx(args.onnx, canonicalize=True, infer_shapes=True, insert_missing_activations=False)
    plan = build_weights_plan_from_ir(g)
    payload = pack_weights_stream_float32(g, plan)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)

    print("[OK] wrote", out_path, "bytes=", len(payload))

    if args.plan_out:
        plan_path = Path(args.plan_out)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan.to_dict(), indent=2))
        print("[OK] wrote", plan_path)


if __name__ == "__main__":
    main()
