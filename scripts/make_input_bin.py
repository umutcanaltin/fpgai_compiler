import argparse
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of float32 inputs")
    ap.add_argument("--out", default="build/input.bin")
    args = ap.parse_args()

    x = (np.arange(args.n, dtype=np.float32) + 1) * 0.1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(x.tobytes(order="C"))
    print("[OK] wrote", out, "bytes=", out.stat().st_size)

if __name__ == "__main__":
    main()
