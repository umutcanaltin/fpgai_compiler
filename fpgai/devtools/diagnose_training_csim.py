#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/training_stream_compare")
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    for item in results.get("results", []):
        design = item.get("design_name") or item.get("name")
        build = root / "artifacts" / design / "build"
        print(f"\n===== {design} =====")
        print("status:", item.get("status"), "returncode:", item.get("returncode"))
        for name in ("weights_before.bin", "grads.bin", "weights_after.bin"):
            hits = [str(p.relative_to(build)) for p in build.rglob(name)] if build.exists() else []
            print(name + ":", hits if hits else "MISSING")
        manifest = build / "manifest.json"
        if manifest.exists():
            try:
                m = json.loads(manifest.read_text(encoding="utf-8"))
                print("manifest.training_compare:", m.get("training_compare"))
                print("manifest.hls:", m.get("hls"))
            except Exception as e:
                print("manifest parse error:", e)
        tb = build / "hls" / "src" / "tb.cpp"
        tbt = read(tb)
        print("tb has output_dir:", "output_dir" in tbt)
        print("tb has write_bin_both:", "write_bin_both" in tbt)
        tcl = build / "hls" / "run_hls.tcl"
        tclt = read(tcl)
        print("tcl has csim_design:", "csim_design" in tclt)
        print("tcl has output dir argv:", "training_hls_outputs" in tclt)
        stdout = build / "hls" / "logs" / "vitis_hls_stdout.log"
        st = read(stdout)
        markers = [line for line in st.splitlines() if "[TB-TRAIN]" in line or "CSim" in line or "csim" in line]
        print("stdout markers:")
        for line in markers[-30:]:
            print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
