from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import yaml


def run(cmd: str, *, cwd: Path, verbose: bool) -> None:
    if verbose:
        print("[CMD]", cmd)
        print("[CWD]", cwd)
    subprocess.check_call(cmd, cwd=str(cwd), shell=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fpgai.yml")
    ap.add_argument("--regen", action="store_true", help="Regenerate artifacts via compiler before running csim")
    ap.add_argument("--vitis-dir", default=None, help="Path to Vitis_HLS install, e.g. /tools/Xilinx/Vitis_HLS/2023.2 (or env VITIS_HLS_DIR)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config).resolve()

    raw = yaml.safe_load(cfg_path.read_text())

    out_dir = Path(raw.get("project", {}).get("out_dir", "build/fpgai")).resolve()
    top_name = raw.get("pipeline", {}).get("outputs", {}).get("top_kernel_name", "deeplearn")

    weights_mode = (
        raw.get("data_movement", {})
           .get("ps_pl", {})
           .get("weights", {})
           .get("mode", "embedded")
    )
    weights_mode = str(weights_mode).lower()

    vitis_dir = args.vitis_dir or os.environ.get("VITIS_HLS_DIR") or "/tools/Xilinx/Vitis_HLS/2023.2"
    vitis_dir = Path(vitis_dir)

    settings = vitis_dir / "settings64.sh"
    vitis_hls = vitis_dir / "bin" / "vitis_hls"

    if not settings.exists():
        raise SystemExit(f"ERROR: settings64.sh not found: {settings}\nSet --vitis-dir or VITIS_HLS_DIR")

    if not vitis_hls.exists():
        raise SystemExit(f"ERROR: vitis_hls not found: {vitis_hls}\nSet --vitis-dir or VITIS_HLS_DIR")

    if args.regen:
        run(f"PYTHONPATH=. python {repo_root/'main.py'} --config {cfg_path}", cwd=repo_root, verbose=args.verbose)

    hls_dir = out_dir / "hls"
    tcl_path = hls_dir / "run_hls.tcl"
    if not tcl_path.exists():
        raise SystemExit(f"ERROR: run_hls.tcl not found: {tcl_path} (did you emit HLS?)")

    input_bin = out_dir / "input.bin"
    if not input_bin.exists():
        raise SystemExit(f"ERROR: input.bin not found: {input_bin}")

    weights_bin = out_dir / "weights.bin"  # you should generate/copy here if stream-mode

    # Patch TCL argv line to pass correct args
    if weights_mode == "stream":
        if not weights_bin.exists():
            raise SystemExit(
                f"ERROR: weights.bin not found: {weights_bin}\n"
                f"Hint: for stream mode, place weights.bin into out_dir or update script."
            )
        argv = f'{weights_bin} {input_bin}'
    else:
        argv = f'{input_bin}'

    tcl = tcl_path.read_text()
    new_lines = []
    for line in tcl.splitlines():
        if line.strip().startswith("csim_design"):
            new_lines.append(f'csim_design -argv "{argv}"')
        else:
            new_lines.append(line)
    tcl_path.write_text("\n".join(new_lines) + "\n")

    cmd = f"bash -lc 'source {settings} && {vitis_hls} -f run_hls.tcl'"
    run(cmd, cwd=hls_dir, verbose=args.verbose)

    # If csim succeeded, you can also run the generated csim.exe directly:
    csim_exe = hls_dir / "fpgai_hls_proj" / "sol1" / "csim" / "build" / "csim.exe"
    if csim_exe.exists():
        print("[OK] csim.exe:", csim_exe)


if __name__ == "__main__":
    main()
