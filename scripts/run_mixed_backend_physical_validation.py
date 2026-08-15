from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from fpgai.implementations import implementation_contract_from_manifest
from fpgai.implementations.mixed_backend import (
    MixedBackendPhysicalRequest,
    emit_mixed_backend_physical_project,
    run_mixed_backend_physical_project,
)


def _run_hls(root: Path, *, part: str, period_ns: float, vitis_hls: str) -> Path:
    hls = root / "hls_scalar_stage"
    src = hls / "src"
    src.mkdir(parents=True, exist_ok=True)
    source = Path("examples/mixed_backend/scale2_hls/scale2_hls.cpp").resolve()
    shutil.copy2(source, src / source.name)
    tcl = hls / "run_hls.tcl"
    tcl.write_text(
        f'''open_project -reset fpgai_mixed_hls\nset_top scale2_hls\nadd_files ./src/scale2_hls.cpp\nopen_solution -reset sol1\nset_part {part}\ncreate_clock -period {period_ns}\ncsynth_design\nexit\n''',
        encoding="utf-8",
    )
    logs = root / "reports"
    logs.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([vitis_hls, "-f", str(tcl)], cwd=hls, text=True, capture_output=True)
    (logs / "mixed_backend_vitis_hls_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (logs / "mixed_backend_vitis_hls_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"Vitis HLS failed with return code {proc.returncode}; inspect {logs}")
    rtl = hls / "fpgai_mixed_hls" / "sol1" / "syn" / "verilog"
    if not rtl.is_dir():
        raise SystemExit(f"HLS RTL directory not found: {rtl}")
    return rtl


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate one physical HLS-RTL -> VHDL-RTL pipeline in Vivado")
    ap.add_argument("--out", default="build/mixed_backend_physical_validation")
    ap.add_argument("--part", default="xck26-sfvc784-2LV-c")
    ap.add_argument("--clock-period-ns", type=float, default=5.0)
    ap.add_argument("--vitis-hls", default="vitis_hls")
    ap.add_argument("--vivado", default="vivado")
    args = ap.parse_args()

    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rtl = _run_hls(root, part=args.part, period_ns=args.clock_period_ns, vitis_hls=args.vitis_hls)
    contract = implementation_contract_from_manifest(Path("examples/packages/scale_bias_vhdl"))
    project = emit_mixed_backend_physical_project(
        MixedBackendPhysicalRequest(
            out_dir=root,
            hls_rtl_dir=rtl,
            hls_top="scale2_hls",
            vhdl_contract=contract,
            part=args.part,
            clock_period_ns=args.clock_period_ns,
        )
    )
    print("project_ok:", project.ok)
    if not project.ok:
        print("issues:", [issue.to_dict() for issue in project.issues])
        return 2
    result = run_mixed_backend_physical_project(project, vivado_executable=args.vivado)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
