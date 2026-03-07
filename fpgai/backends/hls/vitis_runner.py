from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess


@dataclass
class VitisCsimResult:
    hls_dir: Path
    tcl_path: Path
    sol_dir: Path


def _run(cmd: str, *, cwd: Path, verbose: bool = False) -> None:
    if verbose:
        print("[CMD]", cmd)
        print("[CWD]", cwd)
    subprocess.check_call(cmd, cwd=str(cwd), shell=True)


def _emit_run_hls_tcl(
    *,
    top_name: str,
    part: str,
    clock_period_ns: float,
    include_dir: Path,
    src_dir: Path,
    input_bin: Path,
    weights_mode: str,
    weights_bin: Optional[Path],
) -> str:
    layers_inc_dir = include_dir / "layers"

    # Gather design sources that exist
    sources: list[Path] = []

    # Top file could be either <top>.cpp or deeplearn.cpp etc.
    top_cpp = src_dir / f"{top_name}.cpp"
    if top_cpp.exists():
        sources.append(top_cpp)
    else:
        # fall back: some generators always write "deeplearn.cpp"
        alt = src_dir / "deeplearn.cpp"
        if alt.exists():
            sources.append(alt)

    params_cpp = src_dir / "fpgai_params.cpp"
    if params_cpp.exists():
        sources.append(params_cpp)

    layers_dir = src_dir / "layers"
    if layers_dir.exists():
        for p in sorted(layers_dir.glob("*.cpp")):
            sources.append(p)

    tb_cpp = src_dir / "tb.cpp"
    if not tb_cpp.exists():
        raise FileNotFoundError(f"missing testbench: {tb_cpp}")

    if not sources:
        raise FileNotFoundError(f"No design sources found in: {src_dir}")

    cflags = f"-I{include_dir} -I{layers_inc_dir}"

    add_lines = []
    for p in sources:
        add_lines.append(f'add_files "{str(p)}" -cflags "{cflags}"')

    add_tb_line = f'add_files -tb "{str(tb_cpp)}" -cflags "{cflags}"'

    # csim args: pass input.bin always; pass weights.bin only in stream mode
    if weights_mode == "stream":
        if not weights_bin or not weights_bin.exists():
            raise FileNotFoundError(f"weights_mode=stream but weights.bin missing: {weights_bin}")
        argv = f'"{str(input_bin)} {str(weights_bin)}"'
    else:
        argv = f'"{str(input_bin)}"'

    tcl = f"""
open_project -reset fpgai_hls_proj
set_top {top_name}

{chr(10).join(add_lines)}
{add_tb_line}

open_solution -reset sol1
set_part {part}
create_clock -period {clock_period_ns:.6f} -name default

csim_design -argv {argv}
exit
"""
    return tcl.strip() + "\n"


def run_vitis_csim(
    *,
    hls_dir: Path,
    top_name: str,
    part: str,
    clock_period_ns: float,
    vitis_dir: Path,
    weights_mode: str = "embedded",
    verbose: bool = False,
) -> VitisCsimResult:
    """
    Assumes your HLS tree layout:
      hls/
        include/
        src/
        run_hls.tcl (will be written)
    """
    include_dir = hls_dir / "include"
    src_dir = hls_dir / "src"

    input_bin = hls_dir.parent / "input.bin"  # out_dir/input.bin
    if not input_bin.exists():
        # also accept hls/input.bin
        alt = hls_dir / "input.bin"
        if alt.exists():
            input_bin = alt
        else:
            raise FileNotFoundError(f"input.bin not found: {input_bin}")

    weights_bin = hls_dir.parent / "weights.bin"
    if not weights_bin.exists():
        weights_bin = None

    tcl_text = _emit_run_hls_tcl(
        top_name=top_name,
        part=part,
        clock_period_ns=clock_period_ns,
        include_dir=include_dir,
        src_dir=src_dir,
        input_bin=input_bin,
        weights_mode=weights_mode,
        weights_bin=weights_bin,
    )

    tcl_path = hls_dir / "run_hls.tcl"
    tcl_path.write_text(tcl_text, encoding="utf-8")

    settings = vitis_dir / "settings64.sh"
    vitis_hls = vitis_dir / "bin" / "vitis_hls"
    if not settings.exists():
        raise FileNotFoundError(f"settings64.sh not found: {settings}")
    if not vitis_hls.exists():
        raise FileNotFoundError(f"vitis_hls not found: {vitis_hls}")

    cmd = f"bash -lc 'source {settings} && {vitis_hls} -f {tcl_path.name}'"
    _run(cmd, cwd=hls_dir, verbose=verbose)

    sol_dir = hls_dir / "fpgai_hls_proj" / "sol1"
    return VitisCsimResult(hls_dir=hls_dir, tcl_path=tcl_path, sol_dir=sol_dir)
