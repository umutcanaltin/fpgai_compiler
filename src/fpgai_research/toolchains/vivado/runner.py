"""Opt-in Vivado implementation runner."""
from pathlib import Path
from fpgai_research.toolchains.runner import run_tool

def run_vivado(directory: str | Path, *, executable: str = "vivado",
               timeout_seconds: int = 14400,
               allow_verified_readmem_post_absence: bool = False) -> dict:
    root = Path(directory)
    if (root/"vivado"/"run_vivado.tcl").is_file():
        root = root/"vivado"
    return run_tool(tool="vivado",
                    argv=["-mode","batch","-source","run_vivado.tcl","-notrace"],
                    directory=root,executable=executable,
                    timeout_seconds=timeout_seconds,
                    allow_verified_readmem_post_absence=(
                        allow_verified_readmem_post_absence))
