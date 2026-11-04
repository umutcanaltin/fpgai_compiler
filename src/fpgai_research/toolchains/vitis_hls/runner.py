"""Opt-in Vitis HLS runner."""
from pathlib import Path
from fpgai_research.toolchains.runner import run_tool

def run_vitis(directory: str | Path, *, executable: str = "vitis_hls",
              timeout_seconds: int = 7200,
              environment: dict[str, str] | None = None) -> dict:
    return run_tool(tool="vitis_hls",argv=["-f","run_hls.tcl"],
                    directory=directory,executable=executable,
                    timeout_seconds=timeout_seconds,environment=environment)
