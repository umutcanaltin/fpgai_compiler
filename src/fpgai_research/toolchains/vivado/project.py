"""Opt-in standalone Verilog implementation in Vivado, not a VHDL backend."""
from pathlib import Path
import math
import re

PART = "xck26-sfvc784-2LV-c"

def make_vivado_tcl(*, part: str = PART, clock_ns: float = 5.0,
                    top: str = "fpgai_model") -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.+\-]+",part):
        raise ValueError("Invalid FPGA part")
    if not math.isfinite(clock_ns) or clock_ns <= 0:
        raise ValueError("Invalid clock_ns")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",top):
        raise ValueError("Invalid Verilog top")
    return f'''# Model-only research: out-of-context (OOC) implement Vitis HLS Verilog.
# These HLS ports are internal module interfaces, NOT physical KV260 I/O pins.
# OOC mode disables I/O buffer insertion; never override package-pin KLOC DRC.
# No board-level I/O constraints, bitstream deployment or physical I/O timing.
# ROM_READ_ATTESTATION_PROTOCOL_V1: prospective studies only. Vendor
# successful read is independently hash-checked if a working ROM vanishes.
# This is NOT an assertion that the ROM file remains present after Vivado.
# DRC_WARNING_OBSERVATION_PROTOCOL_V1: accept no waivers, suppression or
# changes in Vivado DRC severity. The independent auditor may publish
# routed timing/utilization as a DISCLOSED WARNING-ONLY observation,
# never as strict DRC-clean E4 or board-deployment qualification.
set rtl_dir [file normalize [file join [pwd] .. fpgai_hls_project solution1 syn verilog]]
set sources [concat [glob -nocomplain -directory $rtl_dir *.v] [glob -nocomplain -directory $rtl_dir *.sv]]
if {{[llength $sources] == 0}} {{ error "No Vitis-generated Verilog in $rtl_dir" }}
file mkdir reports
create_project -force research_implementation ./research_implementation -part {part}
foreach source $sources {{
  if {{[string match *.sv $source]}} {{ read_verilog -sv $source }} else {{ read_verilog $source }}
}}
# ROM data originates exclusively from original Vitis-generated RTL.
# $readmemh/$readmemb reference basenames resolved from this cwd.
set rom_sources [concat [glob -nocomplain -directory $rtl_dir *.dat] [glob -nocomplain -directory $rtl_dir *.mem]]
foreach rom $rom_sources {{
  set local_rom [file join [pwd] [file tail $rom]]
  if {{![file isfile $local_rom]}} {{ error "Missing staged HLS ROM file $local_rom" }}
  # Register source-pinned RTL-directory ROM, NOT the cwd readmem copy.
  # The original v4 invocation left its cwd ROM missing after exit 0;
  # why it disappeared is not proven. v5 prospectively isolates the
  # registration from the hash-checked $readmemh working-directory input.
  add_files -fileset sources_1 -norecurse $rom
  set rom_source [get_files -quiet $rom]
  if {{[llength $rom_source] != 1}} {{ error "Vivado failed to register source HLS ROM file $rom" }}
  # Do not set immutable FILE_TYPE for Vivado-classified Data Files.
}}
synth_design -mode out_of_context -top {top} -part {part}
set clocks [get_ports -quiet ap_clk]
if {{[llength $clocks] != 1}} {{ error "Expected a single top-level ap_clk" }}
create_clock -name research_clock -period {clock_ns:g} $clocks
opt_design
place_design
route_design
report_route_status -file reports/route_status.rpt
report_timing_summary -file reports/timing_summary.rpt
report_utilization -file reports/utilization.rpt
report_drc -file reports/drc.rpt
write_checkpoint -force reports/routed_design.dcp
exit
'''

def write_vivado_package(case_directory: str | Path, *, part: str = PART,
                         clock_ns: float = 5.0) -> Path:
    case = Path(case_directory).resolve(strict=True)
    manifest_path = case/"manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Missing source manifest.json")
    import json
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("part",part) != part or manifest.get("clock_ns",clock_ns) != clock_ns:
        raise ValueError("Vivado target part/clock must match the prepared HLS case")
    target = case/"vivado"
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Refusing to overwrite existing Vivado experiment files or evidence")
    script_text=make_vivado_tcl(part=part,clock_ns=clock_ns)
    target.mkdir(exist_ok=True)
    script = target/"run_vivado.tcl"
    script.write_text(script_text)
    return script
