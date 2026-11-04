"""Generate a guarded, project-mode PYNQ-Z2 PS <-> AXI DMA <-> HLS IP design.

This is an EXPERIMENTAL board integration recipe for Vivado 2023.2. It is
not a validated .bit until run, timing/DRC checked and a PYNQ-Z2 board test
passes. User supplies an installed exact PYNQ-Z2 Vivado board_part; no
hard-coded guess about third-party board-file vendor/revision.

The KV260 requires a distinct MPSoC platform/image and is deliberately NOT
misrepresented by the PYNQ-Z2 Zynq-7000 recipe.
"""
from __future__ import annotations
import re

BOARD_PART_RE = re.compile(r"[A-Za-z0-9_.:-]{4,160}\Z", re.ASCII)
PYNQ_Z2_PART = "xc7z020clg400-1"
PYNQ_DMA_CLOCK_NS = 10.0  # 100 MHz PS FCLK_CLK0; fixed in this first recipe.

_PYNQ_TCL = r'''# FPGAI experimental PYNQ-Z2 full-system 2023.2 recipe.
# Inputs: prepared/frozen model.cpp, separately source-synthesized AXI IP.
# All PS pin mux/DDR details MUST come from explicit installed PYNQ-Z2 board_part.
# Never disable safety DRCs, change their severities, or waive timing to get .bit.
# Honor an explicit local third-party board repo, without changing the Vivado install.
# The caller must inspect and trust the board files (PS DDR/MIO presets).
if {[info exists ::env(FPGAI_BOARD_REPO)]} {
  set board_repo [file normalize $::env(FPGAI_BOARD_REPO)]
  if {![file isdirectory $board_repo]} {
    error "FPGAI_BOARD_REPO is not an existing directory"
  }
  set_param board.repoPaths [list $board_repo]
}
set expected_part {@PART@}
set selected_board {@BOARD_PART@}
set installed [get_board_parts -quiet $selected_board]
if {[llength $installed] != 1} {
  error "PYNQ-Z2 board preset is not installed, or board_part is ambiguous: $selected_board"
}
set base [pwd]
set pdir [file normalize [file join $base vivado_project]]
create_project fpgai_pynq $pdir -part $expected_part -force
set_property board_part $selected_board [current_project]
if {[get_property PART [current_project]] ne $expected_part} {
  error "Vivado board part does not match exact PYNQ-Z2 device"
}
set ip_dir [file normalize [file join $base dma_hls fpgai_dma_hls_project solution1 impl ip]]
if {![file isfile [file join $ip_dir component.xml]]} {
  error "Missing original Vitis exported IP catalog component.xml"
}
set_property ip_repo_paths [list $ip_dir] [current_project]
update_ip_catalog -rebuild
set definitions [get_ipdefs -all *:fpgai_dma:*]
if {[llength $definitions] != 1} {
  error "Expected exactly one exported fpgai_dma IP definition"
}
set definition [lindex $definitions 0]

create_bd_design system
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
  -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
  [get_bd_cells ps7]
set_property -dict [list \
  CONFIG.PCW_USE_M_AXI_GP0 {1} \
  CONFIG.PCW_USE_S_AXI_HP0 {1} \
  CONFIG.PCW_EN_CLK0_PORT {1} \
  CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.000000} \
] [get_bd_cells ps7]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list \
  CONFIG.c_include_sg {0} \
  CONFIG.c_sg_length_width {23} \
  CONFIG.c_m_axi_mm2s_data_width {32} \
  CONFIG.c_m_axi_s2mm_data_width {32} \
] [get_bd_cells axi_dma_0]
create_bd_cell -type ip -vlnv $definition fpgai_dma_0
# Explicit contract: HLS AXIS in/out plus AXI-Lite ap_ctrl_hs.
foreach port {axis_in axis_out s_axi_control} {
  if {[llength [get_bd_intf_pins -quiet fpgai_dma_0/$port]] != 1} {
    error "Exported fpgai_dma IP does not expose required $port interface"
  }
}

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 control_sc
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] [get_bd_cells control_sc]
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 ddr_sc
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells ddr_sc]
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_reset
# CONFIG.C_EXT_RESET_HIGH is READ-ONLY in Vivado 2023.2.
# Read back the actual external-reset polarity instead of unsuccessfully
# attempting to set it. The PS FCLK_RESET0_N signal is active-low.
set external_reset_high [string tolower [string trim   [get_property CONFIG.C_EXT_RESET_HIGH [get_bd_cells proc_reset]]]]
if {$external_reset_high ni {0 1 false true}} {
  error "Cannot establish proc_sys_reset external reset polarity: $external_reset_high"
}
puts "FPGAI_PROC_SYS_RESET_EXT_RESET_HIGH=$external_reset_high"

# PS GP0 controls DMA and accelerator. DMA data masters reach DDR via PS HP0.
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins control_sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins control_sc/M00_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins control_sc/M01_AXI] [get_bd_intf_pins fpgai_dma_0/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins ddr_sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins ddr_sc/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins ddr_sc/M00_AXI] [get_bd_intf_pins ps7/S_AXI_HP0]
# The model input/output are streamed through one full-duplex AXI DMA.
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins fpgai_dma_0/axis_in]
connect_bd_intf_net [get_bd_intf_pins fpgai_dma_0/axis_out] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# One proven 100MHz clock domain; no speculative CDC.
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] \
  [get_bd_pins ps7/M_AXI_GP0_ACLK] [get_bd_pins ps7/S_AXI_HP0_ACLK] \
  [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
  [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
  [get_bd_pins fpgai_dma_0/ap_clk] \
  [get_bd_pins control_sc/aclk] [get_bd_pins ddr_sc/aclk] \
  [get_bd_pins proc_reset/slowest_sync_clk]
# Match the actual proc_sys_reset input polarity to the active-low PS reset.
# Use a dedicated 1-bit NOT only when the reset controller requires active-high.
if {$external_reset_high in {1 true}} {
  create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 ps_reset_inverter
  set_property -dict [list CONFIG.C_OPERATION {not} CONFIG.C_SIZE {1}]     [get_bd_cells ps_reset_inverter]
  connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N]     [get_bd_pins ps_reset_inverter/Op1]
  connect_bd_net [get_bd_pins ps_reset_inverter/Res]     [get_bd_pins proc_reset/ext_reset_in]
  puts "FPGAI_PS_RESET_POLARITY_VERIFIED_ACTIVE_HIGH_INVERTED"
} else {
  connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N]     [get_bd_pins proc_reset/ext_reset_in]
  puts "FPGAI_PS_RESET_POLARITY_VERIFIED_ACTIVE_LOW_DIRECT"
}
connect_bd_net [get_bd_pins proc_reset/peripheral_aresetn] \
  [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins fpgai_dma_0/ap_rst_n]
connect_bd_net [get_bd_pins proc_reset/interconnect_aresetn] \
  [get_bd_pins control_sc/aresetn] [get_bd_pins ddr_sc/aresetn]
assign_bd_address
validate_bd_design
save_bd_design

set bd [get_files -quiet *system.bd]
if {[llength $bd] != 1} { error "Expected one complete system block design" }
generate_target all $bd
set wrapper [make_wrapper -files $bd -top]
if {[llength $wrapper] != 1 || ![file isfile [lindex $wrapper 0]]} {
  error "Vivado failed to generate a complete board system wrapper"
}
add_files -norecurse [lindex $wrapper 0]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 2
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
if {![string match "*Complete*" $synth_status]} {
  error "Board-level synthesis did not complete: $synth_status"
}
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
if {![string match "*Complete*" $impl_status]} {
  error "Board-level bitstream implementation did not complete: $impl_status"
}
open_run impl_1
file mkdir [file join $base board_artifacts]
report_timing_summary -file [file join $base board_artifacts timing_summary.rpt]
report_drc -file [file join $base board_artifacts drc.rpt]
report_route_status -file [file join $base board_artifacts route_status.rpt]
report_utilization -file [file join $base board_artifacts utilization.rpt]
set setup_paths [get_timing_paths -setup -max_paths 1]
if {[llength $setup_paths] < 1} {
  error "No valid setup path for board timing qualification"
}
set worst_setup [get_property SLACK [lindex $setup_paths 0]]
if {![string is double -strict $worst_setup] || $worst_setup < 0} {
  error "Board implementation missed routed timing"
}
set impl_dir [get_property DIRECTORY [get_runs impl_1]]
set bits [glob -nocomplain -types f -directory $impl_dir *.bit]
if {[llength $bits] != 1} {
  error "Expected exactly one full board .bit image"
}
set hwhs [glob -nocomplain -types f \
  [file join $pdir fpgai_pynq.gen sources_1 bd system hw_handoff *.hwh]]
if {[llength $hwhs] != 1} {
  error "Expected a matching system.hwh file for PYNQ DMA overlay"
}
file copy -force [lindex $bits 0] [file join $base board_artifacts fpgai_dma.bit]
file copy -force [lindex $hwhs 0] [file join $base board_artifacts fpgai_dma.hwh]
puts "FPGAI_BOARD_ARTIFACTS_EXPORTED"
exit
'''


def pynq_z2_board_tcl(*, board_part: str, part: str, clock_ns: float) -> str:
    if not isinstance(board_part, str) or not BOARD_PART_RE.fullmatch(board_part):
        raise ValueError("Pass an exact, installed Vivado PYNQ-Z2 board_part")
    if part != PYNQ_Z2_PART:
        raise ValueError("PYNQ-Z2 requires xc7z020clg400-1, not a KV260 part")
    if clock_ns != PYNQ_DMA_CLOCK_NS:
        raise ValueError("First PYNQ-Z2 DMA profile has one 100 MHz clock: 10.0 ns")
    return _PYNQ_TCL.replace("@BOARD_PART@", board_part).replace("@PART@", part)
