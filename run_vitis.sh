#!/bin/bash

# Source the Vitis environment setup script
source /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh

# Run the Vitis HLS command with the TCL script
vitis_hls -f /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/generated_tcl_vitis.tcl
