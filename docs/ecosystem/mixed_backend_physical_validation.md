# Mixed HLS/VHDL physical validation

FPGAI now has a maintained physical-validation path for one HLS-generated RTL stage connected directly to an external VHDL stage inside the same Vivado mixed-language project.

The first validated ABI profile is intentionally narrow: signed 16-bit `valid + data`, single input, single output. The maintained experiment synthesizes an HLS `scale2_hls` kernel, stages the generated Verilog RTL, stages the package VHDL RTL, generates a SystemVerilog bridge, runs XSim numeric validation (`7 -> 14 -> 14`), then runs Vivado synthesis and writes utilization/timing reports.

This is a physical backend validation mechanism, not a claim that arbitrary HLS/VHDL boundaries are supported. Wider tensors, AXI-stream, multiple ports, backpressure and compiler graph insertion remain explicit later profiles.
