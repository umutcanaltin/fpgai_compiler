# External VHDL integration foundation

FPGAI now has a first-class external VHDL project-generation contract. The initial `scalar_stream_v1` ABI stages contributor-owned VHDL sources read-only, emits a compiler-owned wrapper, writes a Vivado synthesis Tcl project, and records provenance in `external_vhdl_integration.json`.

This is the E6A foundation. It does not yet claim mixed HLS/VHDL graph composition or numeric RTL simulation; those are subsequent validation stages.
