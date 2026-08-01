# Adding a hardware implementation

Create a package with `asset_type: implementation`, declare `implementation.operator_id`, and provide an `entrypoints.implementation` block. HLS C++, VHDL, Verilog, and SystemVerilog use the same research package contract.

E3C validates and selects metadata. Source integration is introduced in E4A and later RTL backend sprints.
