# Experimental ONNX → AXI DMA → PYNQ-Z2 board-bitstream pilot

**Status (2026-11-04): SOURCE ADDED; NEVER TESTED WITH NATIVE VITIS/VIVADO OR ON A PHYSICAL PYNQ-Z2.** This is an experimental, fail-closed deployment path, not an already qualified board release. It never modifies old OOC experiment folders.

The input is an existing *source-attested, CSim-qualified* FPGAI case and the exact source ONNX model. Frozen weights are checked against its Model IR/source manifest. New HLS synthesis is performed for the **PYNQ-Z2's xc7z020clg400-1**, regardless of the earlier OOC case's KV260 part.

**2026-11-04 local regression:** 1,926 tests passed and 21 skipped on the
user's workstation (feature branch f430ded), but no PYNQ-Z2 Vivado board
part is installed. The first native **HLS-only** AXI IP export can now run
without `--board-part`; it requires only the known Zynq-7020 silicon.
Full-board Vivado compilation still requires the correct board preset.

## First real native HLS diagnostic and next qualification gate

The user's 2026-11-04 workstation passed **35 focused source-only tests**,
and Vivado 2023.2 successfully enumerated the installed board-part preset
`tul.com.tw:pynq-z2:part0:1.0` from the user-owned board repository.

The first original Vitis HLS board-wrapper attempt
(`build/board_pynq_toy_hls_nopreset_01`) **failed** at CSim C++
compilation with `error: ambiguous overload for operator!=` in
`fpgai_dma_tb.cpp`, comparing an `ap_range_ref` with an
`af_range_ref`. Its original logs and receipt must be preserved. The
compiler now generates explicit, equally sized `ap_uint<W>` temporaries
for both raw bit fields before equality comparison. Source-only regression
checks cover W=8,16,24,32; **this fix is not yet native-tested**.

After updating to this revision and running focused tests, repeat the
native HLS stage in a **fresh directory**:

~~~bash
fpgai-board \\
  --onnx build/demo/model.onnx \\
  --case build/neuro_ooc_02 \\
  --output build/board_pynq_toy_hls_nopreset_02 \\
  --board pynq-z2 \\
  --clock-ns 10 \\
  --vitis /tools/Xilinx/Vitis_HLS/2023.2 \\
  --stage hls --execute-native
~~~

Do not proceed to full Vivado block design/bitstream until the original
Vitis CSim marker, C synthesis, IP catalog export and source-attested
receipt all pass. If the next native stage fails, inspect **its own**
`dma_hls/logs/stdout.log` and `stderr.log` rather than editing original
evidence or suppressing tool diagnostics.

## Hardware path and interface

PS DDR ↔ PS S_AXI_HP0 ↔ SmartConnect ↔ AXI DMA (MM2S/S2MM) ↔ AXI4-Stream adapter ↔ original FPGAI model; PS M_AXI_GP0 controls DMA and accelerator AXI-Lite registers.

The adapter uses one 32-bit AXI4-Stream beat per tensor element. Low W bits are raw signed `ap_fixed<W,I,AP_RND,AP_SAT>` bits, high bits unused; AXI TKEEP/TSTRB are all four bytes. Output TLAST is asserted on the final output word. A single `ap_ctrl_hs` start runs one full static input tensor and produces one full output tensor. This is *not* a variable-length streaming protocol or a runtime-updatable weight memory. On malformed input frames, the host must reset/reinitialize the DMA; no automatic resynchronization is implemented.

The board builder generates `fpgai_dma.cpp`, its independent CSim testbench, `run_dma_hls.tcl` (CSim/CSynth/IP Catalog export), and `run_board.tcl`. The Vivado script creates the **full** board project with a PYNQ-Z2 preset, Zynq PS7, AXI DMA, AXI-Lite control, DDR via PS HP0, one 100 MHz PL clock/reset, design validation and implementation to bitstream. If successful it exports `board_artifacts/fpgai_dma.bit` plus matching `fpgai_dma.hwh` and a standalone PYNQ runtime. It does not quietly weaken Vivado DRC or timing.

## 1. Find the installed board part

On your workstation, with Vivado 2023.2:

~~~bash
/tools/Xilinx/Vivado/2023.2/bin/vivado -mode batch \
  -source scripts/list_pynq_z2_board_parts.tcl
~~~

One documented board-part identifier is `tul.com.tw:pynq-z2:part0:1.0`. **Only use an identifier that your own Vivado installation actually lists**. If nothing is listed, install and verify appropriate PYNQ-Z2 board files first. Do not replace the preset with a KV260 configuration or assume a part-only PS config provides correct DDR/MIO.

### Install or register PYNQ-Z2 board definitions without administrator access

AMD Vivado 2023.2 supports third-party board repositories using
`set_param board.repoPaths`. FPGAI's board discovery and full Vivado
scripts now recognize the explicitly configured `FPGAI_BOARD_REPO`
environment variable; neither script copies vendor files into your Vivado
installation.

Obtain the PYNQ-Z2 board definition from an appropriate source and
**inspect the files and license** before using them. The public
[board-file collection](https://github.com/xupsh/pynq-supported-board-file)
contains `pynq-z2/A.0/board.xml`, `preset.xml` and
`part0_pins.xml`, with the TUL vendor and Zynq-7020 part. A local,
user-owned copy can be registered as follows:

~~~bash
mkdir -p ~/Desktop/fpga_board_defs
git clone https://github.com/xupsh/pynq-supported-board-file.git \
  ~/Desktop/fpga_board_defs/pynq-supported-board-file

export FPGAI_BOARD_REPO="$HOME/Desktop/fpga_board_defs/pynq-supported-board-file"
/tools/Xilinx/Vivado/2023.2/bin/vivado -mode batch \
  -source scripts/list_pynq_z2_board_parts.tcl
~~~

Verify the repository actually contains `pynq-z2/A.0/board.xml`,
`preset.xml`, and `part0_pins.xml` before running Vivado. The *same
exported environment variable* must remain set when running
`fpgai-board --stage bitstream`. If Vivado does not discover a board part,
stop and correct repository location or vendor-file compatibility first.

See [AMD Vivado 2023.2 third-party board repositories](https://docs.amd.com/r/2023.2-English/ug895-vivado-system-level-design-entry/Elements-of-the-Platform-Board-Flow)
and [PYNQ board-settings documentation](https://pynq.readthedocs.io/en/v2.7.0/overlay_design_methodology/board_settings.html).
A trusted board-definition source and matching PS preset are essential;
the target FPGA part alone is not a DDR/board configuration.

## 2. Read-only plan

Use the small, already source-qualified MatMul/ReLU experiment before a CNN:

~~~bash
cd ~/Desktop/github_projects/fpgai_compiler
source .venv/bin/activate
python -m pip install --no-deps -e .

fpgai-board \
  --onnx build/demo/model.onnx \
  --case build/neuro_ooc_02 \
  --output build/board_pynq_toy_01 \
  --board pynq-z2 \
  --board-part 'YOUR_VERIFIED_PYNQ_Z2_BOARD_PART' \
  --clock-ns 10 \
  --vitis /tools/Xilinx/Vitis_HLS/2023.2 \
  --vivado /tools/Xilinx/Vivado/2023.2 \
  --plan
~~~

Plan mode does not create files, invoke vendor programs or use GitHub Actions.

## 3. Native HLS export, then full board build

**You can do HLS export even when board-part discovery reported
`NO_PYNQ_Z2_BOARD_PART_INSTALLED`.** For this source-qualified HLS-only
stage, omit `--board-part` (the target silicon remains explicit):

~~~bash
fpgai-board \
  --onnx build/demo/model.onnx \
  --case build/neuro_ooc_02 \
  --output build/board_pynq_toy_hls_no_preset_01 \
  --board pynq-z2 \
  --clock-ns 10 \
  --vitis /tools/Xilinx/Vitis_HLS/2023.2 \
  --stage hls --execute-native
~~~

This runs Vitis HLS C simulation, synthesis and IP-Catalog export for the
Zynq-7020, but **not** Vivado's PS, DMA, DDR integration or a board
bitstream. It can expose real AXI wrapper/RTL interface problems before
spending time installing the third-party board files.

The full bitstream stage below remains blocked until Vivado reports an
installed PYNQ-Z2 board part.

First qualify the new AXI wrapper and the real PYNQ-Z2 IP export:

~~~bash
fpgai-board \
  --onnx build/demo/model.onnx \
  --case build/neuro_ooc_02 \
  --output build/board_pynq_toy_hls_01 \
  --board pynq-z2 \
  --board-part 'YOUR_VERIFIED_PYNQ_Z2_BOARD_PART' \
  --vitis /tools/Xilinx/Vitis_HLS/2023.2 \
  --stage hls --execute-native
~~~

Only after that passes, create an **independent fresh** full-board run:

~~~bash
fpgai-board \
  --onnx build/demo/model.onnx \
  --case build/neuro_ooc_02 \
  --output build/board_pynq_toy_bit_01 \
  --board pynq-z2 \
  --board-part 'YOUR_VERIFIED_PYNQ_Z2_BOARD_PART' \
  --vitis /tools/Xilinx/Vitis_HLS/2023.2 \
  --vivado /tools/Xilinx/Vivado/2023.2 \
  --stage bitstream --execute-native
~~~

The default rejects non-clean full-board DRC. If actual Vivado board reports contain **only** source-listed `DPIP-2`, `DPOP-3`, `DPOP-4` and `RTSTAT-10` warning-level advisories, with no critical, unknown or unconstrained-I/O rules and passing timing, you can choose **a new independent build** with `--accept-board-advisories`. Its status will be `BOARD_BITSTREAM_WITH_ADVISORIES_NOT_DEVICE_TESTED`, *not DRC-clean*. Do not enable it before inspecting the actual board reports.

A successful *strict* build status is `BOARD_BITSTREAM_DRC_CLEAN_UNTESTED_ON_DEVICE`. A finished process or existing E4 OOC receipt alone never qualifies a new board design.

## 4. Make host-side NumPy input and independent output

~~~bash
python examples/make_sample_input.py \
  build/demo/model.onnx build/demo/toy_input.npy

fpgai-board-reference \
  --onnx build/demo/model.onnx \
  --input build/demo/toy_input.npy \
  --output build/demo/toy_expected.npy
~~~

The reference is generated on your PC from the source Model IR. Its input and output shapes must match the ONNX network exactly, with float32 data. Keep private model weights off public Git repositories.

## 5. Actual PYNQ-Z2 DMA inference

**Only after native HLS, Vivado, and real bitstream/HWH verification passes**, copy the `board_artifacts/` bundle plus `toy_input.npy` and `toy_expected.npy` onto the physical board using your approved transfer method.

The board runs a compatible PYNQ Linux image with NumPy and the PYNQ Overlay/DMA/allocate APIs. Run the **standalone** script on the board (no compiler or ONNX package needed):

~~~bash
python3 board_artifacts/fpgai_pynq_runtime.py \
  --bit board_artifacts/fpgai_dma.bit \
  --build-summary board_artifacts/board_build_summary.json \
  --input toy_input.npy \
  --expected toy_expected.npy \
  --output toy_observed.npy \
  --confirm-pynq-z2
~~~

For an accepted warning-only board image, also add `--accept-board-advisories` after reviewing the DRC. The runtime checks source-pinned .bit/.hwh hashes and shapes before programming, uses PYNQ contiguous DMA buffers, invokes AXI-Lite AP_START, polls both DMA channels with a timeout, decodes fixed point, saves actual output and checks an independent reference.

The result `on_device_numerics_verified: true` requires a **real** DMA transfer and numerical pass on hardware. Without a supplied reference, it is explicitly unverified. Time spent includes PS programming/DMA overhead; it is not accelerator-only inference latency. A DMA hang or missing TLAST must be diagnosed on the board and may require reset.

## What remains

- **Native PYNQ-Z2 results and physical on-board measurements have not been executed by the assistant.** The local CPU regression suite has passed (1,926 passed, 21 skipped). Next run the HLS-only IP export independently, then the full board builder after board-part installation. Report the first failure and original logs for corrections on this branch.
- The experimental PYNQ-Z2 profile is fixed to 100 MHz PL clock and uses 32-bit AXI stream words. Large CNNs may not meet Zynq-7020 resource/timing budgets.
- **KV260 is not yet a valid target for this script:** its Kria MPSoC platform, PS HPM/HPC/HP interfaces and Linux device-tree/firmware/runtime require a separate verified recipe. A PYNQ overlay is not interchangeable with the standard KV260 Linux stack.
- Board-specific clocks/resets, address map, DMA driver, bitstream, .hwh and numerical tests must each succeed before claiming 'directly runnable'.

Engineering references: AMD Vitis HLS 2023.2 UG1399 (AXI4-Stream and IP-Catalog export), Vivado IP Integrator, PYNQ DMA and PYNQ overlay .bit/.hwh documentation.
