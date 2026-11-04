# Experimental ONNX → AXI DMA → PYNQ-Z2 board-bitstream pilot

**Status (2026-11-04): SOURCE ADDED; NEVER TESTED WITH NATIVE VITIS/VIVADO OR ON A PHYSICAL PYNQ-Z2.** This is an experimental, fail-closed deployment path, not an already qualified board release. It never modifies old OOC experiment folders.

The input is an existing *source-attested, CSim-qualified* FPGAI case and the exact source ONNX model. Frozen weights are checked against its Model IR/source manifest. New HLS synthesis is performed for the **PYNQ-Z2's xc7z020clg400-1**, regardless of the earlier OOC case's KV260 part.

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

- **Native vendor results and physical on-board measurements have not been executed by the assistant.** Run source-only tests on your workstation, then the HLS pilot, then the full board builder. Report the first failure and original logs for corrections on this branch.
- The experimental PYNQ-Z2 profile is fixed to 100 MHz PL clock and uses 32-bit AXI stream words. Large CNNs may not meet Zynq-7020 resource/timing budgets.
- **KV260 is not yet a valid target for this script:** its Kria MPSoC platform, PS HPM/HPC/HP interfaces and Linux device-tree/firmware/runtime require a separate verified recipe. A PYNQ overlay is not interchangeable with the standard KV260 Linux stack.
- Board-specific clocks/resets, address map, DMA driver, bitstream, .hwh and numerical tests must each succeed before claiming 'directly runnable'.

Engineering references: AMD Vitis HLS 2023.2 UG1399 (AXI4-Stream and IP-Catalog export), Vivado IP Integrator, PYNQ DMA and PYNQ overlay .bit/.hwh documentation.
