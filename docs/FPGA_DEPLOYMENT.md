# From generated model C++ to an FPGA application

**FPGAI Research is an ONNX-to-HLS research compiler, not an automatic
board deployment system.** This page explains the missing stages and
which ones the repository implements today. A working Vitis/Vivado
installation and an FPGA board do not alone make `model.cpp` runnable
on programmable logic.

## Experimental, unverified PYNQ-Z2 AXI DMA board pilot

An **optional new feature-branch prototype** generates a 32-bit AXI4-Stream
interface around the original model, performs a fresh Vitis HLS export,
creates a PYNQ-Z2 PS7/AXI DMA full-system block design, and attempts to
produce a complete Vivado .bit/.hwh overlay. The bundled PYNQ-side runtime
is intended to transfer original fixed-point model input/output and compare
against an independent software reference.

**This source has not yet been run through native board Vitis/Vivado or
verified on a physical PYNQ-Z2.** It does not change or certify the existing
one-command OOC flow, nor does it work on KV260. See the [PYNQ-Z2 DMA board
pilot](PYNQ_Z2_BOARD_DMA_PILOT.md). The rest of this page describes the
already-established research OOC pipeline and the prior manual requirements.

## What FPGAI produces today

```text
Static supported ONNX model
          |
          v
FPGAI typed Model IR + architecture resolution
          |
          v
model.cpp / manifest.json
          |
          +-- optional GCC host numerical reference
          |
          +-- explicit Vitis HLS CSim and C synthesis
          |                |
          |                v
          |          HLS-generated Verilog
          |                |
          |                v
          |       Vivado out-of-context
          |       synth / place / route
          |       timing, utilization, DRC
          |
          +-- NOT a complete board application
```

The current generated top-level is `fpgai_model`, with **one static
input tensor and one static output tensor**, compiled to
`ap_fixed<...>` array interfaces. The compiler does **not** insert a
complete Linux host interface, physical DDR controller, AXI DMA,
processor integration, board clock/reset, device tree, driver or
bitstream workflow around these arrays.

The repository's existing Vivado research script uses
`synth_design -mode out_of_context` followed by place/route/report
commands. It does **not** invoke `write_bitstream`. A routed checkpoint
or HLS synthesis report cannot be programmed as though it were a
complete board image.

## Which users can realistically run what?

| User environment | Available now | Additional work needed |
|---|---|---|
| Laptop without FPGA tools | Import supported ONNX, emit HLS C++, host numerical checks | None for source-only research |
| Workstation with AMD Vitis HLS | Testbench C simulation, generate RTL, estimated cycles/resources | Licensed compatible HLS setup and verified numerical/timing outcomes |
| Workstation with Vivado | Out-of-context placement/routing and timing/resource reports | Prior generated RTL, compatible device, original constraints |
| FPGA engineer with KV260, KR260 or Zynq board | Build a custom embedded hardware/software design using HLS IP | IP export, PS/PL AXI/DMA, clocks, resets, address map, bitstream and software |
| FPGA engineer with PYNQ board | Create an appropriate full Vivado/PYNQ overlay | Board-compatible project, bitstream + HWH metadata and control/data transfer code |
| Datacenter accelerator developer | Build a compatible Vitis kernel/platform application | Kernel packaging, supported accelerator platform, `.xclbin`, host/XRT integration |

This repository does **not** claim to have validated a one-command
bitstream/board deployment for any of those boards.

## Automated model-level evaluation

The local one-command workflow can explicitly return a separate `OOC_ROUTED_WITH_ADVISORIES` *warning-only* result when supported, original source-attested routing and timing pass with only narrowly recognized DRC advisories. This is not strict DRC-clean qualification, hardware execution or deployment. The default remains fail-closed; see [the onboarding guide](NEUROSCIENTIST_QUICKSTART.md#routed-results-with-known-advisory-drc-warnings) for the precise opt-in gates.

For researchers who want to avoid manual Vitis/Vivado interaction, [One-command neuroscientist workflow](NEUROSCIENTIST_QUICKSTART.md) automates the source, Vitis HLS and out-of-context Vivado stages with explicit `--execute-native` authorization. The stages below remain available as advanced manual/research controls. Neither workflow creates a board-programmable image.

## Step 1 — Run the source compiler

Use a synthetic network and generate HLS C++. The
[network gallery](../examples/NETWORK_GALLERY.md) contains convolution,
strided convolution, pooling, dense, and Softmax examples:

```bash
python -m pip install -e ".[onnx]"
mkdir -p build/board_demo
python examples/make_neural_network_onnx.py \
  cnn_pool_dense build/board_demo/model.onnx
python examples/make_sample_input.py \
  build/board_demo/model.onnx build/board_demo/input.npy

python examples/compile_neural_network.py \
  build/board_demo/model.onnx \
  --mapping scalar \
  --verify-host \
  --output build/board_demo/source_check
```

A passing GCC host check means the generated C++ agreed with an
independent NumPy reference for the supplied numerical samples. It
**does not establish FPGA compatibility**.

## Step 2 — Prepare and run Vitis HLS

After [configuring AMD tools](AMD_TOOLCHAIN_SETUP.md), create a fresh
case using the canonical native workflow:

```bash
fpgai-research-tools prepare \
  --onnx build/board_demo/model.onnx \
  --input build/board_demo/input.npy \
  --case build/board_demo/hls_case \
  --part xck26-sfvc784-2LV-c \
  --clock-ns 10 \
  --tolerance 0.01
```

These are example numeric and timing requirements, **not certified
timing constraints for a real board**. Never use a part number just
because a board looks similar: confirm the silicon, speed grade,
platform and tool support first.

The generated `run_hls.tcl` includes source, C simulation and HLS
synthesis. After reviewing the Tcl and testbench:

```bash
fpgai-research-tools run-vitis \
  --case build/board_demo/hls_case \
  --executable "${FPGAI_VITIS_HLS_EXECUTABLE:-vitis_hls}"
```

Observe and independently review CSim failures, timing estimates,
resource estimates, supported interfaces and original RTL before
proceeding. HLS CSim success does not prove an implementation will meet
its clock period.

## Inspect the prepared case and missing deployment pieces

After preparing a case, run the **read-only** hardware inventory:

```bash
python scripts/check_fpga_deployment.py \
  --case build/board_demo/hls_case \
  --target kv260
```

Choose `kv260`, `kr260`, `pynq`, `alveo` or `generic`.
The report identifies source/HLS/OOC file presence and lists the
integration steps still required. Even if every detected OOC report
exists, its `complete_board_runtime_verified` field stays `false`;
a report file does not prove device compatibility, HLS numeric
qualification or an actual hardware execution.

## Step 3 — Export HLS RTL as an integrable IP or kernel

**The established research OOC CLI does not perform this step.** The separate experimental
PYNQ-Z2 builder now contains a native IP-export recipe, but it remains
unverified with AMD tools; see [the pilot](PYNQ_Z2_BOARD_DMA_PILOT.md).

In Vitis HLS, configure/export the synthesized component as a **Vivado
IP Catalog component** (for a Vivado block design), or a supported
**Vitis kernel object** (for an acceleration flow). The original
generated research Tcl stops after `csynth_design`; it does not
perform a complete, production-qualified IP export or kernel
packaging.

The correct export mechanism depends on AMD tool version and the
specific target workflow. For Vitis 2023.2, consult AMD's
[Packaging the RTL Design](https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/Packaging-the-RTL-Design)
and [Vivado IP Development Flow](https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/Vivado-IP-Development-Flow).

**Inspect the actual generated interface.** The `fpgai_model` top
expects fixed-point arrays; it is **not currently a verified AXI4-Lite /
AXI4-Stream / memory-mapped DMA accelerator contract**. Depending on
the tool's interface synthesis and your system requirements, you may
need to add and validate a thin interface wrapper, adjust explicit HLS
interface directives, or package/register a compatible IP. That is
hardware design work, not just setting an environment variable.

Do **not** add speculative `#pragma HLS INTERFACE` directives to the
core emitter and claim universal support without separate functional
and hardware verification.

## Step 4A — Embedded board flow: KV260 / KR260 / Zynq

To produce a software-controlled PL accelerator on a supported board,
you need a **complete system project** in Vivado, not the OOC research
checkpoint:

1. Select the exact board/SOM silicon and supported platform/board
   files, processor subsystem and correct external DDR/interconnect.
2. Add the imported HLS IP plus a proven control/data transport
   interface. Examples may use PS↔PL AXI, AXI DMA and a memory-buffer
   contract, depending on the interface selected. **No DMA driver is
   generated by the current FPGAI research compiler.**
3. Define clocks, resets, AXI address assignments and data widths.
   Reconcile the fixed-point representation and tensor element order
   between the host and the generated HLS function.
4. Add board-specific pin/clock constraints and perform full validation,
   synthesis, implementation, DRC and timing closure. Only then
   generate the **full design bitstream** and matching hardware handoff.
5. Prepare a compatible device tree/firmware or runtime stack and
   write the host program that allocates input/output buffers,
   synchronizes memory, starts the accelerator, and reads outputs.
6. Compare real board outputs against an independent software
   reference across multiple inputs. Measure latency separately from
   programming, DMA and host overhead; record frequency and errors.

For embedded Linux users, a bitstream file alone is often insufficient:
the device tree, platform image, UIO/driver bindings and data-moving
software must agree with the exact design. Do not automatically apply
the host/XRT steps for Alveo to Kria Linux images.

**The repo currently has no supported `fpgai-research --flash`,
`--program-fpga`, or `--run-on-kv260` command.** Avoid shell
instructions that pretend these commands exist.

## Step 4B — PYNQ boards

If using a PYNQ-compatible board and image, one common **downstream**
design is a Vivado-generated `.bit` **plus matching** `.hwh`
metadata, installed onto the PYNQ filesystem and loaded as an overlay:

```python
# Illustrative PYNQ-side snippet. Only after you built and validated a
# matching board-specific bitstream/HWH and installed PYNQ on the board.
from pynq import Overlay
overlay = Overlay("/home/xilinx/overlays/fpgai_example/fpgai_example.bit")
print(overlay.ip_dict)
```

The `Overlay` API loads and describes the hardware, **not the model
inference itself**. Register-level or DMA transfers depend entirely
on your exported IP/control interface. Do not assume the example
bitstream or path already exists.

Official references:
[PYNQ overlay loading](https://pynq.readthedocs.io/en/latest/pynq_overlays/loading_an_overlay.html)
and [matching HWH metadata](https://pynq.readthedocs.io/en/latest/overlay_design_methodology/overlay_design.html).

## Step 4C — Alveo and Vitis kernel/XRT flow

For data-center accelerators, an integrated platform flow usually
requires a supported AMD platform, kernel object, linking into
`.xclbin`, PCIe XRT driver/runtime and a host application. Matching
FPGAI-generated HLS array interfaces to the required accelerator
kernel interfaces requires engineering and verification.

**XRT does not compile ONNX or magically convert `model.cpp` into
`.xclbin`**. You must first produce a valid Vitis kernel and link it
for a compatible installed platform. The initial research OOC flow
is not that end-to-end application acceleration flow.

Reference: [AMD Vitis HLS 2023.2](https://docs.amd.com/r/2023.2-English/Vitis-Tutorials-Getting-Started/Vitis-HLS).

## Step 5 — Program and validate hardware

Once you have a **real** bitstream, board-targeted design and software,
board programming may be performed through the official AMD Hardware
Manager or board-specific software loader. The applicable method
depends on board, processor/OS image, and secure-boot settings.

Do not attempt to program a checkpoint, copy a random `.bit` from a
different board, ignore voltage/pin constraints, or suppress timing
and DRC failures simply to obtain an output file.

For Vivado hardware programming instructions, see
[AMD UG908: Programming the Device](https://docs.amd.com/r/2023.2-English/ug908-vivado-programming-debugging/Programming-the-Device).

## Limitations and contribution opportunities

| Missing product feature | Current state |
|---|---|
| One-command bitstream from ONNX | Experimental separate PYNQ-Z2 board build from an already source-qualified ONNX case; no native/board proof yet |
| Board selection and board-level I/O/AXI metadata | Experimental PYNQ-Z2-specific source template and explicit external board files, not a verified platform system |
| Generated host AXI/DMA drivers | Experimental standalone PYNQ overlay + AXI DMA Python runtime, not yet verified on board |
| Device tree / kernel module generator | Not implemented |
| FPGA programming and runtime inference command | Experimental fpgai-pynq-run / standalone PYNQ runtime requires valid board bitstream and physical testing |
| Board-level power/latency validation | Not established by existing source/HLS studies |

Contributors can propose board-specific wrappers, manifests, host
reference implementations and validation protocols through
[Contributing](../CONTRIBUTING.md), provided they preserve source
correctness and never expose third-party/restricted IP.

### Practical distinction

**Compile a model** → create HLS C++.

**Synthesize the model** → obtain RTL and HLS estimate.

**Implement the logic** → place/route in an FPGA context.

**Run the model on hardware** → create an entire compatible bitstream,
control/data interface and host runtime, then verify results on the
board.

These are four distinct engineering achievements, and no earlier step
is evidence that the later steps have happened.
