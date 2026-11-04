# Installing and configuring AMD Vitis HLS, Vivado and XRT

FPGAI Research generates HLS-oriented C++ from supported static neural networks. **Only source generation and compiler-side validation work without AMD tools.** Native C simulation, HLS synthesis, Vivado out-of-context implementation and board deployment are separate workflows.

This guide describes the codebase's **currently implemented 2023.2-oriented command-line flow**. Vendor installation, operating-system requirements, device support, target platform, licenses and compatible board software remain the responsibility of the user.

## One-command onboarding option

If you already have Vitis HLS and Vivado installed, the [neuroscientist quickstart](NEUROSCIENTIST_QUICKSTART.md) describes `fpgai-run`, which accepts installation directories and loads AMD settings automatically in a child process. It can compile and route a supported static ONNX model without manually opening vendor GUIs. Actual vendor execution requires `--execute-native`. This is model-level HLS and out-of-context implementation, not board programming.

The manual environment and native commands below remain available for advanced research and troubleshooting.

## Who should use which setup?

| Goal | Requires | Does not require |
|---|---|---|
| Generate a synthetic ONNX model and HLS C++ | Python 3.10+, NumPy, ONNX | Vivado, Vitis, FPGA board |
| Check generated C++ against NumPy | GNU C++ compiler and Python dependencies | Vivado, Vitis, XRT |
| Run Vitis HLS C simulation or synthesis | Compatible **licensed** Vitis HLS installation | Attached FPGA board, XRT in the normal HLS flow |
| Produce **out-of-context** Vivado reports | Vitis-generated RTL, Vivado, exact FPGA part | Board programming or XRT |
| Build a board bitstream | Supported Vivado board/platform integration and complete hardware system | Automatic support from FPGAI's research OOC scripts |
| Run an accelerator through Linux/XRT | Complete compatible hardware platform, deployable kernel/bitstream, target runtime and host program | Source compiler alone |

## Linux shell setup (documented release: 2023.2)

Install the AMD tools yourself following AMD's license and installation instructions. For an installation with **both** tool trees under the same parent, use:

```bash
# Example path only; replace with YOUR actual installation parent.
export FPGAI_AMD_ROOT=/tools/Xilinx
export FPGAI_AMD_VERSION=2023.2

# Set paths for this shell; does NOT run any compilation.
source scripts/setup_amd_environment.sh

command -v vitis_hls
command -v vivado

# Diagnostic: discovers executables but never launches vendor tools.
python scripts/check_amd_environment.py --require-hls --require-vivado
```

Equivalent vendor-provided setup scripts are typically:

```bash
source /tools/Xilinx/Vitis/2023.2/settings64.sh
source /tools/Xilinx/Vivado/2023.2/settings64.sh
```

Use only paths that actually exist on your workstation. Running the settings script normally sets `XILINX_VITIS` and `XILINX_VIVADO` and adds the vendor commands to `PATH`. No changes to `~/.bashrc` are required; loading settings in the current shell is safer when different tool versions coexist.

### Installations outside the usual root

If the products are installed in separate directories, source each **existing** vendor `settings64.sh` manually and set the executable paths for the research scripts:

```bash
# These paths are examples, not values automatically discovered by FPGAI.
source /path/to/your/Vitis/2023.2/settings64.sh
source /path/to/your/Vivado/2023.2/settings64.sh

export FPGAI_VITIS_HLS_EXECUTABLE=/path/to/your/Vitis/2023.2/bin/vitis_hls
export FPGAI_VIVADO_EXECUTABLE=/path/to/your/Vivado/2023.2/bin/vivado

python scripts/check_amd_environment.py --require-hls --require-vivado
```

The native workflow accepts these explicitly in `--executable`. The environment checker reports resolved paths but does not validate an AMD license or run a version command. Check vendor versions separately when needed:

```bash
vitis_hls -version
vivado -version
```

These two commands **execute vendor binaries** to print version information and may require functioning installations. They are not run by the Python environment checker.

## Platforms and optional XRT

For a Vitis acceleration flow using an installed compatible platform:

```bash
export PLATFORM_REPO_PATHS=/absolute/path/to/installed/platforms
python scripts/check_amd_environment.py --require-platform
```

Setting `PLATFORM_REPO_PATHS` is **not enough**: the target `.xpfm` must match the FPGA board, shell/platform version, runtime and kernel interfaces. FPGAI does not bundle a platform file or automatically choose one.

For **XRT-based** datacenter acceleration workflows, the documented AMD setup often uses:

```bash
source /opt/xilinx/xrt/setup.sh
```

This is not required for a simple Vitis HLS source/synthesis workflow or the ordinary OOC Vivado study. The repository's setup helper sources XRT **only when explicitly requested**:

```bash
export FPGAI_SOURCE_XRT=1
export FPGAI_XRT_SETUP=/opt/xilinx/xrt/setup.sh
source scripts/setup_amd_environment.sh
python scripts/check_amd_environment.py --require-xrt
```

On embedded Zynq/Kria/PYNQ designs, PS↔PL integration, operating-system image, device tree, memory mapping, DMA and software drivers are board-/image-specific. Do not assume that setting `XRT` makes the compiled code executable on an embedded board.

## Optional embedded Linux root filesystem and sysroot

An embedded software or Vitis platform project may also need a
version-matched **root filesystem** and **cross-compilation sysroot**.
These are *not required* for FPGAI's source-only compiler, basic
HLS synthesis, or its Vivado out-of-context route:

```bash
# Examples only: use files supplied by your matching board platform.
export ROOTFS=/absolute/path/to/embedded/rootfs.ext4
export SYSROOT=/absolute/path/to/sdk/sysroots/your-target
export SDKTARGETSYSROOT="$SYSROOT"

# Checks existence only. Does not compile, deploy or verify compatibility.
python scripts/check_amd_environment.py \
  --require-rootfs --require-sysroot
```

Target architecture, C library, kernel, image, Vitis version and
`.xpfm` must match. For example, a PYNQ or Kria Linux image may
use a different board/runtime mechanism from a generic ZynqMP Vitis
application. Do not copy sysroot paths from another board merely to
make the check pass.

## Generate a reproducible HLS case with the existing tools

This is a **user-controlled** flow. First generate one of the public synthetic ONNX examples and a matching NumPy input tensor:

```bash
mkdir -p build/native_demo
python examples/make_neural_network_onnx.py \
  cnn_pool_dense build/native_demo/model.onnx

python examples/make_sample_input.py \
  build/native_demo/model.onnx build/native_demo/input.npy
```

Next, **prepare** the existing Vitis HLS Tcl, numerical testbench, C++ and Vivado OOC Tcl for your target part. Preparation does not execute either vendor:

```bash
fpgai-research-tools prepare \
  --onnx build/native_demo/model.onnx \
  --input build/native_demo/input.npy \
  --case build/native_demo/hls_case \
  --part xck26-sfvc784-2LV-c \
  --clock-ns 10 \
  --tolerance 0.01
```

The FPGA part above is the repository's research default for **a particular K26 device** and does **not** imply a board-level platform, proper I/O, pins, clock/reset wiring or a loaded Kria image. Replace it with a genuinely supported part for your board. The example must use a **new case directory**; source evidence is create-only.

Inspect the generated files:

```text
build/native_demo/hls_case/
  model.cpp           compiler-generated HLS C++
  tb.cpp              numeric C-simulation testbench
  run_hls.tcl         Vitis HLS source/simulation/synthesis script
  manifest.json       source and target metadata
  vivado/
    run_vivado.tcl    OOC report generation, NOT a bitstream
```

### Explicit Vitis execution

After you inspect the prepared testbench and Tcl, and only if you intend to run licensed vendor tools:

```bash
fpgai-research-tools run-vitis \
  --case build/native_demo/hls_case \
  --executable "${FPGAI_VITIS_HLS_EXECUTABLE:-vitis_hls}"
```

This invokes actual Vitis HLS, and the generated script requests **C simulation and synthesis**. A return code of zero is not by itself an independently qualified timing or numerical result. Check the original CSim output, synthesis XML, effective clock target and source identity before using the observations.

The basic `prepare` CLI currently does not expose an explicit clock-uncertainty argument. For strict research comparisons requiring exact uncertainty, use a protocol that explicitly records and verifies the executed Tcl uncertainty; do **not** silently equate this quickstart with source-pinned publication studies.

### Explicit Vivado out-of-context experiment (optional)

Only after Vitis has generated the RTL sources:

```bash
fpgai-research-tools run-vivado \
  --case build/native_demo/hls_case \
  --executable "${FPGAI_VIVADO_EXECUTABLE:-vivado}"
```

This tool's Tcl **only** synthesizes, places and routes the generated Verilog in an **out-of-context** project, then writes timing, utilization, DRC and routed-checkpoint reports. It does **not** produce a bitstream, IP Integrator design, address map or usable board runtime. DRC warnings and timing failures must be reported as such, never silently treated as board-qualifying success.

## Inspect whether a case is actually board-ready

After source preparation and any deliberately executed native steps,
use the **read-only** [deployment inventory](FPGA_DEPLOYMENT.md):

```bash
python scripts/check_fpga_deployment.py \
  --case build/native_demo/hls_case --target kv260
```

This checks presence of known HLS/Vivado artifact paths, not vendor
license status, design integrity, a bitstream or board compatibility.
It never programs hardware.

## Common setup problems

| Problem | Action |
|---|---|
| `vitis_hls: command not found` | Source the correct vendor `settings64.sh`; check the installed version and `PATH` |
| `vivado: command not found` | Vivado may not be installed or may be installed elsewhere; use its own settings script |
| Tool available but no license | Configure/obtain valid vendor licenses separately under AMD terms; **never commit license files or environment secrets** |
| Unsupported FPGA part | Use an installed device-supported vendor version and an actual part appropriate to your board |
| Missing platform | Install a matching official/custom `.xpfm` separately, then set `PLATFORM_REPO_PATHS` |
| Missing XRT | Only install/configure XRT if your actual runtime flow requires it; HLS source compilation doesn't |
| HLS passes but board cannot run model | Read [FPGA deployment](FPGA_DEPLOYMENT.md): RTL needs integration, control/data interfaces and software |
| Vivado routed reports but no bitstream | Expected: this repository's current Vivado Tcl is **OOC-only** |

## Authoritative technical references

- AMD Vitis 2023.2 environment: https://docs.amd.com/r/2023.2-English/ug1400-vitis-embedded/Setting-Up-the-Environment-to-Run-the-Vitis-Software-Platform
- AMD Vitis HLS 2023.2: https://docs.amd.com/r/2023.2-English/Vitis-Tutorials-Getting-Started/Vitis-HLS
- AMD HLS IP packaging: https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/Packaging-the-RTL-Design

Continue with [FPGA deployment](FPGA_DEPLOYMENT.md), [Using FPGAI](USING_FPGAI.md), and [Scientific reproducibility](SCIENTIFIC_REPRODUCIBILITY.md).
