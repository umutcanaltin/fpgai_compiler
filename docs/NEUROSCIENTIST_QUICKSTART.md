# Run a neural-network model through the FPGAI compiler

This guide is for a researcher who has a **static ONNX model** and wants a single local command to generate HLS C++, run AMD Vitis HLS, and obtain routed Vivado **out-of-context (OOC)** results. It does not require learning vendor GUIs or editing Tcl. The original compiler, generated source and tool reports remain inspectable.

**Scope:** The current workflow generates and evaluates a model-level hardware implementation. It **does not** produce a board bitstream, Linux inference application, AXI DMA subsystem or hardware measurements. A working Vitis/Vivado installation alone cannot supply missing board-specific PS/PL integration.

## 1. Install FPGAI

Use Linux and Python 3.10 or newer:


```bash
git clone https://github.com/umutcanaltin/fpgai_compiler.git
cd fpgai_compiler
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[onnx,test]"
```

The Python package provides a new command, `fpgai-run`. No AMD software is bundled by the repository. Install AMD tools separately and comply with the license for each program.

## 2. Choose a supported model

Provide a local ONNX file, for example `models/network.onnx`. This workflow currently accepts one positive-static-shape runtime input and one output, with operators and architecture combinations supported by the existing compiler. Refer to [Supported features](SUPPORTED_FEATURES.md). Unsupported models fail explicitly rather than being silently transformed.

To generate a public synthetic example (no private weights or data):


```bash
mkdir -p build/demo
python examples/make_neural_network_onnx.py cnn_pool_dense build/demo/model.onnx
```

You can optionally provide `--input path/to/samples.npy` containing real representative **float32** samples, and `--labels path/to/labels.npy` for classification. Without `--input`, FPGAI automatically generates one reproducible synthetic smoke input. Passing that sample in C simulation is a compiler-check result **not** a statement about trained-model accuracy on a real dataset.

## 3. Point to Vitis and Vivado installations

Specify a **product installation directory**, or the actual program executable:


```text
/tools/Xilinx/Vitis_HLS/2023.2
/tools/Xilinx/Vivado/2023.2
```

The command detects a `settings64.sh` when present and sources it in a **child process**. You do not need to manually open Vitis/Vivado or source those files in your shell. Tool locations are resolved before source preparation when native execution is requested. If multiple tool versions exist under one common directory, give the specific version directory.

The target FPGA must also be explicit because the tool locations alone cannot identify the silicon. You can use `--board kv260` (which selects the established K26 part) or specify a verified part with `--part`. The selected part is **not** a complete board platform. Source-only mode uses the repository's existing K26 research default if no target was selected; this is only source packaging, not a hardware compatibility assertion.

## 4. Preview without invoking vendor tools


```bash
fpgai-run \
  --onnx build/demo/model.onnx \
  --output build/demo/case01 \
  --board kv260 \
  --vitis /tools/Xilinx/Vitis_HLS/2023.2 \
  --vivado /tools/Xilinx/Vivado/2023.2 \
  --plan
```

Planning is read-only and never executes native tools, creates a case, programs hardware or starts GitHub Actions. Without `--execute-native`, requesting native stages also produces a read-only plan.

## 5. Run one command on your workstation

When you have reviewed the plan and are ready to invoke the locally installed licensed AMD programs:


```bash
fpgai-run \
  --onnx build/demo/model.onnx \
  --output build/demo/case01 \
  --board kv260 \
  --clock-ns 10 \
  --vitis /tools/Xilinx/Vitis_HLS/2023.2 \
  --vivado /tools/Xilinx/Vivado/2023.2 \
  --stage ooc \
  --execute-native
```

This one command will:

1. Import ONNX, resolve the compiler's architecture constraints and emit HLS C++.
2. Generate a matching numerical C-simulation testbench and HLS Tcl.
3. Initialize the AMD installation environment and run Vitis HLS automatically.
4. Verify that original HLS evidence exists **and** C simulation passed before launching Vivado.
5. Stage source-matched ROM initialization data as needed and run out-of-context Vivado synthesis, placement and routing.
6. Inspect attested routed timing and design-rule reports; report failure if CSim, HLS, route, timing or DRC checks do not qualify.

The command does not claim all ONNX models will synthesize. Actual model support and performance depend on model shape, parameters, numerical precision and your FPGA.

Useful alternatives:


```bash
# HLS-only: Vivado is not needed.
fpgai-run --onnx models/network.onnx --output build/hls01 \
  --part xck26-sfvc784-2LV-c \
  --vitis /tools/Xilinx/Vitis_HLS/2023.2 \
  --stage hls --execute-native

# Generate source/testbench without AMD tools.
fpgai-run --onnx models/network.onnx --output build/source01 \
  --stage source
```

Other options include `--architecture` for JSON architecture mappings, `--tolerance` for fixed-point numerical acceptance, `--clock-ns` for the requested vendor clock, and `--hls-timeout` / `--vivado-timeout` for maximum execution durations.

## 6. Read the results

The case directory contains:

| File or directory | Meaning |
|---|---|
| `model.cpp` | Compiler-generated HLS source |
| `tb.cpp` | Numerical C simulation testbench |
| `manifest.json` | Source and compiler evidence manifest |
| `run_hls.tcl` | Generated Vitis script |
| `vitis_invocation.json` | Native tool invocation receipt |
| `logs/` | Live Vitis stdout/stderr |
| `vivado/run_vivado.tcl` | Generated out-of-context implementation script |
| `vivado/reports/` | Vivado timing, utilization, routing and DRC reports when produced |
| `vivado/vivado_invocation.json` | Native Vivado receipt when invoked |
| `user_run_summary.json` | Simplified qualification status and warnings |

The recorded statuses are `SOURCE_READY`, `HLS_QUALIFIED`, `OOC_QUALIFIED`, `OOC_ROUTED_WITH_ADVISORIES`, or an explicit not-qualified failure. Tool exit code zero is **not** enough for qualification. The summary always states `board_bitstream_generated: false` and `device_programmed: false`.

Use **a fresh output directory for every invocation**. The command intentionally refuses to overwrite prior evidence. When using private ONNX weights, remember that generated HLS C++ and the testbench can contain numeric constants. Keep case folders private and under ignored `build/` unless you have permission to share them.

## When Vivado exits 0 but E4 is rejected

A finished Vivado command is not itself qualified evidence. If the result says `OOC_NOT_QUALIFIED` even though Vivado exits 0, inspect the original run **read-only**:

```bash
fpgai-diagnose --case build/neuro_ooc_01
```

The diagnostic reports the original invocation receipt, staged-ROM integrity changes, original input stability, and whether timing/DRC reports can be parsed as **unpromoted observations**. It does **not** modify the failed case, rerun Vitis/Vivado, or retroactively qualify E4.

The diagnostic also includes individual DRC rule IDs, severities, counts, and whether unconstrained I/O rules appear. A routed model-only out-of-context design may report external port constraints that do not describe an integrated board application; do not suppress such DRCs or claim a deployable image. First inspect all actual rule IDs and resolve their meaning before deciding whether to rerun the vendor tools.

On some Vivado 2023.2 runs, the tool reports it successfully read an original ROM initialization file but the staged working-directory copy is absent after execution. A new run may explicitly opt into the **existing, strict source-pinned and log-hash-verified readmem post-absence protocol**:

```bash
fpgai-run \
  --onnx build/demo/model.onnx \
  --output build/neuro_ooc_02 \
  --board kv260 \
  --clock-ns 10 \
  --vitis /tools/Xilinx/Vitis_HLS/2023.2 \
  --vivado /tools/Xilinx/Vivado/2023.2 \
  --stage ooc \
  --execute-native \
  --allow-verified-readmem-post-absence
```

This is **not** a blanket exception. It accepts only originally hash-attested ROM data, unchanged RTL/source inputs, exact literal Verilog readmem references, and actual vendor-reported successful reads in the matching original stdout log. Modified or unattributed ROM data still fail. The original `neuro_ooc_01` result stays unchanged, and independent routed timing and DRC qualification remain mandatory. Inspect the read-only diagnosis first; if failures have another cause, fix those instead of rerunning with this option.

### Routed results with known advisory DRC warnings

A fully routed, original source-attested model implementation can meet its requested clock and still receive **warning-level** DRC recommendations. For the initial Vitis HLS/Vivado 2023.2 test, these were `DPIP-2`, `DPOP-3`, `DPOP-4` (DSP pipeline optimization recommendations) and `RTSTAT-10` (no routable loads). These are **not** silently deleted or marked as clean.

The default command remains strict: it returns a non-zero status if DRC is not clean, even if the routing and timing are otherwise source-attested. If a researcher explicitly accepts a **warning-only, model-level OOC observation**, a NEW run can use both options:

```bash
fpgai-run \
  --onnx build/demo/model.onnx \
  --output build/neuro_ooc_02 \
  --board kv260 \
  --clock-ns 10 \
  --vitis /tools/Xilinx/Vitis_HLS/2023.2 \
  --vivado /tools/Xilinx/Vivado/2023.2 \
  --stage ooc --execute-native \
  --allow-verified-readmem-post-absence \
  --accept-ooc-advisories
```

This flag allows an exit code of zero only when the original verified E4 report shows complete routing, passing CSim, timing met, and an exact *known* subset of the four warning-only rules with no critical errors or unconstrained I/O. The summary state becomes `OOC_ROUTED_WITH_ADVISORIES`, **never** `OOC_QUALIFIED`. It explicitly records `strict_drc_clean_qualified: false`, the original warning rule counts, and the user's opt-in. Unknown rules, missing/modified/unattributed ROMs, source drift, CSim failure, timing failure and critical DRC still fail closed. No DRC severities, waivers, physical source or vendor Tcl are modified.

This status is appropriate only for clearly labeled *warning-only out-of-context implementation observations*. It does not establish a clean DRC signoff, robust maximum clock, platform integration, a bitstream or hardware deployment. Fixing DSP pipeline utilization is a separate performance/architecture task, not a prerequisite to *reporting* a qualified-with-disclosures research observation. For public results, disclose the source-pinned reports and DRC rule details.

## 7. Validation on the local development PC

From this feature branch, without calling Vitis/Vivado:


```bash
python -m compileall -q src/fpgai_research tests/test_neuroscientist_onboarding.py
python -m pytest -q \
  tests/test_neuroscientist_onboarding.py \
  tests/test_native_tool_identity.py \
  tests/test_public_distribution_surface.py
python -m pytest -q
```

Then run `--plan`, a source-only smoke, and the real native command with installed Vitis/Vivado on the workstation. Preserve the complete logs and `user_run_summary.json` for review before merging this branch. This feature intentionally does **not** add or enable GitHub Actions.

## What remains before board execution?

A verified Vivado OOC implementation is **not** an FPGA bitstream. A deployable KV260/PYNQ/Alveo application still needs explicit board/platform integration, correct AXI/DDR/PS/DMA interfaces, clocks and resets, pins/constraints where applicable, host-side software, packaging and real on-device numerical tests. See [FPGA deployment](FPGA_DEPLOYMENT.md) for that separate work.
