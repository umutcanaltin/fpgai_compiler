# R2z — Unified bounded Conv2D + multi-layer Dense neural-network compiler

**Date:** November 3, 2026  
**Status:** **Implementation and tests committed to `main`, not yet workstation-verified.** No R2z original GCC/CSim/CSynth/Vivado outcomes have been measured on the user's computer. The original R2i–R2x source/vendor studies and the partially executed R2y Vivado study remain frozen; RL remains untouched.

## Why R2z adds an actual supported network path

The research compiler already had real Model IR and HLS source lowerings for **Conv2D, MaxPool, Relu, Flatten, Gemm and MatMul**, but these had mainly been validated in separate NCHW convolution graphs and purely matrix-based MLP chains. A direct end-to-end typed, supported **CNN → Flatten → fully connected NN** entrypoint with per-stage contracts, independently mapped Conv/Dense operators, and complete actual emitted-C++ numerical checks was missing.

R2z implements that missing integration by **reusing existing**, source-reproducible components—not duplicating arithmetic or silently pretending that sequential source arrays are a streaming network:

- `model_ir/neural_chain.py`: immutable typed `NeuralChainIR` containing every original operator identity, rank/shape, independent mapping (`parallelism`, `pes`, `simd`, fold, explicit tile, pipeline intent), and the exact full materialized tensor boundaries between stages. Checks the canonical `FunctionalModel` and existing `architecture.resolver`/ `lowering`/ `verify_refinement` semantics; rejects unsupported graph branches, dynamic weights, missing/duplicate flatten, rank mismatches, mixed numeric formats, dead initializer tensors and nonimplemented spatial mapping on scalar unary layers. Does not claim cycle-level scheduling or early in-place lifetime reuse.
- `compiler/neural_chain.py`: a direct `compile_neural_chain(ArchitecturalModel)` façade, using **the same `compile_architecture` and `emit_hls`** as the rest of the repository. Requires no missing/reordered/rewritten stages, retains the existing `static-conv-trip-v3` bounded Conv tile/guard codegen, the existing `MatMul/Gemm` HLS loops, and emits complete C++ fixed-point intent `ap_fixed<16,6,AP_RND,AP_SAT>` under the default architecture. Its `FPGAI_HOST_FLOAT` mode is explicitly not fixed-point-equivalent. `compiler/__init__.py` exports this new entrypoint without removing the existing compiler API.
- `validation/neural_chain_host.py`: **actual emitted C++** GCC host source execution with the existing `FPGAI_TRACE_TENSORS` macro. It compares *every* intermediate result, and all final outputs, to the existing independent NumPy `validation.reference.evaluate` operator-by-operator; rejects duplicate/missing/extra source traces, wrong stage ordinal or element, nonfinite output, changed source SHA, and numerical mismatch. This is **GCC host E1 only**; no virtual vendor results or hardware performance are invented.
- `experiments/cst_r2z_neural_network_fixtures.py`: three fixed, nontrained, representative topologies: **(1)** padded Conv → Relu → MaxPool → second Conv → Flatten → MatMul Dense → Relu → biased transposed Gemm Dense; **(2)** strided padded Conv with batch size two → Relu → Flatten → biased Gemm → Relu → MatMul; **(3)** two Conv2D layers with both geometry changes → Relu → Flatten → biased transposed Gemm → Softmax. All initializers and inputs are deterministic float32 with independent per-layer PE/SIMD/tile source mapping alongside a scalar-mapped control.
- `experiments/cst_r2z_conv_dense_network_campaign.py`: **six complete original network/source packages** (three topologies × scalar versus independently mapped) and **12 prespecified real GCC host input executions** (two samples per original package). All intermediate tensors must pass the numerical oracle before original source artifact creation. The existing `write_vitis_package` generates six independently source-pinned native-ready Vitis HLS `model.cpp`, `tb.cpp`, `run_hls.tcl`, and `manifest.json` source packages, using **10 ns requested HLS clock and 1 ns HLS uncertainty** and an explicit **0.01 absolute CSim criterion**. **No Vitis CSim, CSynth or Vivado calls occur during preparation.** The create-only original summary/cases and a separate **read-only same-source SHA/rederivation audit** preserve all six source IDs and 12 actual GCC receipts.
- `tests/test_cst_r2z_conv_dense_neural_network.py` and `tests/test_cst_r2z_neural_campaign.py`: verify source identity, actual mixed-layer C++ and every intermediate value across the three configurations and two Phi policies, source mutation rejection, mixed-precision/branch/attribute rejection, native-package creation and create-only/tamper audit. A real small **ONNX Conv → Relu → Flatten → Gemm** network is imported via the existing ONNX frontend and fed to the exact same compiler/GCC pipeline when ONNX is installed. No vendor tool launched by tests.

### Boundaries of implementation

**Supported in R2z:** static NCHW Conv2D (non-grouped, no dilation) with optional constant bias, NCHW 2×2 stride-2 MaxPool, ReLU, GlobalAveragePool, NCHW axis-1 Flatten, static-weight rank-2 Gemm with optional bias and `transB=0/1`, static-weight MatMul, rank-2 Sigmoid/Softmax after Dense, and chains with **multiple convolution and multiple fully connected layers**. The existing ONNX frontend supplies strict shape validation; manually assembled Functional Models must pass the exact same resolver and refinement checks.

**Not claimed:** arbitrary CNN/DAG/residual connections in this particular sequential compiler, dynamically learned weights or training inside HLS, generic object detection/transformer operators, arbitrary grouping/dilation, mixed fixed-point formats without casts, actual device-specific memory banking/ports, stream concurrency, source accumulator in-place fusion, PE/clock/timing realization, Vitis bit-exact CSim, E3 HLS resource QoR, E4 routed device performance, board runtime or neural classification accuracy across any dataset.

The earlier **NCHW graph compiler** and existing **MLP row-chain compiler** are not deleted or replaced. Their distinct evidence/reuse mechanisms remain intact. The R2z full-tensor pipeline is a simpler end-to-end supported use case and cannot, by itself, establish a novel IR or physical memory saving. The fact that standard convolution, matrix multiplication and ONNX import work is engineering coverage; novelty requires comparison to prior literature and source/RTL capabilities.

## Source-only end-to-end publication gate and actual static ONNX path

The public API now also provides `from fpgai_research.compiler import compile_onnx_neural_chain`, which delegates strictly to the existing ONNX importer, canonical resolver and `compile_neural_chain`. The ONNX test verifies **byte-identical canonical generated C++** to the manually imported typed model on the same static Conv→Relu→Flatten→biased Gemm sample; unsupported ONNX input/attributes are rejected. This is a typed source-model compiler, **not native HLS validation**.

A new `scripts/run_compiler_ir_publication_source_gate.sh` provides a one-command **source-only** local gate: verify clean tracked source, compile Python, run focused R2z/R3a/R2y tests and full pytest, create the R2z six-package/twelve-GCC original experiment only if its target path does not already exist, run its independent read-only SHA and denominator audit, run R3a native evidence/three-pair read-only audits, and perform a Vitis **DRY RUN**. It intentionally does not launch Vitis/Vivado and cannot overwrite a prior R2z original output. Existing original native source cases are preserved as is. Run from repo root:

```bash
bash scripts/run_compiler_ir_publication_source_gate.sh
```

Once original R2z E1 source experiments have passed, [R3a original CNN/Dense Vitis E2/E3 protocol](R3A_CNN_DENSE_ORIGINAL_HLS_NATIVE_PROTOCOL_20261103.md) defines the separate opt-in actual six-case HLS vendor evaluation with no automatic physical claims.

## Test and original source-campaign commands

Run in the user's existing `~/Desktop/github_projects/fpgai_research` environment:

```bash
cd ~/Desktop/github_projects/fpgai_research
git status --short
git pull --ff-only origin main
git rev-parse --short HEAD
python -m compileall -q src/fpgai_research experiments/compiler_ir tests

python -m pytest -q \
  tests/test_cst_r2z_conv_dense_neural_network.py \
  tests/test_cst_r2z_neural_campaign.py \
  tests/test_cst_r2y_controlled_physical_pilot.py

python -m pytest -q
```

**Only if both focused and full regressions pass**, create the NEW original, source-only R2z campaign. **Never** delete/overwrite any frozen original R2i–R2y outputs:

```bash
python -m fpgai_research.experiments.cst_r2z_conv_dense_network_campaign \
  --output build/r2z_conv_dense_original_source_6_v1

python -m fpgai_research.experiments.cst_r2z_conv_dense_network_campaign \
  --study build/r2z_conv_dense_original_source_6_v1

cat build/r2z_conv_dense_original_source_6_v1/summary.json
```

Expected *protocol* denominators, **not yet observed original results:** six distinct source packages, twelve original GCC executions, full typed operator coverage, zero vendor invocations. A failing original numerical case is a scientific failure: retain the code and report error, do not remove the failing case or silently relax tolerances until it passes.

The prepared `case_000`…`case_005` folders are source-only HLS-ready. **Native Vitis executions are a distinct future opt-in experiment requiring explicit source/control qualification**. The original R2y physical folder already has `case_000` routed with disclosed DRC and **not strict E4**; never rerun it to rewrite the warnings. R2z does not change or rerun any of those physical cases.

## Publication relevance

The next paper should distinguish clearly: existing per-operator source arithmetic, typed mixed-network composition, independently mapped Conv/MatMul choices, actual host intermediate trace, available native source packages, and (if obtained later) independently qualified real HLS/Vivado outcomes. Until the original R2z workstation tests and source experiment actually complete, the manuscript must state **R2z implemented/unverified**. Even full GCC success only establishes bounded E1 neural source correctness on original probes and not scientific novelty or physical benefits.
