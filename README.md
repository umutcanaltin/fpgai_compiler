# FPGAI Compiler

**Architecture-aware FPGA neural-network research compiler.**

FPGAI explores the relationship between a **functional neural-network graph**, an **explicit operator-level architecture request**, a **resolved implementation**, and generated **HLS C++**. It makes hardware choices such as precision, parallelism, processing elements (PEs), SIMD width, serialization, tiling, and pipeline requests inspectable before vendor synthesis.

The research implementation also investigates bounded, typed source-event contracts for tensor-element production, consumption, last-reader memory reuse and source-order conformance. A separate, optional research environment explores compiler-guided architecture decisions. These are experimental interfaces, **not a finished general-purpose FPGA toolchain**.

**Licensing and permissions:** This public repository has no operative source-code license published at present. GitHub visibility alone does not grant rights to copy, modify, redistribute or commercialize copyrighted code. See [Licensing and permissions](docs/LICENSE_DECISION.md).

## Start here

You can generate HLS C++ locally without installing Vitis or Vivado. The example below uses a tiny generated ONNX MatMul–ReLU model, not private datasets or weights.

**Requirements:** Python 3.10+, pip, and a virtual environment. The ONNX example needs the optional ONNX extra.

```bash
git clone https://github.com/umutcanaltin/fpgai_compiler.git
cd fpgai_compiler
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[onnx]"
mkdir -p build

# Generate a tiny synthetic, self-contained ONNX graph.
python examples/make_toy_onnx.py build/toy.onnx

# Compile ONNX to HLS C++ (no native vendor tools are called).
fpgai-research build/toy.onnx -o build/toy-hls

# Inspect source and the compiler's evidence manifest.
ls build/toy-hls/
```

The last command writes `model.cpp` and `manifest.json`. **Generation is not synthesis, placement, timing closure or proof of numerical correctness.** The source may require an appropriate vendor HLS compiler to synthesize. The `--export-ir` option additionally saves a *reconstructible Model IR artifact that includes model weights*; use it only with data you are permitted to store or distribute.

Configure architecture with an optional JSON contract:

```bash
fpgai-research build/toy.onnx \
  --architecture examples/architecture.json \
  -o build/toy-mapped
```

Requests that exceed the supported shape, attribute, architecture or target constraints should fail explicitly rather than silently substituting an unrelated implementation. For scope-specific controls see [architecture and CLI guide](docs/USING_FPGAI.md).

**Research-paper and release status:** The first C1/C2 compiler paper has original bounded source-contract and source-action evidence, but remains under major scientific revision. The [research artifact and publication guide](docs/RESEARCH_PUBLICATION_STATUS.md) distinguishes the active manuscript from archival studies, records exact source/report identities for the latest verified C1/C2 experiments, and lists the institutional/licensing and independent-science gates still required before a source distribution. This repository is publicly viewable but **is not currently distributed under an operative source-code license**.

## One-command local HLS and Vivado workflow

For supported ONNX networks, the [neuroscientist quickstart](docs/NEUROSCIENTIST_QUICKSTART.md) shows how to supply AMD installation paths and run compiler generation, numerical C simulation, HLS synthesis, and Vivado out-of-context routing through `fpgai-run`. Native execution always requires `--execute-native`; the command never invokes GitHub Actions or generates a board bitstream.

## Explore complete convolutional networks

In addition to the tiny MatMul–ReLU example, the repository includes
**three deterministic convolution/dense ONNX networks** and a separate
**two-layer dense MLP**. They cover convolution–pooling–dense,
stride-2 convolution–dense, two-convolution–Softmax, and Gemm–ReLU–Gemm–Softmax. Each has explicit input/output shapes and uses
the existing strict multi-layer compiler.

```bash
mkdir -p build/gallery
python examples/make_neural_network_onnx.py \
  cnn_pool_dense build/gallery/cnn_pool_dense.onnx
python examples/compile_neural_network.py \
  build/gallery/cnn_pool_dense.onnx \
  --output build/gallery/cnn_pool_dense_hls
```

For source-only numerical checks of **every intermediate layer**, the
example compiler supports `--verify-host` when GNU C++ is available.
See [the illustrated neural-network gallery](examples/NETWORK_GALLERY.md).
None of the examples trains a classifier or claims physical timing.

## What is in the repository?

| Directory | Purpose |
|---|---|
| `src/fpgai_research/frontend/` | Static ONNX importing and attributes |
| `src/fpgai_research/model_ir/` | Typed functional model, architecture mappings, lowering and contracts |
| `src/fpgai_research/architecture/` | Per-operator hardware requests, scoping and source feasibility |
| `src/fpgai_research/compiler/` | Canonical compilation orchestration, bounded neural-chain interfaces |
| `src/fpgai_research/codegen/hls/` | HLS C++ generation |
| `src/fpgai_research/validation/` | Independent numerical and source checks |
| `src/fpgai_research/toolchains/` | Explicit opt-in toolchain packaging and qualified native evidence |
| `examples/` | Small synthetic demonstration models and architecture JSON |
| `tests/` | CPU-side regression and source contract checks |
| `experiments/compiler_ir/` | Research experiment definitions; original hardware measurements require explicit separate execution |
| `rl_environment/` | Separate experimental RL environment, currently frozen for new research development |
| `paper/compiler_ir/` | Detailed academic protocols, evidence, and research history; **not the user manual** |
| `docs/` | Public-facing guides, contribution instructions and release prerequisites |

### Current capabilities and limits

- Typed static functional graph and operator-local architecture requests, with explicit rejection of unsupported shapes/attributes.
- A progressive Functional → Architectural → Lowered Model IR representation and an HLS C++ emitter.
- Bounded neural-network chains with convolution, matrix multiplication / dense, activation, pooling, flattening and some other operators under documented restrictions.
- Bounded MatMul-chain and multi-reader fanout source-event contracts, source scratch-lifetime witnesses and policy-independent source candidate classification.
- Optional native Vitis HLS / Vivado experiments **only** with a compatible local licensed toolchain and explicit execution flags.

Not promised: arbitrary ONNX coverage, general RTL/board integration, formally verified all-input equivalence, physical BRAM savings, automatic successful timing closure, a production-grade optimizer, or improved trained RL performance. See [support matrix](docs/SUPPORTED_FEATURES.md).

## Using and extending

- **Installation, CLI, architecture configuration, outputs:** [Using FPGAI](docs/USING_FPGAI.md).
- **Vitis/Vivado environment variables and HLS workflows:** [AMD toolchain setup](docs/AMD_TOOLCHAIN_SETUP.md).
- **How to integrate and run models on actual FPGA hardware:** [FPGA deployment](docs/FPGA_DEPLOYMENT.md).
- **Source licensing and permissions:** [Licensing and permissions](docs/LICENSE_DECISION.md).
- **Conceptual compiler design and data flow:** [Architecture](docs/ARCHITECTURE.md).
- **What works, what is restricted:** [Supported features](docs/SUPPORTED_FEATURES.md).
- **Running examples without AMD tools:** [Examples](examples/README.md).
- **Contributor development workflow:** [CONTRIBUTING.md](CONTRIBUTING.md).
- **Adding an operator/layer:** [Operator contribution guide](docs/COMPILER_OPERATOR_CONTRIBUTION.md).
- **Research provenance and reproduction:** [Scientific reproducibility](docs/SCIENTIFIC_REPRODUCIBILITY.md).
- **Extended documentation index:** [Documentation](docs/README.md).
- **Research contribution opportunities:** [Roadmap](docs/ROADMAP.md).
- **Project-level changes:** [Changelog](CHANGELOG.md).

## Local development checks

```bash
python -m pip install -e ".[onnx,test]"
python -m compileall -q src/fpgai_research tests
python -m pytest -q
```

These commands do not intentionally launch Vitis/Vivado. A complete test suite covers research modules, so optional dependencies may be required for some specialized tests. **Never run original hardware experiments, enable self-hosted CI, or execute generated scripts solely because installation succeeded.** Native evidence requires separate explicit authorization and a compatible machine.

## Research and scientific use

Experimental findings are separated by the evidence they actually establish: source semantics and generated C++ are not Vitis C simulation; synthesis estimates are not routed physical utilization or frequency; Vivado out-of-context reports are not board latency or power. For a safe overview of research evidence, see [Scientific reproducibility](docs/SCIENTIFIC_REPRODUCIBILITY.md). Historical research studies and sprint notes are maintained for scientific provenance; the current architecture and usage guides define supported user workflows.

The policy-neutral source decision interface is **learning-compatible**, but training improvements belong to a different, future study. Source-level feasibility certificates must not be interpreted as a hardware QoR guarantee.

## Contributing, security, and citation

We welcome reproducible bug reports, technical discussion and scoped contribution proposals. Coordinate contributions with maintainers before submitting copyrighted source. Read [CONTRIBUTING.md](CONTRIBUTING.md), [SUPPORT.md](SUPPORT.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), [SECURITY.md](SECURITY.md), [NOTICE.md](NOTICE.md) and [CITATION.md](CITATION.md). Do not include private model weights, nonredistributable datasets, secrets, vendor logs, university/commercial IP, or personal data in issues and pull requests. For revision-specific scientific citations, see [CITATION.md](CITATION.md).

---
**Project scope:** The compiler supports the documented operator families and research workflows. Scientific results and limitations are maintained in the reproducibility record.
