# Frequently asked questions

### Is FPGAI a complete FPGA product?

No. This is a research compiler and collection of experiment interfaces. The compiler generates HLS-oriented C++ from a supported static neural-network graph, with explicit architecture semantics and source contracts. Device-specific synthesis and deployment are separate.

### Does installation require AMD Vivado or Vitis?

No. Installation, synthetic ONNX generation, Model IR construction and HLS C++ source emission do not invoke licensed vendor programs. Original native experiments require a separate compatible installation and explicit execution.

### Which models can I use?

Start with static MatMul/ReLU, supported dense layers and the bounded NCHW convolution-to-dense paths. Not every ONNX operator or architecture combination works. See [Supported features](SUPPORTED_FEATURES.md).

### Are there examples with convolutional layers?

Yes. The [neural-network gallery](../examples/NETWORK_GALLERY.md)
includes convolution–pooling–dense, stride-2 convolution–dense, and two
convolutional layers with a Softmax output. Each uses static ONNX tensors,
synthetic weights and optional generated-C++ host reference verification.

### Why was my model rejected?

FPGAI's strict parser/resolver rejects unsupported operators, attributes, shapes, numeric formats and architecture mappings. This is preferable to a silent fallback that could invalidate scientific comparisons.

### Do source-certified designs guarantee FPGA timing?

No. Bounded source certification only establishes the specifically checked source event and ownership properties. It does not establish HLS effective timing, routed WNS, physical resources, memory port legality or board execution.

### Why are some HLS results missing?

Some experiments reached C simulation but failed the study's effective synthesis timing criterion. They are reported as failures, not replaced with fabricated cycle/resource comparisons.

### Does the RL environment prove learning improvements?

No. The learning environment is separate from this compiler's correctness claims and is currently frozen for new research changes. A typed observation or candidate mask is not itself a measured trained-policy benefit.

### How do I configure Vitis and Vivado installation locations?

See [AMD toolchain setup](AMD_TOOLCHAIN_SETUP.md) for sourceable settings scripts, explicit executable locations and optional XRT/platform paths. The environment checker is read-only; no vendor tools run unless you explicitly invoke them.

### How can I run a compiled neural network on a KV260, PYNQ or Alveo?

The current compiler only produces HLS C++ and research out-of-context implementation evidence. A full board image needs HLS IP/kernel export, interconnect/AXI and PS/PL integration, a bitstream and host-side software. See the [FPGA deployment guide](FPGA_DEPLOYMENT.md); no one-command board runtime is currently implemented.

### What permissions apply to the source?

This publicly viewable repository has no operative source-code license. Visibility does not grant general permission to copy, modify, redistribute or commercially deploy copyrighted source. See [Licensing and permissions](LICENSE_DECISION.md).

### Can I contribute a new operator?

Yes, after contributor/distribution rights are clarified. Begin with the [contribution guide](../CONTRIBUTING.md) and [operator-extension guide](COMPILER_OPERATOR_CONTRIBUTION.md). Specify its semantic contract, independent numeric reference, supported shapes and emitted-source behavior before claiming native HLS support.

### Where are the paper experiments?

Detailed studies are under `paper/compiler_ir/` and `experiments/compiler_ir/`. Their dated notes are historical/provenance records, **not** user-manual instructions. Prefer [Scientific reproducibility](SCIENTIFIC_REPRODUCIBILITY.md) for evidence terminology.
