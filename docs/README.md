# Documentation

Start with the [project README](../README.md) for a conceptual overview and the simplest compilation path.

| Goal | Document |
|---|---|
| Troubleshooting and common questions | [FAQ](FAQ.md) |
| Prepare a release candidate without exposing research journals | [Clean-room preview](RELEASE_PREVIEW.md) |
| Install and compile a synthetic ONNX model | [Using FPGAI](USING_FPGAI.md) |
| Configure Vitis HLS, Vivado, platform paths and XRT | [AMD toolchain setup](AMD_TOOLCHAIN_SETUP.md) |
| Integrate a compiled model into an actual board and host runtime | [FPGA deployment](FPGA_DEPLOYMENT.md) |
| Who can use or commercially deploy the research code? | [Research-only use policy](WHO_CAN_USE_FPGAI.md) |
| Review proposed academic research-only license terms | [License proposal](ACADEMIC_RESEARCH_LICENSE_PROPOSAL.md) |
| Understand Model IR, HLS pipeline and source certificates | [Architecture](ARCHITECTURE.md) |
| Determine what the compiler does and does not support | [Supported features](SUPPORTED_FEATURES.md) |
| Copy and modify small synthetic demonstrations | [Examples](../examples/README.md) |
| Compile static convolution–dense networks | [Network gallery](../examples/NETWORK_GALLERY.md) |
| Contribute a fix or a new feature | [Contributing](../CONTRIBUTING.md) |
| Implement a new operator or layer | [Operator contribution](COMPILER_OPERATOR_CONTRIBUTION.md) |
| Understand experiment evidence and hardware limitations | [Scientific reproducibility](SCIENTIFIC_REPRODUCIBILITY.md) |
| Propose external disclosure | [Release checklist](OPEN_SOURCE_RELEASE_CHECKLIST.md) and [License decision](LICENSE_DECISION.md) |
| See upcoming research and documentation opportunities | [Roadmap](ROADMAP.md) |
| See project-level changes without internal research diaries | [Changelog](../CHANGELOG.md) |
| Get support or report problems | [Support](../SUPPORT.md) |
| Collaboration expectations | [Code of conduct](../CODE_OF_CONDUCT.md) |
| Understand security disclosures and responsible use | [Security policy](../SECURITY.md) |
| Cite a research artifact without inventing a DOI | [Citation](../CITATION.md) |

## Research-internal resources

`paper/compiler_ir/` contains extensive source-pinned studies, experiment protocols, older hypotheses, and manuscript drafts. These are **not product documentation** and some historical entries describe superseded checkpoints. The separate `paper/rl_experiments/` and `rl_environment/` materials describe a research environment currently frozen for new development.

`docs/maintainers/` preserves historical contributor/README context. Do not rely on that archive for current usage.

**Public release remains blocked** until ownership/IP, license, third-party code, full Git history and distribution rights have been independently reviewed.
