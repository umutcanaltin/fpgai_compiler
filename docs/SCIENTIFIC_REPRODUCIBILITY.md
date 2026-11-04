# Scientific reproducibility and evidence boundaries

FPGAI is a research artifact. **A reproducible compiler result is not automatically a reproduced FPGA result.** The study population, source model, architecture contract, generated C++ identities, tool versions and original vendor invocation reports must be preserved when making measured claims.

## Evidence tiers

| Observation | May support | Does not establish |
|---|---|---|
| Typed model and architecture constraints | Supported static semantics; bounded source legality | Generated C++ numeric correctness or hardware legality |
| Emitted C++ and selected host checks | Materialized source and controlled reference agreement | Actual Vitis synthesis, FPGA performance or all-input proof |
| Vitis C simulation | Original numeric execution under a specific testbench | Hardware frequency or all possible inputs |
| Vitis HLS synthesis | HLS-estimated cycles, resources and effective clock qualification | Postroute utilization, achieved clock, bitstream power or runtime |
| Vivado out-of-context routing | Timing slack and utilization for one implementation context after required DRC/routing gates | Board-level inference, repeated-seed distributions or runtime power |
| Hardware execution | Observed deployment-specific behavior | Source generality or unmeasured models |

Source-event contract certificates currently apply to bounded MatMul-chain and small fanout classes, not every supported CNN. **The presence of an operator in the import/lowering pipeline is not an all-graph compositional proof.**

## Minimal local reproducibility

For a CPU-only smoke example:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[onnx,test]"
mkdir -p build
python examples/make_toy_onnx.py build/toy.onnx
fpgai-research build/toy.onnx -o build/toy-public
python -m pytest -q
```

The synthetic example is **not a numerical/HLS benchmark**. Detailed original research campaigns have separate entry points in `experiments/compiler_ir/` and the source-pinned protocols under `paper/compiler_ir/`; they must use fresh output directories. Do not rerun or overwrite any author's original study.

## Scientific evidence registry

The compiler-paper research currently includes separate populations:

- Finite tensor-event readiness, last-use, scratch occupancy and source-issue order observations.
- Independent finite interpreter comparisons, including varied tensor shapes and operator-local PE/SIMD mappings.
- Policy-independent compiler candidate decisions with source-certified, rejected and uncertified outcomes, and explicit action-screening costs.
- Mixed convolution–dense source/HLS studies, with genuine failed effective-clock cases retained as nonqualified comparisons.
- Earlier, **separate** MNIST MLP/CNN source, HLS and qualified out-of-context physical-routing studies that measure WNS and architecture equivalence. Do **not** present their routed utilization as if it were obtained for the later convolution–dense study.

These are related research questions but **different experimental populations**. They are not independent random samples simply because different software scripts generated them.

For an original study, keep a local record of: Git revision, exact model/weight/input identities, declared candidate population and exclusion rules, numeric/precision policy, FPGA part, requested clock, Vitis/Vivado versions, HLS/project scripts, original source/RTL hashes, reporter/gate result, and raw/provenance-preserving receipt. Research outputs may include proprietary HDL, private model weights, personal paths or nonredistributable data; **do not commit them by default**.

## Reproducing native experiments

Vitis HLS and Vivado are **optional**, proprietary and dependent on a license and supported host machine. Do not run native tools during installation, import, normal test collection or preview generation. A native campaign requires a separately authorized explicit execution flag and a new, isolated output directory.

If a routed case has DRC warnings, a failed timing gate or incomplete source provenance, record it at the weaker supported tier—**do not relabel it as physically qualified**.

## Reporting and writing

Scientific writing should use experiment descriptions, denominators, controlled variables, confidence limitations, relevant equations, and negative findings. Internal development labels, sprint numbers and GitHub issue names are for the provenance index and should not be used as substitutes for scientific methods.

An independent Allo/hls4ml/CODO/other baseline must be executed under a comparable scope before claiming relative capability or speedup. Literature inspection alone cannot establish that another system lacks a capability.

No general formal soundness theorem, universal accelerator performance gain, FPGA power reduction or learned policy benefit follows from source-only success.

## Open-science release

The repository has no approved distribution license. Before releasing source, datasets or original native study outputs, complete [the ownership and source-history checklist](OPEN_SOURCE_RELEASE_CHECKLIST.md), secure rightsholder approval and scrub any affected historical commits if necessary. Source redaction in one file does **not** remove secrets from Git history.

The research manuscript, public documentation and source release should be reviewed separately: a publication-ready paper does not by itself authorize publishing the entire research repository.
