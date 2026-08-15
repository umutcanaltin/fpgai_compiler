# E5B/E5C foundation: Vivado characterization, HLS bottlenecks, tensor liveness

This batch extends the maintained external-operator flow after HLS synthesis.

## Vivado implementation characterization

When `build.stages.vivado_project` / `vivado_implementation` are requested, the ecosystem compiler reuses FPGAI's existing Vivado bridge. The resulting characterization records whole-design WNS/TNS/WHS/THS, derived Fmax, post-implementation resources, power when available, and bitstream/XSA presence. External operator and implementation provenance plus the package lock remain attached to the result. Whole-design measurements are never attributed to a single external node.

Validation levels remain distinct: `hls_synthesized`, `vivado_implemented`, and `bitstream_generated`.

## HLS bottleneck diagnostics

Vitis HLS 200-885 scheduling warnings are normalized into a report with requested/achieved II, loop/module, source location, resource, cause classification, and applicable YAML mechanisms. FPGAI does not silently enable partitioning, unrolling, or storage changes.

## Tensor liveness foundation

The IR analysis records producers, consumers, live ranges, branch/merge points, maximum simultaneously live tensors, and greedy reusable activation-buffer slots. This analysis records metadata independently from code generation; DAG composition consumes it for residual and multi-input graphs.
