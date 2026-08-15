# Multi-output external HLS validation

FPGAI's maintained `SplitScale` example validates the multi-output path end-to-end at the compiler-contract level:

`Input -> external SplitScale -> (identity, scaled) -> Add -> Relu -> Output`.

The external operator owns ONNX import, two-output shape/type inference, and numeric reference behavior. The selected implementation uses `tensor_ports_v1` with one input and two outputs. DAG HLS code generation allocates and tracks both output tensors independently before the built-in `Add` merge.

Maintained configurations provide separate C-simulation, HLS-synthesis, and Vivado-implementation validation levels. FPGAI keeps these levels distinct and does not infer a stronger validation result from a weaker stage.
