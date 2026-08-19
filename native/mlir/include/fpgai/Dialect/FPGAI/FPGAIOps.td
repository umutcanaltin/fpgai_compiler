include "mlir/IR/OpBase.td"
include "fpgai/Dialect/FPGAI/FPGAIDialect.td"

def FPGAI_MatMulOp : FPGAI_Op<"matmul", [Pure]> {
  let summary = "FPGAI matrix multiplication with implementation semantics";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs,
                       OptionalAttr<StrAttr>:$selected_backend,
                       OptionalAttr<I64Attr>:$tile_m,
                       OptionalAttr<I64Attr>:$tile_n,
                       OptionalAttr<I64Attr>:$tile_k,
                       OptionalAttr<StrAttr>:$accumulator_dtype);
  let results = (outs AnyTensor:$result);
}

def FPGAI_SoftmaxOp : FPGAI_Op<"softmax", [Pure]> {
  let summary = "FPGAI softmax/reduction operation";
  let arguments = (ins AnyTensor:$input,
                       DefaultValuedAttr<I64Attr, "-1">:$axis,
                       OptionalAttr<StrAttr>:$selected_backend,
                       OptionalAttr<StrAttr>:$buffer_storage);
  let results = (outs AnyTensor:$result);
}

def FPGAI_LayerNormOp : FPGAI_Op<"layer_norm", [Pure]> {
  let summary = "FPGAI layer normalization";
  let arguments = (ins AnyTensor:$input, AnyTensor:$scale, AnyTensor:$bias,
                       DefaultValuedAttr<I64Attr, "-1">:$axis,
                       OptionalAttr<StrAttr>:$selected_backend);
  let results = (outs AnyTensor:$result);
}

def FPGAI_TransportOp : FPGAI_Op<"transport"> {
  let summary = "Explicit FPGAI tensor transport boundary";
  let arguments = (ins AnyTensor:$input,
                       StrAttr:$protocol,
                       DefaultValuedAttr<BoolAttr, "false">:$ready_valid,
                       OptionalAttr<I64Attr>:$packet_words);
  let results = (outs AnyTensor:$result);
}

def FPGAI_BufferOp : FPGAI_Op<"buffer"> {
  let summary = "Explicit FPGAI memory/buffering contract";
  let arguments = (ins AnyTensor:$input, StrAttr:$storage,
                       OptionalAttr<I64Attr>:$depth_words);
  let results = (outs AnyTensor:$result);
}
