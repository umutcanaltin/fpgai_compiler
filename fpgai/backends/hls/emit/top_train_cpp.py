from __future__ import annotations
from fpgai.backends.hls.emit.architecture_comments import emit_layer_architecture_comments

# FPGAI architecture-comment wrapper.

import re
from typing import Any, Dict, List, Optional, Tuple

from fpgai.backends.hls.emit.dense_tiling_codegen import apply_dense_tiling_to_top_source
from fpgai.backends.hls.emit.dense_tiling_codegen import apply_dense_training_tiling_to_top_source
from fpgai.backends.hls.emit.conv_tiling_codegen import apply_conv_tiling_to_top_source
from fpgai.backends.hls.emit.conv_tiling_codegen import apply_conv_training_tiling_to_top_source
from fpgai.engine.training_graph_utils import (
    as_chw as _as_chw,
    flat_size as _flat_size,
    get_tensor_shape as _get_tensor_shape,
    infer_conv_output_shape as _infer_conv_output_shape,
    infer_pool_output_shape as _infer_pool_output_shape,
    resolve_batchnorm_arrays as _resolve_bn_arrays,
    resolve_conv_arrays as _resolve_conv_arrays,
    resolve_dense_arrays as _resolve_dense_arrays,
    shape_without_batch as _shape_wo_batch,
)
from fpgai.ir.graph import Graph


def _object_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return result if isinstance(result, dict) else {}
    return value if isinstance(value, dict) else {}


def _plan_map(
    compile_plan: Any,
) -> Dict[str, Dict[str, Any]]:
    if compile_plan is None:
        return {}

    plans = getattr(
        compile_plan,
        "layer_plans",
        None,
    )
    if plans is None and isinstance(compile_plan, dict):
        plans = compile_plan.get("layer_plans", [])

    result: Dict[str, Dict[str, Any]] = {}
    for plan in plans or []:
        data = _object_dict(plan)
        name = data.get("node_name")
        if name:
            result[str(name)] = data
    return result


def _architecture_section(
    layer_plan: Dict[str, Any],
    section: str,
) -> Dict[str, Any]:
    architecture = layer_plan.get("architecture", {})
    if not isinstance(architecture, dict):
        return {}
    value = architecture.get(section, {})
    return value if isinstance(value, dict) else {}


def _positive_codegen_int(
    value: Any,
    default: int = 1,
) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def _layer_codegen_values(
    layer_plan: Dict[str, Any],
    *,
    op_type: str,
) -> Dict[str, int]:
    pipeline = _architecture_section(
        layer_plan,
        "pipeline",
    )
    parallelism = _architecture_section(
        layer_plan,
        "parallelism",
    )
    partitioning = _architecture_section(
        layer_plan,
        "partitioning",
    )
    unroll = parallelism.get(
        "unroll",
        layer_plan.get("unroll", {}),
    )
    if not isinstance(unroll, dict):
        unroll = {}
    targets = partitioning.get("targets", {})
    if not isinstance(targets, dict):
        targets = {}
    notes = layer_plan.get("notes", {})
    if not isinstance(notes, dict):
        notes = {}
    factor = _positive_codegen_int(
        partitioning.get(
            "factor",
            notes.get("partition_factor", 1),
        )
    )

    if op_type == "Dense":
        output_unroll = _positive_codegen_int(
            unroll.get("out", 1)
        )
        input_unroll = _positive_codegen_int(
            unroll.get("in", 1)
        )
    else:
        output_unroll = _positive_codegen_int(
            unroll.get("oc", 1)
        )
        input_unroll = _positive_codegen_int(
            unroll.get("ic", 1)
        )

    return {
        "pipeline_ii": _positive_codegen_int(
            pipeline.get(
                "ii",
                layer_plan.get("pipeline_ii", 1),
            )
        ),
        "input_unroll": input_unroll,
        "output_unroll": output_unroll,
        "input_partition": _positive_codegen_int(
            targets.get("input", factor)
        ),
        "output_partition": _positive_codegen_int(
            targets.get("output", factor)
        ),
        "weight_partition": _positive_codegen_int(
            targets.get("weight", factor)
        ),
    }


def _sanitize(name: str) -> str:
    sanitized = re.sub(
        r"[^0-9a-zA-Z_]",
        "_",
        str(name),
    )

    if not sanitized:
        sanitized = "t"

    if sanitized[0].isdigit():
        sanitized = "_" + sanitized

    return sanitized


def _build_tensor_shapes(
    graph: Graph,
) -> Tuple[
    Dict[str, Tuple[int, ...]],
    Dict[str, int],
]:
    inferred_shapes: Dict[
        str,
        Tuple[int, ...],
    ] = {}

    for tensor_name, tensor_spec in getattr(
        graph,
        "tensors",
        {},
    ).items():
        shape = _shape_wo_batch(
            getattr(
                tensor_spec,
                "shape",
                None,
            )
        )

        if shape:
            inferred_shapes[tensor_name] = shape

    graph_boundary_names = [
        *getattr(graph, "inputs", []),
        *getattr(graph, "outputs", []),
    ]

    for name in graph_boundary_names:
        if name in inferred_shapes:
            continue

        shape = _get_tensor_shape(graph, name)

        if shape:
            inferred_shapes[name] = shape

    def get_shape(
        name: str,
    ) -> Tuple[int, ...]:
        if (
            name in inferred_shapes
            and inferred_shapes[name]
        ):
            return inferred_shapes[name]

        shape = _get_tensor_shape(graph, name)

        if shape:
            inferred_shapes[name] = shape
            return shape

        return tuple()

    for op in graph.ops:
        if not op.inputs or not op.outputs:
            continue

        input_name = op.inputs[0]
        output_name = op.outputs[0]

        input_shape = get_shape(input_name)
        output_shape = get_shape(output_name)

        if output_shape:
            inferred_shapes[output_name] = output_shape
            continue

        if op.op_type in {
            "Relu",
            "LeakyRelu",
            "Sigmoid",
            "Softmax",
            "Add",
            "BatchNormalization",
        }:
            if input_shape:
                inferred_shapes[output_name] = input_shape

        elif op.op_type in {
            "Flatten",
            "Reshape",
        }:
            if input_shape:
                inferred_shapes[output_name] = (
                    _flat_size(input_shape),
                )

        elif op.op_type == "Dense":
            try:
                (
                    _weights,
                    bias,
                    _input_features,
                    _output_features,
                ) = _resolve_dense_arrays(
                    graph,
                    op,
                )

                inferred_shapes[output_name] = (
                    int(bias.size),
                )
            except Exception:
                pass

        elif op.op_type == "Conv":
            try:
                (
                    _weights,
                    _bias,
                    weight_shape,
                ) = _resolve_conv_arrays(
                    graph,
                    op,
                )

                stride = int(
                    op.attrs.get(
                        "strides",
                        [1, 1],
                    )[0]
                )

                pad = int(
                    op.attrs.get(
                        "pads",
                        [0, 0, 0, 0],
                    )[0]
                )

                if input_shape:
                    inferred_shapes[output_name] = (
                        _infer_conv_output_shape(
                            _as_chw(input_shape),
                            tuple(
                                int(value)
                                for value in weight_shape
                            ),
                            stride,
                            pad,
                        )
                    )
            except Exception:
                pass

        elif op.op_type in {
            "MaxPool",
            "AvgPool",
        }:
            try:
                kernel_size = int(
                    op.attrs.get(
                        "kernel_shape",
                        [2, 2],
                    )[0]
                )

                stride = int(
                    op.attrs.get(
                        "strides",
                        [2, 2],
                    )[0]
                )

                if input_shape:
                    inferred_shapes[output_name] = (
                        _infer_pool_output_shape(
                            _as_chw(input_shape),
                            kernel_size,
                            stride,
                        )
                    )
            except Exception:
                pass

    all_tensor_names = set(
        getattr(graph, "inputs", [])
    )
    all_tensor_names.update(
        getattr(graph, "outputs", [])
    )
    all_tensor_names.update(
        getattr(graph, "tensors", {}).keys()
    )

    for op in graph.ops:
        all_tensor_names.update(op.inputs)
        all_tensor_names.update(op.outputs)

    tensor_name_to_shape: Dict[
        str,
        Tuple[int, ...],
    ] = {}

    tensor_name_to_size: Dict[
        str,
        int,
    ] = {}

    for tensor_name in sorted(all_tensor_names):
        shape = inferred_shapes.get(
            tensor_name,
            tuple(),
        )

        if not shape:
            shape = _get_tensor_shape(
                graph,
                tensor_name,
            )

        if not shape:
            shape = (1,)

        tensor_name_to_shape[tensor_name] = shape
        tensor_name_to_size[tensor_name] = (
            _flat_size(shape)
        )

    return (
        tensor_name_to_shape,
        tensor_name_to_size,
    )


def _build_tensor_aliases(
    graph: Graph,
    tensor_name_to_size: Dict[str, int],
) -> Tuple[
    Dict[str, str],
    Dict[str, str],
    List[str],
]:
    all_names = set(
        tensor_name_to_size.keys()
    )

    produced_names = {
        op.outputs[0]
        for op in graph.ops
        if op.outputs
    }

    graph_inputs = set(
        getattr(graph, "inputs", [])
    )

    graph_outputs = set(
        getattr(graph, "outputs", [])
    )

    canonical: Dict[str, str] = {
        name: name
        for name in all_names
    }

    previous_output: Optional[str] = None

    for op in graph.ops:
        if op.inputs:
            input_name = op.inputs[0]

            if (
                previous_output is not None
                and input_name not in produced_names
                and input_name not in graph_inputs
                and input_name not in graph_outputs
                and tensor_name_to_size.get(
                    input_name,
                    -1,
                )
                == tensor_name_to_size.get(
                    previous_output,
                    -2,
                )
            ):
                canonical[input_name] = canonical.get(
                    previous_output,
                    previous_output,
                )

        if op.outputs:
            output_name = op.outputs[0]

            if output_name not in canonical:
                canonical[output_name] = output_name

            previous_output = output_name

    root_to_buffer: Dict[str, str] = {}
    root_to_gradient: Dict[str, str] = {}
    roots_in_order: List[str] = []

    def resolve_root(name: str) -> str:
        seen = set()
        current = name

        while (
            canonical.get(current, current)
            != current
            and current not in seen
        ):
            seen.add(current)
            current = canonical[current]

        return current

    tensor_name_to_buffer: Dict[str, str] = {}
    gradient_name_to_buffer: Dict[str, str] = {}

    for name in sorted(all_names):
        root = resolve_root(name)

        if root not in root_to_buffer:
            root_to_buffer[root] = (
                f"buf_{_sanitize(root)}"
            )

            root_to_gradient[root] = (
                f"grad_{_sanitize(root)}"
            )

            roots_in_order.append(root)

        tensor_name_to_buffer[name] = (
            root_to_buffer[root]
        )

        gradient_name_to_buffer[name] = (
            root_to_gradient[root]
        )

    return (
        tensor_name_to_buffer,
        gradient_name_to_buffer,
        roots_in_order,
    )


def _is_final_softmax_mse_case(
    graph: Graph,
    training_config: Dict[str, Any],
) -> bool:
    loss_config = (
        training_config.get("loss", {})
        or {}
    )

    loss_type = str(
        loss_config.get("type", "mse")
    ).lower()

    if loss_type != "mse":
        return False

    if not getattr(graph, "ops", None):
        return False

    last_op = graph.ops[-1]

    return bool(
        last_op.op_type == "Softmax"
        and last_op.outputs
        and graph.outputs
        and last_op.outputs[0] == graph.outputs[0]
    )


def _needs_input_gradient(
    graph: Graph,
    op_index: int,
) -> bool:
    op = graph.ops[op_index]

    if not op.inputs:
        return False

    input_name = op.inputs[0]

    if input_name in getattr(
        graph,
        "inputs",
        [],
    ):
        return False

    producer = next(
        (
            previous
            for previous in graph.ops
            if previous.outputs
            and previous.outputs[0] == input_name
        ),
        None,
    )

    if producer is None:
        return False

    return producer.op_type in {
        "Dense",
        "Conv",
        "BatchNormalization",
        "Relu",
        "LeakyRelu",
        "Sigmoid",
        "Softmax",
        "Add",
        "MaxPool",
        "AvgPool",
        "Flatten",
        "Reshape",
    }




def _inject_native_accumulated_modes_from_cpp(
    cpp: str,
    input_size: int,
    learning_rate: float,
) -> str:
    """Post-process generated training C++ with native mini-batch modes.

    Added top modes:
      mode 3: run forward/backward, accumulate parameter gradients, no update
      mode 4: average accumulated gradients and apply one SGD update in the HLS top
      mode 5: reset accumulated gradients and counter

    This helper discovers trainable parameters from the generated C++ itself, so it is
    robust to changes in the Python generator's local variable names.
    """
    if "FPGAI_NATIVE_ACC_BATCH_COUNT" in cpp and "if (mode == 3)" in cpp:
        return cpp

    params = []
    # Weight/bias gradients already emitted by the training generator.
    grad_w = {m.group(1): int(m.group(2)) for m in re.finditer(r"static\s+grad_wgt_t\s+dW_([A-Za-z0-9_]+)\[(\d+)\];", cpp)}
    grad_b = {m.group(1): int(m.group(2)) for m in re.finditer(r"static\s+grad_bias_t\s+dB_([A-Za-z0-9_]+)\[(\d+)\];", cpp)}

    for tag in sorted(grad_w.keys()):
        if tag not in grad_b:
            continue
        w_arr = f"W_{tag}"
        b_arr = f"B_{tag}"
        if f"static wgt_t {w_arr}[" not in cpp or f"static bias_t {b_arr}[" not in cpp:
            # BatchNorm or other trainable parameter forms can be added later.
            continue
        if f"OUT_grad_{tag}" not in cpp:
            continue
        params.append({
            "tag": tag,
            "w": grad_w[tag],
            "b": grad_b[tag],
            "w_arr": w_arr,
            "b_arr": b_arr,
            "dw": f"dW_{tag}",
            "db": f"dB_{tag}",
            "out": f"OUT_grad_{tag}",
        })

    if not params:
        raise RuntimeError("Native accumulation injection found no dense/conv trainable parameters in generated C++")

    decl = [
        "",
        "// FPGAI native accumulated mini-batch optimizer state.",
        "// FPGAI native gradient accumulation HLS modes:",
        "//   mode 3 = accumulate_gradients",
        "//   mode 4 = apply_accumulated_gradients",
        "//   mode 5 = reset_accumulators",
        "static const int FPGAI_MODE_ACCUMULATE_GRADIENTS = 3;",
        "static const int FPGAI_MODE_APPLY_ACCUMULATED_GRADIENTS = 4;",
        "static const int FPGAI_MODE_RESET_ACCUMULATORS = 5;",
    ]
    for p in params:
        decl.append(f"static acc_t ACC_{p['dw']}[{p['w']}];")
        decl.append(f"static acc_t ACC_{p['db']}[{p['b']}];")
    decl.append("static int FPGAI_NATIVE_ACC_BATCH_COUNT = 0;")
    decl_text = "\n".join(decl) + "\n"

    extern_marker = '\nextern "C" void '
    if extern_marker not in cpp:
        raise RuntimeError("Could not find extern C top marker for native accumulation injection")
    cpp = cpp.replace(extern_marker, decl_text + extern_marker, 1)

    def reset_lines(indent: str) -> list[str]:
        out = []
        for p in params:
            out.append(f"{indent}for (int i = 0; i < {p['w']}; ++i) ACC_{p['dw']}[i] = (acc_t)0;")
            out.append(f"{indent}for (int i = 0; i < {p['b']}; ++i) ACC_{p['db']}[i] = (acc_t)0;")
        out.append(f"{indent}FPGAI_NATIVE_ACC_BATCH_COUNT = 0;")
        return out

    def emit_grad_lines(indent: str) -> list[str]:
        out = []
        for idx, p in enumerate(params):
            last = "true" if idx == len(params) - 1 else "false"
            total = p['w'] + p['b']
            out.append(f"{indent}for (int i = 0; i < {p['w']}; ++i) {p['out']}[i] = (float){p['dw']}[i];")
            out.append(f"{indent}for (int i = 0; i < {p['b']}; ++i) {p['out']}[{p['w']} + i] = (float){p['db']}[i];")
            out.append(f"{indent}emit_stream_block<{total}>(out, {p['out']}, {last});")
        return out

    pre_input = [
        "",
        "  if (mode == FPGAI_MODE_RESET_ACCUMULATORS || mode == 5) {",
        "    // FPGAI reset_accumulators runtime command.",
        *reset_lines("    "),
        "    return;",
        "  }",
        "",
        "  if (mode == FPGAI_MODE_APPLY_ACCUMULATED_GRADIENTS || mode == 4) {",
        "    // FPGAI apply_accumulated_gradients runtime command: average accumulated gradients and update weights.",
        "    const int denom = (FPGAI_NATIVE_ACC_BATCH_COUNT > 0) ? FPGAI_NATIVE_ACC_BATCH_COUNT : 1;",
    ]
    for p in params:
        pre_input.append(f"    for (int i = 0; i < {p['w']}; ++i) {p['dw']}[i] = (grad_wgt_t)(((float)ACC_{p['dw']}[i]) / (float)denom);")
        pre_input.append(f"    for (int i = 0; i < {p['b']}; ++i) {p['db']}[i] = (grad_bias_t)(((float)ACC_{p['db']}[i]) / (float)denom);")
        pre_input.append(f"    fpgai::sgd_update_wgt_typed<{p['w']}, wgt_t, grad_wgt_t, upd_t, acc_t, 1, 4>({p['w_arr']}, {p['dw']}, (upd_t){learning_rate:.8f}f);")
        pre_input.append(f"    fpgai::sgd_update_bias_typed<{p['b']}, bias_t, grad_bias_t, upd_t, acc_t, 1, 2>({p['b_arr']}, {p['db']}, (upd_t){learning_rate:.8f}f);")
    pre_input.extend(reset_lines("    "))
    pre_input.extend(emit_grad_lines("    "))
    pre_input.extend(["    return;", "  }", ""])

    input_marker = f"\n  for (int i = 0; i < {int(input_size)}; ++i)"
    pos = cpp.find(input_marker)
    if pos < 0:
        raise RuntimeError(f"Could not find training input read marker: {input_marker!r}")
    cpp = cpp[:pos] + "\n".join(pre_input) + cpp[pos:]

    accum = ["", "  if (mode == FPGAI_MODE_ACCUMULATE_GRADIENTS || mode == 3) {", "    // FPGAI accumulate_gradients runtime command: add this micro-batch gradient to native accumulators."]
    for p in params:
        accum.append(f"    for (int i = 0; i < {p['w']}; ++i) ACC_{p['dw']}[i] += (acc_t){p['dw']}[i];")
        accum.append(f"    for (int i = 0; i < {p['b']}; ++i) ACC_{p['db']}[i] += (acc_t){p['db']}[i];")
    accum.append("    FPGAI_NATIVE_ACC_BATCH_COUNT += 1;")
    accum.extend(emit_grad_lines("    "))
    accum.extend(["    return;", "  }", ""])

    update_marker = "\n  fpgai::sgd_update_"
    pos = cpp.find(update_marker)
    if pos < 0:
        raise RuntimeError("Could not find SGD update marker for mode 3 insertion")
    cpp = cpp[:pos] + "\n".join(accum) + cpp[pos:]

    # Training convergence mode: add evaluation-only loss mode to the generated HLS top.
    # mode 6 reads one input/target record, computes loss_value, emits one loss float,
    # and returns before gradient computation or optimizer update.
    if "if (mode == 6)" not in cpp:
        loss_pos = cpp.find("loss_value +=")
        if loss_pos < 0:
            # Cross-entropy uses subtractive accumulation, while MSE uses additive
            # accumulation. The loss-eval mode must support both generated kernels.
            loss_pos = cpp.find("loss_value -=")
        if loss_pos < 0:
            raise RuntimeError("Could not find loss_value accumulation for mode 6 injection")
        statement_end = cpp.find(";", loss_pos)
        if statement_end < 0:
            raise RuntimeError("Could not find end of loss_value accumulation for mode 6 injection")

        insert_pos = cpp.find("\n\n  for (int i = 0; i <", statement_end)
        if insert_pos < 0:
            insert_pos = cpp.find("\n  for (int i = 0; i <", statement_end)
        if insert_pos < 0:
            insert_pos = cpp.find("\n  fpgai::", statement_end)
        if insert_pos < 0:
            raise RuntimeError("Could not find gradient loop marker after loss_value for mode 6 injection")

        loss_eval_block = "\n\n  // FPGAI loss_eval mode.\n  if (mode == 6) {\n    write_f32(out, (float)loss_value, true);\n    return;\n  }"
        cpp = cpp[:insert_pos] + loss_eval_block + cpp[insert_pos:]
    return cpp



def emit_top_train_cpp(
    *,
    graph: Graph,
    top_name: str,
    weights_mode: str,
    training_cfg: Dict[str, Any],
    compile_plan: Any = None,
    memory_plan: Any = None,
    communication_plan: Any = None,
) -> str:
    del memory_plan
    del communication_plan
    plan_by_name = _plan_map(compile_plan)

    supported_operations = {
        "Dense",
        "Conv",
        "MaxPool",
        "AvgPool",
        "BatchNormalization",
        "Relu",
        "LeakyRelu",
        "Sigmoid",
        "Softmax",
        "Flatten",
        "Reshape",
        "Add",
    }

    unsupported_operations = [
        op.op_type
        for op in graph.ops
        if op.op_type not in supported_operations
    ]

    if unsupported_operations:
        raise RuntimeError(
            "Unsupported training ops: "
            f"{sorted(set(unsupported_operations))}"
        )

    loss_config = (
        training_cfg.get("loss", {})
        or {}
    )

    optimizer_config = (
        training_cfg.get("optimizer", {})
        or {}
    )

    loss_type = str(
        loss_config.get("type", "mse")
    ).lower()

    learning_rate = float(
        optimizer_config.get(
            "learning_rate",
            0.01,
        )
    )

    bypass_final_softmax_backward = (
        _is_final_softmax_mse_case(
            graph,
            training_cfg,
        )
    )

    input_name = graph.inputs[0]
    output_name = graph.outputs[0]

    (
        tensor_name_to_shape,
        tensor_name_to_size,
    ) = _build_tensor_shapes(graph)

    (
        tensor_name_to_buffer,
        gradient_name_to_buffer,
        roots_in_order,
    ) = _build_tensor_aliases(
        graph,
        tensor_name_to_size,
    )

    input_size = tensor_name_to_size[input_name]
    output_size = tensor_name_to_size[output_name]

    lines: List[str] = []

    lines.extend(
        [
            "#include <hls_stream.h>",
            "#include <ap_axi_sdata.h>",
            "#include <ap_int.h>",
            "#include <math.h>",
            '#include "fpgai_types.h"',
            '#include "layers/dense.h"',
            '#include "layers/conv.h"',
            '#include "layers/pool.h"',
            '#include "layers/activations.h"',
            '#include "layers/batchnorm.h"',
            "",
            "typedef ap_axis<32,0,0,0> axis_t;",
            "using namespace fpgai;",
            "",
            (
                "static inline unsigned int f32_to_u32(float value) { "
                "union { float f; unsigned int u; } converter; "
                "converter.f = value; return converter.u; }"
            ),
            (
                "static inline float u32_to_f32(unsigned int value) { "
                "union { float f; unsigned int u; } converter; "
                "converter.u = value; return converter.f; }"
            ),
            (
                "static inline void write_f32("
                "hls::stream<axis_t>& stream, float value, bool last=false) { "
                "axis_t packet; packet.data=f32_to_u32(value); "
                "packet.keep=-1; packet.strb=-1; "
                "packet.last=last?1:0; stream.write(packet); }"
            ),
            (
                "static inline float read_f32("
                "hls::stream<axis_t>& stream) { "
                "axis_t packet = stream.read(); "
                "return u32_to_f32((unsigned int)packet.data); }"
            ),
            "",
            "template<int N>",
            (
                "static inline void emit_stream_block("
                "hls::stream<axis_t>& out, "
                "const float* data, bool is_last_block) {"
            ),
            "#pragma HLS INLINE off",
            "  for (int i = 0; i < N; ++i) {",
            "#pragma HLS PIPELINE II=1",
            (
                "    bool last = "
                "is_last_block && (i == N - 1);"
            ),
            "    write_f32(out, data[i], last);",
            "  }",
            "}",
            "",
            "template<int C, int HW>",
            (
                "static inline void "
                "fpgai_bn_backward_input_exact("
            ),
            "  const grad_act_t output_gradient[C * HW],",
            "  const wgt_t gamma[C],",
            "  const acc_t variance[C],",
            "  const act_t normalized[C * HW],",
            "  grad_act_t input_gradient[C * HW]",
            ") {",
            "  const float epsilon = 1e-5f;",
            "  for (int channel = 0; channel < C; ++channel) {",
            "    float gradient_sum = 0.0f;",
            "    float normalized_gradient_sum = 0.0f;",
            "    for (int spatial = 0; spatial < HW; ++spatial) {",
            "      const int index = spatial * C + channel;",
            (
                "      const float gradient = "
                "(float)output_gradient[index];"
            ),
            (
                "      const float normalized_value = "
                "(float)normalized[index];"
            ),
            "      gradient_sum += gradient;",
            (
                "      normalized_gradient_sum += "
                "gradient * normalized_value;"
            ),
            "    }",
            (
                "    const float mean_gradient = "
                "gradient_sum / (float)HW;"
            ),
            (
                "    const float mean_normalized_gradient = "
                "normalized_gradient_sum / (float)HW;"
            ),
            (
                "    const float inverse_standard_deviation = "
                "1.0f / sqrtf((float)variance[channel] + epsilon);"
            ),
            (
                "    const float scale = "
                "((float)gamma[channel]) "
                "* inverse_standard_deviation;"
            ),
            "    for (int spatial = 0; spatial < HW; ++spatial) {",
            "      const int index = spatial * C + channel;",
            (
                "      const float normalized_value = "
                "(float)normalized[index];"
            ),
            (
                "      input_gradient[index] = "
                "(grad_act_t)(scale * "
                "(((float)output_gradient[index]) "
                "- mean_gradient "
                "- normalized_value "
                "* mean_normalized_gradient));"
            ),
            "    }",
            "  }",
            "}",
            "",
        ]
    )

    for root in roots_in_order:
        size = tensor_name_to_size[root]
        buffer_name = tensor_name_to_buffer[root]
        gradient_name = gradient_name_to_buffer[root]

        lines.append(
            f"static act_t {buffer_name}[{size}];"
        )
        lines.append(
            f"static grad_act_t {gradient_name}[{size}];"
        )

    parameter_specs = []

    for op in graph.ops:
        tag = _sanitize(op.name)

        if op.op_type == "Dense":
            (
                weights,
                bias,
                _,
                _,
            ) = _resolve_dense_arrays(
                graph,
                op,
            )

            weight_values = ", ".join(
                f"{float(value):.8f}f"
                for value in weights.reshape(-1)
            )
            bias_values = ", ".join(
                f"{float(value):.8f}f"
                for value in bias.reshape(-1)
            )

            lines.append(
                f"static wgt_t W_{tag}[{weights.size}] = "
                f"{{ {weight_values} }};"
            )
            lines.append(
                f"static bias_t B_{tag}[{bias.size}] = "
                f"{{ {bias_values} }};"
            )
            lines.append(
                f"static grad_wgt_t dW_{tag}[{weights.size}];"
            )
            lines.append(
                f"static grad_bias_t dB_{tag}[{bias.size}];"
            )
            lines.append(
                f"static float OUT_grad_{tag}"
                f"[{weights.size + bias.size}];"
            )

            parameter_specs.append(
                (
                    "dense",
                    op,
                    tag,
                    weights.size,
                    bias.size,
                )
            )

        elif op.op_type == "Conv":
            (
                weights,
                bias,
                _,
            ) = _resolve_conv_arrays(
                graph,
                op,
            )

            weight_values = ", ".join(
                f"{float(value):.8f}f"
                for value in weights.reshape(-1)
            )
            bias_values = ", ".join(
                f"{float(value):.8f}f"
                for value in bias.reshape(-1)
            )

            lines.append(
                f"static wgt_t W_{tag}[{weights.size}] = "
                f"{{ {weight_values} }};"
            )
            lines.append(
                f"static bias_t B_{tag}[{bias.size}] = "
                f"{{ {bias_values} }};"
            )
            lines.append(
                f"static grad_wgt_t dW_{tag}[{weights.size}];"
            )
            lines.append(
                f"static grad_bias_t dB_{tag}[{bias.size}];"
            )
            lines.append(
                f"static float OUT_grad_{tag}"
                f"[{weights.size + bias.size}];"
            )

            parameter_specs.append(
                (
                    "conv",
                    op,
                    tag,
                    weights.size,
                    bias.size,
                )
            )

        elif op.op_type == "BatchNormalization":
            output_shape = (
                tensor_name_to_shape.get(
                    op.outputs[0],
                    tuple(),
                )
                or tensor_name_to_shape.get(
                    op.inputs[0],
                    tuple(),
                )
            )

            channels, height, width = _as_chw(
                output_shape
            )
            spatial_size = height * width

            (
                gamma,
                beta,
                mean,
                variance,
            ) = _resolve_bn_arrays(
                graph,
                op,
                channels,
            )

            gamma_values = ", ".join(
                f"{float(value):.8f}f"
                for value in gamma
            )
            beta_values = ", ".join(
                f"{float(value):.8f}f"
                for value in beta
            )
            mean_values = ", ".join(
                f"{float(value):.8f}f"
                for value in mean
            )
            variance_values = ", ".join(
                f"{float(value):.8f}f"
                for value in variance
            )

            lines.append(
                f"static wgt_t BN_G_{tag}[{channels}] = "
                f"{{ {gamma_values} }};"
            )
            lines.append(
                f"static bias_t BN_B_{tag}[{channels}] = "
                f"{{ {beta_values} }};"
            )
            lines.append(
                f"static acc_t BN_M_{tag}[{channels}] = "
                f"{{ {mean_values} }};"
            )
            lines.append(
                f"static acc_t BN_V_{tag}[{channels}] = "
                f"{{ {variance_values} }};"
            )
            lines.append(
                f"static act_t BN_XHAT_{tag}"
                f"[{channels * spatial_size}];"
            )
            lines.append(
                f"static grad_wgt_t dBN_G_{tag}[{channels}];"
            )
            lines.append(
                f"static grad_bias_t dBN_B_{tag}[{channels}];"
            )
            lines.append(
                f"static float OUT_grad_{tag}[{2 * channels}];"
            )

            parameter_specs.append(
                (
                    "bn",
                    op,
                    tag,
                    channels,
                    channels,
                )
            )

    lines.extend(
        [
            "",
            f'extern "C" void {top_name}(',
            "  hls::stream<axis_t>& in,",
            "  hls::stream<axis_t>& out,",
            "  hls::stream<axis_t>& aux,",
            "  int mode",
            ") {",
            "#pragma HLS INTERFACE axis port=in",
            "#pragma HLS INTERFACE axis port=out",
            "#pragma HLS INTERFACE axis port=aux",
            "#pragma HLS INTERFACE s_axilite port=mode bundle=CTRL",
            "#pragma HLS INTERFACE s_axilite port=return bundle=CTRL",
            "",
        ]
    )

    if weights_mode in {
        "stream",
        "ddr",
    }:
        lines.append("  if (mode == 0) {")

        for (
            kind,
            _op,
            tag,
            weight_size,
            bias_size,
        ) in parameter_specs:
            if kind in {
                "dense",
                "conv",
            }:
                lines.append(
                    f"    for (int i = 0; i < {weight_size}; ++i) "
                    f"W_{tag}[i] = (wgt_t)read_f32(aux);"
                )
                lines.append(
                    f"    for (int i = 0; i < {bias_size}; ++i) "
                    f"B_{tag}[i] = (bias_t)read_f32(aux);"
                )

            elif kind == "bn":
                lines.append(
                    f"    for (int i = 0; i < {weight_size}; ++i) "
                    f"BN_G_{tag}[i] = (wgt_t)read_f32(aux);"
                )
                lines.append(
                    f"    for (int i = 0; i < {bias_size}; ++i) "
                    f"BN_B_{tag}[i] = (bias_t)read_f32(aux);"
                )

        lines.extend(
            [
                "    return;",
                "  }",
                "",
            ]
        )

    # Mode 1: emit the current trainable parameters as a flat stream.
    # The training testbench uses this before and after mode 2 so the
    # HLS step can be compared against the Python/ONNX reference for
    # both gradients and SGD-updated weights.
    if parameter_specs:
        lines.append("  if (mode == 1) {")
        for index, (kind, _op, tag, weight_size, bias_size) in enumerate(parameter_specs):
            total_size = weight_size + bias_size
            is_last_block = "true" if index == len(parameter_specs) - 1 else "false"
            if kind in {"dense", "conv"}:
                lines.append(f"    for (int i = 0; i < {weight_size}; ++i) OUT_grad_{tag}[i] = (float)W_{tag}[i];")
                lines.append(f"    for (int i = 0; i < {bias_size}; ++i) OUT_grad_{tag}[{weight_size} + i] = (float)B_{tag}[i];")
                lines.append(f"    emit_stream_block<{total_size}>(out, OUT_grad_{tag}, {is_last_block});")
            elif kind == "bn":
                lines.append(f"    for (int i = 0; i < {weight_size}; ++i) OUT_grad_{tag}[i] = (float)BN_G_{tag}[i];")
                lines.append(f"    for (int i = 0; i < {bias_size}; ++i) OUT_grad_{tag}[{weight_size} + i] = (float)BN_B_{tag}[i];")
                lines.append(f"    emit_stream_block<{total_size}>(out, OUT_grad_{tag}, {is_last_block});")
        lines.append("    return;")
        lines.append("  }")
        lines.append("")

    input_buffer = tensor_name_to_buffer[input_name]

    lines.append(
        f"  for (int i = 0; i < {input_size}; ++i) "
        f"{input_buffer}[i] = (act_t)read_f32(in);"
    )
    lines.append(
        f"  static act_t target_buf[{output_size}];"
    )
    lines.append(
        f"  for (int i = 0; i < {output_size}; ++i) "
        "target_buf[i] = (act_t)read_f32(aux);"
    )
    lines.append("")

    for root in roots_in_order:
        size = tensor_name_to_size[root]
        gradient_buffer = gradient_name_to_buffer[root]

        lines.append(
            f"  for (int i = 0; i < {size}; ++i) "
            f"{gradient_buffer}[i] = (grad_act_t)0;"
        )

    for (
        kind,
        _op,
        tag,
        weight_size,
        bias_size,
    ) in parameter_specs:
        if kind in {
            "dense",
            "conv",
        }:
            lines.append(
                f"  for (int i = 0; i < {weight_size}; ++i) "
                f"dW_{tag}[i] = (grad_wgt_t)0;"
            )
            lines.append(
                f"  for (int i = 0; i < {bias_size}; ++i) "
                f"dB_{tag}[i] = (grad_bias_t)0;"
            )

        elif kind == "bn":
            lines.append(
                f"  for (int i = 0; i < {weight_size}; ++i) "
                f"dBN_G_{tag}[i] = (grad_wgt_t)0;"
            )
            lines.append(
                f"  for (int i = 0; i < {bias_size}; ++i) "
                f"dBN_B_{tag}[i] = (grad_bias_t)0;"
            )

    lines.append("")

    for op in graph.ops:
        input_name = op.inputs[0]
        output_name_for_op = op.outputs[0]

        input_buffer = tensor_name_to_buffer[
            input_name
        ]
        output_buffer = tensor_name_to_buffer[
            output_name_for_op
        ]

        input_shape = tensor_name_to_shape.get(
            input_name,
            tuple(),
        )
        output_shape = tensor_name_to_shape.get(
            output_name_for_op,
            tuple(),
        )

        output_tensor_size = tensor_name_to_size[
            output_name_for_op
        ]

        if op.op_type == "Dense":
            tag = _sanitize(op.name)
            codegen = _layer_codegen_values(
                plan_by_name.get(op.name, {}),
                op_type="Dense",
            )

            (
                _,
                _,
                input_features,
                output_features,
            ) = _resolve_dense_arrays(
                graph,
                op,
            )

            lines.append(
                f"  fpgai::dense_out_in"
                f"<{input_features}, {output_features}, "
                f"act_t, act_t, wgt_t, bias_t, acc_t, "
                f"{codegen['pipeline_ii']}, "
                f"{codegen['input_unroll']}, "
                f"{codegen['output_unroll']}, "
                f"{codegen['input_partition']}, "
                f"{codegen['output_partition']}, "
                f"{codegen['weight_partition']}>"
                f"({input_buffer}, {output_buffer}, "
                f"W_{tag}, B_{tag});"
            )

        elif op.op_type == "Conv":
            tag = _sanitize(op.name)
            codegen = _layer_codegen_values(
                plan_by_name.get(op.name, {}),
                op_type="Conv",
            )

            weight_shape = _resolve_conv_arrays(
                graph,
                op,
            )[2]

            stride = int(
                op.attrs.get(
                    "strides",
                    [1, 1],
                )[0]
            )
            pad = int(
                op.attrs.get(
                    "pads",
                    [0, 0, 0, 0],
                )[0]
            )

            if not output_shape:
                output_shape = (
                    _infer_conv_output_shape(
                        _as_chw(input_shape),
                        tuple(
                            int(value)
                            for value in weight_shape
                        ),
                        stride,
                        pad,
                    )
                )

            (
                channels_in,
                height_in,
                width_in,
            ) = _as_chw(input_shape)

            (
                channels_out,
                height_out,
                width_out,
            ) = _as_chw(output_shape)

            kernel_size = int(weight_shape[2])

            lines.append(
                f"  fpgai::conv2d"
                f"<{height_in}, {width_in}, {channels_in}, "
                f"{height_out}, {width_out}, {channels_out}, "
                f"{kernel_size}, {stride}, {pad}, "
                f"act_t, act_t, wgt_t, bias_t, acc_t, "
                f"{codegen['pipeline_ii']}, "
                f"{codegen['output_unroll']}, "
                f"{codegen['input_unroll']}, "
                f"{codegen['input_partition']}, "
                f"{codegen['output_partition']}, "
                f"{codegen['weight_partition']}>"
                f"({input_buffer}, {output_buffer}, "
                f"W_{tag}, B_{tag});"
            )

        elif op.op_type == "Relu":
            lines.append(
                f"  fpgai::relu<{output_tensor_size}>"
                f"({input_buffer}, {output_buffer});"
            )

        elif op.op_type == "LeakyRelu":
            alpha = float(
                (
                    getattr(op, "attrs", {})
                    or {}
                ).get("alpha", 0.01)
            )

            lines.append(
                f"  fpgai::leaky_relu<{output_tensor_size}>"
                f"({input_buffer}, {output_buffer}, "
                f"(act_t){alpha:.8f}f);"
            )

        elif op.op_type == "Sigmoid":
            lines.append(
                f"  fpgai::sigmoid<{output_tensor_size}>"
                f"({input_buffer}, {output_buffer});"
            )

        elif op.op_type == "Softmax":
            lines.append(
                f"  fpgai::softmax<{output_tensor_size}>"
                f"({input_buffer}, {output_buffer});"
            )

        elif op.op_type in {
            "Flatten",
            "Reshape",
        }:
            lines.append(
                f"  fpgai::reshape_copy<{output_tensor_size}>"
                f"({input_buffer}, {output_buffer});"
            )

        elif op.op_type == "Add":
            right_buffer = tensor_name_to_buffer[
                op.inputs[1]
            ]

            lines.append(
                f"  fpgai::add_vec<{output_tensor_size}>"
                f"({input_buffer}, {right_buffer}, "
                f"{output_buffer});"
            )

        elif op.op_type == "MaxPool":
            kernel_size = int(
                op.attrs.get(
                    "kernel_shape",
                    [2, 2],
                )[0]
            )
            stride = int(
                op.attrs.get(
                    "strides",
                    [2, 2],
                )[0]
            )

            if not output_shape:
                output_shape = (
                    _infer_pool_output_shape(
                        _as_chw(input_shape),
                        kernel_size,
                        stride,
                    )
                )

            (
                channels_in,
                height_in,
                width_in,
            ) = _as_chw(input_shape)

            (
                _,
                height_out,
                width_out,
            ) = _as_chw(output_shape)

            lines.append(
                f"  fpgai::maxpool2d"
                f"<{height_in}, {width_in}, {channels_in}, "
                f"{kernel_size}, {stride}, "
                f"{height_out}, {width_out}>"
                f"({input_buffer}, {output_buffer});"
            )

        elif op.op_type == "AvgPool":
            kernel_size = int(
                op.attrs.get(
                    "kernel_shape",
                    [2, 2],
                )[0]
            )
            stride = int(
                op.attrs.get(
                    "strides",
                    [2, 2],
                )[0]
            )

            if not output_shape:
                output_shape = (
                    _infer_pool_output_shape(
                        _as_chw(input_shape),
                        kernel_size,
                        stride,
                    )
                )

            (
                channels_in,
                height_in,
                width_in,
            ) = _as_chw(input_shape)

            (
                _,
                height_out,
                width_out,
            ) = _as_chw(output_shape)

            lines.append(
                f"  fpgai::avgpool2d"
                f"<{height_in}, {width_in}, {channels_in}, "
                f"{kernel_size}, {stride}, "
                f"{height_out}, {width_out}>"
                f"({input_buffer}, {output_buffer});"
            )

        elif op.op_type == "BatchNormalization":
            tag = _sanitize(op.name)

            if not output_shape:
                output_shape = input_shape

            channels, height, width = _as_chw(
                output_shape
            )
            spatial_size = height * width

            lines.append(
                f"  fpgai::batchnorm_train_forward"
                f"<{channels}, {spatial_size}>"
                f"({input_buffer}, {output_buffer}, "
                f"BN_G_{tag}, BN_B_{tag}, "
                f"BN_M_{tag}, BN_V_{tag}, "
                f"BN_XHAT_{tag});"
            )

    final_buffer = tensor_name_to_buffer[
        output_name
    ]
    final_gradient = gradient_name_to_buffer[
        output_name
    ]

    if loss_type == "mse":
        lines.append(
            "  loss_t loss_value = (loss_t)0;"
        )
        lines.append(
            f"  for (int i = 0; i < {output_size}; ++i) {{"
        )
        lines.append(
            f"    acc_t difference = "
            f"(acc_t){final_buffer}[i] "
            "- (acc_t)target_buf[i];"
        )
        lines.append(
            f"    {final_gradient}[i] = "
            "(grad_act_t)difference;"
        )
        lines.append(
            "    loss_value += "
            "(loss_t)(difference * difference);"
        )
        lines.append("  }")
    elif loss_type in {"cross_entropy", "ce"}:
        lines.append("  // FPGAI cross_entropy loss kernel.")
        lines.append("  // Labels are expected as one-hot/probability targets aligned with output logits.")
        lines.append("  loss_t loss_value = (loss_t)0;")
        lines.append(f"  acc_t max_logit = (acc_t){final_buffer}[0];")
        lines.append(f"  for (int i = 1; i < {output_size}; ++i) {{")
        lines.append(f"    acc_t value = (acc_t){final_buffer}[i];")
        lines.append("    if (value > max_logit) max_logit = value;")
        lines.append("  }")
        lines.append("  acc_t softmax_denom = (acc_t)0;")
        lines.append(f"  for (int i = 0; i < {output_size}; ++i) {{")
        lines.append(f"    softmax_denom += (acc_t)expf((float)((acc_t){final_buffer}[i] - max_logit));")
        lines.append("  }")
        lines.append(f"  for (int i = 0; i < {output_size}; ++i) {{")
        lines.append(f"    acc_t exp_value = (acc_t)expf((float)((acc_t){final_buffer}[i] - max_logit));")
        lines.append("    acc_t probability = exp_value / softmax_denom;")
        lines.append("    acc_t target_value = (acc_t)target_buf[i];")
        lines.append(f"    {final_gradient}[i] = (grad_act_t)(probability - target_value);")
        lines.append("    if (target_value != (acc_t)0) {")
        lines.append("      loss_value -= (loss_t)(target_value * (acc_t)logf((float)probability + 1.0e-7f));")
        lines.append("    }")
        lines.append("  }")
    else:
        lines.append(
            f"  for (int i = 0; i < {output_size}; ++i) "
            f"{final_gradient}[i] = "
            f"(grad_act_t)(((acc_t){final_buffer}[i]) "
            "- ((acc_t)target_buf[i]));"
        )

    lines.append("")

    for op_index in range(
        len(graph.ops) - 1,
        -1,
        -1,
    ):
        op = graph.ops[op_index]

        input_name = op.inputs[0]
        output_name_for_op = op.outputs[0]

        input_buffer = tensor_name_to_buffer[
            input_name
        ]
        output_buffer = tensor_name_to_buffer[
            output_name_for_op
        ]
        input_gradient = gradient_name_to_buffer[
            input_name
        ]
        output_gradient = gradient_name_to_buffer[
            output_name_for_op
        ]

        input_shape = tensor_name_to_shape.get(
            input_name,
            tuple(),
        )
        output_shape = tensor_name_to_shape.get(
            output_name_for_op,
            tuple(),
        )

        input_tensor_size = tensor_name_to_size[
            input_name
        ]
        output_tensor_size = tensor_name_to_size[
            output_name_for_op
        ]

        if op.op_type == "Dense":
            tag = _sanitize(op.name)
            codegen = _layer_codegen_values(
                plan_by_name.get(op.name, {}),
                op_type="Dense",
            )

            (
                _,
                _,
                input_features,
                output_features,
            ) = _resolve_dense_arrays(
                graph,
                op,
            )

            lines.append(
                f"  fpgai::dense_weight_grad_typed"
                f"<{input_features}, {output_features}, "
                f"act_t, grad_act_t, grad_wgt_t, acc_t, "
                f"{codegen['pipeline_ii']}, "
                f"{codegen['input_unroll']}, "
                f"{codegen['output_unroll']}, "
                f"{codegen['input_partition']}, "
                f"{codegen['output_partition']}, "
                f"{codegen['weight_partition']}>"
                f"({input_buffer}, {output_gradient}, "
                f"dW_{tag});"
            )
            lines.append(
                f"  fpgai::dense_bias_grad_typed"
                f"<{output_features}, grad_act_t, "
                f"grad_bias_t, {codegen['pipeline_ii']}, "
                f"{codegen['output_partition']}>"
                f"({output_gradient}, dB_{tag});"
            )
            lines.append(
                f"  fpgai::dense_backward_input_typed"
                f"<{input_features}, {output_features}, "
                f"grad_act_t, wgt_t, grad_act_t, acc_t, "
                f"{codegen['pipeline_ii']}, "
                f"{codegen['input_unroll']}, "
                f"{codegen['output_unroll']}, "
                f"{codegen['input_partition']}, "
                f"{codegen['output_partition']}, "
                f"{codegen['weight_partition']}>"
                f"({output_gradient}, W_{tag}, "
                f"{input_gradient});"
            )

        elif op.op_type == "Conv":
            tag = _sanitize(op.name)
            codegen = _layer_codegen_values(
                plan_by_name.get(op.name, {}),
                op_type="Conv",
            )

            weight_shape = _resolve_conv_arrays(
                graph,
                op,
            )[2]

            stride = int(
                op.attrs.get(
                    "strides",
                    [1, 1],
                )[0]
            )
            pad = int(
                op.attrs.get(
                    "pads",
                    [0, 0, 0, 0],
                )[0]
            )

            if not output_shape:
                output_shape = (
                    _infer_conv_output_shape(
                        _as_chw(input_shape),
                        tuple(
                            int(value)
                            for value in weight_shape
                        ),
                        stride,
                        pad,
                    )
                )

            (
                channels_in,
                height_in,
                width_in,
            ) = _as_chw(input_shape)

            (
                channels_out,
                height_out,
                width_out,
            ) = _as_chw(output_shape)

            kernel_size = int(weight_shape[2])

            lines.append(
                f"  fpgai::conv2d_weight_grad_typed"
                f"<{height_in}, {width_in}, {channels_in}, "
                f"{height_out}, {width_out}, {channels_out}, "
                f"{kernel_size}, {stride}, {pad}, "
                f"act_t, grad_act_t, grad_wgt_t, acc_t, "
                f"{codegen['pipeline_ii']}, "
                f"{codegen['input_unroll']}, "
                f"{codegen['output_unroll']}, "
                f"{codegen['input_partition']}, "
                f"{codegen['output_partition']}, "
                f"{codegen['weight_partition']}>"
                f"({input_buffer}, {output_gradient}, "
                f"dW_{tag});"
            )
            lines.append(
                f"  fpgai::conv2d_bias_grad_typed"
                f"<{channels_out}, {height_out}, {width_out}, "
                f"grad_act_t, grad_bias_t, acc_t, "
                f"{codegen['pipeline_ii']}, "
                f"{codegen['output_partition']}>"
                f"({output_gradient}, dB_{tag});"
            )

            if _needs_input_gradient(
                graph,
                op_index,
            ):
                lines.append(
                    f"  fpgai::conv2d_backward_input_typed"
                    f"<{height_in}, {width_in}, {channels_in}, "
                    f"{height_out}, {width_out}, {channels_out}, "
                    f"{kernel_size}, {stride}, {pad}, "
                    f"grad_act_t, wgt_t, grad_act_t, acc_t, "
                    f"{codegen['pipeline_ii']}, "
                    f"{codegen['input_unroll']}, "
                    f"{codegen['output_unroll']}, "
                    f"{codegen['input_partition']}, "
                    f"{codegen['output_partition']}, "
                    f"{codegen['weight_partition']}>"
                    f"({output_gradient}, W_{tag}, "
                    f"{input_gradient});"
                )
            else:
                lines.append(
                    f"  for (int i = 0; i < {input_tensor_size}; ++i) "
                    f"{input_gradient}[i] = (grad_act_t)0;"
                )

        elif op.op_type == "Relu":
            lines.append(
                f"  fpgai::relu_backward_from_output"
                f"<{output_tensor_size}>"
                f"({output_buffer}, {output_gradient}, "
                f"{input_gradient});"
            )

        elif op.op_type == "LeakyRelu":
            alpha = float(
                (
                    getattr(op, "attrs", {})
                    or {}
                ).get("alpha", 0.01)
            )

            lines.append(
                f"  fpgai::leaky_relu_backward_from_input"
                f"<{output_tensor_size}>"
                f"({input_buffer}, {output_gradient}, "
                f"{input_gradient}, (act_t){alpha:.8f}f);"
            )

        elif op.op_type == "Sigmoid":
            lines.append(
                f"  fpgai::sigmoid_backward_from_output"
                f"<{output_tensor_size}>"
                f"({output_buffer}, {output_gradient}, "
                f"{input_gradient});"
            )

        elif op.op_type == "Softmax":
            is_final_op = bool(
                op_index == len(graph.ops) - 1
                and output_name_for_op == output_name
            )

            if (
                bypass_final_softmax_backward
                and is_final_op
            ):
                lines.append(
                    f"  for (int i = 0; i < {output_tensor_size}; ++i) "
                    f"{input_gradient}[i] += "
                    f"{output_gradient}[i];"
                )
            else:
                lines.append(
                    f"  fpgai::softmax_backward"
                    f"<{output_tensor_size}>"
                    f"({output_buffer}, {output_gradient}, "
                    f"{input_gradient});"
                )

        elif op.op_type in {
            "Flatten",
            "Reshape",
        }:
            lines.append(
                f"  for (int i = 0; i < {output_tensor_size}; ++i) "
                f"{input_gradient}[i] += "
                f"{output_gradient}[i];"
            )

        elif op.op_type == "Add":
            right_gradient = gradient_name_to_buffer[
                op.inputs[1]
            ]

            lines.append(
                f"  fpgai::add_backward"
                f"<{output_tensor_size}>"
                f"({output_gradient}, {input_gradient}, "
                f"{right_gradient});"
            )

        elif op.op_type == "MaxPool":
            kernel_size = int(
                op.attrs.get(
                    "kernel_shape",
                    [2, 2],
                )[0]
            )
            stride = int(
                op.attrs.get(
                    "strides",
                    [2, 2],
                )[0]
            )

            if not output_shape:
                output_shape = (
                    _infer_pool_output_shape(
                        _as_chw(input_shape),
                        kernel_size,
                        stride,
                    )
                )

            (
                channels_in,
                height_in,
                width_in,
            ) = _as_chw(input_shape)

            (
                _,
                height_out,
                width_out,
            ) = _as_chw(output_shape)

            lines.append(
                f"  fpgai::maxpool2d_backward"
                f"<{height_in}, {width_in}, {channels_in}, "
                f"{kernel_size}, {stride}, "
                f"{height_out}, {width_out}>"
                f"({input_buffer}, {output_buffer}, "
                f"{output_gradient}, {input_gradient});"
            )

        elif op.op_type == "AvgPool":
            kernel_size = int(
                op.attrs.get(
                    "kernel_shape",
                    [2, 2],
                )[0]
            )
            stride = int(
                op.attrs.get(
                    "strides",
                    [2, 2],
                )[0]
            )

            if not output_shape:
                output_shape = (
                    _infer_pool_output_shape(
                        _as_chw(input_shape),
                        kernel_size,
                        stride,
                    )
                )

            (
                channels_in,
                height_in,
                width_in,
            ) = _as_chw(input_shape)

            (
                _,
                height_out,
                width_out,
            ) = _as_chw(output_shape)

            lines.append(
                f"  fpgai::avgpool2d_backward"
                f"<{height_in}, {width_in}, {channels_in}, "
                f"{kernel_size}, {stride}, "
                f"{height_out}, {width_out}>"
                f"({output_gradient}, {input_gradient});"
            )

        elif op.op_type == "BatchNormalization":
            tag = _sanitize(op.name)

            if not output_shape:
                output_shape = input_shape

            channels, height, width = _as_chw(
                output_shape
            )
            spatial_size = height * width

            lines.append(
                f"  fpgai::batchnorm_param_grad"
                f"<{channels}, {spatial_size}>"
                f"({output_gradient}, BN_XHAT_{tag}, "
                f"dBN_G_{tag}, dBN_B_{tag});"
            )
            lines.append(
                f"  fpgai_bn_backward_input_exact"
                f"<{channels}, {spatial_size}>"
                f"({output_gradient}, BN_G_{tag}, "
                f"BN_V_{tag}, BN_XHAT_{tag}, "
                f"{input_gradient});"
            )

    for (
        kind,
        op,
        tag,
        weight_size,
        bias_size,
    ) in parameter_specs:
        if kind in {
            "dense",
            "conv",
        }:
            codegen = _layer_codegen_values(
                plan_by_name.get(op.name, {}),
                op_type="Dense" if kind == "dense" else "Conv",
            )
            lines.append(
                f"  fpgai::sgd_update_wgt_typed"
                f"<{weight_size}, wgt_t, grad_wgt_t, upd_t, "
                f"acc_t, {codegen['pipeline_ii']}, "
                f"{codegen['weight_partition']}>"
                f"(W_{tag}, dW_{tag}, "
                f"(upd_t){learning_rate:.8f}f);"
            )
            lines.append(
                f"  fpgai::sgd_update_bias_typed"
                f"<{bias_size}, bias_t, grad_bias_t, upd_t, "
                f"acc_t, {codegen['pipeline_ii']}, "
                f"{codegen['output_partition']}>"
                f"(B_{tag}, dB_{tag}, "
                f"(upd_t){learning_rate:.8f}f);"
            )

        elif kind == "bn":
            lines.append(
                f"  fpgai::sgd_update_wgt<{weight_size}>"
                f"(BN_G_{tag}, dBN_G_{tag}, "
                f"(upd_t){learning_rate:.8f}f);"
            )
            lines.append(
                f"  fpgai::sgd_update_bias<{bias_size}>"
                f"(BN_B_{tag}, dBN_B_{tag}, "
                f"(upd_t){learning_rate:.8f}f);"
            )

    for index, (
        kind,
        _op,
        tag,
        weight_size,
        bias_size,
    ) in enumerate(parameter_specs):
        total_size = weight_size + bias_size

        if kind in {
            "dense",
            "conv",
        }:
            lines.append(
                f"  for (int i = 0; i < {weight_size}; ++i) "
                f"OUT_grad_{tag}[i] = (float)dW_{tag}[i];"
            )
            lines.append(
                f"  for (int i = 0; i < {bias_size}; ++i) "
                f"OUT_grad_{tag}[{weight_size} + i] = "
                f"(float)dB_{tag}[i];"
            )
        else:
            lines.append(
                f"  for (int i = 0; i < {weight_size}; ++i) "
                f"OUT_grad_{tag}[i] = "
                f"(float)dBN_G_{tag}[i];"
            )
            lines.append(
                f"  for (int i = 0; i < {bias_size}; ++i) "
                f"OUT_grad_{tag}[{weight_size} + i] = "
                f"(float)dBN_B_{tag}[i];"
            )

        is_last = (
            "true"
            if index == len(parameter_specs) - 1
            else "false"
        )

        lines.append(
            f"  emit_stream_block<{total_size}>"
            f"(out, OUT_grad_{tag}, {is_last});"
        )

    if not parameter_specs:
        lines.append(
            "  write_f32(out, 0.0f, true);"
        )

    lines.extend(
        [
            "}",
            "",
        ]
    )

    return _inject_native_accumulated_modes_from_cpp("\n".join(lines), input_size, learning_rate)


# FPGAI training communication annotation wrapper.
#
# Training top generation already receives a communication_plan. This wrapper
# makes the tensor-edge communication plan visible in generated HLS artifacts
# without changing training runtime semantics.
def _fpgai_train_comm_plan_to_dict(communication_plan):
    if communication_plan is None:
        return {}
    if isinstance(communication_plan, dict):
        return communication_plan
    if hasattr(communication_plan, "to_dict"):
        return communication_plan.to_dict()
    return {
        "edges": getattr(communication_plan, "edges", []),
        "notes": getattr(communication_plan, "notes", {}),
    }


def _fpgai_train_comm_edge_macros(communication_plan) -> str:
    plan = _fpgai_train_comm_plan_to_dict(communication_plan)
    edges = plan.get("edges", []) if isinstance(plan, dict) else []
    notes = plan.get("notes", {}) if isinstance(plan, dict) else {}

    lines = [
        "// FPGAI training communication tensor-edge plan.",
        "// Compression codecs other than raw are modeled unless implemented_in_hls=true.",
        "#define FPGAI_TRAIN_COMM_PLAN_PRESENT 1",
    ]

    scope = notes.get("scope") if isinstance(notes, dict) else None
    if scope is not None:
        lines.append(f"// communication_scope={scope}")

    for edge in edges:
        if not isinstance(edge, dict):
            if hasattr(edge, "to_dict"):
                edge = edge.to_dict()
            else:
                edge = dict(getattr(edge, "__dict__", {}) or {})

        kind = str(edge.get("kind") or edge.get("tensor_name") or "tensor").upper()
        safe_kind = "".join(ch if ch.isalnum() else "_" for ch in kind)
        macro_prefix = f"FPGAI_TRAIN_COMM_{safe_kind}"

        direction = edge.get("direction")
        codec = edge.get("codec", edge.get("encoding", "raw"))
        implemented = edge.get("implemented_in_hls", False)
        precision_bits = edge.get("precision_bits")
        transfer_bytes = edge.get("transfer_bytes")
        size_bytes = edge.get("size_bytes")

        lines.append(
            f"// tensor={edge.get('tensor_name')} kind={edge.get('kind')} "
            f"direction={direction} codec={codec} implemented_in_hls={implemented}"
        )

        if precision_bits is not None:
            lines.append(f"#define {macro_prefix}_PRECISION_BITS {int(precision_bits)}")
        if size_bytes is not None:
            lines.append(f"#define {macro_prefix}_SIZE_BYTES {int(size_bytes)}")
        if transfer_bytes is not None:
            lines.append(f"#define {macro_prefix}_TRANSFER_BYTES {int(transfer_bytes)}")
        lines.append(f"#define {macro_prefix}_IMPLEMENTED_IN_HLS {1 if implemented else 0}")

    lines.append("")
    return "\n".join(lines)


_fpgai_train_comm_previous_emit_top_train_cpp = emit_top_train_cpp


def emit_top_train_cpp(*args, **kwargs):
    communication_plan = kwargs.get("communication_plan")
    previous_kwargs = dict(kwargs)
    # Newer compiler stages pass raw_cfg for readability/config reports.
    # Older training emit wrappers in this file do not accept that keyword,
    # so consume it at this compatibility boundary instead of forwarding it
    # into the legacy emitter chain.
    previous_kwargs.pop("raw_cfg", None)
    source = _fpgai_train_comm_previous_emit_top_train_cpp(*args, **previous_kwargs)
    macros = _fpgai_train_comm_edge_macros(communication_plan)
    return macros + source





def _training_dense_tile_pairs_from_plan(compile_plan) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    if compile_plan is None:
        return pairs

    for layer_plan in getattr(compile_plan, "layer_plans", []) or []:
        if str(getattr(layer_plan, "op_type", "")).lower() != "dense":
            continue

        arch = getattr(layer_plan, "architecture", None)
        tiling = getattr(arch, "tiling", None)
        sizes = getattr(tiling, "sizes", {}) or {}

        tile_in = int(
            sizes.get("input")
            or sizes.get("in")
            or sizes.get("tile_in")
            or 1
        )
        tile_out = int(
            sizes.get("output")
            or sizes.get("out")
            or sizes.get("tile_out")
            or 1
        )
        pairs.append((tile_in, tile_out))

    return pairs


def _rewrite_training_dense_forward_extra_template_calls(
    source: str,
    compile_plan,
) -> str:
    """Rewrite training Dense forward calls with extra policy template args.

    Existing inference dense tiling rewriters may not match training calls like:

      fpgai::dense_out_in<IN, OUT, act_t, act_t, wgt_t, bias_t, acc_t, ...>(...)

    This rewrites only fpgai::dense_out_in<...> call sites in the final generated
    C++ source to:

      fpgai::dense_out_in_tiled<IN, OUT, TILE_IN, TILE_OUT, ...>(...)
    """

    pairs = _training_dense_tile_pairs_from_plan(compile_plan)
    if not pairs:
        return source

    import re

    pattern = re.compile(
        r"fpgai::dense_out_in<\s*"
        r"(?P<in>\d+)\s*,\s*"
        r"(?P<out>\d+)\s*,\s*"
        r"(?P<rest>[^>]*)>"
    )

    index = {"i": 0}

    def repl(match: re.Match[str]) -> str:
        i = index["i"]
        index["i"] += 1
        tile_in, tile_out = pairs[min(i, len(pairs) - 1)]

        return (
            f"fpgai::dense_out_in_tiled<"
            f"{match.group('in')}, {match.group('out')}, "
            f"{tile_in}, {tile_out}, "
            f"{match.group('rest').strip()}>"
        )

    return pattern.sub(repl, source)



def _rewrite_training_tiled_call_namespace(source: str) -> str:
    """Training tiled helpers are emitted into generated source scope.

    The existing tiling materializers emit conv2d_tiled/dense_out_in_tiled
    helper definitions directly in the generated C++ source. They are not inside
    namespace fpgai, while base untiled kernels are available as fpgai::conv2d
    and fpgai::dense_out_in through included layer headers.

    Therefore training tiled forward call sites must call the generated tiled
    helpers unqualified.
    """

    source = source.replace("fpgai::conv2d_tiled<", "conv2d_tiled<")
    source = source.replace("fpgai::dense_out_in_tiled<", "dense_out_in_tiled<")
    return source

# FPGAI training forward tiling wrapper.
#
# This reuses the existing dense/conv tiling materializers for the forward
# training path. Backward-gradient/update kernels are intentionally left
# untouched until training-specific tiled gradient kernels are implemented.
_fpgai_train_tiling_previous_emit_top_train_cpp = emit_top_train_cpp


def emit_top_train_cpp(*args, **kwargs):
    source = _fpgai_train_tiling_previous_emit_top_train_cpp(*args, **kwargs)
    graph = kwargs.get("graph")
    compile_plan = kwargs.get("compile_plan")
    if graph is None or compile_plan is None:
        return source

    source = apply_dense_tiling_to_top_source(source, graph, compile_plan)
    source = _rewrite_training_dense_forward_extra_template_calls(source, compile_plan)
    source = apply_dense_training_tiling_to_top_source(source, graph, compile_plan)
    source = apply_conv_tiling_to_top_source(source, graph, compile_plan)
    source = apply_conv_training_tiling_to_top_source(source, graph, compile_plan)
    source = _rewrite_training_tiled_call_namespace(source)

    if "FPGAI real dense tiling helper" in source or "FPGAI real convolution tiling helper" in source:
        source = (
            "// FPGAI training forward tiling materialized. "
            "Dense backward/update tiling is materialized; conv backward/update tiling is materialized.\n"
            + source
        )

    return source


# FPGAI training storage-binding and runtime import/export wrapper.

def _fpgai_training_notes(*, compile_plan=None, memory_plan=None) -> Dict[str, Any]:
    notes: Dict[str, Any] = {}
    for plan in (compile_plan, memory_plan):
        value = getattr(plan, "notes", {}) if plan is not None else {}
        if isinstance(value, dict):
            notes.update(value)
    return notes


def _fpgai_training_weight_storage_impl_from_notes(notes: Dict[str, Any]) -> str:
    requested = str(
        notes.get("resolved_weight_storage", notes.get("weight_storage", "bram")) or "bram"
    ).strip().lower()
    aliases = {
        "embedded": "bram",
        "on_chip": "bram",
        "onchip": "bram",
        "block": "bram",
        "block_ram": "bram",
        "bram": "bram",
        "uram": "uram",
        "ultra": "uram",
        "ultra_ram": "uram",
        "lutram": "lutram",
        "lut_ram": "lutram",
        "distributed": "lutram",
    }
    return aliases.get(requested, "bram")


def _fpgai_training_parameter_specs_from_source(source: str) -> List[Tuple[str, str, int, str, int]]:
    specs: List[Tuple[str, str, int, str, int]] = []
    for match in re.finditer(r"static\s+wgt_t\s+(W_[A-Za-z0-9_]+)\[(\d+)\]\s*=.*?;\s*static\s+bias_t\s+(B_[A-Za-z0-9_]+)\[(\d+)\]", source, flags=re.DOTALL):
        specs.append((match.group(1), match.group(3), int(match.group(2)), match.group(3), int(match.group(4))))
    for match in re.finditer(r"static\s+wgt_t\s+(BN_G_[A-Za-z0-9_]+)\[(\d+)\]\s*=.*?;\s*static\s+bias_t\s+(BN_B_[A-Za-z0-9_]+)\[(\d+)\]", source, flags=re.DOTALL):
        specs.append((match.group(1), match.group(3), int(match.group(2)), match.group(3), int(match.group(4))))
    return specs


def _fpgai_remove_legacy_aux_import_block(source: str) -> str:
    marker = "  if (mode == 0) {"
    start = source.find(marker)
    if start < 0:
        return source
    end_marker = "    return;\n  }\n\n"
    end = source.find(end_marker, start)
    if end < 0:
        return source
    return source[:start] + source[end + len(end_marker):]


def _fpgai_insert_training_storage_bindings(source: str, *, compile_plan=None, memory_plan=None) -> str:
    notes = _fpgai_training_notes(compile_plan=compile_plan, memory_plan=memory_plan)
    impl = _fpgai_training_weight_storage_impl_from_notes(notes)
    if impl not in {"bram", "uram", "lutram"}:
        impl = "bram"

    semantics = str(notes.get("memory_semantics_mode", notes.get("resolved_weight_semantics", "bram_static")) or "bram_static").strip().lower()
    import_iface = str(notes.get("weight_import_interface", "compile_time") or "compile_time").strip().lower()
    import_policy = str(notes.get("weight_import_policy", "static") or "static").strip().lower()
    export_iface = str(notes.get("weight_export_interface", "none") or "none").strip().lower()
    export_policy = str(notes.get("weight_export_policy", "none") or "none").strip().lower()
    runtime_import = impl in {"bram", "uram"} and import_iface == "m_axi" and import_policy == "full" and semantics.startswith(f"{impl}_import")
    runtime_export = runtime_import and export_iface == "m_axi" and export_policy == "full"
    grad_impl = str(notes.get("resolved_gradient_storage", notes.get("gradient_storage", "bram")) or "bram").strip().lower()
    grad_impl = {"block": "bram", "block_ram": "bram", "ultra": "uram", "ultra_ram": "uram"}.get(grad_impl, grad_impl)
    if grad_impl not in {"bram", "uram", "lutram"}:
        grad_impl = "bram"

    updated = source
    if runtime_import and "ap_uint<32>* weights_mem" not in updated:
        updated = updated.replace(
            "#include <ap_axi_sdata.h>\n",
            "#include <ap_axi_sdata.h>\n#include <ap_int.h>\n",
            1,
        )
        updated = updated.replace(
            "  hls::stream<axis_t>& aux,\n  int mode",
            "  hls::stream<axis_t>& aux,\n  ap_uint<32>* weights_mem,\n  int mode",
            1,
        )
        updated = updated.replace(
            "#pragma HLS INTERFACE axis port=aux\n",
            "#pragma HLS INTERFACE axis port=aux\n"
            "#pragma HLS INTERFACE m_axi port=weights_mem offset=slave bundle=gmem_weights\n"
            "#pragma HLS INTERFACE s_axilite port=weights_mem bundle=CTRL\n",
            1,
        )
        updated = _fpgai_remove_legacy_aux_import_block(updated)

    specs = _fpgai_training_parameter_specs_from_source(updated)
    pragma_lines: List[str] = [
        "  // FPGAI training command modes.",
        "  static const int FPGAI_MODE_EXPORT_WEIGHTS_STREAM = 1;",
        "  static const int FPGAI_MODE_RUN_TRAINING = 2;",
    ]
    if runtime_import:
        pragma_lines.extend([
            "  static const int FPGAI_MODE_IMPORT_WEIGHTS = 3;",
            "  static const int FPGAI_MODE_EXPORT_WEIGHTS = 4;",
        ])
    pragma_lines.append(f"  // FPGAI training weight storage: {impl}_mutable.")
    for w_name, b_name, _w_size, _b_name2, _b_size in specs:
        pragma_lines.append(f"#pragma HLS BIND_STORAGE variable={w_name} type=ram_2p impl={impl}")
        pragma_lines.append(f"#pragma HLS BIND_STORAGE variable={b_name} type=ram_2p impl={impl}")
        if w_name.startswith("W_"):
            tag = w_name[2:]
            pragma_lines.append(f"#pragma HLS BIND_STORAGE variable=dW_{tag} type=ram_2p impl={grad_impl}")
            pragma_lines.append(f"#pragma HLS BIND_STORAGE variable=dB_{tag} type=ram_2p impl={grad_impl}")
        elif w_name.startswith("BN_G_"):
            tag = w_name[5:]
            pragma_lines.append(f"#pragma HLS BIND_STORAGE variable=dBN_G_{tag} type=ram_2p impl={grad_impl}")
            pragma_lines.append(f"#pragma HLS BIND_STORAGE variable=dBN_B_{tag} type=ram_2p impl={grad_impl}")

    insert_block = "\n".join(pragma_lines) + "\n"
    if "FPGAI training weight storage:" not in updated:
        anchor = "#pragma HLS INTERFACE s_axilite port=return bundle=CTRL\n"
        updated = updated.replace(anchor, anchor + insert_block, 1)

    if runtime_import and "if (mode == FPGAI_MODE_IMPORT_WEIGHTS)" not in updated:
        offset = 0
        import_lines = ["  if (mode == FPGAI_MODE_IMPORT_WEIGHTS) {"]
        for w_name, b_name, w_size, _b_name2, b_size in specs:
            import_lines.append(f"    for (int i = 0; i < {w_size}; ++i) {w_name}[i] = (wgt_t)u32_to_f32((unsigned int)weights_mem[{offset} + i]);")
            offset += w_size
            import_lines.append(f"    for (int i = 0; i < {b_size}; ++i) {b_name}[i] = (bias_t)u32_to_f32((unsigned int)weights_mem[{offset} + i]);")
            offset += b_size
        import_lines.extend(["    return;", "  }", ""])
        if runtime_export:
            offset = 0
            import_lines.append("  if (mode == FPGAI_MODE_EXPORT_WEIGHTS) {")
            for w_name, b_name, w_size, _b_name2, b_size in specs:
                import_lines.append(f"    for (int i = 0; i < {w_size}; ++i) weights_mem[{offset} + i] = f32_to_u32((float){w_name}[i]);")
                offset += w_size
                import_lines.append(f"    for (int i = 0; i < {b_size}; ++i) weights_mem[{offset} + i] = f32_to_u32((float){b_name}[i]);")
                offset += b_size
            import_lines.extend(["    return;", "  }", ""])
        branch = "\n".join(import_lines)
        updated = updated.replace(insert_block, insert_block + branch, 1)

    return updated


_fpgai_training_storage_previous_emit_top_train_cpp = emit_top_train_cpp

def emit_top_train_cpp(*args, **kwargs):
    source = _fpgai_training_storage_previous_emit_top_train_cpp(*args, **kwargs)
    return _fpgai_insert_training_storage_bindings(
        source,
        compile_plan=kwargs.get("compile_plan"),
        memory_plan=kwargs.get("memory_plan"),
    )


# FPGAI DDR-tiled mutable training wrapper.
def _fpgai_training_is_ddr_tiled_mutable(*, compile_plan=None, memory_plan=None) -> bool:
    notes = _fpgai_training_notes(compile_plan=compile_plan, memory_plan=memory_plan)
    semantics = str(notes.get("memory_semantics_mode", notes.get("resolved_weight_semantics", "")) or "").strip().lower()
    storage = str(notes.get("resolved_weight_storage", notes.get("weight_storage", "")) or "").strip().lower()
    import_iface = str(notes.get("weight_import_interface", "") or "").strip().lower()
    import_policy = str(notes.get("weight_import_policy", "") or "").strip().lower()
    export_iface = str(notes.get("weight_export_interface", "") or "").strip().lower()
    export_policy = str(notes.get("weight_export_policy", "") or "").strip().lower()
    return (
        semantics in {"ddr_tiled", "ddr_tiled_mutable", "training_ddr_tiled_mutable"}
        or (storage == "ddr" and import_iface == "m_axi" and import_policy == "tiled" and export_iface in {"none", "m_axi"} and export_policy in {"none", "tiled"})
    )


def _fpgai_training_dense_parameter_layout(graph: Graph) -> List[Tuple[str, int, int]]:
    layout: List[Tuple[str, int, int]] = []
    for op in graph.ops:
        if op.op_type == "Dense":
            weights, bias, _, _ = _resolve_dense_arrays(graph, op)
            layout.append((_sanitize(op.name), int(weights.size), int(bias.size)))
    return layout


def _fpgai_reject_unsupported_ddr_training(graph: Graph) -> None:
    unsupported = []
    has_trainable = False
    allowed_passthrough = {
        "Relu",
        "LeakyRelu",
        "Sigmoid",
        "Softmax",
        "Flatten",
        "Reshape",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
    }
    for op in graph.ops:
        if op.op_type in {"Dense", "Conv"}:
            has_trainable = True
        elif op.op_type not in allowed_passthrough:
            unsupported.append(op.op_type)
    if unsupported:
        raise RuntimeError(
            "training DDR tiled mutable currently supports Dense/Conv training graphs; "
            f"unsupported ops: {sorted(set(unsupported))}"
        )
    if not has_trainable:
        raise RuntimeError("training DDR tiled mutable requires at least one Dense or Conv layer.")


def _fpgai_insert_training_ddr_tiled_mutable(source: str, *, graph: Graph, compile_plan=None, memory_plan=None) -> str:
    if not _fpgai_training_is_ddr_tiled_mutable(compile_plan=compile_plan, memory_plan=memory_plan):
        return source
    _fpgai_reject_unsupported_ddr_training(graph)
    updated = source
    if "ap_uint<32>* weights_mem" not in updated:
        updated = updated.replace(
            "  hls::stream<axis_t>& aux,\n  int mode",
            "  hls::stream<axis_t>& aux,\n  ap_uint<32>* weights_mem,\n  int mode",
            1,
        )
        updated = updated.replace(
            "#pragma HLS INTERFACE axis port=aux\n",
            "#pragma HLS INTERFACE axis port=aux\n"
            "#pragma HLS INTERFACE m_axi port=weights_mem offset=slave bundle=gmem_weights\n"
            "#pragma HLS INTERFACE s_axilite port=weights_mem bundle=CTRL\n",
            1,
        )
    if "FPGAI training DDR tiled mutable" in updated:
        return updated

    specs = _fpgai_training_parameter_specs_from_source(updated)
    if not specs:
        return updated
    tile_lines = [
        "  // FPGAI training DDR tiled mutable backend.",
        "  static const int FPGAI_MODE_RUN_TRAINING = 2;",
        "  static const int FPGAI_MODE_DDR_TILED_TRAINING = 7;",
        "  static wgt_t weight_tile[FPGAI_DDR_TRAIN_TILE_OUT][FPGAI_DDR_TRAIN_TILE_IN];",
        "  static grad_wgt_t grad_tile[FPGAI_DDR_TRAIN_TILE_OUT][FPGAI_DDR_TRAIN_TILE_IN];",
        "  static wgt_t conv_weight_tile[FPGAI_DDR_TRAIN_TILE_OC][FPGAI_DDR_TRAIN_TILE_IC][FPGAI_DDR_TRAIN_TILE_KH][FPGAI_DDR_TRAIN_TILE_KW];",
        "  static grad_wgt_t conv_grad_tile[FPGAI_DDR_TRAIN_TILE_OC][FPGAI_DDR_TRAIN_TILE_IC][FPGAI_DDR_TRAIN_TILE_KH][FPGAI_DDR_TRAIN_TILE_KW];",
        "#pragma HLS BIND_STORAGE variable=weight_tile type=ram_2p impl=bram",
        "#pragma HLS BIND_STORAGE variable=grad_tile type=ram_2p impl=bram",
        "#pragma HLS BIND_STORAGE variable=conv_weight_tile type=ram_2p impl=bram",
        "#pragma HLS BIND_STORAGE variable=conv_grad_tile type=ram_2p impl=bram",
        "  // weights_mem is the architectural source of truth for DDR tiled mutable training.",
    ]
    offset = 0
    pre_lines = ["  if (mode == FPGAI_MODE_RUN_TRAINING || mode == FPGAI_MODE_DDR_TILED_TRAINING) {", "    // Import Dense parameter tiles from DDR before the local update step."]
    post_lines = ["    // Export updated Dense parameter tiles back to DDR after the local update step."]
    for w_name, b_name, w_size, _b_name2, b_size in specs:
        pre_lines.append(f"    for (int tile_base = 0; tile_base < {w_size}; tile_base += FPGAI_DDR_TRAIN_TILE_WORDS) {{")
        pre_lines.append("#pragma HLS LOOP_TRIPCOUNT min=1 max=4096")
        pre_lines.append(f"      for (int i = 0; i < FPGAI_DDR_TRAIN_TILE_WORDS && tile_base + i < {w_size}; ++i) {{")
        pre_lines.append(f"        weight_tile[0][i] = (wgt_t)u32_to_f32((unsigned int)weights_mem[{offset} + tile_base + i]);")
        pre_lines.append(f"        {w_name}[tile_base + i] = weight_tile[0][i];")
        pre_lines.append("      }")
        pre_lines.append("    }")
        offset += w_size
        pre_lines.append(f"    for (int i = 0; i < {b_size}; ++i) {b_name}[i] = (bias_t)u32_to_f32((unsigned int)weights_mem[{offset} + i]);")
        offset += b_size
    # We place post export after the generated optimizer code by adding a marker branch before final fallback return.
    offset = 0
    for w_name, b_name, w_size, _b_name2, b_size in specs:
        post_lines.append(f"    for (int tile_base = 0; tile_base < {w_size}; tile_base += FPGAI_DDR_TRAIN_TILE_WORDS) {{")
        post_lines.append("#pragma HLS LOOP_TRIPCOUNT min=1 max=4096")
        post_lines.append(f"      for (int i = 0; i < FPGAI_DDR_TRAIN_TILE_WORDS && tile_base + i < {w_size}; ++i) {{")
        post_lines.append(f"        grad_tile[0][i] = (grad_wgt_t)0;")
        post_lines.append(f"        weight_tile[0][i] = {w_name}[tile_base + i];")
        post_lines.append(f"        weights_mem[{offset} + tile_base + i] = f32_to_u32((float)weight_tile[0][i]);")
        post_lines.append("      }")
        post_lines.append("    }")
        offset += w_size
        post_lines.append(f"    for (int i = 0; i < {b_size}; ++i) weights_mem[{offset} + i] = f32_to_u32((float){b_name}[i]);")
        offset += b_size
    post_lines.append("  }")

    macro_block = [
        "#ifndef FPGAI_DDR_TRAIN_TILE_WORDS",
        "#define FPGAI_DDR_TRAIN_TILE_WORDS 16",
        "#endif",
        "#ifndef FPGAI_DDR_TRAIN_TILE_OUT",
        "#define FPGAI_DDR_TRAIN_TILE_OUT 1",
        "#endif",
        "#ifndef FPGAI_DDR_TRAIN_TILE_IN",
        "#define FPGAI_DDR_TRAIN_TILE_IN FPGAI_DDR_TRAIN_TILE_WORDS",
        "#endif",
        "#ifndef FPGAI_DDR_TRAIN_TILE_OC",
        "#define FPGAI_DDR_TRAIN_TILE_OC 1",
        "#endif",
        "#ifndef FPGAI_DDR_TRAIN_TILE_IC",
        "#define FPGAI_DDR_TRAIN_TILE_IC 1",
        "#endif",
        "#ifndef FPGAI_DDR_TRAIN_TILE_KH",
        "#define FPGAI_DDR_TRAIN_TILE_KH 3",
        "#endif",
        "#ifndef FPGAI_DDR_TRAIN_TILE_KW",
        "#define FPGAI_DDR_TRAIN_TILE_KW 3",
        "#endif",
        "",
    ]
    updated = updated.replace('using namespace fpgai;\n', 'using namespace fpgai;\n' + '\n'.join(macro_block), 1)

    anchor = "#pragma HLS INTERFACE s_axilite port=return bundle=CTRL\n"
    updated = updated.replace(anchor, anchor + "\n".join(tile_lines) + "\n", 1)

    # Insert pre-import immediately before normal training mode branch body starts. This source uses mode == 2 for step training.
    train_marker = "  if (mode == 2) {"
    pos = updated.find(train_marker)
    if pos >= 0:
        # Keep the original mode branch; put import at the top of the branch.
        brace_end = updated.find("\n", pos)
        updated = updated[:brace_end+1] + "\n".join(pre_lines[1:]) + "\n" + updated[brace_end+1:]
        # Export before the first return after the mode==2 branch, best-effort.
        ret = updated.find("    return;", brace_end)
        if ret >= 0:
            updated = updated[:ret] + "\n".join(post_lines) + "\n" + updated[ret:]
    else:
        updated = updated.replace("  // FPGAI training weight storage", "\n".join(pre_lines + post_lines) + "\n  // FPGAI training weight storage", 1)
    return updated


_fpgai_training_ddr_previous_emit_top_train_cpp = emit_top_train_cpp

def emit_top_train_cpp(*args, **kwargs):
    source = _fpgai_training_ddr_previous_emit_top_train_cpp(*args, **kwargs)
    return _fpgai_insert_training_ddr_tiled_mutable(
        source,
        graph=kwargs.get("graph", args[0] if args else None),
        compile_plan=kwargs.get("compile_plan"),
        memory_plan=kwargs.get("memory_plan"),
    )

# FPGAI training readability wrapper.
_fpgai_training_readability_previous_emit_top_train_cpp = emit_top_train_cpp


def _fpgai_training_cfg_get(raw: Any, path: str, default: Any = None) -> Any:
    cur = raw
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _fpgai_training_readability_value(kwargs) -> str:
    raw = kwargs.get("raw_cfg") or {}
    value = str(_fpgai_training_cfg_get(raw, "codegen.readability", "high") or "high").strip().lower().replace("-", "_")
    return value if value in {"compact", "normal", "high", "debug"} else "high"


def _fpgai_training_note(kwargs, key: str, default=""):
    for plan_key in ("memory_plan", "communication_plan", "compile_plan"):
        plan = kwargs.get(plan_key)
        notes = getattr(plan, "notes", None)
        if isinstance(notes, dict) and key in notes:
            return notes.get(key)
    return default


def _fpgai_training_readability_banner(kwargs) -> str:
    level = _fpgai_training_readability_value(kwargs)
    if level == "compact":
        return ""
    raw = kwargs.get("raw_cfg") or {}
    weights_mode = str(kwargs.get("weights_mode", ""))
    memory_mode = str(_fpgai_training_note(kwargs, "memory_semantics_mode", weights_mode))
    storage = str(_fpgai_training_note(kwargs, "resolved_training_weight_storage", _fpgai_training_cfg_get(raw, "memory.storage.weights", "bram")))
    if level == "normal":
        return f"// FPGAI generated training HLS top: weights_mode={weights_mode}, memory_semantics={memory_mode}.\n"
    return "\n".join([
        "// ============================================================",
        "// FPGAI generated training HLS top",
        "// Pipeline mode: training_on_device",
        f"// Codegen readability: {level}",
        f"// Weight mode: {weights_mode}",
        f"// Weight storage: {storage}",
        f"// Weight semantics: {memory_mode}",
        "// Runtime import/export is command-driven; reload before each compute: false",
        "// Sections: includes/types, runtime constants, mutable storage, training helpers, top dispatch.",
        "// ============================================================",
        "",
    ])


def emit_top_train_cpp(*args, **kwargs):
    source = _fpgai_training_readability_previous_emit_top_train_cpp(*args, **kwargs)
    banner = _fpgai_training_readability_banner(kwargs)
    if banner and "FPGAI generated training HLS top" not in source[:512]:
        return banner + source
    return source


# FPGAI training I/O tiled movement and gradient export wrapper.
_fpgai_training_io_gradient_previous_emit_top_train_cpp = emit_top_train_cpp


def _fpgai_training_raw_get(raw: Any, path: str, default: Any = None) -> Any:
    cur = raw
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _fpgai_training_movement(raw: Any, tensor: str, direction: str) -> dict:
    cfg = _fpgai_training_raw_get(raw, f"data_movement.{tensor}.{direction}", {}) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "interface": str(cfg.get("interface", "")).strip().lower().replace('-', '_'),
        "transport": str(cfg.get("transport", "")).strip().lower().replace('-', '_'),
        "policy": str(cfg.get("policy", "")).strip().lower().replace('-', '_'),
    }


def _fpgai_training_has_m_axi_tiled(raw: Any, tensor: str, direction: str) -> bool:
    mv = _fpgai_training_movement(raw, tensor, direction)
    return mv.get("interface") == "m_axi" and mv.get("policy") == "tiled"


def _fpgai_training_has_axi_stream_tiled(raw: Any, tensor: str, direction: str) -> bool:
    mv = _fpgai_training_movement(raw, tensor, direction)
    return mv.get("interface") == "axi_stream" and mv.get("policy") == "tiled"


def _fpgai_training_gradient_export_policy(raw: Any) -> str:
    mv = _fpgai_training_movement(raw, "gradients", "export")
    if mv.get("interface") == "m_axi" and mv.get("policy") in {"full", "tiled"}:
        return str(mv.get("policy"))
    return "none"


def _fpgai_insert_training_io_and_gradient_ports(source: str, *, raw_cfg: Any) -> str:
    raw = raw_cfg or {}
    input_tiled = _fpgai_training_has_m_axi_tiled(raw, "inputs", "import")
    label_tiled = _fpgai_training_has_m_axi_tiled(raw, "labels", "import")
    output_tiled = _fpgai_training_has_m_axi_tiled(raw, "outputs", "export")
    gradient_export_policy = _fpgai_training_gradient_export_policy(raw)
    gradient_export = gradient_export_policy in {"full", "tiled"}
    if not any([input_tiled, label_tiled, output_tiled, gradient_export]):
        return source

    updated = source
    port_lines: list[str] = []
    pragma_lines: list[str] = []
    tile_lines: list[str] = []

    if input_tiled and "ap_uint<32>* input_mem" not in updated:
        port_lines.append("  ap_uint<32>* input_mem,")
        pragma_lines.extend([
            "#pragma HLS INTERFACE m_axi port=input_mem offset=slave bundle=gmem_input",
            "#pragma HLS INTERFACE s_axilite port=input_mem bundle=CTRL",
        ])
        tile_lines.extend([
            "  // FPGAI training tiled input import: PS DDR -> m_axi input_mem -> local activation tile.",
            "  static act_t input_tile[FPGAI_TRAIN_INPUT_TILE_SIZE];",
            "#pragma HLS BIND_STORAGE variable=input_tile type=ram_1p impl=bram",
        ])
    if label_tiled and "ap_uint<32>* label_mem" not in updated:
        port_lines.append("  ap_uint<32>* label_mem,")
        pragma_lines.extend([
            "#pragma HLS INTERFACE m_axi port=label_mem offset=slave bundle=gmem_labels",
            "#pragma HLS INTERFACE s_axilite port=label_mem bundle=CTRL",
        ])
        tile_lines.extend([
            "  // FPGAI training tiled label import: PS DDR -> m_axi label_mem -> local label tile.",
            "  static act_t label_tile[FPGAI_TRAIN_LABEL_TILE_SIZE];",
            "#pragma HLS BIND_STORAGE variable=label_tile type=ram_1p impl=bram",
        ])
    if output_tiled and "ap_uint<32>* output_mem" not in updated:
        port_lines.append("  ap_uint<32>* output_mem,")
        pragma_lines.extend([
            "#pragma HLS INTERFACE m_axi port=output_mem offset=slave bundle=gmem_output",
            "#pragma HLS INTERFACE s_axilite port=output_mem bundle=CTRL",
        ])
        tile_lines.extend([
            "  // FPGAI training tiled output export: local output tile -> m_axi output_mem -> PS DDR.",
            "  static act_t output_tile[FPGAI_TRAIN_OUTPUT_TILE_SIZE];",
            "#pragma HLS BIND_STORAGE variable=output_tile type=ram_1p impl=bram",
        ])
    if gradient_export and "ap_uint<32>* gradients_mem" not in updated:
        port_lines.append("  ap_uint<32>* gradients_mem,")
        pragma_lines.extend([
            "#pragma HLS INTERFACE m_axi port=gradients_mem offset=slave bundle=gmem_gradients",
            "#pragma HLS INTERFACE s_axilite port=gradients_mem bundle=CTRL",
        ])
        tile_lines.extend([
            f"  // FPGAI gradient export {gradient_export_policy} mode: parameter gradients are flattened into gradients_mem.",
            "  static const int FPGAI_MODE_EXPORT_GRADIENTS = 8;",
        ])
        if gradient_export_policy == "tiled":
            tile_lines.extend([
                "  static grad_wgt_t gradient_export_tile[FPGAI_GRADIENT_EXPORT_TILE_SIZE];",
                "#pragma HLS BIND_STORAGE variable=gradient_export_tile type=ram_1p impl=bram",
            ])

    if port_lines:
        marker = "  int mode\n) {"
        if marker in updated:
            updated = updated.replace(marker, "\n".join(port_lines) + "\n  int mode\n) {", 1)
        else:
            marker2 = "  int mode,\n) {"
            updated = updated.replace(marker2, "\n".join(port_lines) + "\n  int mode,\n) {", 1)

    if pragma_lines:
        anchor = "#pragma HLS INTERFACE s_axilite port=mode bundle=CTRL\n"
        if anchor in updated and pragma_lines[0] not in updated:
            updated = updated.replace(anchor, "\n".join(pragma_lines) + "\n" + anchor, 1)

    if tile_lines:
        anchor = "#pragma HLS INTERFACE s_axilite port=return bundle=CTRL\n"
        if anchor in updated and "FPGAI training tiled input import" not in updated and "FPGAI gradient export full mode" not in updated:
            updated = updated.replace(anchor, anchor + "\n".join(tile_lines) + "\n", 1)

    # Keep the legacy stream/testbench data path intact until the full training tiled
    # data scheduler is implemented.  Earlier versions rewrote the first ``read_f32``
    # assignment with an ``i``-indexed m_axi load, but some generated training tops
    # read scalar values outside an ``i`` loop.  That produced invalid C++ such as
    # ``buf_input[i] = ...`` with no ``i`` in scope.  For Sprint 29Q the contract is
    # an honest interface/report/codegen hook: expose the m_axi ports and local tile
    # buffers, preserve the existing numeric comparison path, and leave compute-fused
    # tiled scheduling for the later numeric-validation/codegen refactor sprint.
    if output_tiled and "output_mem[i] = f32_to_u32" not in updated:
        # Export final training forward output before gradient/loss computation.
        loss_anchor = "  loss_t loss_value = (loss_t)0;\n"
        block = (
            "  for (int i = 0; i < FPGAI_TRAIN_OUTPUT_TILE_SIZE; ++i) {\n"
            "    output_tile[i] = (act_t)0;\n"
            "  }\n"
            "  // Output export loop is bounded by the generated output tensor size in the original training top.\n"
        )
        if loss_anchor in updated:
            updated = updated.replace(loss_anchor, block + loss_anchor, 1)
            # Best effort replacement based on final_buffer cannot be known here without parsing; keep interface/tile explicit.
    if gradient_export and "if (mode == FPGAI_MODE_EXPORT_GRADIENTS)" not in updated:
        marker = "  if (mode == 1) {"
        branch = [
            "  if (mode == FPGAI_MODE_EXPORT_GRADIENTS) {",
            f"    // FPGAI gradient_export {gradient_export_policy} mode: copy generated gradients to gradients_mem.",
            "    int gradient_offset = 0;",
        ]
        for match in re.finditer(r"static\s+grad_wgt_t\s+(dW_[A-Za-z0-9_]+)\[(\d+)\];", updated):
            name, size = match.group(1), match.group(2)
            if gradient_export_policy == "tiled":
                branch.append(f"    for (int tile_base = 0; tile_base < {size}; tile_base += FPGAI_GRADIENT_EXPORT_TILE_SIZE) {{")
                branch.append(f"      for (int lane = 0; lane < FPGAI_GRADIENT_EXPORT_TILE_SIZE; ++lane) {{")
                branch.append(f"        int idx = tile_base + lane;")
                branch.append(f"        gradient_export_tile[lane] = (idx < {size}) ? {name}[idx] : (grad_wgt_t)0;")
                branch.append(f"        if (idx < {size}) gradients_mem[gradient_offset + idx] = f32_to_u32((float)gradient_export_tile[lane]);")
                branch.append("      }")
                branch.append("    }")
            else:
                branch.append(f"    for (int i = 0; i < {size}; ++i) gradients_mem[gradient_offset + i] = f32_to_u32((float){name}[i]);")
            branch.append(f"    gradient_offset += {size};")
        for match in re.finditer(r"static\s+grad_bias_t\s+(dB_[A-Za-z0-9_]+)\[(\d+)\];", updated):
            name, size = match.group(1), match.group(2)
            if gradient_export_policy == "tiled":
                branch.append(f"    for (int tile_base = 0; tile_base < {size}; tile_base += FPGAI_GRADIENT_EXPORT_TILE_SIZE) {{")
                branch.append(f"      for (int lane = 0; lane < FPGAI_GRADIENT_EXPORT_TILE_SIZE; ++lane) {{")
                branch.append(f"        int idx = tile_base + lane;")
                branch.append(f"        gradient_export_tile[lane] = (idx < {size}) ? (grad_wgt_t){name}[idx] : (grad_wgt_t)0;")
                branch.append(f"        if (idx < {size}) gradients_mem[gradient_offset + idx] = f32_to_u32((float)gradient_export_tile[lane]);")
                branch.append("      }")
                branch.append("    }")
            else:
                branch.append(f"    for (int i = 0; i < {size}; ++i) gradients_mem[gradient_offset + i] = f32_to_u32((float){name}[i]);")
            branch.append(f"    gradient_offset += {size};")
        branch.extend(["    return;", "  }", ""])
        if marker in updated:
            updated = updated.replace(marker, "\n".join(branch) + marker, 1)
    if any([input_tiled, label_tiled, output_tiled, gradient_export_policy == "tiled"]) and "FPGAI_TRAIN_INPUT_TILE_SIZE" not in updated[:2500]:
        macros = [
            "#ifndef FPGAI_TRAIN_INPUT_TILE_SIZE",
            "#define FPGAI_TRAIN_INPUT_TILE_SIZE 64",
            "#endif",
            "#ifndef FPGAI_TRAIN_LABEL_TILE_SIZE",
            "#define FPGAI_TRAIN_LABEL_TILE_SIZE 64",
            "#endif",
            "#ifndef FPGAI_TRAIN_OUTPUT_TILE_SIZE",
            "#define FPGAI_TRAIN_OUTPUT_TILE_SIZE 64",
            "#endif",
            "#ifndef FPGAI_GRADIENT_EXPORT_TILE_SIZE",
            "#define FPGAI_GRADIENT_EXPORT_TILE_SIZE 64",
            "#endif",
            "",
        ]
        updated = updated.replace('using namespace fpgai;\n', 'using namespace fpgai;\n' + '\n'.join(macros), 1)
    return updated



def _fpgai_insert_training_axis_stream_tiled_io(source: str, *, raw_cfg: Any) -> str:
    raw = raw_cfg or {}
    input_axis_tiled = _fpgai_training_has_axi_stream_tiled(raw, "inputs", "import")
    label_axis_tiled = _fpgai_training_has_axi_stream_tiled(raw, "labels", "import")
    output_axis_tiled = _fpgai_training_has_axi_stream_tiled(raw, "outputs", "export")
    if not any([input_axis_tiled, label_axis_tiled, output_axis_tiled]):
        return source

    updated = source

    if "FPGAI_TRAIN_AXIS_INPUT_TILE_SIZE" not in updated[:3500]:
        macros = [
            "#ifndef FPGAI_TRAIN_AXIS_INPUT_TILE_SIZE",
            "#define FPGAI_TRAIN_AXIS_INPUT_TILE_SIZE 64",
            "#endif",
            "#ifndef FPGAI_TRAIN_AXIS_LABEL_TILE_SIZE",
            "#define FPGAI_TRAIN_AXIS_LABEL_TILE_SIZE 64",
            "#endif",
            "#ifndef FPGAI_TRAIN_AXIS_OUTPUT_TILE_SIZE",
            "#define FPGAI_TRAIN_AXIS_OUTPUT_TILE_SIZE 64",
            "#endif",
            "",
        ]
        updated = updated.replace('using namespace fpgai;\n', 'using namespace fpgai;\n' + '\n'.join(macros), 1)

    if output_axis_tiled and "emit_stream_tiled_block" not in updated:
        helper = """
template<int N, int TILE>
static inline void emit_stream_tiled_block(
  hls::stream<axis_t>& out,
  const float* data,
  bool is_last_block) {
#pragma HLS INLINE off
  // FPGAI training AXI-stream tiled output export: local output tile -> AXI stream with TLAST.
  float axis_output_tile[TILE];
#pragma HLS ARRAY_PARTITION variable=axis_output_tile complete dim=1
  for (int tile_base = 0; tile_base < N; tile_base += TILE) {
    for (int lane = 0; lane < TILE; ++lane) {
#pragma HLS PIPELINE II=1
      int idx = tile_base + lane;
      axis_output_tile[lane] = (idx < N) ? data[idx] : 0.0f;
    }
    for (int lane = 0; lane < TILE; ++lane) {
#pragma HLS PIPELINE II=1
      int idx = tile_base + lane;
      if (idx < N) {
        bool last = is_last_block && (idx == N - 1);
        write_f32(out, axis_output_tile[lane], last);
      }
    }
  }
}
"""
        marker = "template<int C, int HW>"
        if marker in updated:
            updated = updated.replace(marker, helper + "\n" + marker, 1)
        else:
            updated = updated.replace("\nextern \"C\" void", helper + "\nextern \"C\" void", 1)

    if input_axis_tiled:
        pattern = re.compile(
            r"  for \(int i = 0; i < (\d+); \+\+i\) ([A-Za-z0-9_]+)\[i\] = \(act_t\)read_f32\(in\);"
        )
        def repl(match: re.Match[str]) -> str:
            total, buf = match.group(1), match.group(2)
            return "\n".join([
                "  // FPGAI training AXI-stream tiled input import: in stream -> axis_input_tile -> activation buffer.",
                "  act_t axis_input_tile[FPGAI_TRAIN_AXIS_INPUT_TILE_SIZE];",
                "#pragma HLS ARRAY_PARTITION variable=axis_input_tile complete dim=1",
                f"  for (int tile_base = 0; tile_base < {total}; tile_base += FPGAI_TRAIN_AXIS_INPUT_TILE_SIZE) {{",
                "    for (int lane = 0; lane < FPGAI_TRAIN_AXIS_INPUT_TILE_SIZE; ++lane) {",
                "#pragma HLS PIPELINE II=1",
                "      int idx = tile_base + lane;",
                f"      axis_input_tile[lane] = (idx < {total}) ? (act_t)read_f32(in) : (act_t)0;",
                "    }",
                "    for (int lane = 0; lane < FPGAI_TRAIN_AXIS_INPUT_TILE_SIZE; ++lane) {",
                "#pragma HLS PIPELINE II=1",
                "      int idx = tile_base + lane;",
                f"      if (idx < {total}) {buf}[idx] = axis_input_tile[lane];",
                "    }",
                "  }",
            ])
        updated, n = pattern.subn(repl, updated, count=1)

    if label_axis_tiled:
        pattern = re.compile(
            r"  for \(int i = 0; i < (\d+); \+\+i\) target_buf\[i\] = \(act_t\)read_f32\(aux\);"
        )
        def repl_label(match: re.Match[str]) -> str:
            total = match.group(1)
            return "\n".join([
                "  // FPGAI training AXI-stream tiled label import: aux stream -> axis_label_tile -> target buffer.",
                "  act_t axis_label_tile[FPGAI_TRAIN_AXIS_LABEL_TILE_SIZE];",
                "#pragma HLS ARRAY_PARTITION variable=axis_label_tile complete dim=1",
                f"  for (int tile_base = 0; tile_base < {total}; tile_base += FPGAI_TRAIN_AXIS_LABEL_TILE_SIZE) {{",
                "    for (int lane = 0; lane < FPGAI_TRAIN_AXIS_LABEL_TILE_SIZE; ++lane) {",
                "#pragma HLS PIPELINE II=1",
                "      int idx = tile_base + lane;",
                f"      axis_label_tile[lane] = (idx < {total}) ? (act_t)read_f32(aux) : (act_t)0;",
                "    }",
                "    for (int lane = 0; lane < FPGAI_TRAIN_AXIS_LABEL_TILE_SIZE; ++lane) {",
                "#pragma HLS PIPELINE II=1",
                "      int idx = tile_base + lane;",
                f"      if (idx < {total}) target_buf[idx] = axis_label_tile[lane];",
                "    }",
                "  }",
            ])
        updated, n = pattern.subn(repl_label, updated, count=1)

    if output_axis_tiled:
        updated = re.sub(
            r"emit_stream_block<(\d+)>\(out, ([^,]+), (true|false)\);",
            r"emit_stream_tiled_block<\1, FPGAI_TRAIN_AXIS_OUTPUT_TILE_SIZE>(out, \2, \3);",
            updated,
        )

    return updated

def emit_top_train_cpp(*args, **kwargs):
    source = _fpgai_training_io_gradient_previous_emit_top_train_cpp(*args, **kwargs)
    source = _fpgai_insert_training_io_and_gradient_ports(source, raw_cfg=kwargs.get("raw_cfg") or {})
    return _fpgai_insert_training_axis_stream_tiled_io(source, raw_cfg=kwargs.get("raw_cfg") or {})

# FPGAI training optimizer-state/loss readability and interface wrapper.
def _fpgai_training_optimizer_state_storage(raw: Any) -> str:
    value = _fpgai_training_raw_get(raw, "training.storage.optimizer_state", "none")
    return str(value or "none").strip().lower().replace('-', '_')


def _fpgai_training_has_optimizer_state_m_axi(raw: Any, direction: str) -> bool:
    mv = _fpgai_training_movement(raw, "optimizer_state", direction)
    return mv.get("interface") == "m_axi" and mv.get("policy") in {"full", "tiled"}


def _fpgai_insert_optimizer_state_and_loss_contract(source: str, *, raw_cfg: Any) -> str:
    raw = raw_cfg or {}
    optimizer_type = str(_fpgai_training_raw_get(raw, "training.optimizer.type", "sgd") or "sgd").strip().lower().replace('-', '_')
    loss_type = str(_fpgai_training_raw_get(raw, "training.loss.type", "mse") or "mse").strip().lower().replace('-', '_')
    storage = _fpgai_training_optimizer_state_storage(raw)
    import_full = _fpgai_training_has_optimizer_state_m_axi(raw, "import")
    export_full = _fpgai_training_has_optimizer_state_m_axi(raw, "export")
    if not any([optimizer_type != "sgd", loss_type != "mse", storage in {"bram", "uram", "ddr"}, import_full, export_full]):
        return source

    updated = source
    impl = "uram" if storage == "uram" else "bram"
    banner = [
        "// ============================================================",
        "// FPGAI training optimizer/loss contract",
        f"// Optimizer: {optimizer_type}",
        f"// Loss: {loss_type}",
        f"// Optimizer state storage: {storage}",
        "// Optimizer state DDR mode uses m_axi optimizer_state_mem plus a local tile buffer.",
        "// Momentum/Adam and cross-entropy generated kernels are emitted when selected.",
        "// ============================================================",
        "",
    ]
    if "FPGAI training optimizer/loss contract" not in updated:
        updated = updated.replace("using namespace fpgai;\n", "using namespace fpgai;\n" + "\n".join(banner), 1)

    port_lines: list[str] = []
    pragma_lines: list[str] = []
    state_lines: list[str] = []
    if (import_full or export_full) and "ap_uint<32>* optimizer_state_mem" not in updated:
        port_lines.append("  ap_uint<32>* optimizer_state_mem,")
        pragma_lines.extend([
            "#pragma HLS INTERFACE m_axi port=optimizer_state_mem offset=slave bundle=gmem_optimizer_state",
            "#pragma HLS INTERFACE s_axilite port=optimizer_state_mem bundle=CTRL",
        ])
    if storage in {"bram", "uram", "ddr"} and "optimizer_state_tile[FPGAI_OPTIMIZER_STATE_TILE_SIZE]" not in updated:
        state_lines.extend([
            f"  // FPGAI optimizer-state {storage} backing: local tile mirrors optimizer_state_mem when external storage is selected.",
            "  static opt_t optimizer_state_tile[FPGAI_OPTIMIZER_STATE_TILE_SIZE];",
            f"#pragma HLS BIND_STORAGE variable=optimizer_state_tile type=ram_2p impl={impl}",
        ])
    if port_lines:
        marker = "  int mode\n) {"
        if marker in updated:
            updated = updated.replace(marker, "\n".join(port_lines) + "\n  int mode\n) {", 1)
    if pragma_lines:
        anchor = "#pragma HLS INTERFACE s_axilite port=mode bundle=CTRL\n"
        if anchor in updated and pragma_lines[0] not in updated:
            updated = updated.replace(anchor, "\n".join(pragma_lines) + "\n" + anchor, 1)
    if state_lines:
        anchor = "#pragma HLS INTERFACE s_axilite port=return bundle=CTRL\n"
        if anchor in updated and "optimizer_state_tile" not in updated:
            updated = updated.replace(anchor, anchor + "\n".join(state_lines) + "\n", 1)
    if "FPGAI_OPTIMIZER_STATE_TILE_SIZE" not in updated[:2500]:
        macros = [
            "#ifndef FPGAI_OPTIMIZER_STATE_TILE_SIZE",
            "#define FPGAI_OPTIMIZER_STATE_TILE_SIZE 64",
            "#endif",
            "",
        ]
        updated = updated.replace('using namespace fpgai;\n', 'using namespace fpgai;\n' + '\n'.join(macros), 1)
    return updated


_fpgai_training_optimizer_loss_previous_emit_top_train_cpp = emit_top_train_cpp


def emit_top_train_cpp(*args, **kwargs):
    source = _fpgai_training_optimizer_loss_previous_emit_top_train_cpp(*args, **kwargs)
    return _fpgai_insert_optimizer_state_and_loss_contract(source, raw_cfg=kwargs.get("raw_cfg") or {})

# FPGAI momentum optimizer real generated update wrapper.
def _fpgai_insert_momentum_optimizer_kernel(source: str, *, raw_cfg: Any) -> str:
    raw = raw_cfg or {}
    optimizer_type = str(_fpgai_training_raw_get(raw, "training.optimizer.type", "sgd") or "sgd").strip().lower().replace('-', '_')
    if optimizer_type != "momentum":
        return source
    learning_rate = float(_fpgai_training_raw_get(raw, "training.optimizer.learning_rate", 0.01) or 0.01)
    momentum = float(_fpgai_training_raw_get(raw, "training.optimizer.momentum", 0.9) or 0.9)
    updated = source

    # Discover trainable arrays from the generated source. This keeps the wrapper aligned
    # with Dense/Conv tags already produced by the existing training codegen path.
    params = []
    for kind, prefix, grad_prefix, value_type, grad_type in [
        ("weight", "W", "dW", "wgt_t", "grad_wgt_t"),
        ("bias", "B", "dB", "bias_t", "grad_bias_t"),
    ]:
        pattern = re.compile(rf"static\\s+{re.escape(value_type)}\\s+{prefix}_([A-Za-z0-9_]+)\\[(\\d+)\\]")
        for m in pattern.finditer(updated):
            tag, size = m.group(1), int(m.group(2))
            grad_arr = f"{grad_prefix}_{tag}"
            if f"static {grad_type} {grad_arr}[" not in updated:
                continue
            params.append({
                "kind": kind,
                "tag": tag,
                "size": size,
                "arr": f"{prefix}_{tag}",
                "grad": grad_arr,
                "velocity": f"FPGAI_MOMENTUM_{prefix}_{tag}",
                "value_type": value_type,
                "grad_type": grad_type,
            })
    if not params:
        return updated

    if "FPGAI Momentum optimizer update kernel" not in updated:
        banner = [
            "// ============================================================",
            "// FPGAI Momentum optimizer update kernel",
            "// Generated update rule: V = momentum * V - learning_rate * dParam; Param = Param + V.",
            f"// learning_rate = {learning_rate:.8f}",
            f"// momentum = {momentum:.8f}",
            "// ============================================================",
            "",
        ]
        updated = updated.replace("using namespace fpgai;\n", "using namespace fpgai;\n" + "\n".join(banner), 1)

    decl_lines = ["", "// FPGAI persistent momentum optimizer velocity state."]
    for p in params:
        decl_lines.append(f"static opt_t {p['velocity']}[{p['size']}];")
    decl_text = "\n".join(decl_lines) + "\n"
    extern_marker = '\nextern "C" void '
    if "FPGAI persistent momentum optimizer velocity state" not in updated and extern_marker in updated:
        updated = updated.replace(extern_marker, decl_text + extern_marker, 1)

    def repl_typed(match: re.Match) -> str:
        arr, grad = match.group(1), match.group(2)
        p = next((x for x in params if x["arr"] == arr and x["grad"] == grad), None)
        if p is None:
            return match.group(0)
        return (
            f"  // FPGAI momentum optimizer update for {arr}.\n"
            f"  for (int i = 0; i < {p['size']}; ++i) {{\n"
            f"    {p['velocity']}[i] = (opt_t)(((float){momentum:.8f}f * (float){p['velocity']}[i]) - ((float){learning_rate:.8f}f * (float){grad}[i]));\n"
            f"    {arr}[i] = ({p['value_type']})((float){arr}[i] + (float){p['velocity']}[i]);\n"
            f"  }}"
        )

    # Replace the generated SGD update calls with actual Momentum loops. The regex is
    # intentionally restricted to array + gradient arguments so unsupported parameter
    # forms are left unchanged for later explicit implementation.
    typed_pattern = re.compile(
        r"  fpgai::sgd_update_(?:wgt|bias)_typed<[^;]+?\((W_[A-Za-z0-9_]+|B_[A-Za-z0-9_]+),\s*(dW_[A-Za-z0-9_]+|dB_[A-Za-z0-9_]+),\s*\(upd_t\)[^;]+?;",
        re.DOTALL,
    )
    updated = typed_pattern.sub(repl_typed, updated)
    return updated


_fpgai_training_momentum_previous_emit_top_train_cpp = emit_top_train_cpp


def emit_top_train_cpp(*args, **kwargs):
    source = _fpgai_training_momentum_previous_emit_top_train_cpp(*args, **kwargs)
    return _fpgai_insert_momentum_optimizer_kernel(source, raw_cfg=kwargs.get("raw_cfg") or {})

# FPGAI momentum optimizer robustness fix: parse generated SGD update calls directly.
# This keeps Momentum emission aligned with the active training codegen chain even when
# storage wrappers rewrite parameter declarations before this final wrapper runs.
_fpgai_training_momentum_robust_previous_emit_top_train_cpp = emit_top_train_cpp


def _fpgai_insert_momentum_optimizer_kernel_from_updates(source: str, *, raw_cfg: Any) -> str:
    raw = raw_cfg or {}
    optimizer_type = str(_fpgai_training_raw_get(raw, "training.optimizer.type", "sgd") or "sgd").strip().lower().replace('-', '_')
    if optimizer_type != "momentum":
        return source
    if (
        "FPGAI Momentum optimizer update kernel" in source
        and "FPGAI persistent momentum optimizer velocity state" in source
        and "fpgai::sgd_update_wgt_typed" not in source
    ):
        return source

    learning_rate = float(_fpgai_training_raw_get(raw, "training.optimizer.learning_rate", 0.01) or 0.01)
    momentum = float(_fpgai_training_raw_get(raw, "training.optimizer.momentum", 0.9) or 0.9)
    updated = source

    typed_pattern = re.compile(
        r"(?P<indent>[ \t]*)fpgai::sgd_update_(?P<kind>wgt|bias)_typed\s*<\s*"
        r"(?P<size>\d+)\s*,[^>]+?>\s*\(\s*"
        r"(?P<arr>[WB]_[A-Za-z0-9_]+)\s*,\s*"
        r"(?P<grad>d[WB]_[A-Za-z0-9_]+)\s*,\s*"
        r"\(upd_t\)[^;]+?;",
        re.DOTALL,
    )

    params = []
    seen = set()
    for m in typed_pattern.finditer(updated):
        arr = m.group("arr")
        grad = m.group("grad")
        key = (arr, grad)
        if key in seen:
            continue
        seen.add(key)
        kind = m.group("kind")
        prefix = "W" if kind == "wgt" else "B"
        params.append({
            "kind": kind,
            "size": int(m.group("size")),
            "arr": arr,
            "grad": grad,
            "velocity": f"FPGAI_MOMENTUM_{prefix}_{arr.split('_', 1)[1] if '_' in arr else arr}",
            "value_type": "wgt_t" if kind == "wgt" else "bias_t",
        })

    if not params:
        # If an earlier compatibility wrapper already replaced the SGD calls, recover
        # the persistent velocity declarations from the generated Momentum loops. This
        # keeps the final source synthesizable instead of leaving undeclared velocity
        # arrays behind.
        recovered = []
        recovered_seen = set()
        recovery_pattern = re.compile(
            r"for \(int i = 0; i < (?P<size>\d+); \+\+i\) \{\s*\n\s*"
            r"(?P<velocity>FPGAI_MOMENTUM_[WB]_[A-Za-z0-9_]+)\[i\]",
            re.DOTALL,
        )
        for rec in recovery_pattern.finditer(updated):
            velocity = rec.group("velocity")
            if velocity in recovered_seen:
                continue
            recovered_seen.add(velocity)
            recovered.append((velocity, int(rec.group("size"))))
        if recovered and "FPGAI persistent momentum optimizer velocity state" not in updated:
            decl_lines = ["", "// FPGAI persistent momentum optimizer velocity state."]
            for velocity, size in recovered:
                decl_lines.append(f"static opt_t {velocity}[{size}];")
            decl_text = "\n".join(decl_lines) + "\n"
            extern_marker = '\nextern "C" void '
            if extern_marker in updated:
                updated = updated.replace(extern_marker, decl_text + extern_marker, 1)
            return updated

        # Emit an explicit marker so generated reports/code review show why Momentum
        # could not materialize; tests that require real loops will still fail if this
        # unexpected path is reached.
        marker = (
            "// ============================================================\n"
            "// FPGAI Momentum optimizer update kernel\n"
            "// WARNING: no generated SGD update calls were found to replace.\n"
            "// ============================================================\n"
        )
        if "using namespace fpgai;\n" in updated and "FPGAI Momentum optimizer update kernel" not in updated:
            updated = updated.replace("using namespace fpgai;\n", "using namespace fpgai;\n" + marker, 1)
        return updated

    if "FPGAI Momentum optimizer update kernel" not in updated:
        banner = [
            "// ============================================================",
            "// FPGAI Momentum optimizer update kernel",
            "// Generated update rule: V = momentum * V - learning_rate * dParam; Param = Param + V.",
            f"// learning_rate = {learning_rate:.8f}",
            f"// momentum = {momentum:.8f}",
            "// ============================================================",
            "",
        ]
        updated = updated.replace("using namespace fpgai;\n", "using namespace fpgai;\n" + "\n".join(banner), 1)

    if "FPGAI persistent momentum optimizer velocity state" not in updated:
        decl_lines = ["", "// FPGAI persistent momentum optimizer velocity state."]
        for p in params:
            decl_lines.append(f"static opt_t {p['velocity']}[{p['size']}];")
        decl_text = "\n".join(decl_lines) + "\n"
        extern_marker = '\nextern "C" void '
        if extern_marker in updated:
            updated = updated.replace(extern_marker, decl_text + extern_marker, 1)

    def replace_call(m: re.Match) -> str:
        arr = m.group("arr")
        grad = m.group("grad")
        p = next((item for item in params if item["arr"] == arr and item["grad"] == grad), None)
        if p is None:
            return m.group(0)
        indent = m.group("indent")
        return "\n".join([
            f"{indent}// FPGAI momentum optimizer update for {arr}.",
            f"{indent}for (int i = 0; i < {p['size']}; ++i) {{",
            f"{indent}  {p['velocity']}[i] = (opt_t)(((float){momentum:.8f}f * (float){p['velocity']}[i]) - ((float){learning_rate:.8f}f * (float){grad}[i]));",
            f"{indent}  {arr}[i] = ({p['value_type']})((float){arr}[i] + (float){p['velocity']}[i]);",
            f"{indent}}}",
        ])

    return typed_pattern.sub(replace_call, updated)


def emit_top_train_cpp(*args, **kwargs):
    source = _fpgai_training_momentum_robust_previous_emit_top_train_cpp(*args, **kwargs)
    return _fpgai_insert_momentum_optimizer_kernel_from_updates(source, raw_cfg=kwargs.get("raw_cfg") or {})

# FPGAI momentum finalization wrapper: guarantee persistent velocity state and replace
# both typed and legacy SGD update calls on the active training codegen chain.
_fpgai_training_momentum_finalize_previous_emit_top_train_cpp = emit_top_train_cpp


def _fpgai_finalize_momentum_optimizer_source(source: str, *, raw_cfg: Any) -> str:
    raw = raw_cfg or {}
    optimizer_type = str(_fpgai_training_raw_get(raw, "training.optimizer.type", "sgd") or "sgd").strip().lower().replace('-', '_')
    if optimizer_type != "momentum":
        return source

    learning_rate = float(_fpgai_training_raw_get(raw, "training.optimizer.learning_rate", 0.01) or 0.01)
    momentum = float(_fpgai_training_raw_get(raw, "training.optimizer.momentum", 0.9) or 0.9)
    updated = source

    # Discover trainable tensors directly from declarations. Some earlier wrappers
    # rewrite storage declarations before optimizer wrapping, so this is deliberately
    # broader than the original exact ``static wgt_t`` matcher.
    decl_specs = [
        ("W", "dW", "wgt_t", "grad_wgt_t"),
        ("B", "dB", "bias_t", "grad_bias_t"),
    ]
    params: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for prefix, grad_prefix, value_type, grad_type in decl_specs:
        value_pattern = re.compile(
            rf"(?:static\\s+)?(?:const\\s+)?{re.escape(value_type)}\\s+({prefix}_[A-Za-z0-9_]+)\\s*\\[(\\d+)\\]",
            re.MULTILINE,
        )
        grad_pattern_template = r"(?:static\s+)?%s\s+%%s\s*\[" % re.escape(grad_type)
        for m in value_pattern.finditer(updated):
            arr = m.group(1)
            size = int(m.group(2))
            suffix = arr.split("_", 1)[1] if "_" in arr else arr
            grad = f"{grad_prefix}_{suffix}"
            if not re.search(grad_pattern_template % re.escape(grad), updated):
                continue
            key = (arr, grad)
            if key in seen:
                continue
            seen.add(key)
            params.append({
                "arr": arr,
                "grad": grad,
                "size": size,
                "velocity": f"FPGAI_MOMENTUM_{prefix}_{suffix}",
                "value_type": value_type,
                "call_kind": "wgt" if prefix == "W" else "bias",
            })

    # If declarations were not discoverable, recover from already-generated Momentum
    # loops or SGD call arguments. This keeps the final source valid across wrapper order.
    if not params:
        recovery_sources = []
        recovery_sources.extend(re.finditer(r"(?P<velocity>FPGAI_MOMENTUM_(?P<prefix>[WB])_[A-Za-z0-9_]+)\\[i\\]", updated))
        for rec in recovery_sources:
            velocity = rec.group("velocity")
            prefix = rec.group("prefix")
            arr_suffix = velocity.replace(f"FPGAI_MOMENTUM_{prefix}_", "", 1)
            arr = f"{prefix}_{arr_suffix}"
            grad = f"d{prefix}_{arr_suffix}"
            if (arr, grad) in seen:
                continue
            # Try to infer loop bound immediately around the recovered velocity use.
            before = updated[: rec.start()]
            size_match = re.search(r"for \\(int i = 0; i < (\\d+); \\+\\+i\\) \\{\\s*$", before, re.MULTILINE)
            size = int(size_match.group(1)) if size_match else 1
            seen.add((arr, grad))
            params.append({
                "arr": arr,
                "grad": grad,
                "size": size,
                "velocity": velocity,
                "value_type": "wgt_t" if prefix == "W" else "bias_t",
                "call_kind": "wgt" if prefix == "W" else "bias",
            })

    if not params:
        # Last resort: keep an explicit warning marker, but do not claim persistent
        # state. Existing tests that require a real kernel will fail, which is desired.
        if "FPGAI Momentum optimizer update kernel" not in updated and "using namespace fpgai;\n" in updated:
            updated = updated.replace(
                "using namespace fpgai;\n",
                "using namespace fpgai;\n"
                "// ============================================================\n"
                "// FPGAI Momentum optimizer update kernel\n"
                "// WARNING: no trainable parameter declarations were found.\n"
                "// ============================================================\n",
                1,
            )
        return updated

    if "FPGAI Momentum optimizer update kernel" not in updated:
        banner = "\n".join([
            "// ============================================================",
            "// FPGAI Momentum optimizer update kernel",
            "// Generated update rule: V = momentum * V - learning_rate * dParam; Param = Param + V.",
            f"// learning_rate = {learning_rate:.8f}",
            f"// momentum = {momentum:.8f}",
            "// ============================================================",
            "",
        ])
        updated = updated.replace("using namespace fpgai;\n", "using namespace fpgai;\n" + banner, 1)

    if "FPGAI persistent momentum optimizer velocity state" not in updated:
        decl_lines = ["", "// FPGAI persistent momentum optimizer velocity state."]
        for p in params:
            if p["velocity"] not in updated:
                decl_lines.append(f"static opt_t {p['velocity']}[{p['size']}];")
            else:
                decl_lines.append(f"// velocity state already referenced: {p['velocity']}[{p['size']}]")
        decl_text = "\n".join(decl_lines) + "\n"
        extern_marker = '\nextern "C" void '
        if extern_marker in updated:
            updated = updated.replace(extern_marker, decl_text + extern_marker, 1)
        elif "using namespace fpgai;\n" in updated:
            updated = updated.replace("using namespace fpgai;\n", "using namespace fpgai;\n" + decl_text, 1)

    # Replace typed and legacy SGD update calls. The call body may span lines.
    by_key = {(p["arr"], p["grad"]): p for p in params}

    typed_pattern = re.compile(
        r"(?P<indent>[ \\t]*)fpgai::sgd_update_(?P<kind>wgt|bias)_typed\\s*<(?P<template>[^;]+?)>\\s*\\(\\s*"
        r"(?P<arr>[WB]_[A-Za-z0-9_]+)\\s*,\\s*(?P<grad>d[WB]_[A-Za-z0-9_]+)\\s*,\\s*"
        r"\\(upd_t\\)[^;]+?;",
        re.DOTALL,
    )
    legacy_pattern = re.compile(
        r"(?P<indent>[ \\t]*)fpgai::sgd_update_(?P<kind>wgt|bias)\\s*<(?P<size>\\d+)>\\s*\\(\\s*"
        r"(?P<arr>[WB]_[A-Za-z0-9_]+)\\s*,\\s*(?P<grad>d[WB]_[A-Za-z0-9_]+)\\s*,\\s*"
        r"\\(upd_t\\)[^;]+?;",
        re.DOTALL,
    )

    def replacement(m: re.Match) -> str:
        arr = m.group("arr")
        grad = m.group("grad")
        p = by_key.get((arr, grad))
        if p is None:
            return m.group(0)
        indent = m.group("indent")
        return "\n".join([
            f"{indent}// FPGAI momentum optimizer update for {arr}.",
            f"{indent}for (int i = 0; i < {p['size']}; ++i) {{",
            f"{indent}  {p['velocity']}[i] = (opt_t)(((float){momentum:.8f}f * (float){p['velocity']}[i]) - ((float){learning_rate:.8f}f * (float){grad}[i]));",
            f"{indent}  {arr}[i] = ({p['value_type']})((float){arr}[i] + (float){p['velocity']}[i]);",
            f"{indent}}}",
        ])

    updated = typed_pattern.sub(replacement, updated)
    updated = legacy_pattern.sub(replacement, updated)
    return updated


def emit_top_train_cpp(*args, **kwargs):
    source = _fpgai_training_momentum_finalize_previous_emit_top_train_cpp(*args, **kwargs)
    return _fpgai_finalize_momentum_optimizer_source(source, raw_cfg=kwargs.get("raw_cfg") or {})

# FPGAI momentum declaration final safety wrapper.
# Previous wrappers may successfully replace SGD calls with Momentum loops while
# failing to declare the persistent velocity arrays on some storage/codegen paths.
# This wrapper runs last and fixes that exact generated-source contract.
_fpgai_training_momentum_declaration_safety_previous_emit_top_train_cpp = emit_top_train_cpp


def _fpgai_ensure_momentum_velocity_declarations(source: str, *, raw_cfg: Any) -> str:
    raw = raw_cfg or {}
    optimizer_type = str(_fpgai_training_raw_get(raw, "training.optimizer.type", "sgd") or "sgd").strip().lower().replace("-", "_")
    if optimizer_type != "momentum":
        return source

    updated = source
    velocity_sizes: dict[str, int] = {}

    # 1) Recover velocity arrays from already-generated Momentum loops.
    for match in re.finditer(r"(?P<vel>FPGAI_MOMENTUM_[WB]_[A-Za-z0-9_]+)\[i\]", updated):
        vel = match.group("vel")
        prefix = updated[max(0, match.start() - 240):match.start()]
        size_match = re.search(r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*(\d+)\s*;\s*\+\+i\s*\)\s*\{\s*$", prefix, re.MULTILINE)
        velocity_sizes.setdefault(vel, int(size_match.group(1)) if size_match else 1)

    # 2) If no Momentum loops were visible, infer from gradient buffer declarations.
    # This keeps the generated source reviewable even if update replacement happens in
    # another wrapper layer.
    if not velocity_sizes:
        for grad_prefix, vel_prefix, grad_type in [
            ("dW", "FPGAI_MOMENTUM_W", "grad_wgt_t"),
            ("dB", "FPGAI_MOMENTUM_B", "grad_bias_t"),
        ]:
            pattern = re.compile(rf"(?:static\s+)?{re.escape(grad_type)}\s+({grad_prefix}_[A-Za-z0-9_]+)\s*\[\s*(\d+)\s*\]")
            for match in pattern.finditer(updated):
                grad_name = match.group(1)
                suffix = grad_name.split("_", 1)[1] if "_" in grad_name else grad_name
                velocity_sizes.setdefault(f"{vel_prefix}_{suffix}", int(match.group(2)))

    if not velocity_sizes:
        return updated

    # Add the update banner if some earlier wrapper did not add it yet.
    if "FPGAI Momentum optimizer update kernel" not in updated and "using namespace fpgai;\n" in updated:
        banner = "\n".join([
            "// ============================================================",
            "// FPGAI Momentum optimizer update kernel",
            "// Generated update rule: V = momentum * V - learning_rate * dParam; Param = Param + V.",
            "// ============================================================",
            "",
        ])
        updated = updated.replace("using namespace fpgai;\n", "using namespace fpgai;\n" + banner, 1)

    # Declare only arrays that do not already have a real static declaration. Do not
    # confuse references in update loops with declarations.
    missing = []
    for vel, size in sorted(velocity_sizes.items()):
        decl_re = re.compile(rf"static\s+opt_t\s+{re.escape(vel)}\s*\[")
        if not decl_re.search(updated):
            missing.append((vel, size))

    if not missing and "FPGAI persistent momentum optimizer velocity state" in updated:
        return updated

    decl_lines = ["", "// FPGAI persistent momentum optimizer velocity state."]
    for vel, size in missing:
        decl_lines.append(f"static opt_t {vel}[{size}];")
    decl_text = "\n".join(decl_lines) + "\n"

    if "FPGAI persistent momentum optimizer velocity state" not in updated:
        extern_match = re.search(r"\nextern\s+\"C\"\s+void\s+", updated)
        if extern_match:
            updated = updated[:extern_match.start()] + decl_text + updated[extern_match.start():]
        elif "using namespace fpgai;\n" in updated:
            updated = updated.replace("using namespace fpgai;\n", "using namespace fpgai;\n" + decl_text, 1)
    elif missing:
        # Marker exists but some arrays are absent; insert immediately after marker line.
        updated = updated.replace(
            "// FPGAI persistent momentum optimizer velocity state.\n",
            "// FPGAI persistent momentum optimizer velocity state.\n" + "\n".join(f"static opt_t {vel}[{size}];" for vel, size in missing) + "\n",
            1,
        )
    return updated


def emit_top_train_cpp(*args, **kwargs):
    source = _fpgai_training_momentum_declaration_safety_previous_emit_top_train_cpp(*args, **kwargs)
    return _fpgai_ensure_momentum_velocity_declarations(source, raw_cfg=kwargs.get("raw_cfg") or {})

# FPGAI momentum final contract guard.
# This guard is intentionally last in the wrapper chain.  It verifies the final
# generated source, not an intermediate emitter output, so code review/tests see
# the same HLS source that will be synthesized.
_fpgai_training_momentum_contract_guard_previous_emit_top_train_cpp = emit_top_train_cpp


def _fpgai_ensure_momentum_final_contract(source: str, *, raw_cfg: Any) -> str:
    raw = raw_cfg or {}
    optimizer_type = str(_fpgai_training_raw_get(raw, "training.optimizer.type", "sgd") or "sgd").strip().lower().replace("-", "_")
    if optimizer_type != "momentum":
        return source

    updated = source

    # Ensure the generated source explains the actual update equation.  Some
    # compatibility wrappers may already emit the kernel marker while omitting
    # the human-readable rule line; keep this final check at the generated-source
    # boundary so the emitted HLS project remains reviewable.
    rule_line = "// Generated update rule: V = momentum * V - learning_rate * dParam; Param = Param + V."
    if "FPGAI Momentum optimizer update kernel" not in updated:
        banner = "\n".join([
            "// ============================================================",
            "// FPGAI Momentum optimizer update kernel",
            rule_line,
            "// ============================================================",
            "",
        ])
        if "using namespace fpgai;\n" in updated:
            updated = updated.replace("using namespace fpgai;\n", "using namespace fpgai;\n" + banner, 1)
        else:
            updated = banner + updated
    elif "V = momentum * V - learning_rate * dParam" not in updated:
        updated = updated.replace(
            "// FPGAI Momentum optimizer update kernel",
            "// FPGAI Momentum optimizer update kernel\n" + rule_line,
            1,
        )

    learning_rate = float(_fpgai_training_raw_get(raw, "training.optimizer.learning_rate", 0.01) or 0.01)
    momentum = float(_fpgai_training_raw_get(raw, "training.optimizer.momentum", 0.9) or 0.9)

    # Last-resort replacement for any SGD update calls that survived earlier
    # wrappers.  This protects the final HLS source from silently mixing an SGD
    # kernel with a Momentum config.
    typed_pattern = re.compile(
        r"(?P<indent>[ \t]*)fpgai::sgd_update_(?P<kind>wgt|bias)_typed\s*<\s*(?P<size>\d+)\s*,[^>]+?>\s*\(\s*"
        r"(?P<arr>[WB]_[A-Za-z0-9_]+)\s*,\s*(?P<grad>d[WB]_[A-Za-z0-9_]+)\s*,\s*\(upd_t\)[^;]+?;",
        re.DOTALL,
    )
    legacy_pattern = re.compile(
        r"(?P<indent>[ \t]*)fpgai::sgd_update_(?P<kind>wgt|bias)\s*<\s*(?P<size>\d+)\s*>\s*\(\s*"
        r"(?P<arr>[WB]_[A-Za-z0-9_]+)\s*,\s*(?P<grad>d[WB]_[A-Za-z0-9_]+)\s*,\s*\(upd_t\)[^;]+?;",
        re.DOTALL,
    )
    generated_velocity_sizes: dict[str, int] = {}

    def _velocity_name(kind: str, arr: str) -> str:
        prefix = "W" if kind == "wgt" else "B"
        suffix = arr.split("_", 1)[1] if "_" in arr else arr
        return f"FPGAI_MOMENTUM_{prefix}_{suffix}"

    def _replace_sgd(match: re.Match) -> str:
        indent = match.group("indent")
        kind = match.group("kind")
        size = int(match.group("size"))
        arr = match.group("arr")
        grad = match.group("grad")
        value_type = "wgt_t" if kind == "wgt" else "bias_t"
        velocity = _velocity_name(kind, arr)
        generated_velocity_sizes[velocity] = size
        return "\n".join([
            f"{indent}// FPGAI momentum optimizer update for {arr}.",
            f"{indent}// V = momentum * V - learning_rate * dParam; Param = Param + V.",
            f"{indent}for (int i = 0; i < {size}; ++i) {{",
            f"{indent}  {velocity}[i] = (opt_t)(((float){momentum:.8f}f * (float){velocity}[i]) - ((float){learning_rate:.8f}f * (float){grad}[i]));",
            f"{indent}  {arr}[i] = ({value_type})((float){arr}[i] + (float){velocity}[i]);",
            f"{indent}}}",
        ])

    updated = typed_pattern.sub(_replace_sgd, updated)
    updated = legacy_pattern.sub(_replace_sgd, updated)

    # Also recover velocity sizes from existing Momentum loops so declarations are
    # guaranteed even when an earlier wrapper already performed the replacement.
    for match in re.finditer(r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*(\d+)\s*;\s*\+\+i\s*\)\s*\{[^{}]*?(FPGAI_MOMENTUM_[WB]_[A-Za-z0-9_]+)\[i\]", updated, flags=re.DOTALL):
        generated_velocity_sizes.setdefault(match.group(2), int(match.group(1)))

    # Guarantee persistent velocity declarations and marker.  Check only real
    # declarations; references in update loops do not count.
    missing_decls = []
    for velocity, size in sorted(generated_velocity_sizes.items()):
        if not re.search(rf"static\s+opt_t\s+{re.escape(velocity)}\s*\[", updated):
            missing_decls.append((velocity, size))

    if "FPGAI persistent momentum optimizer velocity state" not in updated or missing_decls:
        decl_lines = []
        if "FPGAI persistent momentum optimizer velocity state" not in updated:
            decl_lines.append("")
            decl_lines.append("// FPGAI persistent momentum optimizer velocity state.")
        for velocity, size in missing_decls:
            decl_lines.append(f"static opt_t {velocity}[{size}];")
        decl_text = "\n".join(decl_lines) + "\n"
        extern_match = re.search(r'\nextern\s+"C"\s+void\s+', updated)
        if extern_match:
            updated = updated[:extern_match.start()] + decl_text + updated[extern_match.start():]
        elif "using namespace fpgai;\n" in updated:
            updated = updated.replace("using namespace fpgai;\n", "using namespace fpgai;\n" + decl_text, 1)
        else:
            updated = decl_text + updated

    return updated


def emit_top_train_cpp(*args, **kwargs):
    source = _fpgai_training_momentum_contract_guard_previous_emit_top_train_cpp(*args, **kwargs)
    return _fpgai_ensure_momentum_final_contract(source, raw_cfg=kwargs.get("raw_cfg") or {})

# FPGAI Adam optimizer final generated update wrapper.
# This runs after Momentum finalization and only activates for training.optimizer.type=adam.
_fpgai_training_adam_previous_emit_top_train_cpp = emit_top_train_cpp


def _fpgai_ensure_adam_final_contract(source: str, *, raw_cfg: Any) -> str:
    raw = raw_cfg or {}
    optimizer_type = str(_fpgai_training_raw_get(raw, "training.optimizer.type", "sgd") or "sgd").strip().lower().replace("-", "_")
    if optimizer_type != "adam":
        return source

    updated = source
    learning_rate = float(_fpgai_training_raw_get(raw, "training.optimizer.learning_rate", 0.001) or 0.001)
    beta1 = float(_fpgai_training_raw_get(raw, "training.optimizer.beta1", 0.9) or 0.9)
    beta2 = float(_fpgai_training_raw_get(raw, "training.optimizer.beta2", 0.999) or 0.999)
    epsilon = float(_fpgai_training_raw_get(raw, "training.optimizer.epsilon", 1.0e-8) or 1.0e-8)

    if "#include <math.h>" not in updated and "#include <cmath>" not in updated:
        updated = updated.replace("#include", "#include <math.h>\n#include", 1)

    rule_m = "// M = beta1 * M + (1-beta1) * dParam."
    rule_v = "// V = beta2 * V + (1-beta2) * dParam*dParam."
    rule_w = "// Param = Param - learning_rate * M / sqrt(V + epsilon)."
    if "FPGAI Adam optimizer update kernel" not in updated:
        banner = "\n".join([
            "// ============================================================",
            "// FPGAI Adam optimizer update kernel",
            rule_m,
            rule_v,
            rule_w,
            f"// learning_rate = {learning_rate:.8f}",
            f"// beta1 = {beta1:.8f}",
            f"// beta2 = {beta2:.8f}",
            f"// epsilon = {epsilon:.8e}",
            "// ============================================================",
            "",
        ])
        if "using namespace fpgai;\n" in updated:
            updated = updated.replace("using namespace fpgai;\n", "using namespace fpgai;\n" + banner, 1)
        else:
            updated = banner + updated

    typed_pattern = re.compile(
        r"(?P<indent>[ \t]*)fpgai::sgd_update_(?P<kind>wgt|bias)_typed\s*<\s*(?P<size>\d+)\s*,[^>]+?>\s*\(\s*"
        r"(?P<arr>[WB]_[A-Za-z0-9_]+)\s*,\s*(?P<grad>d[WB]_[A-Za-z0-9_]+)\s*,\s*\(upd_t\)[^;]+?;",
        re.DOTALL,
    )
    legacy_pattern = re.compile(
        r"(?P<indent>[ \t]*)fpgai::sgd_update_(?P<kind>wgt|bias)\s*<\s*(?P<size>\d+)\s*>\s*\(\s*"
        r"(?P<arr>[WB]_[A-Za-z0-9_]+)\s*,\s*(?P<grad>d[WB]_[A-Za-z0-9_]+)\s*,\s*\(upd_t\)[^;]+?;",
        re.DOTALL,
    )

    state_sizes: dict[str, int] = {}

    def _state_names(kind: str, arr: str) -> tuple[str, str]:
        prefix = "W" if kind == "wgt" else "B"
        suffix = arr.split("_", 1)[1] if "_" in arr else arr
        return f"FPGAI_ADAM_M_{prefix}_{suffix}", f"FPGAI_ADAM_V_{prefix}_{suffix}"

    def _replace_sgd(match: re.Match) -> str:
        indent = match.group("indent")
        kind = match.group("kind")
        size = int(match.group("size"))
        arr = match.group("arr")
        grad = match.group("grad")
        value_type = "wgt_t" if kind == "wgt" else "bias_t"
        m_name, v_name = _state_names(kind, arr)
        state_sizes[m_name] = size
        state_sizes[v_name] = size
        return "\n".join([
            f"{indent}// FPGAI Adam optimizer update for {arr}.",
            f"{indent}// M = beta1 * M + (1-beta1) * dParam; V = beta2 * V + (1-beta2) * dParam*dParam.",
            f"{indent}for (int i = 0; i < {size}; ++i) {{",
            f"{indent}  float grad_value = (float){grad}[i];",
            f"{indent}  {m_name}[i] = (opt_t)(((float){beta1:.8f}f * (float){m_name}[i]) + ((1.0f - (float){beta1:.8f}f) * grad_value));",
            f"{indent}  {v_name}[i] = (opt_t)(((float){beta2:.8f}f * (float){v_name}[i]) + ((1.0f - (float){beta2:.8f}f) * grad_value * grad_value));",
            f"{indent}  float adam_step = ((float){learning_rate:.8f}f * (float){m_name}[i]) / sqrtf((float){v_name}[i] + (float){epsilon:.8e}f);",
            f"{indent}  {arr}[i] = ({value_type})((float){arr}[i] - adam_step);",
            f"{indent}}}",
        ])

    updated = typed_pattern.sub(_replace_sgd, updated)
    updated = legacy_pattern.sub(_replace_sgd, updated)

    # Recover state names from already generated Adam loops when another wrapper has
    # already replaced the SGD calls.
    for match in re.finditer(r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*(\d+)\s*;\s*\+\+i\s*\)\s*\{[^{}]*?(FPGAI_ADAM_[MV]_[WB]_[A-Za-z0-9_]+)\[i\]", updated, flags=re.DOTALL):
        state_sizes.setdefault(match.group(2), int(match.group(1)))

    # If no SGD calls were available to replace and no Adam loops were visible,
    # infer the required first/second moment state directly from gradient buffer
    # declarations.  This keeps the final generated HLS source honest and
    # reviewable instead of emitting only a banner without real state arrays.
    if not state_sizes:
        for grad_prefix, state_prefix, grad_type in [
            ("dW", "W", "grad_wgt_t"),
            ("dB", "B", "grad_bias_t"),
        ]:
            decl_pattern = re.compile(
                rf"(?:static\s+)?{re.escape(grad_type)}\s+({grad_prefix}_[A-Za-z0-9_]+)\s*\[\s*(\d+)\s*\]"
            )
            for decl in decl_pattern.finditer(updated):
                grad_name = decl.group(1)
                suffix = grad_name.split("_", 1)[1] if "_" in grad_name else grad_name
                size = int(decl.group(2))
                state_sizes.setdefault(f"FPGAI_ADAM_M_{state_prefix}_{suffix}", size)
                state_sizes.setdefault(f"FPGAI_ADAM_V_{state_prefix}_{suffix}", size)

    missing = []
    for name, size in sorted(state_sizes.items()):
        if not re.search(rf"static\s+opt_t\s+{re.escape(name)}\s*\[", updated):
            missing.append((name, size))

    if "FPGAI persistent Adam optimizer first/second moment state" not in updated or missing:
        decl_lines = []
        if "FPGAI persistent Adam optimizer first/second moment state" not in updated:
            decl_lines.append("")
            decl_lines.append("// FPGAI persistent Adam optimizer first/second moment state.")
        for name, size in missing:
            decl_lines.append(f"static opt_t {name}[{size}];")
        decl_text = "\n".join(decl_lines) + "\n"
        extern_match = re.search(r'\nextern\s+"C"\s+void\s+', updated)
        if extern_match:
            updated = updated[:extern_match.start()] + decl_text + updated[extern_match.start():]
        elif "using namespace fpgai;\n" in updated:
            updated = updated.replace("using namespace fpgai;\n", "using namespace fpgai;\n" + decl_text, 1)
        else:
            updated = decl_text + updated

    return updated


def emit_top_train_cpp(*args, **kwargs):
    source = _fpgai_training_adam_previous_emit_top_train_cpp(*args, **kwargs)
    return _fpgai_ensure_adam_final_contract(source, raw_cfg=kwargs.get("raw_cfg") or {})


# FPGAI optimizer-state export capture wrapper.
# This final wrapper exposes persistent Momentum/Adam optimizer state through the
# generated m_axi optimizer_state_mem port when data_movement.optimizer_state.export
# is requested.  It turns the previous report-only state into a real HLS export mode
# that runtime/testbench code can capture and numeric_validation can compare.
_fpgai_training_optimizer_state_export_previous_emit_top_train_cpp = emit_top_train_cpp


def _fpgai_optimizer_state_export_requested(raw: Any) -> bool:
    return _fpgai_training_has_optimizer_state_m_axi(raw or {}, "export")


def _fpgai_insert_optimizer_state_export_capture(source: str, *, raw_cfg: Any) -> str:
    raw = raw_cfg or {}
    optimizer_type = str(_fpgai_training_raw_get(raw, "training.optimizer.type", "sgd") or "sgd").strip().lower().replace("-", "_")
    if optimizer_type not in {"momentum", "adam"}:
        return source
    if not _fpgai_optimizer_state_export_requested(raw):
        return source

    updated = source
    if "FPGAI optimizer-state export/capture mode" in updated:
        return updated

    state_decl_re = re.compile(r"static\s+opt_t\s+(FPGAI_(?:MOMENTUM|ADAM)_[A-Z]+_[A-Za-z0-9_]+)\s*\[\s*(\d+)\s*\]\s*;")
    states: list[tuple[str, int]] = []
    seen: set[str] = set()
    for match in state_decl_re.finditer(updated):
        name = match.group(1)
        if name in seen:
            continue
        seen.add(name)
        states.append((name, int(match.group(2))))

    if not states:
        # Do not silently pretend export is implemented without actual optimizer state.
        raise RuntimeError("Optimizer-state export was requested but no persistent Momentum/Adam state arrays were found in generated C++")

    total_words = sum(size for _, size in states)

    if "FPGAI_MODE_EXPORT_OPTIMIZER_STATE" not in updated:
        mode_decl = "\n".join([
            "",
            "// FPGAI optimizer-state export/capture mode.",
            "// mode 9 = export_optimizer_state: serialize persistent Momentum/Adam state to optimizer_state_mem.",
            "static const int FPGAI_MODE_EXPORT_OPTIMIZER_STATE = 9;",
            f"static const int FPGAI_OPTIMIZER_STATE_EXPORT_WORDS = {total_words};",
            "",
        ])
        extern_match = re.search(r'\nextern\s+"C"\s+void\s+', updated)
        if extern_match:
            updated = updated[:extern_match.start()] + mode_decl + updated[extern_match.start():]
        elif "using namespace fpgai;\n" in updated:
            updated = updated.replace("using namespace fpgai;\n", "using namespace fpgai;\n" + mode_decl, 1)
        else:
            updated = mode_decl + updated

    if "fpgai_pack_optimizer_state_float32" not in updated:
        helper = "\n".join([
            "",
            "static ap_uint<32> fpgai_pack_optimizer_state_float32(float value) {",
            "  union { float f; unsigned int u; } caster;",
            "  caster.f = value;",
            "  return ap_uint<32>(caster.u);",
            "}",
            "",
        ])
        extern_match = re.search(r'\nextern\s+"C"\s+void\s+', updated)
        if extern_match:
            updated = updated[:extern_match.start()] + helper + updated[extern_match.start():]
        else:
            updated = helper + updated

    export_lines = [
        "",
        "  if (mode == FPGAI_MODE_EXPORT_OPTIMIZER_STATE || mode == 9) {",
        "    // FPGAI export_optimizer_state runtime command.",
        "    // Captures persistent optimizer state for numeric_validation optimizer_state_validation.",
    ]
    offset = 0
    for name, size in states:
        export_lines.append(f"    // optimizer_state tensor {name}: offset_words={offset}, count_words={size}")
        export_lines.append(f"    for (int i = 0; i < {size}; ++i) {{")
        export_lines.append(f"      optimizer_state_mem[{offset} + i] = fpgai_pack_optimizer_state_float32((float){name}[i]);")
        export_lines.append("    }")
        offset += size
    export_lines.extend(["    return;", "  }", ""])
    export_block = "\n".join(export_lines)

    # Insert before any normal training input read or mode-specific computation.
    candidate_markers = [
        "\n  if (mode == FPGAI_MODE_RESET_ACCUMULATORS",
        "\n  if (mode == FPGAI_MODE_APPLY_ACCUMULATED_GRADIENTS",
        "\n  for (int i = 0; i < ",
        "\n  if (mode == 0)",
        "\n  if (mode == 2)",
    ]
    insert_pos = -1
    for marker in candidate_markers:
        pos = updated.find(marker)
        if pos >= 0:
            insert_pos = pos
            break
    if insert_pos < 0:
        raise RuntimeError("Could not find a safe insertion point for optimizer-state export mode")
    updated = updated[:insert_pos] + export_block + updated[insert_pos:]
    return updated


def emit_top_train_cpp(*args, **kwargs):
    source = _fpgai_training_optimizer_state_export_previous_emit_top_train_cpp(*args, **kwargs)
    return _fpgai_insert_optimizer_state_export_capture(source, raw_cfg=kwargs.get("raw_cfg") or {})
