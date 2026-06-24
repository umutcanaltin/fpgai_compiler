from __future__ import annotations
from fpgai.backends.hls.emit.architecture_comments import emit_layer_architecture_comments

# FPGAI architecture-comment wrapper.

import re
from typing import Any, Dict, List, Optional, Tuple

from fpgai.backends.hls.emit.dense_tiling_codegen import apply_dense_tiling_to_top_source
<<<<<<< HEAD
from fpgai.backends.hls.emit.dense_tiling_codegen import apply_dense_training_tiling_to_top_source
=======
>>>>>>> 901de078132a537e425cac7602bc09eef226e2d3
from fpgai.backends.hls.emit.conv_tiling_codegen import apply_conv_tiling_to_top_source
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

    decl = ["", "// FPGAI native accumulated mini-batch optimizer state."]
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
        "  if (mode == 5) {",
        *reset_lines("    "),
        "    return;",
        "  }",
        "",
        "  if (mode == 4) {",
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

    accum = ["", "  if (mode == 3) {"]
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
    source = _fpgai_train_comm_previous_emit_top_train_cpp(*args, **kwargs)
    macros = _fpgai_train_comm_edge_macros(communication_plan)
    return macros + source



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
<<<<<<< HEAD
    source = apply_dense_training_tiling_to_top_source(source, graph, compile_plan)
=======
>>>>>>> 901de078132a537e425cac7602bc09eef226e2d3
    source = apply_conv_tiling_to_top_source(source, graph, compile_plan)

    if "FPGAI real dense tiling helper" in source or "FPGAI real convolution tiling helper" in source:
        source = (
            "// FPGAI training forward tiling materialized. "
<<<<<<< HEAD
            "Dense backward/update tiling is materialized; conv backward/update tiling remains training-specific future work.\n"
=======
            "Backward/update tiling remains training-specific future work.\n"
>>>>>>> 901de078132a537e425cac7602bc09eef226e2d3
            + source
        )

    return source
