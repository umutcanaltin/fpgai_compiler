"""CSim-only selected-parameter probes for generated training HLS.

The emitted globals and assignments are excluded from synthesis with
``__SYNTHESIS__`` guards.  They exist only to expose one selected Dense weight
update to the generated C simulation testbench without changing hardware.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

PROBE_STAGES = (
    "forward_input",
    "backward_output_gradient",
    "parameter_gradient_term",
    "parameter_gradient_accumulated",
    "optimizer_m",
    "optimizer_v",
    "optimizer_delta",
    "parameter_before",
    "parameter_after",
    "dense_forward_output",
    "activation_forward_output",
    "activation_upstream_gradient",
    "activation_backward_output",
)


def _lookup(raw: Mapping[str, Any] | None, path: str, default: Any = None) -> Any:
    cur: Any = raw or {}
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def selected_dense_weight_probe(raw_cfg: Mapping[str, Any] | None) -> dict[str, Any] | None:
    probes = _lookup(raw_cfg, "validation.numeric.probes", {})
    if not isinstance(probes, Mapping) or not bool(probes.get("enabled", False)):
        return None
    selectors = probes.get("selectors", [])
    if not isinstance(selectors, list) or len(selectors) != 1 or not isinstance(selectors[0], Mapping):
        raise ValueError("HLS training probes currently require exactly one selector.")
    selector = selectors[0]
    operator = str(selector.get("operator", "")).strip()
    parameter = str(selector.get("parameter", "weight")).strip().lower()
    index = selector.get("tensor_index")
    if not operator or parameter != "weight" or not isinstance(index, list) or len(index) != 2:
        raise ValueError(
            "HLS training probes currently support one Dense weight selector with tensor_index=[row, col]."
        )
    return {"operator": operator, "row": int(index[0]), "col": int(index[1])}


def instrument_fused_dense_adam_probe(source: str, *, raw_cfg: Mapping[str, Any] | None) -> str:
    """Instrument one fused Dense/Adam update with CSim-only scalar probes."""
    selector = selected_dense_weight_probe(raw_cfg)
    if selector is None:
        return source

    tag = re.sub(r"[^A-Za-z0-9_]", "_", selector["operator"])
    marker = f"// FPGAI fused_update Adam update for W_{tag};"
    marker_pos = source.find(marker)
    if marker_pos < 0:
        raise ValueError(
            f"validation.numeric.probes selector {selector['operator']}.weight requires "
            "a generated fused Dense/Adam update for that operator."
        )

    input_feature_match = re.search(
        r"const int input_index = gradient_index % (\d+);",
        source[marker_pos:],
    )
    if input_feature_match is None:
        raise RuntimeError("Could not resolve Dense input feature count for HLS training probe.")
    input_features = int(input_feature_match.group(1))
    selected_flat = selector["row"] * input_features + selector["col"]

    declaration = """
#ifndef __SYNTHESIS__
extern \"C\" float fpgai_training_probe_values[16];
float fpgai_training_probe_values[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#endif
"""
    if "fpgai_training_probe_values[16]" not in source:
        include_pos = source.find("\n", source.find("#include"))
        if include_pos < 0:
            source = declaration + source
        else:
            source = source[: include_pos + 1] + declaration + source[include_pos + 1 :]
        marker_pos = source.find(marker)

    loop_end = source.find("\n}", marker_pos)
    if loop_end < 0:
        raise RuntimeError("Could not find the end of the selected fused-update loop.")
    loop = source[marker_pos:loop_end]

    grad_line = re.search(r"(?m)^(\s*)const float fused_grad_value = ([^;]+);", loop)
    if grad_line is None:
        raise RuntimeError("Could not find fused_grad_value in selected HLS update.")
    indent = grad_line.group(1)
    # Use the canonical local input/output indices, avoiding source-level tensor-name assumptions.
    capture_grad = "\n".join([
        grad_line.group(0),
        f"{indent}#ifndef __SYNTHESIS__",
        f"{indent}if (gradient_index == 0) fpgai_training_probe_values[9] = 1.0f;",
        f"{indent}if (gradient_index == {selected_flat}) {{",
        f"{indent}  fpgai_training_probe_values[10] = 1.0f;",
        f"{indent}  fpgai_training_probe_values[0] = (float){_input_expr(loop)};",
        f"{indent}  fpgai_training_probe_values[1] = (float){_output_grad_expr(loop)};",
        f"{indent}  fpgai_training_probe_values[2] = fused_grad_value;",
        f"{indent}  fpgai_training_probe_values[3] = fused_grad_value;",
        f"{indent}}}",
        f"{indent}#endif",
    ])
    loop = loop.replace(grad_line.group(0), capture_grad, 1)

    m_line = re.search(rf"(?m)^(\s*)float adam_m_used = \(float\)FPGAI_ADAM_M_W_{re.escape(tag)}\[gradient_index\];", loop)
    v_line = re.search(rf"(?m)^(\s*)float adam_v_used = \(float\)FPGAI_ADAM_V_W_{re.escape(tag)}\[gradient_index\];", loop)
    delta_line = re.search(r"(?m)^(\s*)float adam_delta = [^;]+;", loop)
    update_line = re.search(rf"(?m)^(\s*)W_{re.escape(tag)}\[gradient_index\] = \(wgt_t\)\(\(float\)W_{re.escape(tag)}\[gradient_index\] - adam_delta\);", loop)
    if not all((m_line, v_line, delta_line, update_line)):
        raise RuntimeError("Could not find Adam state/update statements for selected HLS training probe.")

    loop = loop.replace(
        m_line.group(0),
        m_line.group(0) + "\n" + "\n".join([
            f"{m_line.group(1)}#ifndef __SYNTHESIS__",
            f"{m_line.group(1)}if (gradient_index == {selected_flat}) fpgai_training_probe_values[4] = adam_m_used;",
            f"{m_line.group(1)}#endif",
        ]), 1,
    )
    loop = loop.replace(
        v_line.group(0),
        v_line.group(0) + "\n" + "\n".join([
            f"{v_line.group(1)}#ifndef __SYNTHESIS__",
            f"{v_line.group(1)}if (gradient_index == {selected_flat}) fpgai_training_probe_values[5] = adam_v_used;",
            f"{v_line.group(1)}#endif",
        ]), 1,
    )
    loop = loop.replace(
        delta_line.group(0),
        delta_line.group(0) + "\n" + "\n".join([
            f"{delta_line.group(1)}#ifndef __SYNTHESIS__",
            f"{delta_line.group(1)}if (gradient_index == {selected_flat}) {{",
            f"{delta_line.group(1)}  fpgai_training_probe_values[6] = -adam_delta;",
            f"{delta_line.group(1)}  fpgai_training_probe_values[7] = (float)W_{tag}[gradient_index];",
            f"{delta_line.group(1)}}}",
            f"{delta_line.group(1)}#endif",
        ]), 1,
    )
    loop = loop.replace(
        update_line.group(0),
        update_line.group(0) + "\n" + "\n".join([
            f"{update_line.group(1)}#ifndef __SYNTHESIS__",
            f"{update_line.group(1)}if (gradient_index == {selected_flat}) {{",
            f"{update_line.group(1)}  fpgai_training_probe_values[8] = (float)W_{tag}[gradient_index];",
            f"{update_line.group(1)}  fpgai_training_probe_values[11] = 1.0f;",
            f"{update_line.group(1)}}}",
            f"{update_line.group(1)}#endif",
        ]), 1,
    )

    source = source[:marker_pos] + loop + source[loop_end:]

    # Capture the activation boundary that owns the selected Dense output
    # gradient.  This distinguishes a Dense backward defect from a legitimate
    # ReLU branch change caused by fixed-point forward arithmetic.
    output_grad_buffer = _output_grad_expr(loop).split("[", 1)[0]
    relu_pattern = re.compile(
        r"fpgai::relu_backward_from_(?P<owner>output|mask)<(?P<size>\d+)>\s*\(\s*"
        r"(?P<owner_buffer>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
        r"(?P<upstream>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
        + re.escape(output_grad_buffer)
        + r"\s*\);"
    )
    relu_match = relu_pattern.search(source)
    if relu_match is not None:
        if selector["row"] >= int(relu_match.group("size")):
            raise ValueError("Selected Dense output row is outside the owning ReLU tensor.")
        dense_output_buffer = None
        dense_call = re.search(
            rf"fpgai::dense(?:_tiled)?<[^>]+>\s*\([^;]*?,\s*(?P<out>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*W_{re.escape(tag)}\s*,",
            source,
            flags=re.DOTALL,
        )
        if dense_call is not None:
            dense_output_buffer = dense_call.group("out")
        upstream = relu_match.group("upstream")
        owner_kind = relu_match.group("owner")
        owner_buffer = relu_match.group("owner_buffer")
        activation = owner_buffer
        if owner_kind == "mask":
            forward_match = re.search(
                rf"fpgai::relu_with_mask<{relu_match.group('size')}>\s*\(\s*"
                rf"(?P<input>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
                rf"(?P<output>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
                + re.escape(owner_buffer)
                + r"\s*\);",
                source,
            )
            if forward_match is not None:
                activation = forward_match.group("output")
                if dense_output_buffer is None:
                    dense_output_buffer = forward_match.group("input")
        dense_expr = dense_output_buffer or activation
        capture = "\n".join([
            relu_match.group(0),
            "  #ifndef __SYNTHESIS__",
            f"  fpgai_training_probe_values[12] = (float){dense_expr}[{selector['row']}];",
            f"  fpgai_training_probe_values[13] = (float){activation}[{selector['row']}];",
            f"  fpgai_training_probe_values[14] = (float){upstream}[{selector['row']}];",
            f"  fpgai_training_probe_values[15] = (float){output_grad_buffer}[{selector['row']}];",
            "  #endif",
        ])
        source = source[:relu_match.start()] + capture + source[relu_match.end():]

    return source


def _input_expr(loop: str) -> str:
    match = re.search(r"const float fused_grad_value = \(\(float\)([A-Za-z0-9_]+)\[input_index\]", loop)
    if match is None:
        raise RuntimeError("Could not resolve fused-update input buffer for HLS probe.")
    return f"{match.group(1)}[input_index]"


def _output_grad_expr(loop: str) -> str:
    match = re.search(r"\* \(float\)([A-Za-z0-9_]+)\[output_index\]\)", loop)
    if match is None:
        raise RuntimeError("Could not resolve fused-update output-gradient buffer for HLS probe.")
    return f"{match.group(1)}[output_index]"
