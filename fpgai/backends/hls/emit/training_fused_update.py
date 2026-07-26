"""Dense/Adam fused parameter-gradient lowering for generated training HLS.

This module owns the source-to-source transformation for
``training.gradients.computation=fused_update``.  The public training emitter
remains in :mod:`top_train_cpp`; dependencies on its graph/source discovery
helpers are injected explicitly to keep this mechanism independently testable.
"""

from __future__ import annotations

import re
from typing import Any, Callable


RawGet = Callable[[Any, str, Any], Any]
GradientPolicy = Callable[[Any], tuple[str, str]]
DenseGradientSpecs = Callable[[str], list[dict[str, Any]]]
BalancedBraceEnd = Callable[[str, int], int]


def _replace_adam_weight_update(
    source: str,
    *,
    spec: dict[str, Any],
    balanced_brace_end: BalancedBraceEnd,
    prefer_direct_path: bool = True,
) -> str:
    """Replace one materialized Dense Adam weight update with fused recompute.

    The generated loop uses a canonical row-major ``gradient_index`` and keeps
    the gradient in ``float`` for Adam arithmetic.  Casting to ``grad_wgt_t``
    before the optimizer update caused unnecessary early quantization and made
    the optimizer path differ from the reference update contract.
    """

    tag = spec["tag"]
    dw = spec["dw"]
    weight = f"W_{tag}"
    marker = f"// FPGAI Adam optimizer update for {weight}."
    marker_positions = [match.start() for match in re.finditer(re.escape(marker), source)]
    if not marker_positions:
        raise RuntimeError(
            f"fused_update could not find the final Adam update owner for {weight}"
        )

    # Native-accumulation support emits an Adam update inside mode 4 before the
    # ordinary direct-training update. Fused-update is currently restricted to
    # direct single-record schedules, so lowering the first marker incorrectly
    # transformed the dormant mode-4 branch and left the executed mode-2 path
    # without its materialized dW producer. Select the final marker for the
    # direct path, while preserving the single-marker behavior used by compact
    # unit fixtures and emitters without native accumulation.
    marker_pos = marker_positions[-1] if prefer_direct_path else marker_positions[0]
    loop_pos = source.find("for (int i = 0; i < ", marker_pos)
    if loop_pos < 0:
        raise RuntimeError(f"fused_update could not find the Adam loop for {weight}")
    open_brace = source.find("{", loop_pos)
    loop_end = balanced_brace_end(source, open_brace)
    body = source[open_brace + 1 : loop_end - 1]
    if f"{dw}[i]" not in body:
        raise RuntimeError(
            f"fused_update Adam update for {weight} does not consume {dw}[i]"
        )

    # Adam consumes the recomputed gradient as float.  The exported gradient
    # still follows the configured grad_wgt_t contract in the export branch.
    expression = (
        f"((float){spec['input']}[input_index] * "
        f"(float){spec['output_grad']}[output_index])"
    )
    transformed_body = body.replace(f"(float){dw}[i]", "fused_grad_value")
    transformed_body = transformed_body.replace(f"{dw}[i]", "(grad_wgt_t)fused_grad_value")
    transformed_body = transformed_body.replace("[i]", "[gradient_index]")
    transformed_body = "\n".join(
        ("  " + line if line.strip() else line) for line in transformed_body.splitlines()
    )

    count = int(spec["input_features"]) * int(spec["output_features"])
    replacement = "\n".join(
        [
            f"// FPGAI fused_update Adam update for {weight}; no complete or tiled {dw} buffer is materialized.",
            f"for (int gradient_index = 0; gradient_index < {count}; ++gradient_index) {{",
            "#pragma HLS PIPELINE II=1",
            f"  const int output_index = gradient_index / {int(spec['input_features'])};",
            f"  const int input_index = gradient_index % {int(spec['input_features'])};",
            f"  const float fused_grad_value = {expression};",
            transformed_body,
            "}",
        ]
    )
    return source[:marker_pos] + replacement + source[loop_end:]


def materialize_dense_fused_update(
    source: str,
    *,
    raw_cfg: Any,
    raw_get: RawGet,
    parameter_gradient_policy: GradientPolicy,
    dense_gradient_specs: DenseGradientSpecs,
    balanced_brace_end: BalancedBraceEnd,
) -> str:
    """Lower Dense Adam parameter gradients into direct optimizer updates."""

    raw = raw_cfg or {}
    computation, storage = parameter_gradient_policy(raw)
    if computation != "fused_update":
        return source

    optimizer = str(
        raw_get(raw, "training.optimizer.type", "sgd") or "sgd"
    ).strip().lower().replace("-", "_")
    batch_mode = str(
        raw_get(raw, "training.batch.mode", "direct") or "direct"
    ).strip().lower().replace("-", "_")
    batch_size = int(raw_get(raw, "training.batch.size", 1) or 1)
    accumulation_steps = int(
        raw_get(raw, "training.gradient_accumulation.steps", 1) or 1
    )
    export_policy = str(
        raw_get(raw, "training.gradients.export_policy", "recompute") or "recompute"
    ).strip().lower().replace("-", "_")

    if optimizer != "adam":
        raise ValueError(
            "training.gradients.computation=fused_update currently supports optimizer.type=adam."
        )
    if (
        batch_mode in {"accumulated", "accumulate", "gradient_accumulation"}
        or batch_size != 1
        or accumulation_steps != 1
    ):
        raise ValueError(
            "training.gradients.computation=fused_update currently requires direct single-record updates "
            "(training.batch.mode=direct, training.batch.size=1, gradient_accumulation.steps=1)."
        )
    if export_policy not in {"recompute", "disabled"}:
        raise ValueError(
            "training.gradients.export_policy for fused_update must be recompute or disabled."
        )

    specs = dense_gradient_specs(source)
    declared = set(
        re.findall(r"static\s+grad_wgt_t\s+dW_([A-Za-z0-9_]+)\[\d+\];", source)
    )
    mapped = {spec["tag"] for spec in specs}
    unsupported = sorted(declared - mapped)
    if unsupported:
        raise ValueError(
            "fused_update currently supports Dense parameter gradients only; "
            f"unlowered trainable gradient owners: {unsupported}."
        )
    if not specs:
        raise RuntimeError("fused_update found no Dense parameter-gradient calls to lower")

    updated = source
    for spec in reversed(specs):
        start = int(spec["call_start"])
        end = int(spec["call_end"])
        if start < 0 or end <= start or updated[start:end].strip() != spec["call"].strip():
            raise RuntimeError(
                f"Could not remove full gradient kernel for {spec['dw']}: "
                "the discovered generated-source span no longer matches"
            )
        replacement = (
            "\n  // FPGAI fused_update: full Dense dW kernel removed; "
            "gradient is consumed immediately by Adam."
        )
        updated = updated[:start] + replacement + updated[end:]

    for spec in specs:
        dw = spec["dw"]
        acc_dw = f"ACC_{dw}"
        updated = re.sub(
            rf"(?m)^\s*static\s+grad_wgt_t\s+{re.escape(dw)}\[\d+\];\s*\n?",
            "",
            updated,
        )
        updated = re.sub(
            rf"(?m)^\s*static\s+acc_t\s+{re.escape(acc_dw)}\[\d+\];\s*\n?",
            "",
            updated,
        )
        updated = re.sub(
            rf"(?m)^\s*#pragma\s+HLS\s+BIND_STORAGE\s+variable=(?:{re.escape(dw)}|{re.escape(acc_dw)})[^\n]*\n?",
            "",
            updated,
        )
        updated = re.sub(
            rf"(?m)^\s*for\s*\(int\s+i\s*=\s*0;[^\n]*\)\s*{re.escape(acc_dw)}\[i\]\s*(?:=|\+=)[^;]*;\s*$",
            f"  // FPGAI fused_update: removed obsolete full accumulator loop for {acc_dw}.",
            updated,
        )
        updated = re.sub(
            rf"(?m)^\s*for\s*\(int\s+i\s*=\s*0;[^\n]*\)\s*{re.escape(dw)}\[i\]\s*=\s*[^;]*;\s*$",
            f"  // FPGAI fused_update: removed obsolete materialized-gradient loop for {dw}.",
            updated,
        )
        updated = _replace_adam_weight_update(
            updated,
            spec=spec,
            balanced_brace_end=balanced_brace_end,
            prefer_direct_path=True,
        )

        expression_i = (
            f"(grad_wgt_t)((acc_t){spec['input']}[i % {spec['input_features']}] * "
            f"(acc_t){spec['output_grad']}[i / {spec['input_features']}])"
        )
        expression_idx = (
            f"(grad_wgt_t)((acc_t){spec['input']}[idx % {spec['input_features']}] * "
            f"(acc_t){spec['output_grad']}[idx / {spec['input_features']}])"
        )
        if export_policy == "recompute":
            updated = re.sub(
                rf"\(float\){re.escape(acc_dw)}\[i\]", f"(float){expression_i}", updated
            )
            updated = re.sub(
                rf"\(float\){re.escape(acc_dw)}\[idx\]", f"(float){expression_idx}", updated
            )
            updated = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(dw)}\[i\]", expression_i, updated
            )
            updated = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(dw)}\[idx\]", expression_idx, updated
            )

        if re.search(rf"\b(?:ACC_)?{re.escape(dw)}\s*\[", updated):
            raise RuntimeError(
                f"fused_update left a materialized-gradient reference for {dw}"
            )
        if re.search(r"\([^;=]+\)\s*=", updated):
            raise RuntimeError(
                "fused_update generated an assignment to a temporary expression; "
                "materialized-gradient owner cleanup is incomplete"
            )
        if f"FPGAI_DW_TILE_{spec['tag']}" in updated:
            raise RuntimeError(
                f"fused_update unexpectedly materialized a gradient tile for {dw}"
            )

    banner = (
        "// FPGAI real parameter-gradient fused_update lowering: dense_only=true; optimizer=adam; "
        f"storage={storage}; export_policy={export_policy}; full_dW_arrays=false; gradient_tiles=false; "
        "adam_gradient_arithmetic=float_before_state_cast.\n"
    )
    return banner + updated
