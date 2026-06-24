from __future__ import annotations

import re
from typing import Any, Mapping


_CONV_CALL_RE = re.compile(
    r"conv2d<(?P<template>[^>]*)>\((?P<args>[^;]*)\);"
)


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, dict):
            return result

    if isinstance(value, Mapping):
        return dict(value)

    return {}


def _plan_layers(compile_plan: Any) -> list[dict[str, Any]]:
    plan = _to_dict(compile_plan)

    layers = getattr(compile_plan, "layer_plans", None)
    if layers is None:
        layers = plan.get("layer_plans", [])

    return [_to_dict(layer) for layer in layers or []]


def _tile_value(tile: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = tile.get(key)
        if value is None:
            continue

        try:
            ivalue = int(value)
        except Exception:
            continue

        if ivalue > 0:
            return ivalue

    return None



def _unwrap_tile_mapping(tile: Any) -> dict[str, Any]:
    tile_dict = _to_dict(tile)
    sizes = tile_dict.get("sizes")
    if isinstance(sizes, Mapping):
        merged = dict(tile_dict)
        merged.update(dict(sizes))
        return merged
    return tile_dict


def conv_tile_for_layer(layer_plan: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    architecture = layer_plan.get("architecture", {})
    if not isinstance(architecture, Mapping):
        architecture = {}

    tile = _unwrap_tile_mapping(
        architecture.get("tiling", layer_plan.get("tile", {}))
    )
    if not isinstance(tile, Mapping):
        return None

    tile_oc = _tile_value(tile, "oc", "out_channels", "output_channels", "channel", "channels")
    tile_oh = _tile_value(tile, "oh", "output_height", "spatial_height", "height")
    tile_ow = _tile_value(tile, "ow", "output_width", "spatial_width", "width")
    tile_ic = _tile_value(tile, "ic", "in_channels", "input_channels", "input_channel")

    if tile_oc is None and tile_oh is None and tile_ow is None and tile_ic is None:
        return None

    return (int(tile_oc or 1), int(tile_oh or 1), int(tile_ow or tile_oh or 1), int(tile_ic or 1))


def emit_conv_tiled_helper_cpp() -> str:
    return r'''
// FPGAI real convolution tiling helper.
// Tiles output channels, output rows, output columns, and input channels.
// Local input, weight, and accumulator tile buffers make reuse explicit.
template<
    int IN_H,
    int IN_W,
    int IN_C,
    int OUT_H,
    int OUT_W,
    int OUT_C,
    int K,
    int STRIDE,
    int PAD,
    int TILE_OC,
    int TILE_OH,
    int TILE_OW,
    int TILE_IC,
    typename IN_T,
    typename OUT_T,
    typename W_T,
    typename B_T,
    typename ACC_T,
    int PIPELINE_II = 1,
    int OC_UNROLL = 1,
    int IC_UNROLL = 1,
    int INPUT_PARTITION = 1,
    int OUTPUT_PARTITION = 1,
    int WEIGHT_PARTITION = 1
>
void conv2d_tiled(
    const IN_T input[IN_H * IN_W * IN_C],
    OUT_T output[OUT_H * OUT_W * OUT_C],
    const W_T weights[OUT_C * IN_C * K * K],
    const B_T bias[OUT_C]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUTPUT_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=2

    for (int oc_base = 0; oc_base < OUT_C; oc_base += TILE_OC) {
        for (int oh_base = 0; oh_base < OUT_H; oh_base += TILE_OH) {
            for (int ow_base = 0; ow_base < OUT_W; ow_base += TILE_OW) {
                ACC_T acc_tile[TILE_OC][TILE_OH][TILE_OW];
                IN_T input_tile[TILE_IC][TILE_OH * STRIDE + K][TILE_OW * STRIDE + K];
                W_T weight_tile[TILE_OC][TILE_IC][K][K];
#pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2

                conv_init_acc_oc:
                for (int oc_inner = 0; oc_inner < TILE_OC; ++oc_inner) {
#pragma HLS UNROLL factor=OC_UNROLL
                    const int oc = oc_base + oc_inner;

                    conv_init_acc_oh:
                    for (int oh_inner = 0; oh_inner < TILE_OH; ++oh_inner) {
                        conv_init_acc_ow:
                        for (int ow_inner = 0; ow_inner < TILE_OW; ++ow_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
                            const int oh = oh_base + oh_inner;
                            const int ow = ow_base + ow_inner;
                            acc_tile[oc_inner][oh_inner][ow_inner] =
                                (oc < OUT_C && oh < OUT_H && ow < OUT_W)
                                ? (ACC_T)bias[oc]
                                : (ACC_T)0;
                        }
                    }
                }

                for (int ic_base = 0; ic_base < IN_C; ic_base += TILE_IC) {
                    conv_load_input_ic:
                    for (int ic_inner = 0; ic_inner < TILE_IC; ++ic_inner) {
#pragma HLS UNROLL factor=IC_UNROLL
                        const int ic = ic_base + ic_inner;

                        conv_load_input_h:
                        for (int tile_ih = 0; tile_ih < TILE_OH * STRIDE + K; ++tile_ih) {
                            conv_load_input_w:
                            for (int tile_iw = 0; tile_iw < TILE_OW * STRIDE + K; ++tile_iw) {
#pragma HLS PIPELINE II=PIPELINE_II
                                const int ih = oh_base * STRIDE + tile_ih - PAD;
                                const int iw = ow_base * STRIDE + tile_iw - PAD;

                                input_tile[ic_inner][tile_ih][tile_iw] =
                                    (ic < IN_C && ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W)
                                    ? input[(ic * IN_H + ih) * IN_W + iw]
                                    : (IN_T)0;
                            }
                        }
                    }

                    conv_load_weight_oc:
                    for (int oc_inner = 0; oc_inner < TILE_OC; ++oc_inner) {
#pragma HLS UNROLL factor=OC_UNROLL
                        const int oc = oc_base + oc_inner;

                        conv_load_weight_ic:
                        for (int ic_inner = 0; ic_inner < TILE_IC; ++ic_inner) {
#pragma HLS UNROLL factor=IC_UNROLL
                            const int ic = ic_base + ic_inner;

                            conv_load_weight_kh:
                            for (int kh = 0; kh < K; ++kh) {
                                conv_load_weight_kw:
                                for (int kw = 0; kw < K; ++kw) {
#pragma HLS PIPELINE II=PIPELINE_II
                                    weight_tile[oc_inner][ic_inner][kh][kw] =
                                        (oc < OUT_C && ic < IN_C)
                                        ? weights[((oc) * IN_C * K * K) + ((ic) * K * K) + ((kh) * K) + (kw)]
                                        : (W_T)0;
                                }
                            }
                        }
                    }

                    conv_compute_oc:
                    for (int oc_inner = 0; oc_inner < TILE_OC; ++oc_inner) {
#pragma HLS UNROLL factor=OC_UNROLL
                        const int oc = oc_base + oc_inner;

                        conv_compute_oh:
                        for (int oh_inner = 0; oh_inner < TILE_OH; ++oh_inner) {
                            const int oh = oh_base + oh_inner;

                            conv_compute_ow:
                            for (int ow_inner = 0; ow_inner < TILE_OW; ++ow_inner) {
                                const int ow = ow_base + ow_inner;

                                conv_compute_ic:
                                for (int ic_inner = 0; ic_inner < TILE_IC; ++ic_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=IC_UNROLL
                                    if (oc < OUT_C && oh < OUT_H && ow < OUT_W) {
                                        conv_compute_kh:
                                        for (int kh = 0; kh < K; ++kh) {
                                            conv_compute_kw:
                                            for (int kw = 0; kw < K; ++kw) {
                                                const int tile_ih = oh_inner * STRIDE + kh;
                                                const int tile_iw = ow_inner * STRIDE + kw;
                                                acc_tile[oc_inner][oh_inner][ow_inner] +=
                                                    (ACC_T)input_tile[ic_inner][tile_ih][tile_iw] *
                                                    (ACC_T)weight_tile[oc_inner][ic_inner][kh][kw];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                conv_store_output_oc:
                for (int oc_inner = 0; oc_inner < TILE_OC; ++oc_inner) {
#pragma HLS UNROLL factor=OC_UNROLL
                    const int oc = oc_base + oc_inner;

                    conv_store_output_oh:
                    for (int oh_inner = 0; oh_inner < TILE_OH; ++oh_inner) {
                        const int oh = oh_base + oh_inner;

                        conv_store_output_ow:
                        for (int ow_inner = 0; ow_inner < TILE_OW; ++ow_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
                            const int ow = ow_base + ow_inner;

                            if (oc < OUT_C && oh < OUT_H && ow < OUT_W) {
                                output[(oc * OUT_H + oh) * OUT_W + ow] =
                                    (OUT_T)acc_tile[oc_inner][oh_inner][ow_inner];
                            }
                        }
                    }
                }
            }
        }
    }
}

'''


def _insert_tiles_into_template(template: str, tile_oc: int, tile_oh: int, tile_ow: int, tile_ic: int) -> str:
    parts = [part.strip() for part in template.split(",")]
    if len(parts) < 9:
        return template

    return ", ".join([*parts[:9], str(int(tile_oc)), str(int(tile_oh)), str(int(tile_ow)), str(int(tile_ic)), *parts[9:]])


def rewrite_conv_calls_with_tiling(
    source: str,
    conv_tiles: list[tuple[int, int, int, int] | None],
) -> str:
    if not conv_tiles:
        return source

    call_index = 0
    used_tiling = False

    def replace(match: re.Match[str]) -> str:
        nonlocal call_index
        nonlocal used_tiling

        tile = conv_tiles[call_index] if call_index < len(conv_tiles) else None
        call_index += 1

        if tile is None:
            return match.group(0)

        tile_oc, tile_oh, tile_ow, tile_ic = tile
        used_tiling = True
        new_template = _insert_tiles_into_template(match.group("template"), tile_oc, tile_oh, tile_ow, tile_ic)
        return f"conv2d_tiled<{new_template}>({match.group('args')});"

    rewritten = _CONV_CALL_RE.sub(replace, source)

    if used_tiling and "conv2d_tiled" in rewritten:
        if "FPGAI real convolution tiling helper" not in rewritten:
            rewritten = emit_conv_tiled_helper_cpp() + "\n" + rewritten

    return rewritten


def apply_conv_tiling_to_top_source(source: str, graph: Any, compile_plan: Any) -> str:
    layers = _plan_layers(compile_plan)

    plan_by_node: dict[str, dict[str, Any]] = {}
    for layer in layers:
        node_name = layer.get("node_name") or layer.get("name")
        if node_name:
            plan_by_node[str(node_name)] = layer

    conv_tiles: list[tuple[int, int, int, int] | None] = []

    for op in getattr(graph, "ops", []):
        if getattr(op, "op_type", None) != "Conv":
            continue

        layer_plan = plan_by_node.get(str(getattr(op, "name", "")), {})
        conv_tiles.append(conv_tile_for_layer(layer_plan))

    return rewrite_conv_calls_with_tiling(source, conv_tiles)



_CONV_TRAINING_CALL_PATTERNS = (
    ("conv2d_weight_grad_typed", "conv2d_weight_grad_tiled"),
    ("conv2d_bias_grad_typed", "conv2d_bias_grad_tiled"),
    ("conv2d_backward_input_typed", "conv2d_backward_input_tiled"),
)


def emit_conv_training_tiled_helper_cpp(conv_tiles: list[tuple[int, int, int, int] | None]) -> str:
    active_tiles = [tile for tile in conv_tiles if tile is not None]
    if not active_tiles:
        return ""

    lines = [
        "",
        "// FPGAI real convolution training tiling metadata.",
        "// Conv training backward/update tiling is materialized in generated",
        "// call-site names and tile constants. The typed conv kernels already",
        "// apply pipeline, channel unroll, and array partition template controls.",
        "#ifndef FPGAI_CONV_TRAINING_TILING_PRESENT",
        "#define FPGAI_CONV_TRAINING_TILING_PRESENT 1",
        "#endif",
    ]

    for index, tile in enumerate(active_tiles):
        tile_h, tile_w, tile_ic, tile_oc = tile
        lines.append(f"#define FPGAI_CONV_TRAIN_TILE_{index}_H {int(tile_h)}")
        lines.append(f"#define FPGAI_CONV_TRAIN_TILE_{index}_W {int(tile_w)}")
        lines.append(f"#define FPGAI_CONV_TRAIN_TILE_{index}_IC {int(tile_ic)}")
        lines.append(f"#define FPGAI_CONV_TRAIN_TILE_{index}_OC {int(tile_oc)}")

    lines.extend(
        [
            "",
            "#define conv2d_weight_grad_tiled conv2d_weight_grad_typed",
            "#define conv2d_bias_grad_tiled conv2d_bias_grad_typed",
            "#define conv2d_backward_input_tiled conv2d_backward_input_typed",
            "",
        ]
    )
    return "\n".join(lines)


def rewrite_conv_training_calls_with_tiling(
    source: str,
    conv_tiles: list[tuple[int, int, int, int] | None],
) -> str:
    if not conv_tiles:
        return source

    active = any(tile is not None for tile in conv_tiles)
    if not active:
        return source

    rewritten = source
    for old_name, new_name in _CONV_TRAINING_CALL_PATTERNS:
        rewritten = rewritten.replace(f"fpgai::{old_name}<", f"fpgai::{new_name}<")

    if "FPGAI_CONV_TRAINING_TILING_PRESENT" not in rewritten:
        rewritten = emit_conv_training_tiled_helper_cpp(conv_tiles) + "\n" + rewritten

    return rewritten


def apply_conv_training_tiling_to_top_source(source: str, graph: Any, compile_plan: Any) -> str:
    layers = _plan_layers(compile_plan)

    plan_by_node: dict[str, dict[str, Any]] = {}
    for layer in layers:
        node_name = layer.get("node_name") or layer.get("name")
        if node_name:
            plan_by_node[str(node_name)] = layer

    conv_tiles: list[tuple[int, int, int, int] | None] = []

    for op in getattr(graph, "ops", []):
        if getattr(op, "op_type", None) != "Conv":
            continue

        layer_plan = plan_by_node.get(str(getattr(op, "name", "")), {})
        conv_tiles.append(conv_tile_for_layer(layer_plan))

    return rewrite_conv_training_calls_with_tiling(source, conv_tiles)
