from __future__ import annotations

import re
from typing import Any, Mapping


_DENSE_CALL_RE = re.compile(
    r"dense_out_in\s*<(?P<template>[^>]*)>\s*\((?P<args>[^;]*)\)\s*;",
    re.MULTILINE | re.DOTALL,
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
        return dict(sizes)

    if hasattr(sizes, "to_dict"):
        sizes_dict = sizes.to_dict()
        if isinstance(sizes_dict, Mapping):
            return dict(sizes_dict)

    return tile_dict


def dense_tile_for_layer(
    layer_plan: Mapping[str, Any],
) -> tuple[int, int] | None:
    architecture = _to_dict(layer_plan.get("architecture", {}))

    tile = architecture.get(
        "tiling",
        layer_plan.get("tile", {}),
    )
    tile = _unwrap_tile_mapping(tile)

    if not isinstance(tile, Mapping):
        return None

    tile_in = _tile_value(tile, "in", "input", "input_features", "input_tile", "tile_in")
    tile_out = _tile_value(tile, "out", "output", "output_features", "output_tile", "tile_out")

    if tile_in is None and tile_out is None:
        return None

    return (int(tile_in or 1), int(tile_out or 1))


def emit_dense_tiled_helper_cpp() -> str:
    return r'''

// FPGAI real dense tiling helper.
// Supports Dense weights emitted as either flat W[OUT * IN] or 2-D W[OUT][IN].
template<
    int IN,
    int OUT,
    int TILE_IN,
    int TILE_OUT,
    typename IN_T,
    typename OUT_T,
    typename W_T,
    typename B_T,
    typename ACC_T,
    int PIPELINE_II = 1,
    int IN_UNROLL = 1,
    int OUT_UNROLL = 1,
    int IN_PARTITION = 1,
    int OUT_PARTITION = 1,
    int WEIGHT_PARTITION = 1
>
void dense_out_in_tiled(
    const IN_T input[IN],
    OUT_T output[OUT],
    const W_T weights[OUT * IN],
    const B_T bias[OUT]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=1

    for (int out_base = 0; out_base < OUT; out_base += TILE_OUT) {
        ACC_T acc_tile[TILE_OUT];
#pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1

        dense_init_output_tile_flat:
        for (int out_inner = 0; out_inner < TILE_OUT; ++out_inner) {
#pragma HLS UNROLL factor=OUT_UNROLL
            const int out_idx = out_base + out_inner;
            acc_tile[out_inner] = (out_idx < OUT) ? (ACC_T)bias[out_idx] : (ACC_T)0;
        }

        for (int in_base = 0; in_base < IN; in_base += TILE_IN) {
            IN_T input_tile[TILE_IN];
            W_T weight_tile[TILE_OUT][TILE_IN];
#pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2

            dense_load_input_tile:
            for (int in_inner = 0; in_inner < TILE_IN; ++in_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=IN_UNROLL
                const int in_idx = in_base + in_inner;
                input_tile[in_inner] = (in_idx < IN) ? input[in_idx] : (IN_T)0;
            }

            dense_load_weight_tile_out:
            for (int out_inner = 0; out_inner < TILE_OUT; ++out_inner) {
#pragma HLS UNROLL factor=OUT_UNROLL
                const int out_idx = out_base + out_inner;

                dense_load_weight_tile_in:
                for (int in_inner = 0; in_inner < TILE_IN; ++in_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=IN_UNROLL
                    const int in_idx = in_base + in_inner;
                    weight_tile[out_inner][in_inner] =
                        (out_idx < OUT && in_idx < IN)
                        ? weights[out_idx * IN + in_idx]
                        : (W_T)0;
                }
            }

            dense_compute_tile_out:
            for (int out_inner = 0; out_inner < TILE_OUT; ++out_inner) {
#pragma HLS UNROLL factor=OUT_UNROLL
                const int out_idx = out_base + out_inner;

                dense_compute_tile_in:
                for (int in_inner = 0; in_inner < TILE_IN; ++in_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=IN_UNROLL
                    if (out_idx < OUT) {
                        acc_tile[out_inner] +=
                            (ACC_T)input_tile[in_inner] *
                            (ACC_T)weight_tile[out_inner][in_inner];
                    }
                }
            }
        }

        dense_store_output_tile:
        for (int out_inner = 0; out_inner < TILE_OUT; ++out_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=OUT_UNROLL
            const int out_idx = out_base + out_inner;
            if (out_idx < OUT) {
                output[out_idx] = (OUT_T)acc_tile[out_inner];
            }
        }
    }
}

template<
    int IN,
    int OUT,
    int TILE_IN,
    int TILE_OUT,
    typename IN_T,
    typename OUT_T,
    typename W_T,
    typename B_T,
    typename ACC_T,
    int PIPELINE_II = 1,
    int IN_UNROLL = 1,
    int OUT_UNROLL = 1,
    int IN_PARTITION = 1,
    int OUT_PARTITION = 1,
    int WEIGHT_PARTITION = 1
>
void dense_out_in_tiled(
    const IN_T input[IN],
    OUT_T output[OUT],
    const W_T weights[OUT][IN],
    const B_T bias[OUT]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=IN_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_PARTITION dim=1
#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=WEIGHT_PARTITION dim=2

    for (int out_base = 0; out_base < OUT; out_base += TILE_OUT) {
        ACC_T acc_tile[TILE_OUT];
#pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1

        dense_init_output_tile_2d:
        for (int out_inner = 0; out_inner < TILE_OUT; ++out_inner) {
#pragma HLS UNROLL factor=OUT_UNROLL
            const int out_idx = out_base + out_inner;
            acc_tile[out_inner] = (out_idx < OUT) ? (ACC_T)bias[out_idx] : (ACC_T)0;
        }

        for (int in_base = 0; in_base < IN; in_base += TILE_IN) {
            IN_T input_tile[TILE_IN];
            W_T weight_tile[TILE_OUT][TILE_IN];
#pragma HLS ARRAY_PARTITION variable=input_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_tile complete dim=2

            dense_load_input_tile_2d:
            for (int in_inner = 0; in_inner < TILE_IN; ++in_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=IN_UNROLL
                const int in_idx = in_base + in_inner;
                input_tile[in_inner] = (in_idx < IN) ? input[in_idx] : (IN_T)0;
            }

            dense_load_weight_tile_out_2d:
            for (int out_inner = 0; out_inner < TILE_OUT; ++out_inner) {
#pragma HLS UNROLL factor=OUT_UNROLL
                const int out_idx = out_base + out_inner;

                dense_load_weight_tile_in_2d:
                for (int in_inner = 0; in_inner < TILE_IN; ++in_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=IN_UNROLL
                    const int in_idx = in_base + in_inner;
                    weight_tile[out_inner][in_inner] =
                        (out_idx < OUT && in_idx < IN)
                        ? weights[out_idx][in_idx]
                        : (W_T)0;
                }
            }

            dense_compute_tile_out_2d:
            for (int out_inner = 0; out_inner < TILE_OUT; ++out_inner) {
#pragma HLS UNROLL factor=OUT_UNROLL
                const int out_idx = out_base + out_inner;

                dense_compute_tile_in_2d:
                for (int in_inner = 0; in_inner < TILE_IN; ++in_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=IN_UNROLL
                    if (out_idx < OUT) {
                        acc_tile[out_inner] +=
                            (ACC_T)input_tile[in_inner] *
                            (ACC_T)weight_tile[out_inner][in_inner];
                    }
                }
            }
        }

        dense_store_output_tile_2d:
        for (int out_inner = 0; out_inner < TILE_OUT; ++out_inner) {
#pragma HLS PIPELINE II=PIPELINE_II
#pragma HLS UNROLL factor=OUT_UNROLL
            const int out_idx = out_base + out_inner;
            if (out_idx < OUT) {
                output[out_idx] = (OUT_T)acc_tile[out_inner];
            }
        }
    }
}

'''


def _insert_tiles_into_template(template: str, tile_in: int, tile_out: int) -> str:
    parts = [part.strip() for part in template.split(",")]
    if len(parts) < 2:
        return template

    return ", ".join([parts[0], parts[1], str(int(tile_in)), str(int(tile_out)), *parts[2:]])


def rewrite_dense_calls_with_tiling(
    source: str,
    dense_tiles: list[tuple[int, int] | None],
) -> str:
    if not dense_tiles:
        return source

    call_index = 0
    used_tiling = False

    def replace(match: re.Match[str]) -> str:
        nonlocal call_index
        nonlocal used_tiling

        tile = dense_tiles[call_index] if call_index < len(dense_tiles) else None
        call_index += 1

        if tile is None:
            return match.group(0)

        tile_in, tile_out = tile
        used_tiling = True
        new_template = _insert_tiles_into_template(match.group("template"), tile_in, tile_out)
        return f"dense_out_in_tiled<{new_template}>({match.group('args')});"

    rewritten = _DENSE_CALL_RE.sub(replace, source)

    if used_tiling and "dense_out_in_tiled" in rewritten:
        if "FPGAI real dense tiling helper" not in rewritten:
            rewritten = emit_dense_tiled_helper_cpp() + "\n" + rewritten

    return rewritten


def apply_dense_tiling_to_top_source(source: str, graph: Any, compile_plan: Any) -> str:
    layers = _plan_layers(compile_plan)

    plan_by_node: dict[str, dict[str, Any]] = {}
    for layer in layers:
        node_name = layer.get("node_name") or layer.get("name")
        if node_name:
            plan_by_node[str(node_name)] = layer

    dense_tiles: list[tuple[int, int] | None] = []

    for op in getattr(graph, "ops", []):
        if getattr(op, "op_type", None) != "Dense":
            continue

        layer_plan = plan_by_node.get(str(getattr(op, "name", "")), {})
        dense_tiles.append(dense_tile_for_layer(layer_plan))

    return rewrite_dense_calls_with_tiling(source, dense_tiles)
