from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from fpgai.config.access import get_path
from fpgai.engine.models import (
    ArchitecturePlan,
    BufferingPlan,
    CompilePlan,
    LayerDescriptor,
    LayerMemoryPlan,
    LayerPlan,
    ParallelismPlan,
    PartitionPlan,
    PipelinePlan,
    PrecisionPlan,
    TilingPlan,
)


_cfg_get = get_path


@dataclass
class Policy:
    name: str
    target_clock_mhz: float

    pe: int
    simd: int
    unroll_factor: int
    partition_factor: int
    pipeline_style: str

    conv_oh: int
    conv_ow: int
    conv_oc: int
    dense_in: int
    dense_out: int

    weight_region_preference: List[str]
    activation_region_preference: List[str]

    allow_double_buffer: bool

    axi_word_bits: int
    burst_len: int
    enable_bitpack: bool
    enable_compression: bool

    array_partition_mode: str
    mac_style: str
    accum_strategy: str
    activation_impl: str
    round_mode: str
    sat_mode: str


POLICIES: Dict[str, Policy] = {
    "Fit-First": Policy(
        name="Fit-First",
        target_clock_mhz=180.0,
        pe=1, simd=1, unroll_factor=1, partition_factor=1, pipeline_style="conservative",
        conv_oh=4, conv_ow=4, conv_oc=2, dense_in=16, dense_out=4,
        weight_region_preference=["BRAM", "URAM", "DDR"],
        activation_region_preference=["BRAM", "DDR", "URAM"],
        allow_double_buffer=False,
        axi_word_bits=64, burst_len=16, enable_bitpack=False, enable_compression=False,
        array_partition_mode="block", mac_style="serial", accum_strategy="serial",
        activation_impl="small_lut", round_mode="trn", sat_mode="wrap",
    ),
    "DSP-Saver": Policy(
        name="DSP-Saver",
        target_clock_mhz=190.0,
        pe=1, simd=1, unroll_factor=1, partition_factor=1, pipeline_style="balanced",
        conv_oh=4, conv_ow=4, conv_oc=4, dense_in=32, dense_out=8,
        weight_region_preference=["BRAM", "URAM", "DDR"],
        activation_region_preference=["BRAM", "DDR", "URAM"],
        allow_double_buffer=False,
        axi_word_bits=64, burst_len=16, enable_bitpack=True, enable_compression=False,
        array_partition_mode="block", mac_style="serial", accum_strategy="serial",
        activation_impl="small_lut", round_mode="trn", sat_mode="wrap",
    ),
    "BRAM-Saver": Policy(
        name="BRAM-Saver",
        target_clock_mhz=190.0,
        pe=1, simd=2, unroll_factor=1, partition_factor=1, pipeline_style="balanced",
        conv_oh=4, conv_ow=4, conv_oc=4, dense_in=32, dense_out=8,
        weight_region_preference=["URAM", "BRAM", "DDR"],
        activation_region_preference=["DDR", "BRAM", "URAM"],
        allow_double_buffer=False,
        axi_word_bits=128, burst_len=32, enable_bitpack=True, enable_compression=True,
        array_partition_mode="block", mac_style="balanced", accum_strategy="tree",
        activation_impl="small_lut", round_mode="trn", sat_mode="wrap",
    ),
    "Memory-First": Policy(
        name="Memory-First",
        target_clock_mhz=190.0,
        pe=1, simd=2, unroll_factor=1, partition_factor=1, pipeline_style="balanced",
        conv_oh=4, conv_ow=4, conv_oc=4, dense_in=32, dense_out=8,
        weight_region_preference=["BRAM", "DDR", "URAM"],
        activation_region_preference=["BRAM", "DDR", "URAM"],
        allow_double_buffer=False,
        axi_word_bits=128, burst_len=32, enable_bitpack=True, enable_compression=True,
        array_partition_mode="block", mac_style="balanced", accum_strategy="tree",
        activation_impl="small_lut", round_mode="trn", sat_mode="wrap",
    ),

    "Balanced": Policy(
        name="Balanced",
        target_clock_mhz=200.0,
        pe=2, simd=2, unroll_factor=2, partition_factor=2, pipeline_style="balanced",
        conv_oh=8, conv_ow=8, conv_oc=8, dense_in=64, dense_out=16,
        weight_region_preference=["BRAM", "URAM", "DDR"],
        activation_region_preference=["BRAM", "URAM", "DDR"],
        allow_double_buffer=True,
        axi_word_bits=128, burst_len=32, enable_bitpack=True, enable_compression=True,
        array_partition_mode="cyclic", mac_style="balanced", accum_strategy="tree",
        activation_impl="medium_lut", round_mode="trn", sat_mode="wrap",
    ),
    "Throughput-First": Policy(
        name="Throughput-First",
        target_clock_mhz=210.0,
        pe=2, simd=4, unroll_factor=2, partition_factor=4, pipeline_style="aggressive",
        conv_oh=8, conv_ow=8, conv_oc=8, dense_in=64, dense_out=16,
        weight_region_preference=["BRAM", "URAM", "DDR"],
        activation_region_preference=["BRAM", "URAM", "DDR"],
        allow_double_buffer=True,
        axi_word_bits=128, burst_len=64, enable_bitpack=True, enable_compression=True,
        array_partition_mode="cyclic", mac_style="tree", accum_strategy="tree",
        activation_impl="medium_lut", round_mode="trn", sat_mode="wrap",
    ),
    "Latency-First": Policy(
        name="Latency-First",
        target_clock_mhz=220.0,
        pe=4, simd=4, unroll_factor=4, partition_factor=4, pipeline_style="aggressive",
        conv_oh=16, conv_ow=16, conv_oc=16, dense_in=128, dense_out=32,
        weight_region_preference=["BRAM", "URAM", "DDR"],
        activation_region_preference=["BRAM", "URAM", "DDR"],
        allow_double_buffer=True,
        axi_word_bits=128, burst_len=64, enable_bitpack=True, enable_compression=True,
        array_partition_mode="cyclic", mac_style="tree", accum_strategy="tree",
        activation_impl="large_lut", round_mode="trn", sat_mode="wrap",
    ),
}


def _numerics_defaults(cfg) -> Dict[str, Any]:
    return _cfg_get(cfg.raw, "numerics.defaults", {}) or {}


def _normalize_ap_spec(spec: Dict[str, Any] | None, default_tb: int, default_ib: int) -> Dict[str, Any]:
    spec = spec or {}
    return {
        "type": str(spec.get("type", "ap_fixed")),
        "total_bits": int(spec.get("total_bits", default_tb)),
        "int_bits": int(spec.get("int_bits", default_ib)),
    }


def _precision_info_from_specs(
    act: Dict[str, Any],
    weight: Dict[str, Any],
    bias: Dict[str, Any],
    accum: Dict[str, Any],
) -> Dict[str, Any]:
    act_type = str(act.get("type", "float")).lower()
    weight_type = str(weight.get("type", "float")).lower()

    precision_mode = (
        "fixed"
        if act_type.startswith("ap_fixed")
        or weight_type.startswith("ap_fixed")
        or act_type == "ap_fixed"
        or weight_type == "ap_fixed"
        else "float"
    )

    return {
        "precision_mode": precision_mode,
        "act_bits": int(act["total_bits"]) if "total_bits" in act else None,
        "weight_bits": int(weight["total_bits"]) if "total_bits" in weight else None,
        "bias_bits": int(bias["total_bits"]) if "total_bits" in bias else None,
        "accum_bits": int(accum["total_bits"]) if "total_bits" in accum else None,
        "act_int_bits": int(act["int_bits"]) if "int_bits" in act else None,
        "weight_int_bits": int(weight["int_bits"]) if "int_bits" in weight else None,
        "bias_int_bits": int(bias["int_bits"]) if "int_bits" in bias else None,
        "accum_int_bits": int(accum["int_bits"]) if "int_bits" in accum else None,
        "act_type": act_type,
        "weight_type": weight_type,
    }


def _default_precision_info(cfg) -> Dict[str, Any]:
    numerics = _numerics_defaults(cfg)
    act = _normalize_ap_spec(numerics.get("activation", {}), 16, 6)
    weight = _normalize_ap_spec(numerics.get("weight", {}), 16, 6)
    bias = _normalize_ap_spec(numerics.get("bias", {}), 24, 10)
    accum = _normalize_ap_spec(numerics.get("accum", {}), 24, 10)
    return _precision_info_from_specs(act, weight, bias, accum)


def _descriptor_precision_info(desc: LayerDescriptor, default_precision: Dict[str, Any]) -> Dict[str, Any]:
    raw = (desc.attrs or {}).get("precision")
    if not isinstance(raw, dict):
        return dict(default_precision)

    act = _normalize_ap_spec(raw.get("activation", {}), default_precision.get("act_bits") or 16, default_precision.get("act_int_bits") or 6)
    weight = _normalize_ap_spec(raw.get("weight", {}), default_precision.get("weight_bits") or 16, default_precision.get("weight_int_bits") or 6)
    bias = _normalize_ap_spec(raw.get("bias", {}), default_precision.get("bias_bits") or 24, default_precision.get("bias_int_bits") or 10)
    accum = _normalize_ap_spec(raw.get("accum", {}), default_precision.get("accum_bits") or 24, default_precision.get("accum_int_bits") or 10)
    return _precision_info_from_specs(act, weight, bias, accum)


def _policy_name(cfg) -> str:
    raw = cfg.raw
    return str(
        _cfg_get(
            raw,
            "optimization.parallel_policy",
            _cfg_get(raw, "analysis.design_space.policy_name", "Balanced"),
        )
    )


def _pick_policy(cfg) -> Policy:
    name = _policy_name(cfg)
    try:
        return POLICIES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown parallel policy {name!r}; "
            f"expected one of {sorted(POLICIES)}"
        ) from exc


def _list_cfg(raw: Dict[str, Any], path: str, default: List[str]) -> List[str]:
    value = _cfg_get(raw, path, None)
    if isinstance(value, list) and value:
        return [str(x).upper() for x in value]
    if isinstance(value, str) and value.strip():
        return [x.strip().upper() for x in value.split(",") if x.strip()]
    return list(default)


def _bool_cfg(raw: Dict[str, Any], path: str, default: bool) -> bool:
    value = _cfg_get(raw, path, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "double"}
    return bool(value)


def _override_policy_from_cfg(cfg, base: Policy) -> Policy:
    raw = cfg.raw
    return Policy(
        name=base.name,
        target_clock_mhz=float(_cfg_get(raw, "targets.platform.clocks.0.target_mhz", base.target_clock_mhz)),
        pe=max(1, int(_cfg_get(raw, "optimization.parallel.pe", base.pe))),
        simd=max(1, int(_cfg_get(raw, "optimization.parallel.simd", base.simd))),
        unroll_factor=max(1, int(_cfg_get(raw, "optimization.parallel.unroll_factor", base.unroll_factor))),
        partition_factor=max(1, int(_cfg_get(raw, "optimization.parallel.partition_factor", base.partition_factor))),
        pipeline_style=str(_cfg_get(raw, "optimization.parallel.pipeline_style", base.pipeline_style)),
        conv_oh=base.conv_oh,
        conv_ow=base.conv_ow,
        conv_oc=base.conv_oc,
        dense_in=base.dense_in,
        dense_out=base.dense_out,
        weight_region_preference=_list_cfg(raw, "memory.weight_region_preference", base.weight_region_preference),
        activation_region_preference=_list_cfg(raw, "memory.activation_region_preference", base.activation_region_preference),
        allow_double_buffer=_bool_cfg(raw, "memory.allow_double_buffer", base.allow_double_buffer),
        axi_word_bits=base.axi_word_bits,
        burst_len=base.burst_len,
        enable_bitpack=base.enable_bitpack,
        enable_compression=base.enable_compression,
        array_partition_mode=str(_cfg_get(raw, "optimization.parallel.array_partition_mode", base.array_partition_mode)),
        mac_style=base.mac_style,
        accum_strategy=base.accum_strategy,
        activation_impl=base.activation_impl,
        round_mode=base.round_mode,
        sat_mode=base.sat_mode,
    )


def _choose_weight_mode(desc: LayerDescriptor, raw_cfg: Dict[str, Any]) -> str:
    del desc
    requested = str(
        _cfg_get(
            raw_cfg,
            "data_movement.ps_pl.weights.mode",
            _cfg_get(raw_cfg, "memory.weight_storage", "embedded"),
        )
    ).lower().replace("-", "_")
    if requested in ("dma_ddr", "external", "external_ddr"):
        requested = "ddr"
    if requested in ("stream", "streaming"):
        requested = "stream"
    if requested in ("embedded", "on_chip", "onchip", "bram", "uram"):
        requested = "embedded"

    if requested in ("stream", "ddr"):
        return requested

    return "embedded"


def _buffering_for(weights_mode: str, policy: Policy) -> str:
    if not policy.allow_double_buffer:
        return "single"
    return "double" if weights_mode in ("stream", "ddr") else "single"



def _pipeline_style_from_cfg(raw: Dict[str, Any], policy: Policy) -> str:
    value = _cfg_get(
        raw,
        "optimization.pipeline.style",
        _cfg_get(raw, "optimization.parallel.pipeline_style", policy.pipeline_style),
    )
    style = str(value or policy.pipeline_style).strip().lower()
    if style in {"aggressive", "balanced", "conservative"}:
        return style
    return str(policy.pipeline_style or "balanced").strip().lower()


def _pipeline_ii_override_from_cfg(raw: Dict[str, Any]) -> int | None:
    value = _cfg_get(raw, "optimization.pipeline.ii", None)
    if value is None:
        value = _cfg_get(raw, "optimization.pipeline_ii", None)
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def _pipeline_ii_for(policy: Policy, compute_hint: str = "", raw: Dict[str, Any] | None = None) -> int:
    """Lower pipeline_style into distinct HLS initiation intervals.

    Pipeline policy goal: conservative, balanced, and aggressive must produce
    different generated HLS directives even when PE/SIMD/unroll/partition
    are held constant by the sweep.

    These values are intentionally simple and artifact-driven:
    - conservative prioritizes timing/resource safety, so it relaxes II.
    - balanced uses moderate pipelining.
    - aggressive targets II=1.
    """
    if raw is not None:
        override = _pipeline_ii_override_from_cfg(raw)
        if override is not None:
            return override
        style = _pipeline_style_from_cfg(raw, policy)
    else:
        style = str(getattr(policy, "pipeline_style", "balanced") or "balanced").lower()
    hint = str(compute_hint or "").lower()

    if style == "aggressive":
        return 1
    if style == "balanced":
        return 2
    if style == "conservative":
        return 4 if hint == "memory_bound" else 3
    return 2


def _positive_tile_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def _as_mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _merge_tile_values(
    target: Dict[str, int],
    source: Any,
    aliases: Dict[str, tuple[str, ...]],
) -> Dict[str, int]:
    section = _as_mapping(source)
    if not section:
        return target

    sizes = section.get("sizes")
    if isinstance(sizes, dict):
        merged = dict(section)
        merged.update(sizes)
        section = merged

    out = dict(target)

    for canonical, keys in aliases.items():
        for key in keys:
            value = _positive_tile_int(section.get(key))
            if value is not None:
                out[canonical] = value
                break

    return out


def _layer_specific_tile_section(raw: Dict[str, Any], node_name: str) -> Dict[str, Any]:
    layers = _cfg_get(raw, "optimization.tiling.layers", {})

    if isinstance(layers, dict):
        section = layers.get(node_name)
        return section if isinstance(section, dict) else {}

    if isinstance(layers, list):
        for item in layers:
            if not isinstance(item, dict):
                continue
            match = item.get("match", {})
            if isinstance(match, dict) and match.get("name") == node_name:
                return item
            if item.get("name") == node_name:
                return item

    return {}


def _dense_tile_from_cfg(
    raw: Dict[str, Any],
    node_name: str,
    default_tile: Dict[str, int],
    *,
    in_features: int,
    out_features: int,
) -> Dict[str, int]:
    aliases = {
        "in": (
            "in",
            "input",
            "input_features",
            "input_tile",
            "tile_in",
            "dense_tile_in",
        ),
        "out": (
            "out",
            "output",
            "output_features",
            "output_tile",
            "tile_out",
            "dense_tile_out",
        ),
    }

    tile = dict(default_tile)
    tile = _merge_tile_values(
        tile,
        _cfg_get(raw, "optimization.tiling.dense", {}),
        aliases,
    )
    tile = _merge_tile_values(
        tile,
        _layer_specific_tile_section(raw, node_name),
        aliases,
    )

    tile["in"] = max(1, min(int(in_features or 1), int(tile.get("in", 1))))
    tile["out"] = max(1, min(int(out_features or 1), int(tile.get("out", 1))))
    return tile


def _conv_tile_from_cfg(
    raw: Dict[str, Any],
    node_name: str,
    default_tile: Dict[str, int],
) -> Dict[str, int]:
    aliases = {
        "ic": (
            "ic",
            "in_channels",
            "input_channels",
            "input_channel",
            "tile_ic",
            "conv_tile_ic",
        ),
        "oc": (
            "oc",
            "out_channels",
            "output_channels",
            "channel",
            "channels",
            "tile_oc",
            "conv_tile_oc",
        ),
        "oh": (
            "oh",
            "output_height",
            "spatial_height",
            "height",
            "tile_oh",
            "conv_tile_oh",
        ),
        "ow": (
            "ow",
            "output_width",
            "spatial_width",
            "width",
            "tile_ow",
            "conv_tile_ow",
        ),
    }

    tile = dict(default_tile)
    tile = _merge_tile_values(
        tile,
        _cfg_get(raw, "optimization.tiling.conv", {}),
        aliases,
    )
    tile = _merge_tile_values(
        tile,
        _layer_specific_tile_section(raw, node_name),
        aliases,
    )

    return {
        key: max(1, int(value))
        for key, value in tile.items()
    }


def _layer_notes(desc: LayerDescriptor, precision: Dict[str, Any], policy: Policy) -> Dict[str, Any]:
    return {
        "policy_name": policy.name,
        "compute_hint": desc.compute_hint,
        "precision_tag": (desc.attrs or {}).get("precision_tag"),
        "partition_factor": policy.partition_factor,
        "partition_mode": policy.array_partition_mode,
        "pipeline_style": policy.pipeline_style,
        "mac_style": policy.mac_style,
        "accum_strategy": policy.accum_strategy,
        "activation_impl": policy.activation_impl,
        "round_mode": policy.round_mode,
        "sat_mode": policy.sat_mode,
        "requested_act_bits": precision["act_bits"],
        "requested_weight_bits": precision["weight_bits"],
        "requested_bias_bits": precision["bias_bits"],
        "requested_accum_bits": precision["accum_bits"],
        "requested_act_int_bits": precision["act_int_bits"],
        "requested_weight_int_bits": precision["weight_int_bits"],
        "requested_bias_int_bits": precision["bias_int_bits"],
        "requested_accum_int_bits": precision["accum_int_bits"],
    }


def _architecture_plan(
    *,
    precision: Dict[str, Any],
    policy: Policy,
    tile: Dict[str, int],
    unroll: Dict[str, int],
    pipeline_ii: int,
    pipeline_style: str,
    weight_mode: str,
    activation_mode: str,
    buffering: str,
) -> ArchitecturePlan:
    pe = unroll.get(
        "out",
        unroll.get(
            "oc",
            unroll.get("element", 1),
        ),
    )
    simd = unroll.get("in", unroll.get("ic", 1))

    partition_targets = {
        "input": max(policy.partition_factor, simd),
        "output": max(policy.partition_factor, pe),
        "weight": max(
            policy.partition_factor,
            pe * simd,
        ),
        "gradient": max(policy.partition_factor, pe),
    }

    return ArchitecturePlan(
        precision=PrecisionPlan(
            mode=precision["precision_mode"],
            activation_bits=precision["act_bits"],
            weight_bits=precision["weight_bits"],
            bias_bits=precision["bias_bits"],
            accumulator_bits=precision["accum_bits"],
            activation_int_bits=precision["act_int_bits"],
            weight_int_bits=precision["weight_int_bits"],
            bias_int_bits=precision["bias_int_bits"],
            accumulator_int_bits=precision["accum_int_bits"],
        ),
        pipeline=PipelinePlan(
            ii=pipeline_ii,
            style=pipeline_style,
        ),
        parallelism=ParallelismPlan(
            pe=pe,
            simd=simd,
            unroll=unroll,
        ),
        partitioning=PartitionPlan(
            factor=policy.partition_factor,
            mode=policy.array_partition_mode,
            targets=partition_targets,
        ),
        tiling=TilingPlan(sizes=tile),
        buffering=BufferingPlan(mode=buffering),
        memory=LayerMemoryPlan(
            weight_mode=weight_mode,
            activation_mode=activation_mode,
            weight_region=policy.weight_region_preference[0],
            activation_region=policy.activation_region_preference[0],
        ),
    )


def _plan_conv(desc: LayerDescriptor, precision: Dict[str, Any], weights_mode: str, policy: Policy, raw: Dict[str, Any]) -> LayerPlan:
    tile = _conv_tile_from_cfg(
        raw,
        desc.node_name,
        {
            "oh": policy.conv_oh,
            "ow": policy.conv_ow,
            "oc": policy.conv_oc,
            "ic": policy.simd,
        },
    )
    unroll = {"ic": policy.simd, "oc": policy.pe}
    pipeline_style = _pipeline_style_from_cfg(raw, policy)
    pipeline_ii = _pipeline_ii_for(policy, desc.compute_hint, raw)
    activation_mode = (
        "stream"
        if policy.name in ("Latency-First", "Throughput-First")
        else "buffer"
    )
    buffering = _buffering_for(weights_mode, policy)
    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile=tile,
        unroll=unroll,
        pipeline_ii=pipeline_ii,
        weight_mode=weights_mode,
        activation_mode=activation_mode,
        buffering=buffering,
        backend_kernel=desc.backend_kernel or "conv",
        notes=_layer_notes(desc, precision, policy),
        architecture=_architecture_plan(
            precision=precision,
            policy=policy,
            tile=tile,
            unroll=unroll,
            pipeline_ii=pipeline_ii,
            pipeline_style=pipeline_style,
            weight_mode=weights_mode,
            activation_mode=activation_mode,
            buffering=buffering,
        ),
    )


def _plan_dense(desc: LayerDescriptor, precision: Dict[str, Any], weights_mode: str, policy: Policy, raw: Dict[str, Any]) -> LayerPlan:
    out_features = int(desc.attrs.get("out_features", 1) or 1)
    in_features = int(desc.attrs.get("in_features", 1) or 1)

    tile = _dense_tile_from_cfg(
        raw,
        desc.node_name,
        {
            "in": min(in_features, policy.dense_in),
            "out": min(out_features, policy.dense_out),
        },
        in_features=in_features,
        out_features=out_features,
    )
    unroll = {"in": policy.simd, "out": policy.pe}
    pipeline_style = _pipeline_style_from_cfg(raw, policy)
    pipeline_ii = _pipeline_ii_for(policy, desc.compute_hint, raw)
    activation_mode = (
        "stream"
        if policy.name in ("Latency-First", "Throughput-First")
        else "buffer"
    )
    buffering = _buffering_for(weights_mode, policy)

    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile=tile,
        unroll=unroll,
        pipeline_ii=pipeline_ii,
        weight_mode=weights_mode,
        activation_mode=activation_mode,
        buffering=buffering,
        backend_kernel=desc.backend_kernel or "dense",
        notes=_layer_notes(desc, precision, policy),
        architecture=_architecture_plan(
            precision=precision,
            policy=policy,
            tile=tile,
            unroll=unroll,
            pipeline_ii=pipeline_ii,
            pipeline_style=pipeline_style,
            weight_mode=weights_mode,
            activation_mode=activation_mode,
            buffering=buffering,
        ),
    )


def _plan_pool(desc: LayerDescriptor, precision: Dict[str, Any], policy: Policy) -> LayerPlan:
    spatial = max(4, min(policy.conv_oh, policy.conv_ow))
    tile = {"oh": spatial, "ow": spatial}
    pipeline_ii = _pipeline_ii_for(policy, desc.compute_hint)
    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile=tile,
        unroll={},
        pipeline_ii=pipeline_ii,
        weight_mode="embedded",
        activation_mode="stream",
        buffering="single",
        backend_kernel=desc.backend_kernel or desc.op_type.lower(),
        notes=_layer_notes(desc, precision, policy),
        architecture=_architecture_plan(
            precision=precision,
            policy=policy,
            tile=tile,
            unroll={},
            pipeline_ii=pipeline_ii,
            pipeline_style=policy.pipeline_style,
            weight_mode="embedded",
            activation_mode="stream",
            buffering="single",
        ),
    )


def _plan_elementwise(desc: LayerDescriptor, precision: Dict[str, Any], policy: Policy) -> LayerPlan:
    unroll = {"element": policy.unroll_factor}
    pipeline_ii = _pipeline_ii_for(policy, desc.compute_hint)
    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile={},
        unroll=unroll,
        pipeline_ii=pipeline_ii,
        weight_mode="embedded",
        activation_mode="stream",
        buffering="single",
        backend_kernel=desc.backend_kernel or desc.op_type.lower(),
        notes=_layer_notes(desc, precision, policy),
        architecture=_architecture_plan(
            precision=precision,
            policy=policy,
            tile={},
            unroll=unroll,
            pipeline_ii=pipeline_ii,
            pipeline_style=policy.pipeline_style,
            weight_mode="embedded",
            activation_mode="stream",
            buffering="single",
        ),
    )


def _plan_generic(desc: LayerDescriptor, precision: Dict[str, Any], weights_mode: str, policy: Policy) -> LayerPlan:
    pipeline_ii = _pipeline_ii_for(policy, desc.compute_hint)
    buffering = _buffering_for(weights_mode, policy)
    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile={},
        unroll={},
        pipeline_ii=pipeline_ii,
        weight_mode=weights_mode,
        activation_mode="buffer",
        buffering=buffering,
        backend_kernel=desc.backend_kernel or desc.op_type.lower(),
        notes=_layer_notes(desc, precision, policy),
        architecture=_architecture_plan(
            precision=precision,
            policy=policy,
            tile={},
            unroll={},
            pipeline_ii=pipeline_ii,
            pipeline_style=policy.pipeline_style,
            weight_mode=weights_mode,
            activation_mode="buffer",
            buffering=buffering,
        ),
    )


def make_compile_plan(cfg, descriptors: List[LayerDescriptor]) -> CompilePlan:
    raw = cfg.raw
    default_precision = _default_precision_info(cfg)
    policy = _override_policy_from_cfg(cfg, _pick_policy(cfg))

    target_board = str(_cfg_get(raw, "targets.platform.board", "unknown"))
    target_part = str(_cfg_get(raw, "targets.platform.part", "unknown"))

    layer_plans: List[LayerPlan] = []
    execution_order: List[str] = []

    for desc in descriptors:
        execution_order.append(desc.node_name)
        weights_mode = _choose_weight_mode(desc, raw)
        precision = _descriptor_precision_info(desc, default_precision)

        if desc.op_type == "Conv":
            plan = _plan_conv(desc, precision, weights_mode, policy, raw)
        elif desc.op_type == "Dense":
            plan = _plan_dense(desc, precision, weights_mode, policy, raw)
        elif desc.op_type in ("MaxPool", "AvgPool"):
            plan = _plan_pool(desc, precision, policy)
        elif desc.op_type in ("Relu", "LeakyRelu", "Sigmoid", "Softmax", "Add", "Reshape", "Flatten"):
            plan = _plan_elementwise(desc, precision, policy)
        else:
            plan = _plan_generic(desc, precision, weights_mode, policy)

        layer_plans.append(plan)

    total_param_bytes = sum(d.param_bytes for d in descriptors)
    total_activation_in = sum(d.activation_bytes_in for d in descriptors)
    total_activation_out = sum(d.activation_bytes_out for d in descriptors)

    return CompilePlan(
        target_board=target_board,
        target_part=target_part,
        clock_mhz=policy.target_clock_mhz,
        execution_order=execution_order,
        layer_plans=layer_plans,
        global_resource_budget={
            "total_param_bytes": total_param_bytes,
            "total_activation_bytes_in": total_activation_in,
            "total_activation_bytes_out": total_activation_out,
        },
        notes={
            "planner": "policy_driven_v6_precision_aware",
            "global_weights_mode_requested": str(
                _cfg_get(
                    raw,
                    "data_movement.ps_pl.weights.mode",
                    _cfg_get(raw, "memory.weight_storage", "embedded"),
                )
            ).lower(),
            "memory_strategy": str(_cfg_get(raw, "memory.strategy", "policy_default")),
            "weight_storage": str(
                _cfg_get(
                    raw,
                    "memory.weight_storage",
                    _cfg_get(raw, "data_movement.ps_pl.weights.mode", "embedded"),
                )
            ).lower(),
            "precision_mode": default_precision["precision_mode"],
            "act_bits": default_precision["act_bits"],
            "weight_bits": default_precision["weight_bits"],
            "bias_bits": default_precision["bias_bits"],
            "accum_bits": default_precision["accum_bits"],
            "act_int_bits": default_precision["act_int_bits"],
            "weight_int_bits": default_precision["weight_int_bits"],
            "bias_int_bits": default_precision["bias_int_bits"],
            "accum_int_bits": default_precision["accum_int_bits"],
            "parallel_policy": policy.name,
            "requested_clock_mhz": _cfg_get(
                raw,
                "targets.platform.clocks.0.target_mhz",
                None,
            ),
            "effective_clock_mhz": policy.target_clock_mhz,
            "parallel_pe": policy.pe,
            "parallel_simd": policy.simd,
            "parallel_unroll_factor": policy.unroll_factor,
            "parallel_partition_factor": policy.partition_factor,
            "parallel_pipeline_style": policy.pipeline_style,
            "pipeline_style_requested": str(_cfg_get(raw, "optimization.pipeline.style", _cfg_get(raw, "optimization.parallel.pipeline_style", policy.pipeline_style))),
            "pipeline_ii_requested": _cfg_get(raw, "optimization.pipeline.ii", _cfg_get(raw, "optimization.pipeline_ii", None)),
            "weight_region_preference": list(policy.weight_region_preference),
            "activation_region_preference": list(policy.activation_region_preference),
            "axi_word_bits": policy.axi_word_bits,
            "burst_len": policy.burst_len,
            "enable_bitpack": policy.enable_bitpack,
            "enable_compression": policy.enable_compression,
            "array_partition_mode": policy.array_partition_mode,
            "mac_style": policy.mac_style,
            "accum_strategy": policy.accum_strategy,
            "activation_impl": policy.activation_impl,
            "round_mode": policy.round_mode,
            "sat_mode": policy.sat_mode,
            "allow_double_buffer": policy.allow_double_buffer,
        },
    )
