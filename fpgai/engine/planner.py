from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from fpgai.engine.models import CompilePlan, LayerDescriptor, LayerPlan


def _cfg_get(raw: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = raw
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


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


def _default_precision_info(cfg) -> Dict[str, Any]:
    numerics = _numerics_defaults(cfg)
    act = numerics.get("activation", {}) or {}
    weight = numerics.get("weight", {}) or {}
    bias = numerics.get("bias", {}) or {}
    accum = numerics.get("accum", {}) or {}

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
    return POLICIES.get(name, POLICIES["Balanced"])


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
        weight_region_preference=list(base.weight_region_preference),
        activation_region_preference=list(base.activation_region_preference),
        allow_double_buffer=base.allow_double_buffer,
        axi_word_bits=base.axi_word_bits,
        burst_len=base.burst_len,
        enable_bitpack=base.enable_bitpack,
        enable_compression=base.enable_compression,
        array_partition_mode=base.array_partition_mode,
        mac_style=base.mac_style,
        accum_strategy=base.accum_strategy,
        activation_impl=base.activation_impl,
        round_mode=base.round_mode,
        sat_mode=base.sat_mode,
    )


def _choose_weight_mode(desc: LayerDescriptor, raw_cfg: Dict[str, Any]) -> str:
    """
    Keep the full pipeline stable:
    - only use stream/ddr if the user explicitly requests it
    - otherwise stay embedded for small/medium models
    """
    requested = str(_cfg_get(raw_cfg, "data_movement.ps_pl.weights.mode", "embedded")).lower()
    if requested == "dma_ddr":
        requested = "ddr"

    if requested in ("stream", "ddr"):
        return requested

    # Safe default for now: keep robust full pipeline
    return "embedded"


def _buffering_for(weights_mode: str, policy: Policy) -> str:
    if not policy.allow_double_buffer:
        return "single"
    return "double" if weights_mode in ("stream", "ddr") else "single"


def _pipeline_ii_for(policy: Policy, compute_hint: str = "") -> int:
    if policy.pipeline_style == "aggressive":
        return 1
    if policy.pipeline_style == "conservative":
        return 2 if compute_hint == "memory_bound" else 1
    return 1


def _layer_notes(desc: LayerDescriptor, precision: Dict[str, Any], policy: Policy) -> Dict[str, Any]:
    return {
        "policy_name": policy.name,
        "compute_hint": desc.compute_hint,
        "partition_factor": policy.partition_factor,
        "partition_mode": policy.array_partition_mode,
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


def _plan_conv(desc: LayerDescriptor, precision: Dict[str, Any], weights_mode: str, policy: Policy) -> LayerPlan:
    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile={"oh": policy.conv_oh, "ow": policy.conv_ow, "oc": policy.conv_oc},
        unroll={"ic": policy.simd, "oc": policy.pe},
        pipeline_ii=_pipeline_ii_for(policy, desc.compute_hint),
        weight_mode=weights_mode,
        activation_mode="stream" if policy.name in ("Latency-First", "Throughput-First") else "buffer",
        buffering=_buffering_for(weights_mode, policy),
        backend_kernel=desc.backend_kernel or "conv",
        notes=_layer_notes(desc, precision, policy),
    )


def _plan_dense(desc: LayerDescriptor, precision: Dict[str, Any], weights_mode: str, policy: Policy) -> LayerPlan:
    out_features = int(desc.attrs.get("out_features", 1) or 1)
    in_features = int(desc.attrs.get("in_features", 1) or 1)

    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile={"in": min(in_features, policy.dense_in), "out": min(out_features, policy.dense_out)},
        unroll={"in": policy.simd, "out": policy.pe},
        pipeline_ii=_pipeline_ii_for(policy, desc.compute_hint),
        weight_mode=weights_mode,
        activation_mode="stream" if policy.name in ("Latency-First", "Throughput-First") else "buffer",
        buffering=_buffering_for(weights_mode, policy),
        backend_kernel=desc.backend_kernel or "dense",
        notes=_layer_notes(desc, precision, policy),
    )


def _plan_pool(desc: LayerDescriptor, precision: Dict[str, Any], policy: Policy) -> LayerPlan:
    spatial = max(4, min(policy.conv_oh, policy.conv_ow))
    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile={"oh": spatial, "ow": spatial},
        unroll={},
        pipeline_ii=_pipeline_ii_for(policy, desc.compute_hint),
        weight_mode="embedded",
        activation_mode="stream",
        buffering="single",
        backend_kernel=desc.backend_kernel or desc.op_type.lower(),
        notes=_layer_notes(desc, precision, policy),
    )


def _plan_elementwise(desc: LayerDescriptor, precision: Dict[str, Any], policy: Policy) -> LayerPlan:
    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile={},
        unroll={},
        pipeline_ii=_pipeline_ii_for(policy, desc.compute_hint),
        weight_mode="embedded",
        activation_mode="stream",
        buffering="single",
        backend_kernel=desc.backend_kernel or desc.op_type.lower(),
        notes=_layer_notes(desc, precision, policy),
    )


def _plan_generic(desc: LayerDescriptor, precision: Dict[str, Any], weights_mode: str, policy: Policy) -> LayerPlan:
    return LayerPlan(
        node_name=desc.node_name,
        op_type=desc.op_type,
        precision_mode=precision["precision_mode"],
        act_bits=precision["act_bits"],
        weight_bits=precision["weight_bits"],
        tile={},
        unroll={},
        pipeline_ii=_pipeline_ii_for(policy, desc.compute_hint),
        weight_mode=weights_mode,
        activation_mode="buffer",
        buffering=_buffering_for(weights_mode, policy),
        backend_kernel=desc.backend_kernel or desc.op_type.lower(),
        notes=_layer_notes(desc, precision, policy),
    )


def make_compile_plan(cfg, descriptors: List[LayerDescriptor]) -> CompilePlan:
    raw = cfg.raw
    precision = _default_precision_info(cfg)
    policy = _override_policy_from_cfg(cfg, _pick_policy(cfg))

    target_board = str(_cfg_get(raw, "targets.platform.board", "unknown"))
    target_part = str(_cfg_get(raw, "targets.platform.part", "unknown"))

    layer_plans: List[LayerPlan] = []
    execution_order: List[str] = []

    for desc in descriptors:
        execution_order.append(desc.node_name)
        weights_mode = _choose_weight_mode(desc, raw)

        if desc.op_type == "Conv":
            plan = _plan_conv(desc, precision, weights_mode, policy)
        elif desc.op_type == "Dense":
            plan = _plan_dense(desc, precision, weights_mode, policy)
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
            "planner": "policy_driven_v5_stable",
            "global_weights_mode_requested": str(_cfg_get(raw, "data_movement.ps_pl.weights.mode", "embedded")).lower(),
            "precision_mode": precision["precision_mode"],
            "act_bits": precision["act_bits"],
            "weight_bits": precision["weight_bits"],
            "bias_bits": precision["bias_bits"],
            "accum_bits": precision["accum_bits"],
            "act_int_bits": precision["act_int_bits"],
            "weight_int_bits": precision["weight_int_bits"],
            "bias_int_bits": precision["bias_int_bits"],
            "accum_int_bits": precision["accum_int_bits"],
            "parallel_policy": policy.name,
            "parallel_pe": policy.pe,
            "parallel_simd": policy.simd,
            "parallel_unroll_factor": policy.unroll_factor,
            "parallel_partition_factor": policy.partition_factor,
            "parallel_pipeline_style": policy.pipeline_style,
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