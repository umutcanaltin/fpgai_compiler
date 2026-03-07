from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import struct
import numpy as np


@dataclass(frozen=True)
class DenseWeightsSpec:
    layer_name: str
    in_features: int
    out_features: int
    weight_name: str
    bias_name: str
    layout: str = "out_in"  # W[out][in]


@dataclass(frozen=True)
class WeightsPlan:
    """
    Host-visible description of how the HLS kernel expects weights.
    stream mode expects, per Dense layer:
      - W[out][in] float32 (row-major by out then in)
      - B[out] float32
    """
    dense_layers: List[DenseWeightsSpec]

    def to_dict(self) -> dict:
        return {
            "dense_layers": [
                {
                    "layer_name": x.layer_name,
                    "in_features": x.in_features,
                    "out_features": x.out_features,
                    "weight_name": x.weight_name,
                    "bias_name": x.bias_name,
                    "layout": x.layout,
                }
                for x in self.dense_layers
            ]
        }


def build_weights_plan_from_ir(graph) -> WeightsPlan:
    dense_specs: List[DenseWeightsSpec] = []
    for op in graph.ops:
        if op.op_type != "Dense":
            continue
        w = op.attrs.get("weight")
        b = op.attrs.get("bias")
        in_f = int(op.attrs.get("in_features") or 0)
        out_f = int(op.attrs.get("out_features") or 0)
        if not isinstance(w, str) or not isinstance(b, str):
            raise ValueError(f"Dense {op.name} missing weight/bias attr")
        if in_f <= 0 or out_f <= 0:
            raise ValueError(f"Dense {op.name} missing in/out features")
        dense_specs.append(
            DenseWeightsSpec(
                layer_name=op.name,
                in_features=in_f,
                out_features=out_f,
                weight_name=w,
                bias_name=b,
                layout=str(op.attrs.get("layout", "out_in")),
            )
        )
    return WeightsPlan(dense_layers=dense_specs)


def _as_out_in(W: np.ndarray, out_f: int, in_f: int) -> np.ndarray:
    W = np.asarray(W)
    if W.shape == (out_f, in_f):
        return W
    if W.shape == (in_f, out_f):
        return W.T
    raise ValueError(f"Weight shape {W.shape} incompatible with OUT={out_f}, IN={in_f}")


def pack_weights_stream_float32(graph, plan: WeightsPlan) -> bytes:
    """
    Produces the exact byte payload expected after cmd=3 in stream mode.

    Format: float32 little-endian words:
      for each Dense layer:
        W[out][in] then B[out]
    """
    out = bytearray()
    for spec in plan.dense_layers:
        W = graph.params[spec.weight_name]
        B = graph.params[spec.bias_name]

        W2 = _as_out_in(W, spec.out_features, spec.in_features).astype(np.float32, copy=False)
        B2 = np.asarray(B).reshape(-1).astype(np.float32, copy=False)

        if B2.shape[0] != spec.out_features:
            raise ValueError(f"{spec.layer_name} bias length {B2.shape[0]} != out_features {spec.out_features}")

        # W: out-major then in
        for o in range(spec.out_features):
            for i in range(spec.in_features):
                out += struct.pack("<f", float(W2[o, i]))

        # B
        for o in range(spec.out_features):
            out += struct.pack("<f", float(B2[o]))

    return bytes(out)


def pack_cmd_word(cmd: int) -> bytes:
    """cmd word is sent as uint32 on the AXIS. Little-endian."""
    return struct.pack("<I", int(cmd) & 0xFFFFFFFF)
