from __future__ import annotations

from typing import Any, Dict, List

from fpgai.engine.models import CommunicationEdge, CommunicationPlan, MemoryPlan


def _cfg_get(raw: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = raw
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _edge_direction(kind: str, region: str) -> str:
    kind_u = str(kind).lower()
    region_u = str(region).upper()

    if kind_u in ("input", "weight") and region_u in ("HOST", "DDR"):
        return "PS_TO_PL"
    if kind_u == "output":
        return "PL_TO_PS"
    if region_u in ("BRAM", "URAM", "LUTRAM"):
        return "PL_TO_PL"
    return "PS_TO_PL"


def _choose_encoding(
    *,
    policy_name: str,
    enable_bitpack: bool,
    enable_compression: bool,
    kind: str,
    region: str,
    size_bytes: int,
) -> str:
    kind_u = str(kind).lower()
    region_u = str(region).upper()

    if kind_u == "output":
        return "raw"

    if region_u == "DDR" and enable_compression and size_bytes >= 64 * 1024:
        return "rle"

    if enable_bitpack and kind_u in ("weight", "activation", "input"):
        return "bitpack"

    return "raw"


def _packed_bits_for(notes: Dict[str, Any]) -> int | None:
    act_bits = notes.get("act_bits")
    weight_bits = notes.get("weight_bits")
    if isinstance(weight_bits, int):
        return weight_bits
    if isinstance(act_bits, int):
        return act_bits
    return None


def make_communication_plan(cfg, memory_plan: MemoryPlan) -> CommunicationPlan:
    raw = cfg.raw
    cnotes = memory_plan.notes or {}

    policy_name = str(cnotes.get("policy_name", "Balanced"))

    # compiler notes may not be inside memory plan notes in older runs
    axi_word_bits = int(_cfg_get(raw, "communication.axi.word_bits", 128))
    burst_len = int(_cfg_get(raw, "communication.axi.burst_len", 32))
    enable_bitpack = bool(_cfg_get(raw, "data_movement.ps_pl.compression.enabled", True))
    enable_compression = bool(_cfg_get(raw, "data_movement.ps_pl.compression.enabled", True))

    # policy-aware defaults if explicit config is absent
    if policy_name == "Fit-First":
        axi_word_bits = int(_cfg_get(raw, "communication.axi.word_bits", 64))
        burst_len = int(_cfg_get(raw, "communication.axi.burst_len", 16))
        enable_bitpack = bool(_cfg_get(raw, "data_movement.ps_pl.compression.enabled", False))
    elif policy_name == "Latency-First":
        axi_word_bits = int(_cfg_get(raw, "communication.axi.word_bits", 128))
        burst_len = int(_cfg_get(raw, "communication.axi.burst_len", 64))
        enable_bitpack = bool(_cfg_get(raw, "data_movement.ps_pl.compression.enabled", True))

    edges: List[CommunicationEdge] = []

    for p in memory_plan.placements:
        direction = _edge_direction(p.kind, p.region)
        encoding = _choose_encoding(
            policy_name=policy_name,
            enable_bitpack=enable_bitpack,
            enable_compression=enable_compression,
            kind=p.kind,
            region=p.region,
            size_bytes=int(p.size_bytes),
        )

        packed_bits = None
        if encoding == "bitpack":
            packed_bits = _packed_bits_for(p.notes)

        unpack_in_pl = encoding in ("bitpack", "rle") and direction != "PL_TO_PS"

        edges.append(
            CommunicationEdge(
                tensor_name=p.tensor_name,
                direction=direction,
                encoding=encoding,
                packed_bits=packed_bits,
                axi_word_bits=axi_word_bits,
                burst_len=burst_len,
                unpack_in_pl=unpack_in_pl,
                notes={
                    "policy_name": policy_name,
                    "kind": p.kind,
                    "region": p.region,
                    "double_buffer": p.double_buffer,
                    "reason": p.notes.get("reason"),
                },
            )
        )

    return CommunicationPlan(
        edges=edges,
        notes={
            "planner": "policy_comm_v2",
            "policy_name": policy_name,
            "axi_word_bits": axi_word_bits,
            "burst_len": burst_len,
            "enable_bitpack": enable_bitpack,
            "enable_compression": enable_compression,
        },
    )