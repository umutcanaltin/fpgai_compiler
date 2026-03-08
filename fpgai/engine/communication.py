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


def _default_axi_word_bits(cfg) -> int:
    raw = cfg.raw
    return int(_cfg_get(raw, "data_movement.ps_pl.axi_word_bits", 32))


def _compression_enabled(cfg) -> bool:
    raw = cfg.raw
    return bool(_cfg_get(raw, "data_movement.ps_pl.compression.enabled", True))


def _pick_encoding(kind: str, size_bytes: int, cfg) -> tuple[str, bool]:
    compression = _compression_enabled(cfg)

    if not compression:
        return "raw", False

    # very simple v1 policy
    if kind == "input" and size_bytes >= 64:
        return "bitpack", True
    if kind == "output":
        return "raw", False
    return "raw", False


def make_communication_plan(cfg, memory_plan: MemoryPlan) -> CommunicationPlan:
    axi_word_bits = _default_axi_word_bits(cfg)
    edges: List[CommunicationEdge] = []

    for placement in memory_plan.placements:
        kind = placement.kind
        size_bytes = placement.size_bytes

        if placement.region == "HOST" and kind == "input":
            encoding, unpack_in_pl = _pick_encoding(kind, size_bytes, cfg)
            edges.append(
                CommunicationEdge(
                    tensor_name=placement.tensor_name,
                    direction="PS_TO_PL",
                    encoding=encoding,
                    packed_bits=None,
                    axi_word_bits=axi_word_bits,
                    burst_len=16,
                    unpack_in_pl=unpack_in_pl,
                )
            )

        elif placement.region == "HOST" and kind == "output":
            edges.append(
                CommunicationEdge(
                    tensor_name=placement.tensor_name,
                    direction="PL_TO_PS",
                    encoding="raw",
                    packed_bits=None,
                    axi_word_bits=axi_word_bits,
                    burst_len=16,
                    unpack_in_pl=False,
                )
            )

        elif placement.kind == "activation":
            edges.append(
                CommunicationEdge(
                    tensor_name=placement.tensor_name,
                    direction="PL_TO_PL",
                    encoding="raw",
                    packed_bits=None,
                    axi_word_bits=axi_word_bits,
                    burst_len=16,
                    unpack_in_pl=False,
                )
            )

        elif placement.kind == "weight":
            # for now treat all weights as PS/DDR managed or preloaded abstractly
            # later this will depend on weight_mode + memory region
            if placement.region == "DDR":
                edges.append(
                    CommunicationEdge(
                        tensor_name=placement.tensor_name,
                        direction="PS_TO_PL",
                        encoding="raw",
                        packed_bits=None,
                        axi_word_bits=axi_word_bits,
                        burst_len=32,
                        unpack_in_pl=False,
                    )
                )

    return CommunicationPlan(
        edges=edges,
        notes={
            "planner": "communication_heuristic_v1",
            "axi_word_bits": axi_word_bits,
            "compression_enabled": _compression_enabled(cfg),
        },
    )