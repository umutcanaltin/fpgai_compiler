from __future__ import annotations

from typing import Any, Dict, List, Mapping

from fpgai.config.access import get_path
from fpgai.engine.models import CommunicationEdge, CommunicationPlan, MemoryPlan


_cfg_get = get_path


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _positive_int(value: Any, default: int) -> int:
    if value is None or isinstance(value, bool):
        return int(default)
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return parsed if parsed > 0 else int(default)


def _bool_cfg(raw: Dict[str, Any], path: str, default: bool) -> bool:
    value = _cfg_get(raw, path, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return bool(default if value is None else value)


def _edge_direction(kind: str, region: str, mode: str = "") -> str:
    kind_u = str(kind).lower()
    region_u = str(region).upper()
    mode_u = str(mode).lower()

    if kind_u in {"input", "weight", "bias", "target", "aux"}:
        return "PS_TO_PL"
    if kind_u in {"output", "loss", "gradient", "updated_weight"}:
        return "PL_TO_PS"
    if mode_u in {"stream", "streamed", "ddr", "dma_ddr", "external_ddr"}:
        return "PS_TO_PL"
    if region_u in {"BRAM", "URAM", "LUTRAM"}:
        return "PL_TO_PL"
    if region_u in {"HOST", "DDR"}:
        return "PS_TO_PL"
    return "PS_TO_PL"



def _first_cfg_dict(raw: Dict[str, Any], *paths: str) -> Dict[str, Any]:
    for path in paths:
        value = _cfg_get(raw, path, None)
        if isinstance(value, dict):
            return _as_dict(value)
    return {}


def _first_cfg_value(raw: Dict[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        value = _cfg_get(raw, path, None)
        if value is not None:
            return value
    return default


def _edge_cfg_paths(kind: str) -> tuple[str, ...]:
    kind_u = str(kind).lower()
    if kind_u in {"input", "inputs", "activation_in"}:
        return (
            "data_movement.inputs.import",
            "data_movement.inputs",
            "data_movement.input.load",
            "data_movement.ps_pl.input",
        )
    if kind_u in {"weight", "weights", "bias", "param", "parameter"}:
        return (
            "data_movement.weights.import",
            "data_movement.weights.load",
            "data_movement.weights",
            "data_movement.ps_pl.weights",
        )
    if kind_u in {"target", "targets", "label", "labels", "aux"}:
        return (
            "data_movement.labels.import",
            "data_movement.labels",
            "data_movement.aux.load",
            "data_movement.ps_pl.aux",
        )
    if kind_u in {"output", "outputs", "activation_out"}:
        return (
            "data_movement.outputs.export",
            "data_movement.outputs",
            "data_movement.output.store",
            "data_movement.pl_ps.output",
        )
    if kind_u in {"loss"}:
        return (
            "data_movement.loss.export",
            "data_movement.loss.store",
            "data_movement.pl_ps.loss",
        )
    if kind_u in {"gradient", "gradients", "updated_weight"}:
        return (
            "data_movement.gradients.export",
            "data_movement.gradients.store",
            "data_movement.gradients",
            "data_movement.pl_ps.gradients",
        )
    if kind_u in {"optimizer_state", "optimizer", "optimizer_states"}:
        return (
            "data_movement.optimizer_state.export",
            "data_movement.optimizer_state.store",
            "data_movement.optimizer_state",
            "data_movement.pl_ps.optimizer_state",
        )
    return (f"data_movement.tensor_edges.{kind_u}",)




def _edge_tiling(edge_cfg: Dict[str, Any]) -> tuple[bool, int | None]:
    policy = str(edge_cfg.get("policy") or "").strip().lower().replace("-", "_")
    tiled = edge_cfg.get("tiled")
    enabled = policy == "tiled"
    tile_size = edge_cfg.get("tile_size", edge_cfg.get("tile_words"))
    if isinstance(tiled, Mapping):
        if "enabled" in tiled:
            enabled = _bool_cfg({"tiled": tiled}, "tiled.enabled", enabled)
        elif any(k in tiled for k in {"tile_size", "size", "words"}):
            enabled = True
        tile_size = tiled.get("tile_size", tiled.get("size", tiled.get("words", tile_size)))
    elif tiled is not None:
        if isinstance(tiled, bool):
            enabled = tiled
        elif isinstance(tiled, str):
            enabled = tiled.strip().lower() in {"1", "true", "yes", "on", "enabled", "tiled"}
        else:
            enabled = bool(tiled)
    parsed_tile = _positive_int(tile_size, 64 if enabled else 0)
    return bool(enabled), (parsed_tile if enabled else None)


def _edge_size_bytes(raw: Dict[str, Any], kind: str, default_words: int) -> int:
    cfg = _edge_cfg(raw, kind)

    size_bytes = cfg.get("size_bytes")
    if size_bytes is not None:
        return int(size_bytes)

    size_words = cfg.get("size_words")
    if size_words is not None:
        return int(size_words) * 4

    nbytes = cfg.get("bytes")
    if nbytes is not None:
        return int(nbytes)

    words = cfg.get("words")
    if words is not None:
        return int(words) * 4

    for path in _edge_cfg_paths(kind):
        value = _synthetic_size(raw, path, default_words)
        if value != default_words * 4:
            return value

    return default_words * 4

def _edge_path(kind: str) -> str:
    kind_u = str(kind).lower()
    if kind_u in {"input", "inputs", "activation_in"}:
        return "data_movement.inputs.import"
    if kind_u in {"weight", "weights", "bias", "param", "parameter"}:
        return "data_movement.weights.import"
    if kind_u in {"target", "targets", "label", "labels", "aux"}:
        return "data_movement.labels.import"
    if kind_u in {"output", "outputs", "activation_out"}:
        return "data_movement.outputs.export"
    if kind_u in {"loss"}:
        return "data_movement.loss.export"
    if kind_u in {"gradient", "gradients", "updated_weight"}:
        return "data_movement.gradients.export"
    return f"data_movement.tensor_edges.{kind_u}"


def _edge_cfg(raw: Dict[str, Any], kind: str) -> Dict[str, Any]:
    return _first_cfg_dict(raw, *_edge_cfg_paths(kind))


def _global_compression_cfg(raw: Dict[str, Any], direction: str) -> Dict[str, Any]:
    if direction == "PL_TO_PS":
        cfg = _cfg_get(raw, "data_movement.pl_ps.compression", None)
        if cfg is None:
            cfg = _cfg_get(raw, "data_movement.compression", {})
        return _as_dict(cfg)
    cfg = _cfg_get(raw, "data_movement.ps_pl.compression", None)
    if cfg is None:
        cfg = _cfg_get(raw, "data_movement.compression", {})
    return _as_dict(cfg)


def _precision_bits_from_spec(spec: Any) -> int | None:
    spec_d = _as_dict(spec)
    value = spec_d.get("total_bits")
    if value is None:
        value = spec_d.get("bits")
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _precision_bits_for(kind: str, placement_notes: Dict[str, Any], edge_cfg: Dict[str, Any]) -> int | None:
    explicit = _precision_bits_from_spec(edge_cfg.get("precision"))
    if explicit is not None:
        return explicit

    kind_u = str(kind).lower()
    if kind_u in {"weight", "bias", "param", "parameter"}:
        value = placement_notes.get("weight_bits")
        if value is None:
            value = placement_notes.get("bias_bits")
    else:
        value = placement_notes.get("act_bits")

    if isinstance(value, int) and value > 0:
        return value
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _compression_policy(raw: Dict[str, Any], kind: str, direction: str, edge_cfg: Dict[str, Any]) -> tuple[bool, str]:
    edge_compression = _as_dict(edge_cfg.get("compression"))
    global_compression = _global_compression_cfg(raw, direction)

    enabled = edge_compression.get("enabled", global_compression.get("enabled", False))
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    enabled = bool(enabled)

    codec = str(
        edge_compression.get(
            "codec",
            edge_compression.get(
                "encoding",
                global_compression.get("codec", global_compression.get("encoding", "raw")),
            ),
        )
        or "raw"
    ).strip().lower()

    if not enabled:
        return False, "raw"

    if codec in {"none", "off", "false"}:
        return False, "raw"

    if codec not in {"raw", "bitpack", "rle", "delta", "sparse"}:
        codec = "bitpack"

    return True, codec


def _estimate_transfer_bytes(
    *,
    size_bytes: int,
    precision_bits: int | None,
    compression_enabled: bool,
    codec: str,
    default_payload_bits: int = 32,
) -> tuple[int, int | None]:
    size_bytes = max(0, int(size_bytes))
    packed_bits = precision_bits if precision_bits and precision_bits > 0 else None

    if not compression_enabled or codec == "raw":
        return size_bytes, packed_bits

    if codec == "bitpack":
        bits = packed_bits or default_payload_bits
        # Existing tensors are byte-sized estimates. Interpret as default 32-bit
        # elements unless precision metadata says otherwise.
        elements = max(1, (size_bytes * 8 + default_payload_bits - 1) // default_payload_bits)
        return max(1, (elements * bits + 7) // 8), bits

    if codec == "rle":
        return max(1, size_bytes // 2), packed_bits

    if codec == "delta":
        return max(1, (size_bytes * 3) // 4), packed_bits

    if codec == "sparse":
        return max(1, (size_bytes * 35) // 100), packed_bits

    return size_bytes, packed_bits


def _implemented_in_hls(kind: str, direction: str, codec: str, mode: str) -> bool:
    # Current generated HLS implements raw AXI stream I/O and raw runtime
    # weight preload through stream/DDR. Compression codecs are modeled unless
    # a dedicated codec decoder/encoder is generated.
    if codec != "raw":
        return False

    kind_u = str(kind).lower()
    mode_u = str(mode).lower()
    if kind_u in {"input", "output"}:
        return True
    if kind_u in {"weight", "bias", "param", "parameter"} and mode_u in {
        "embedded",
        "stream",
        "streamed",
        "ddr",
        "dma_ddr",
        "external_ddr",
    }:
        return True
    return direction in {"PS_TO_PL", "PL_TO_PS", "PL_TO_PL"}


def _default_mode_for(kind: str, region: str, edge_cfg: Dict[str, Any]) -> str:
    configured = edge_cfg.get("mode")
    if configured is not None:
        return str(configured).strip().lower().replace("-", "_")
    interface = str(edge_cfg.get("interface", "") or "").strip().lower().replace("-", "_")
    transport = str(edge_cfg.get("transport", "") or "").strip().lower().replace("-", "_")
    if interface in {"axi_stream", "axis", "stream"} or transport in {"dma", "axi_dma"}:
        return "stream"
    if interface in {"m_axi", "maxi", "ddr"}:
        return "ddr"
    if interface in {"compile_time", "static", "embedded"}:
        return "embedded"

    kind_u = str(kind).lower()
    region_u = str(region).upper()
    if kind_u in {"input", "output", "target", "aux", "loss", "gradient"}:
        return "stream"
    if kind_u in {"weight", "bias", "param", "parameter"}:
        if region_u == "DDR":
            return "ddr"
        return "embedded"
    return "internal"


def _make_edge(
    *,
    raw: Dict[str, Any],
    policy_name: str,
    tensor_name: str,
    kind: str,
    region: str,
    size_bytes: int,
    placement_notes: Dict[str, Any],
    double_buffer: bool = False,
    default_axi_word_bits: int,
    default_burst_len: int,
) -> CommunicationEdge:
    edge_cfg = _edge_cfg(raw, kind)
    tiled_enabled, tile_size = _edge_tiling(edge_cfg)
    if tiled_enabled and not edge_cfg.get("policy"):
        edge_cfg = {**edge_cfg, "policy": "tiled"}
    mode = _default_mode_for(kind, region, edge_cfg)
    direction = _edge_direction(kind, region, mode)
    precision_bits = _precision_bits_for(kind, placement_notes, edge_cfg)

    compression_enabled, codec = _compression_policy(raw, kind, direction, edge_cfg)
    transfer_bytes, packed_bits = _estimate_transfer_bytes(
        size_bytes=size_bytes,
        precision_bits=precision_bits,
        compression_enabled=compression_enabled,
        codec=codec,
    )

    axi_word_bits = _positive_int(
        edge_cfg.get("axi_word_bits", _cfg_get(raw, "communication.axi.word_bits", default_axi_word_bits)),
        default_axi_word_bits,
    )
    burst_len = _positive_int(
        edge_cfg.get("burst_len", _cfg_get(raw, "communication.axi.burst_len", default_burst_len)),
        default_burst_len,
    )

    unpack_in_pl = codec in {"bitpack", "rle", "delta", "sparse"} and direction == "PS_TO_PL"

    return CommunicationEdge(
        tensor_name=str(tensor_name),
        direction=direction,
        encoding=codec,
        packed_bits=packed_bits if codec == "bitpack" else None,
        axi_word_bits=axi_word_bits,
        burst_len=burst_len,
        unpack_in_pl=unpack_in_pl,
        size_bytes=max(0, int(size_bytes)),
        transfer_bytes=int(transfer_bytes),
        precision_bits=precision_bits,
        compression_enabled=compression_enabled,
        codec=codec,
        source="PL" if direction == "PL_TO_PS" else ("PL" if direction == "PL_TO_PL" else "HOST"),
        destination="HOST" if direction == "PL_TO_PS" else "PL",
        implemented_in_hls=_implemented_in_hls(kind, direction, codec, mode),
        notes={
            "policy_name": policy_name,
            "kind": kind,
            "region": region,
            "mode": mode,
            "interface": edge_cfg.get("interface"),
            "transport": edge_cfg.get("transport"),
            "policy": edge_cfg.get("policy"),
            "tiled": bool(tiled_enabled),
            "tile_size": tile_size,
            "direction_name": "export" if direction == "PL_TO_PS" else "import",
            "double_buffer": bool(double_buffer),
            "reason": placement_notes.get("reason"),
            "edge_config_path": _edge_path(kind),
            "modeled_transfer": codec != "raw",
            "hardware_codec": codec == "raw",
            "original_size_bytes": max(0, int(size_bytes)),
            "estimated_transfer_bytes": int(transfer_bytes),
        },
    )


def _synthetic_size(raw: Dict[str, Any], path: str, default: int) -> int:
    return _positive_int(
        _cfg_get(raw, f"{path}.size_bytes", _cfg_get(raw, f"{path}.bytes", default)),
        default,
    )


def _append_synthetic_io_edges(
    *,
    raw: Dict[str, Any],
    edges: List[CommunicationEdge],
    policy_name: str,
    default_axi_word_bits: int,
    default_burst_len: int,
    existing_kinds: set[str],
) -> None:
    # Ensure input/output communication is always represented even when the
    # memory plan only contains parameter/internal activation placements.
    if "input" not in existing_kinds:
        input_cfg = _edge_cfg(raw, "input")
        edges.append(
            _make_edge(
                raw=raw,
                policy_name=policy_name,
                tensor_name=str(input_cfg.get("tensor_name", "input")),
                kind="input",
                region=str(input_cfg.get("region", "HOST")),
                size_bytes=_edge_size_bytes(raw, "input", 4),
                placement_notes={},
                default_axi_word_bits=default_axi_word_bits,
                default_burst_len=default_burst_len,
            )
        )

    if "output" not in existing_kinds:
        output_cfg = _edge_cfg(raw, "output")
        edges.append(
            _make_edge(
                raw=raw,
                policy_name=policy_name,
                tensor_name=str(output_cfg.get("tensor_name", "output")),
                kind="output",
                region=str(output_cfg.get("region", "HOST")),
                size_bytes=_edge_size_bytes(raw, "output", 4),
                placement_notes={},
                default_axi_word_bits=default_axi_word_bits,
                default_burst_len=default_burst_len,
            )
        )

    aux_cfg = _edge_cfg(raw, "aux")
    if aux_cfg.get("enabled", False) and "aux" not in existing_kinds and "target" not in existing_kinds:
        edges.append(
            _make_edge(
                raw=raw,
                policy_name=policy_name,
                tensor_name=str(aux_cfg.get("tensor_name", "aux")),
                kind="aux",
                region=str(aux_cfg.get("region", "HOST")),
                size_bytes=_edge_size_bytes(raw, "aux", 4),
                placement_notes={},
                default_axi_word_bits=default_axi_word_bits,
                default_burst_len=default_burst_len,
            )
        )


def make_communication_plan(cfg, memory_plan: MemoryPlan) -> CommunicationPlan:
    raw = cfg.raw
    cnotes = memory_plan.notes or {}

    policy_name = str(cnotes.get("policy_name", "Balanced"))

    axi_word_bits = int(_cfg_get(raw, "communication.axi.word_bits", 128))
    burst_len = int(_cfg_get(raw, "communication.axi.burst_len", 32))

    if policy_name in ("Fit-First", "Memory-First"):
        axi_word_bits = int(_cfg_get(raw, "communication.axi.word_bits", 64))
        burst_len = int(_cfg_get(raw, "communication.axi.burst_len", 16))
    elif policy_name == "Latency-First":
        axi_word_bits = int(_cfg_get(raw, "communication.axi.word_bits", 128))
        burst_len = int(_cfg_get(raw, "communication.axi.burst_len", 64))

    edges: List[CommunicationEdge] = []
    existing_kinds: set[str] = set()

    for p in memory_plan.placements:
        notes = getattr(p, "notes", {}) or {}
        kind = str(getattr(p, "kind", "tensor")).lower()
        region = str(getattr(p, "region", "BRAM"))
        existing_kinds.add(kind)

        edges.append(
            _make_edge(
                raw=raw,
                policy_name=policy_name,
                tensor_name=str(getattr(p, "tensor_name", kind)),
                kind=kind,
                region=region,
                size_bytes=int(getattr(p, "size_bytes", 0) or 0),
                placement_notes=notes,
                double_buffer=bool(getattr(p, "double_buffer", False)),
                default_axi_word_bits=axi_word_bits,
                default_burst_len=burst_len,
            )
        )

    _append_synthetic_io_edges(
        raw=raw,
        edges=edges,
        policy_name=policy_name,
        default_axi_word_bits=axi_word_bits,
        default_burst_len=burst_len,
        existing_kinds=existing_kinds,
    )

    total_original_bytes = sum(int(edge.size_bytes) for edge in edges)
    total_transfer_bytes = sum(int(edge.transfer_bytes) for edge in edges)

    return CommunicationPlan(
        edges=edges,
        notes={
            "planner": "tensor_edge_comm_v1",
            "policy_name": policy_name,
            "axi_word_bits": axi_word_bits,
            "burst_len": burst_len,
            "total_original_bytes": total_original_bytes,
            "total_transfer_bytes": total_transfer_bytes,
            "modeled_transfer_reduction_bytes": total_original_bytes - total_transfer_bytes,
            "contains_modeled_codecs": any(edge.codec != "raw" for edge in edges),
            "contains_hardware_codecs": any(edge.codec != "raw" and edge.implemented_in_hls for edge in edges),
            "scope": "input_weight_output_aux_tensor_edges",
        },
    )
