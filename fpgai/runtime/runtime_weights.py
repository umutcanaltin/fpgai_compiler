"""Runtime weight payload parsing and packaging."""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any


_RUNTIME_WEIGHT_ARRAY_RE = re.compile(
    r"const\s+[A-Za-z_][A-Za-z0-9_:<>]*\s+([WB]\d+)_init\s*\[\s*(\d+)\s*\]\s*=\s*\{(.*?)\};",
    re.DOTALL,
)


def _normalise_weights_mode(weights_mode: str | None) -> str:
    mode = str(weights_mode or "").strip().lower().replace("-", "_")
    aliases = {
        "bram": "bram_static",
        "embedded": "bram_static",
        "embedded_bram": "bram_static",
        "onchip": "bram_static",
        "on_chip": "bram_static",
        "static": "bram_static",
        "const": "bram_static",
        "ddr": "bram_import_full",
        "dma_ddr": "bram_import_full",
        "runtime": "bram_import_full",
        "preload": "bram_import_full",
        "direct": "bram_import_full",
        "uram": "uram_import_full",
    }
    return aliases.get(mode, mode)


def _float_to_packed32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _parse_float_initializer_values(body: str) -> list[float]:
    values: list[float] = []
    for raw in body.replace("\n", " ").split(","):
        token = raw.strip()
        if not token:
            continue
        token = token.replace("f", "")
        values.append(float(token))
    return values


def _runtime_param_source(root: Path) -> Path | None:
    candidates = [
        root / "hls" / "src" / "fpgai_params.cpp",
        root / "hls" / "fpgai_params.cpp",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    hits = sorted(root.glob("hls/**/fpgai_params.cpp"))
    for hit in hits:
        if hit.is_file():
            return hit
    return None


def _parse_runtime_weight_entries(root: Path) -> tuple[Path | None, list[dict[str, Any]], list[int]]:
    src = _runtime_param_source(root)
    if src is None:
        return None, [], []

    text = src.read_text(encoding="utf-8", errors="replace")
    entries: list[dict[str, Any]] = []
    words: list[int] = []
    offset = 0

    for match in _RUNTIME_WEIGHT_ARRAY_RE.finditer(text):
        base_name = match.group(1)
        declared_count = int(match.group(2))
        values = _parse_float_initializer_values(match.group(3))

        if len(values) != declared_count:
            raise ValueError(
                f"Runtime weight initializer {base_name}_init declares {declared_count} values "
                f"but parser found {len(values)} in {src}"
            )

        packed = [_float_to_packed32(v) for v in values]
        words.extend(packed)
        entries.append(
            {
                "name": base_name,
                "kind": "bias" if base_name.startswith("B") else "weight",
                "offset_words": offset,
                "count_words": declared_count,
            }
        )
        offset += declared_count

    return src, entries, words


def _runtime_weight_payload_required(weights_mode: str | None, entries: list[dict[str, Any]]) -> bool:
    mode = _normalise_weights_mode(weights_mode)
    if mode in {
        "bram_import_full",
        "bram_import_export_full",
        "uram_import_full",
        "uram_import_export_full",
        "ddr_tiled",
        "ddr_tiled_mutable",
        "tile_cached",
    }:
        return True
    if mode in {"bram_static", "uram_static", "none", "static", "const"}:
        return False

    return bool(entries)


def _emit_runtime_weight_payload(
    root: Path,
    package_dir: Path,
    *,
    weights_mode: str | None,
) -> dict[str, Any]:
    src, entries, words = _parse_runtime_weight_entries(root)
    required = _runtime_weight_payload_required(weights_mode, entries)

    summary: dict[str, Any] = {
        "required": required,
        "present": False,
        "weights_mode": _normalise_weights_mode(weights_mode),
        "import_required": required,
        "export_supported": _normalise_weights_mode(weights_mode) in {"bram_import_export_full", "uram_import_export_full", "ddr_tiled_mutable"},
        "reload_before_each_compute": False,
        "format": "packed32",
        "source": src.as_posix() if src is not None else None,
        "total_words": 0,
        "word_bytes": 4,
        "weights_bin": None,
        "weight_layout": None,
        "status": "not_required",
    }

    if not required:
        return {"summary": summary, "files": {}}

    if not entries:
        summary["status"] = "missing_generated_runtime_parameters"
        return {"summary": summary, "files": {}}

    weights_dir = package_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    weights_bin = weights_dir / "weights.bin"
    weights_bin.write_bytes(b"".join(struct.pack("<I", word) for word in words))

    layout = {
        "format": "packed32",
        "source": "generated_hls_parameters",
        "parameter_source": src.as_posix() if src is not None else None,
        "entries": entries,
        "total_words": len(words),
        "word_bytes": 4,
    }

    layout_path = weights_dir / "weight_layout.json"
    layout_path.write_text(json.dumps(layout, indent=2, sort_keys=True), encoding="utf-8")

    summary.update(
        {
            "present": True,
            "total_words": len(words),
            "weights_bin": "weights/weights.bin",
            "weight_layout": "weights/weight_layout.json",
            "status": "created",
        }
    )

    return {
        "summary": summary,
        "files": {
            "weights_bin": {
                "source": src.as_posix() if src is not None else None,
                "package_path": "weights/weights.bin",
                "bytes": weights_bin.stat().st_size,
            },
            "weight_layout": {
                "source": layout_path.as_posix(),
                "package_path": "weights/weight_layout.json",
                "bytes": layout_path.stat().st_size,
            },
        },
    }
