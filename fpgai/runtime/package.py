from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from pathlib import Path
from typing import Any, Mapping


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _copy_if_exists(src: Path, dst: Path) -> dict[str, Any] | None:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": src.as_posix(),
        "package_path": dst.as_posix(),
        "bytes": dst.stat().st_size,
    }


def _first_existing(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        hits = sorted(root.glob(pattern))
        for hit in hits:
            if hit.is_file():
                return hit
    return None


def _collect_existing(root: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        out.extend([p for p in sorted(root.glob(pattern)) if p.is_file()])
    return out


def _artifact_status(path: Path | None) -> dict[str, Any]:
    return {
        "present": bool(path is not None and path.exists()),
        "path": path.as_posix() if path is not None else None,
    }



_RUNTIME_WEIGHT_ARRAY_RE = re.compile(
    r"const\s+[A-Za-z_][A-Za-z0-9_:<>]*\s+([WB]\d+)_init\s*\[\s*(\d+)\s*\]\s*=\s*\{(.*?)\};",
    re.DOTALL,
)


def _normalise_weights_mode(weights_mode: str | None) -> str:
    mode = str(weights_mode or "").strip().lower()
    aliases = {
        "bram": "embedded",
        "embedded_bram": "embedded",
        "onchip": "embedded",
        "on_chip": "embedded",
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
    if mode in {"uram", "ddr", "runtime", "preload", "direct", "tile_cached"}:
        return True
    if mode in {"embedded", "none", "static", "const"}:
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

def emit_runtime_package(
    out_dir: str | Path,
    *,
    board: str | None = None,
    pipeline_mode: str | None = None,
    top_name: str | None = None,
    hls_artifacts: Mapping[str, Any] | None = None,
    weights_mode: str | None = None,
) -> dict[str, Any]:
    """Create a self-describing runtime package from existing compile artifacts.

    This function does not run Vivado, deploy to a board, or infer that hardware
    artifacts exist. It packages files that are already present and records
    bitstream/XSA/HWH status truthfully.
    """

    root = Path(out_dir).resolve()
    package_dir = root / "runtime_package"
    package_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Any] = {}

    copy_plan = {
        "compile_manifest": (root / "manifest.json", package_dir / "manifest.json"),
        "input_bin": (root / "input.bin", package_dir / "inputs" / "input.bin"),
        "output_bin": (root / "output.bin", package_dir / "outputs" / "output.bin"),
        "hls_artifact_metadata": (
            root / "hls_artifact_metadata.json",
            package_dir / "hls" / "hls_artifact_metadata.json",
        ),
        "hls_schedule_summary": (
            root / "hls_schedule_summary.json",
            package_dir / "hls" / "hls_schedule_summary.json",
        ),
        "hls_ii_comparison": (
            root / "hls_ii_comparison.json",
            package_dir / "hls" / "hls_ii_comparison.json",
        ),
    }

    for name, (src, dst) in copy_plan.items():
        copied = _copy_if_exists(src, dst)
        if copied is not None:
            copied["package_path"] = _safe_rel(Path(copied["package_path"]), package_dir)
            files[name] = copied

    # Capture HLS run logs when present.
    hls_logs = _collect_existing(root, ["hls/logs/*.log", "hls/logs/*.json"])
    copied_logs: list[dict[str, Any]] = []
    for src in hls_logs:
        copied = _copy_if_exists(src, package_dir / "hls" / "logs" / src.name)
        if copied is not None:
            copied["package_path"] = _safe_rel(Path(copied["package_path"]), package_dir)
            copied_logs.append(copied)
    if copied_logs:
        files["hls_logs"] = copied_logs

    # Runtime hardware handoff/status. These are presence checks only.
    bitstream = _first_existing(
        root,
        [
            "vivado_bridge/bitstream/*.bit",
            "vivado_bridge/project/**/*.bit",
            "**/*.bit",
        ],
    )
    hwh = _first_existing(
        root,
        [
            "vivado_bridge/bitstream/*.hwh",
            "vivado_bridge/project/**/*.hwh",
            "**/*.hwh",
        ],
    )
    xsa = _first_existing(
        root,
        [
            "vivado_bridge/bitstream/*.xsa",
            "vivado_bridge/project/**/*.xsa",
            "**/*.xsa",
        ],
    )

    hardware = {
        "bitstream": _artifact_status(bitstream),
        "hwh": _artifact_status(hwh),
        "xsa": _artifact_status(xsa),
        "deployable_overlay_present": bool(bitstream is not None and (hwh is not None or xsa is not None)),
    }

    for name, src in {"bitstream": bitstream, "hwh": hwh, "xsa": xsa}.items():
        if src is None:
            continue
        copied = _copy_if_exists(src, package_dir / "hardware" / src.name)
        if copied is not None:
            copied["package_path"] = _safe_rel(Path(copied["package_path"]), package_dir)
            files[name] = copied

    weight_payload = _emit_runtime_weight_payload(root, package_dir, weights_mode=weights_mode)
    files.update(weight_payload["files"])

    payload: dict[str, Any] = {
        "schema_version": 1,
        "package_kind": "fpgai_runtime_package",
        "status": "created",
        "package_dir": package_dir.as_posix(),
        "source_out_dir": root.as_posix(),
        "board": board,
        "pipeline_mode": pipeline_mode,
        "top_name": top_name,
        "hls_artifacts": dict(hls_artifacts or {}),
        "hardware": hardware,
        "runtime_weights": weight_payload["summary"],
        "files": files,
        "notes": [
            "Runtime package records and copies existing artifacts only.",
            "It does not run Vivado, deploy to hardware, or infer missing bitstream/XSA/HWH files.",
        ],
    }

    manifest_path = package_dir / "package_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    readme = package_dir / "README_RUNTIME.md"
    readme.write_text(
        "\n".join(
            [
                "# FPGAI Runtime Package",
                "",
                "This package contains runtime-facing artifacts copied from an FPGAI compile output.",
                "",
                f"- board: `{board}`",
                f"- pipeline_mode: `{pipeline_mode}`",
                f"- top_name: `{top_name}`",
                f"- bitstream present: `{hardware['bitstream']['present']}`",
                f"- hwh present: `{hardware['hwh']['present']}`",
                f"- xsa present: `{hardware['xsa']['present']}`",
                f"- deployable overlay present: `{hardware['deployable_overlay_present']}`",
                f"- runtime weight payload required: `{weight_payload['summary']['required']}`",
                f"- runtime weight payload present: `{weight_payload['summary']['present']}`",
                "",
                "The package is truthful: missing hardware handoff files are recorded as missing.",
                "Use the Vivado bridge flow to generate bitstream/XSA artifacts before board deployment.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "path": "runtime_package/package_manifest.json",
        "package_dir": "runtime_package",
        "status": payload["status"],
        "deployable_overlay_present": hardware["deployable_overlay_present"],
        "bitstream_present": hardware["bitstream"]["present"],
        "hwh_present": hardware["hwh"]["present"],
        "xsa_present": hardware["xsa"]["present"],
        "runtime_weight_payload_required": weight_payload["summary"]["required"],
        "runtime_weight_payload_present": weight_payload["summary"]["present"],
        "runtime_weight_total_words": weight_payload["summary"]["total_words"],
        "file_count": len(files),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create an FPGAI runtime package from a compile output directory.")
    parser.add_argument("out_dir")
    parser.add_argument("--board")
    parser.add_argument("--pipeline-mode")
    parser.add_argument("--top-name")
    ns = parser.parse_args(argv)

    result = emit_runtime_package(
        ns.out_dir,
        board=ns.board,
        pipeline_mode=ns.pipeline_mode,
        top_name=ns.top_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
