from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("paper_experiments/full_pipeline_gate/sprint26_paper_matrix")
MATRIX = ROOT / "paper_design_matrix.json"
OUT_DIR = ROOT / "configs"
MANIFEST_JSON = ROOT / "generated_config_manifest.json"
MANIFEST_CSV = ROOT / "generated_config_manifest.csv"
MANIFEST_MD = ROOT / "generated_config_manifest.md"

INFERENCE_BASE = Path("configs/examples/inference_compile.yml")
TRAINING_BASE = Path("configs/examples/training_compile_smoke.yml")


PRECISION_DEFAULTS: dict[str, dict[str, dict[str, Any]]] = {
    "fx8_3": {
        "activation": {"type": "ap_fixed", "total_bits": 8, "int_bits": 3},
        "weight": {"type": "ap_fixed", "total_bits": 8, "int_bits": 3},
        "bias": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
        "accum": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
    },
    "fx12_4": {
        "activation": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
        "weight": {"type": "ap_fixed", "total_bits": 12, "int_bits": 4},
        "bias": {"type": "ap_fixed", "total_bits": 20, "int_bits": 8},
        "accum": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
    },
    "fx16_6": {
        "activation": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
        "weight": {"type": "ap_fixed", "total_bits": 16, "int_bits": 6},
        "bias": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
        "accum": {"type": "ap_fixed", "total_bits": 24, "int_bits": 10},
    },
}


BOARD_PARTS = {
    "pynq_z2": "xc7z020clg400-1",
    "kv260": "xck26-sfvc784-2LV-c",
    "kr260": "xck26-sfvc784-2LV-c",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not load as a YAML mapping")
    return data


def _set_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = cfg
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _parse_tile(tile: str) -> dict[str, int]:
    parts = [int(x) for x in str(tile).lower().split("x")]
    if len(parts) != 3:
        raise ValueError(f"Expected tile AxBxC, got {tile!r}")
    tm, tn, tk = parts
    return {"tm": tm, "tn": tn, "tk": tk}


def _policy_for(point: dict[str, Any]) -> str:
    if str(point["pipeline_style"]).lower() == "aggressive" or int(point["pe"]) >= 8:
        return "Latency-First"
    if int(point["pe"]) <= 2:
        return "Balanced"
    return "Balanced"


def _apply_common(cfg: dict[str, Any], point: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)

    name = str(point["name"])
    board = str(point["board"])
    precision = str(point["precision"])
    mode = str(point["mode"])
    storage = str(point["weight_storage"])

    if precision not in PRECISION_DEFAULTS:
        raise KeyError(f"Unknown precision mode {precision!r}")

    # Project/output location.
    _set_path(cfg, "project.name", name)
    _set_path(cfg, "project.out_dir", str(ROOT / "runs" / name))
    _set_path(cfg, "project.clean", True)

    # Existing configs use pipeline.mode, not project.mode.
    _set_path(cfg, "pipeline.mode", mode)
    _set_path(cfg, "pipeline.outputs.top_kernel_name", "deeplearn")

    # Board and part.
    _set_path(cfg, "targets.platform.board", board)
    _set_path(cfg, "targets.platform.part", BOARD_PARTS.get(board, BOARD_PARTS["kv260"]))
    _set_path(
        cfg,
        "targets.platform.clocks",
        [{"name": "pl_clk0", "target_mhz": 100}],
    )

    # Keep board/clock on the schema-supported targets.platform path.
    # Do not add unsupported top-level target.* keys because config validation
    # should reject unknown sections instead of silently accepting them.

    # Precision.
    _set_path(cfg, "numerics.precision_mode", precision)
    _set_path(cfg, "numerics.defaults", copy.deepcopy(PRECISION_DEFAULTS[precision]))

    # Hardware knobs.
    _set_path(cfg, "optimization.parallel_policy", _policy_for(point))
    _set_path(cfg, "optimization.parallel.pe", int(point["pe"]))
    _set_path(cfg, "optimization.parallel.simd", int(point["simd"]))
    _set_path(cfg, "optimization.parallel.unroll_factor", int(point["unroll"]))
    _set_path(cfg, "optimization.parallel.partition_factor", int(point["partition"]))
    _set_path(cfg, "optimization.parallel.array_partition_mode", str(point["partition_mode"]))

    _set_path(cfg, "optimization.pipeline.style", str(point["pipeline_style"]))
    _set_path(cfg, "optimization.pipeline.ii", int(point["ii"]))

    _set_path(cfg, "optimization.tiling.dense", _parse_tile(str(point["dense_tile"])))
    _set_path(cfg, "optimization.tiling.conv", _parse_tile(str(point["conv_tile"])))

    _set_path(cfg, "memory.weight_storage", storage)
    _set_path(cfg, "memory.allow_double_buffer", bool(point["double_buffer"]))
    _set_path(cfg, "memory.storage.weights", storage)
    _set_path(cfg, "memory.storage.activations", "bram")
    _set_path(cfg, "memory.storage.gradients", storage if mode == "training_on_device" else "bram")
    _set_path(cfg, "memory.storage.optimizer_state", storage if mode == "training_on_device" else "bram")

    # Keep training storage aligned for training cases so there is no hidden contradiction.
    if mode == "training_on_device":
        _set_path(cfg, "training.storage.weights", storage)
        _set_path(cfg, "training.storage.gradients", storage)
        _set_path(cfg, "training.storage.optimizer_state", storage)
        _set_path(cfg, "training.storage.activations", "bram")

    # Stage 1 is prediction/codegen only. Do not add unsupported top-level
    # vivado.* keys here; Vivado subset configs will be generated by a later
    # tool using schema-supported flow controls.

    # Keep HLS source/codegen enabled. Real csynth is a later selected subset.
    _set_path(cfg, "backends.hls.enabled", True)
    _set_path(cfg, "backends.hls.vitis.enabled", True)
    _set_path(cfg, "backends.hls.vitis.mode", "csim")

    # Trace paper intent under the existing project section so validation
    # remains strict and unknown top-level sections are not hidden.
    _set_path(cfg, "project.paper_experiment_name", name)
    _set_path(cfg, "project.paper_experiment_stage", str(point["stage"]))
    _set_path(cfg, "project.paper_experiment_purpose", str(point["purpose"]))
    _set_path(cfg, "project.paper_experiment_source_matrix", str(MATRIX))

    return cfg


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    points = json.loads(MATRIX.read_text(encoding="utf-8"))
    if not isinstance(points, list):
        raise TypeError("paper design matrix must be a list")

    inference_base = _load_yaml(INFERENCE_BASE)
    training_base = _load_yaml(TRAINING_BASE)

    manifest: list[dict[str, Any]] = []

    for point in points:
        if point["mode"] == "training_on_device":
            base = training_base
            base_config = str(TRAINING_BASE)
        else:
            base = inference_base
            base_config = str(INFERENCE_BASE)

        cfg = _apply_common(base, point)

        out_path = OUT_DIR / f"{point['name']}.yml"
        out_path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, width=100),
            encoding="utf-8",
        )

        manifest.append(
            {
                "name": point["name"],
                "config_path": str(out_path),
                "base_config": base_config,
                "stage": point["stage"],
                "mode": point["mode"],
                "board": point["board"],
                "precision": point["precision"],
                "pe": point["pe"],
                "simd": point["simd"],
                "unroll": point["unroll"],
                "partition": point["partition"],
                "pipeline_style": point["pipeline_style"],
                "ii": point["ii"],
                "dense_tile": point["dense_tile"],
                "conv_tile": point["conv_tile"],
                "weight_storage": point["weight_storage"],
                "double_buffer": point["double_buffer"],
                "purpose": point["purpose"],
            }
        )

    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    lines = [
        "# Generated paper configs",
        "",
        f"Generated configs: {len(manifest)}",
        "",
        "| name | mode | board | precision | knobs | config |",
        "|---|---|---|---|---|---|",
    ]
    for row in manifest:
        knobs = (
            f"pe={row['pe']}, simd={row['simd']}, unroll={row['unroll']}, "
            f"part={row['partition']}, {row['pipeline_style']} II={row['ii']}, "
            f"mem={row['weight_storage']}"
        )
        lines.append(
            f"| `{row['name']}` | {row['mode']} | {row['board']} | {row['precision']} | "
            f"{knobs} | `{row['config_path']}` |"
        )

    MANIFEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] generated configs: {len(manifest)}")
    print(f"[OK] wrote {MANIFEST_JSON}")
    print(f"[OK] wrote {MANIFEST_CSV}")
    print(f"[OK] wrote {MANIFEST_MD}")
    print(f"[OK] config dir: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
