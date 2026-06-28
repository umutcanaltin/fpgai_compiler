from __future__ import annotations

import csv
import json
from pathlib import Path


OUT = Path("paper_experiments/full_pipeline_gate/sprint26_paper_matrix")
OUT.mkdir(parents=True, exist_ok=True)


def design_points():
    boards = ["pynq_z2", "kv260", "kr260"]

    points = []

    # Safe board-aware baselines.
    for board in boards:
        points.append(
            {
                "name": f"{board}_baseline_safe_fx16",
                "stage": "prediction+hls_candidate",
                "mode": "inference",
                "board": board,
                "precision": "fx16_6",
                "pe": 2,
                "simd": 2,
                "unroll": 2,
                "partition": 2,
                "partition_mode": "cyclic",
                "pipeline_style": "balanced",
                "ii": 2,
                "dense_tile": "8x8x8",
                "conv_tile": "4x4x4",
                "weight_storage": "bram",
                "double_buffer": False,
                "purpose": "safe baseline for each board",
            }
        )

    # Precision effect on same hardware/policy.
    for precision in ["fx16_6", "fx12_4", "fx8_3"]:
        points.append(
            {
                "name": f"kv260_precision_{precision}",
                "stage": "prediction+hls_candidate",
                "mode": "inference",
                "board": "kv260",
                "precision": precision,
                "pe": 4,
                "simd": 4,
                "unroll": 4,
                "partition": 4,
                "partition_mode": "cyclic",
                "pipeline_style": "balanced",
                "ii": 2,
                "dense_tile": "16x16x16",
                "conv_tile": "8x8x8",
                "weight_storage": "bram",
                "double_buffer": False,
                "purpose": "precision effect",
            }
        )

    # Parallelism effect.
    for factor in [1, 2, 4, 8]:
        points.append(
            {
                "name": f"kv260_parallel_x{factor}",
                "stage": "prediction+hls_candidate",
                "mode": "inference",
                "board": "kv260",
                "precision": "fx16_6",
                "pe": factor,
                "simd": factor,
                "unroll": factor,
                "partition": factor,
                "partition_mode": "cyclic" if factor < 8 else "complete",
                "pipeline_style": "balanced",
                "ii": 2,
                "dense_tile": "16x16x16",
                "conv_tile": "8x8x8",
                "weight_storage": "bram",
                "double_buffer": False,
                "purpose": "parallelism scaling effect",
            }
        )

    # Pipeline effect.
    for style, ii in [("balanced", 2), ("aggressive", 1)]:
        points.append(
            {
                "name": f"kv260_pipeline_{style}_ii{ii}",
                "stage": "prediction+hls_candidate",
                "mode": "inference",
                "board": "kv260",
                "precision": "fx16_6",
                "pe": 4,
                "simd": 4,
                "unroll": 4,
                "partition": 4,
                "partition_mode": "cyclic",
                "pipeline_style": style,
                "ii": ii,
                "dense_tile": "16x16x16",
                "conv_tile": "8x8x8",
                "weight_storage": "bram",
                "double_buffer": False,
                "purpose": "pipeline II effect",
            }
        )

    # Tiling effect.
    for label, dense, conv in [
        # The paper MLP is intentionally small. These values are chosen so
        # small/medium/large remain distinct after per-layer dimension
        # clamping in the planner/codegen path.
        ("small", "1x1x1", "1x1x1"),
        ("medium", "2x2x2", "2x2x2"),
        ("large", "4x4x4", "4x4x4"),
    ]:
        points.append(
            {
                "name": f"kv260_tiling_{label}",
                "stage": "prediction+hls_candidate",
                "mode": "inference",
                "board": "kv260",
                "precision": "fx16_6",
                "pe": 4,
                "simd": 4,
                "unroll": 4,
                "partition": 4,
                "partition_mode": "cyclic",
                "pipeline_style": "balanced",
                "ii": 2,
                "dense_tile": dense,
                "conv_tile": conv,
                "weight_storage": "bram",
                "double_buffer": False,
                "purpose": "tiling effect",
            }
        )

    # Memory storage effect.
    for storage in ["bram", "uram"]:
        points.append(
            {
                "name": f"kv260_memory_{storage}",
                "stage": "prediction+hls_candidate",
                "mode": "inference",
                "board": "kv260",
                "precision": "fx16_6",
                "pe": 4,
                "simd": 4,
                "unroll": 4,
                "partition": 4,
                "partition_mode": "cyclic",
                "pipeline_style": "balanced",
                "ii": 2,
                "dense_tile": "16x16x16",
                "conv_tile": "8x8x8",
                "weight_storage": storage,
                "double_buffer": storage == "uram",
                "purpose": "BRAM vs URAM effect",
            }
        )

    # Combined aggressive.
    points.append(
        {
            "name": "kv260_combined_aggressive_fx8",
            "stage": "prediction+hls+vivado_candidate",
            "mode": "inference",
            "board": "kv260",
            "precision": "fx8_3",
            "pe": 8,
            "simd": 8,
            "unroll": 8,
            "partition": 8,
            "partition_mode": "complete",
            "pipeline_style": "aggressive",
            "ii": 1,
            "dense_tile": "32x32x32",
            "conv_tile": "16x16x16",
            "weight_storage": "uram",
            "double_buffer": True,
            "purpose": "combined aggressive design",
        }
    )

    # Training points.
    for label, factor, precision, storage, ii in [
        ("safe", 2, "fx16_6", "bram", 2),
        ("aggressive", 8, "fx8_3", "uram", 1),
    ]:
        points.append(
            {
                "name": f"training_kv260_{label}_{precision}",
                "stage": "prediction+hls+vivado_candidate",
                "mode": "training_on_device",
                "board": "kv260",
                "precision": precision,
                "pe": factor,
                "simd": factor,
                "unroll": factor,
                "partition": factor,
                "partition_mode": "cyclic" if factor < 8 else "complete",
                "pipeline_style": "balanced" if ii == 2 else "aggressive",
                "ii": ii,
                "dense_tile": "8x8x8" if factor == 2 else "32x32x32",
                "conv_tile": "4x4x4" if factor == 2 else "16x16x16",
                "weight_storage": storage,
                "double_buffer": storage == "uram",
                "purpose": "training hardware decision effect",
            }
        )

    return points


def main() -> int:
    points = design_points()

    json_path = OUT / "paper_design_matrix.json"
    csv_path = OUT / "paper_design_matrix.csv"
    md_path = OUT / "paper_design_matrix.md"

    json_path.write_text(json.dumps(points, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0].keys()))
        writer.writeheader()
        writer.writerows(points)

    lines = [
        "# FPGAI paper design matrix",
        "",
        "This matrix is intentionally stratified. It does not run every Cartesian combination.",
        "The goal is to isolate each hardware decision and then include a few combined designs.",
        "",
        f"Total design points: {len(points)}",
        "",
        "| name | stage | mode | board | precision | pe/simd/unroll/partition | pipeline | tiling | memory | purpose |",
        "|---|---|---|---|---|---:|---|---|---|---|",
    ]

    for p in points:
        lines.append(
            f"| `{p['name']}` | {p['stage']} | {p['mode']} | {p['board']} | {p['precision']} | "
            f"{p['pe']}/{p['simd']}/{p['unroll']}/{p['partition']} | "
            f"{p['pipeline_style']} II={p['ii']} | dense {p['dense_tile']}, conv {p['conv_tile']} | "
            f"{p['weight_storage']}, db={p['double_buffer']} | {p['purpose']} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {md_path}")
    print(f"[OK] design points: {len(points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
