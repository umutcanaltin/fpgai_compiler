#!/usr/bin/env python3
"""Create a focused Sprint 9 sweep that compares real precision modes.

The normal inference_precision sweep can enumerate all policies for fx8_3 first.
When run with --max-design-points 4, that gives policy variation but no precision
variation. Sprint 9 needs at least two precision modes so numerics.defaults and
ap_fixed widths can be checked.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Run this inside the FPGAI venv.") from exc


DEFAULT_PRECISIONS = ["fx8_3", "fx16_6"]
DEFAULT_POLICIES = ["balanced", "latency_first"]


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML in {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _find_parameters_mapping(sweep: dict[str, Any]) -> dict[str, Any]:
    params = sweep.get("parameters")
    if isinstance(params, dict):
        return params
    # Some older sweep schemas nest design parameters under matrix/parameters.
    matrix = sweep.get("matrix")
    if isinstance(matrix, dict) and isinstance(matrix.get("parameters"), dict):
        return matrix["parameters"]
    raise ValueError("Could not find a parameters mapping in the sweep YAML")


def create_precision_effect_sweep(
    input_path: Path,
    output_path: Path,
    precisions: list[str] | None = None,
    policies: list[str] | None = None,
) -> dict[str, Any]:
    sweep = _load_yaml(input_path)
    focused = copy.deepcopy(sweep)
    params = _find_parameters_mapping(focused)

    precisions = precisions or DEFAULT_PRECISIONS
    policies = policies or DEFAULT_POLICIES

    if "precision_mode" not in params:
        raise ValueError("Input sweep has no precision_mode parameter")
    if "policy" not in params:
        raise ValueError("Input sweep has no policy parameter")

    params["precision_mode"] = precisions
    params["policy"] = policies

    # Keep the experiment small and explicit. Do not add unsupported schema keys.
    focused["name"] = focused.get("name", "inference_precision")
    focused["description"] = (
        "Sprint 9 focused precision-effect sweep: compares fx8_3 and fx16_6 "
        "under balanced and latency_first policies."
    )

    _write_yaml(output_path, focused)
    return focused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("configs/sweeps/inference_precision.yml"),
        help="Existing inference precision sweep to copy and narrow.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("configs/sweeps/sprint9_precision_effect.yml"),
        help="Output focused Sprint 9 sweep path.",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=DEFAULT_PRECISIONS,
        help="Precision modes to include.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=DEFAULT_POLICIES,
        help="Policies to include.",
    )
    args = parser.parse_args(argv)

    focused = create_precision_effect_sweep(args.input, args.out, args.precisions, args.policies)
    params = _find_parameters_mapping(focused)
    print(f"[OK] Wrote focused Sprint 9 sweep: {args.out}")
    print(f"precision_mode: {params['precision_mode']}")
    print(f"policy: {params['policy']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
