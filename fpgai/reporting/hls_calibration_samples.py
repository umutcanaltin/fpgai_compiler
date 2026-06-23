#!/usr/bin/env python3
"""Collect and aggregate FPGAI HLS calibration sample datasets.

Sprint 10B helper.

Two collection modes are supported:

1. Existing dataset mode: collect already-written ``estimate_vs_hls.json`` files.
2. Expanded project-root mode: find every ``build/calibration/compile_plan_for_calibration.json``
   and rebuild the calibration dataset from that design's own HLS reports.

Expanded mode is useful after Sprint 9 because every materialized design has its
own preserved build directory. It avoids the case where cached
``estimate_vs_hls.json`` files are identical or too small and therefore collapse
back to the original five operator samples.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

DEFAULT_OUTPUT = Path("experiments/hls_calibration_dataset_aggregated/estimate_vs_hls_aggregated.json")
DEFAULT_DATASET_NAME = "estimate_vs_hls.json"
DEFAULT_COMPILE_PLAN_NAME = "compile_plan_for_calibration.json"

DatasetBuilder = Callable[[Path], dict[str, Any]]


def discover_dataset_paths(roots: Iterable[str | Path], *, dataset_name: str = DEFAULT_DATASET_NAME) -> list[Path]:
    """Return sorted unique calibration dataset paths under the given roots."""
    found: set[Path] = set()
    for root_value in roots:
        root = Path(root_value)
        if not root.exists():
            continue
        if root.is_file():
            if root.name == dataset_name:
                found.add(root.resolve())
            continue
        for path in root.rglob(dataset_name):
            if path.is_file():
                # Avoid collecting this script's own aggregated output when
                # scanning experiments/ again.
                if "hls_calibration_dataset_aggregated" in path.parts:
                    continue
                found.add(path.resolve())
    return sorted(found, key=lambda p: str(p))


def discover_project_roots(
    roots: Iterable[str | Path], *, compile_plan_name: str = DEFAULT_COMPILE_PLAN_NAME
) -> list[Path]:
    """Return build/project roots that can be rebuilt into calibration datasets.

    A root is inferred from ``<project_root>/calibration/compile_plan_for_calibration.json``.
    """
    found: set[Path] = set()
    for root_value in roots:
        root = Path(root_value)
        if not root.exists():
            continue
        candidates: list[Path]
        if root.is_file():
            candidates = [root] if root.name == compile_plan_name else []
        else:
            candidates = [p for p in root.rglob(compile_plan_name) if p.is_file()]
        for compile_plan in candidates:
            if compile_plan.parent.name == "calibration":
                found.add(compile_plan.parent.parent.resolve())
    return sorted(found, key=lambda p: str(p))


def load_dataset(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset is not a JSON object: {path}")
    return data


def build_dataset_from_project_root(project_root: str | Path) -> dict[str, Any]:
    """Rebuild one calibration dataset from a design build/project root."""
    from fpgai.analysis.hls_calibration_dataset import build_calibration_dataset

    project_root = Path(project_root)
    compile_plan = project_root / "calibration" / DEFAULT_COMPILE_PLAN_NAME
    if not compile_plan.exists():
        raise FileNotFoundError(compile_plan)

    preferred_report_dir = project_root / "hls" / "fpgai_hls_proj" / "sol1" / "syn" / "report"
    hls_report_dir = preferred_report_dir if preferred_report_dir.exists() else project_root / "hls"
    design_name = infer_design_name(project_root)

    return build_calibration_dataset(
        compile_plan,
        hls_report_dir,
        output_path=None,
        project=design_name or project_root.name,
    )


def aggregate_datasets(paths: Iterable[str | Path], *, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Merge samples from multiple estimate_vs_hls datasets."""
    repo_root_path = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    unique_paths = _unique_resolved_paths(paths)

    samples: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for path in unique_paths:
        try:
            data = load_dataset(path)
        except Exception as exc:
            warnings.append({"source": _safe_relative(path, repo_root_path), "warning": f"load_failed: {exc}"})
            continue
        _append_dataset_samples(
            data,
            source_id=_safe_relative(path, repo_root_path),
            source_path=path,
            samples=samples,
            sources=sources,
            warnings=warnings,
            repo_root=repo_root_path,
        )

    return _make_aggregate_payload(samples, sources, warnings)


def aggregate_project_roots(
    project_roots: Iterable[str | Path],
    *,
    repo_root: str | Path | None = None,
    builder: DatasetBuilder | None = None,
) -> dict[str, Any]:
    """Rebuild and merge datasets from project roots containing HLS reports."""
    repo_root_path = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    unique_roots = _unique_resolved_paths(project_roots)
    build = builder or build_dataset_from_project_root

    samples: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for project_root in unique_roots:
        try:
            data = build(project_root)
        except Exception as exc:
            warnings.append({"source": _safe_relative(project_root, repo_root_path), "warning": f"rebuild_failed: {exc}"})
            continue
        _append_dataset_samples(
            data,
            source_id=_safe_relative(project_root, repo_root_path),
            source_path=project_root,
            samples=samples,
            sources=sources,
            warnings=warnings,
            repo_root=repo_root_path,
        )

    payload = _make_aggregate_payload(samples, sources, warnings)
    payload["collection_mode"] = "expanded_project_roots"
    return payload


def _append_dataset_samples(
    data: dict[str, Any],
    *,
    source_id: str,
    source_path: Path,
    samples: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    repo_root: Path,
) -> None:
    source_samples = data.get("samples", [])
    if not isinstance(source_samples, list):
        warnings.append({"source": source_id, "warning": "samples_not_list"})
        return

    design_name = infer_design_name(source_path)
    source_record: dict[str, Any] = {
        "path": source_id,
        "sample_count": len(source_samples),
    }
    if design_name:
        source_record["design_name"] = design_name
    for key in ("project", "board", "part", "clock_target_mhz"):
        if data.get(key) is not None:
            source_record[key] = data.get(key)
    sources.append(source_record)

    for index, sample in enumerate(source_samples):
        if not isinstance(sample, dict):
            warnings.append({"source": source_id, "sample_index": index, "warning": "sample_not_object"})
            continue
        merged_sample = dict(sample)
        merged_sample.setdefault("features", {})
        if isinstance(merged_sample["features"], dict):
            merged_sample["features"] = dict(merged_sample["features"])
            merged_sample["features"].setdefault("source_dataset", source_id)
            if design_name:
                merged_sample["features"].setdefault("design_name", design_name)
            for key in ("board", "part", "clock_target_mhz"):
                if data.get(key) is not None:
                    merged_sample["features"].setdefault(key, data.get(key))
        merged_sample["source_dataset"] = source_id
        if design_name:
            merged_sample["source_design"] = design_name
        samples.append(merged_sample)


def _make_aggregate_payload(
    samples: list[dict[str, Any]], sources: list[dict[str, Any]], warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    operator_counts = Counter(str(sample.get("operator", "Unknown")) for sample in samples)
    metric_names = _infer_metric_names(samples)
    return {
        "schema_version": 1,
        "format": "fpgai.hls_calibration_dataset.aggregate.v2",
        "project": "aggregated_hls_calibration_dataset",
        "collection_mode": "existing_datasets",
        "source_count": len(sources),
        "sample_count": len(samples),
        "operator_counts": dict(sorted(operator_counts.items())),
        "metrics": metric_names,
        "sources": sources,
        "samples": samples,
        "warnings": warnings,
    }


def write_aggregated_dataset(dataset: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def infer_design_name(path: str | Path) -> str | None:
    """Infer design name from experiments/<exp>/artifacts/<design>/... paths."""
    parts = Path(path).parts
    for index, part in enumerate(parts):
        if part == "artifacts" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _infer_metric_names(samples: list[dict[str, Any]]) -> list[str]:
    metrics: set[str] = set()
    for sample in samples:
        for section in ("estimated", "hls_actual", "calibrated_estimate"):
            data = sample.get(section, {}) or {}
            if isinstance(data, dict):
                metrics.update(str(key) for key in data.keys())
    preferred = ["lut", "ff", "dsp", "bram", "latency_cycles"]
    ordered = [metric for metric in preferred if metric in metrics]
    ordered.extend(sorted(metrics.difference(ordered)))
    return ordered


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _unique_resolved_paths(paths: Iterable[str | Path]) -> list[Path]:
    unique_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique_paths.append(path)
    return unique_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate FPGAI HLS calibration datasets.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["build", "experiments"],
        help="Root directories or explicit dataset/project files to scan. Default: build experiments",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset file name to collect in existing-dataset mode. Default: estimate_vs_hls.json",
    )
    parser.add_argument(
        "--expand-project-roots",
        action="store_true",
        help="Rebuild samples from every calibration/compile_plan_for_calibration.json and HLS report tree instead of only reading cached estimate_vs_hls.json files.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used for relative source paths. Default: current directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.expand_project_roots:
        project_roots = discover_project_roots(args.roots)
        if not project_roots:
            print(f"[FAIL] No calibration/{DEFAULT_COMPILE_PLAN_NAME} files found under: {', '.join(args.roots)}")
            return 2
        dataset = aggregate_project_roots(project_roots, repo_root=args.repo_root)
    else:
        paths = discover_dataset_paths(args.roots, dataset_name=args.dataset_name)
        if not paths:
            print(f"[FAIL] No {args.dataset_name} files found under: {', '.join(args.roots)}")
            return 2
        dataset = aggregate_datasets(paths, repo_root=args.repo_root)

    output = write_aggregated_dataset(dataset, args.out)

    print(f"[OK] Aggregated HLS calibration dataset: {output}")
    print(f"Mode: {dataset.get('collection_mode', 'unknown')}")
    print(f"Sources: {dataset['source_count']}")
    print(f"Samples: {dataset['sample_count']}")
    print(f"Operators: {', '.join(f'{k}={v}' for k, v in dataset['operator_counts'].items())}")
    if dataset.get("warnings"):
        print(f"Warnings: {len(dataset['warnings'])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
