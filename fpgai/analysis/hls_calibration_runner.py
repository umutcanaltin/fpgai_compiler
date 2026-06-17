"""Compiler/benchmark integration glue for Sprint 5 HLS calibration.

This module is intentionally best-effort. It should never break normal FPGAI
compilation unless the user explicitly sets:

    analysis.hls_calibration.fail_on_empty: true

It writes the Sprint 5 artifacts under:

    <out_dir>/<output_dir>/hls_operator_dataset.json
    <out_dir>/<output_dir>/calibrated_model.json
    <out_dir>/<output_dir>/estimate_vs_hls.json
    <out_dir>/<output_dir>/summary.txt

Default output_dir is ``calibration``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any
import json
import os
import re
import traceback

from fpgai.analysis.hls_calibration_dataset import build_calibration_dataset
from fpgai.analysis.hls_calibration_model import fit_calibration_model
from fpgai.analysis.hls_estimate_report import write_estimate_vs_hls_report


@dataclass(frozen=True)
class HLSCalibrationResult:
    out_dir: Path
    dataset_json: Path | None = None
    calibrated_model_json: Path | None = None
    estimate_vs_hls_json: Path | None = None
    summary_txt: Path | None = None
    compile_plan_json: Path | None = None
    sample_count: int = 0
    skipped: bool = False
    reason: str | None = None


def run_hls_calibration(
    *,
    out_dir: str | Path,
    raw_cfg: dict[str, Any] | None = None,
    compile_plan: Any | None = None,
    hls_report_dir: str | Path | None = None,
    clock_mhz: float | None = None,
    verbose: bool = False,
) -> HLSCalibrationResult:
    """Run Sprint 5 calibration from inside the compiler flow.

    Parameters are deliberately broad so this can be called from both the
    inference compiler path and the benchmark path without changing their data
    structures.
    """
    raw_cfg = raw_cfg or {}
    cfg = _get_path(raw_cfg, "analysis.hls_calibration", {})
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}

    enabled = bool(cfg.get("enabled", True))
    output_dir_name = str(cfg.get("output_dir", "calibration"))
    fail_on_empty = bool(cfg.get("fail_on_empty", cfg.get("fail_on_missing_hls_report", False)))
    fail_on_error = bool(cfg.get("fail_on_error", False))

    out_dir = Path(out_dir)
    calibration_dir = out_dir / output_dir_name
    calibration_dir.mkdir(parents=True, exist_ok=True)

    if not enabled:
        reason = "analysis.hls_calibration.enabled is false"
        _write_skip(calibration_dir, reason)
        return HLSCalibrationResult(out_dir=calibration_dir, skipped=True, reason=reason)

    try:
        compile_plan_path = _materialize_compile_plan(
            out_dir=out_dir,
            calibration_dir=calibration_dir,
            compile_plan=compile_plan,
            raw_cfg=raw_cfg,
            clock_mhz=clock_mhz,
        )
        if compile_plan_path is None:
            reason = "no compile plan could be found or materialized"
            _write_skip(calibration_dir, reason)
            if fail_on_empty:
                raise RuntimeError(f"HLS calibration failed: {reason}")
            return HLSCalibrationResult(out_dir=calibration_dir, skipped=True, reason=reason)

        report_root = Path(hls_report_dir) if hls_report_dir is not None else out_dir
        dataset_json = calibration_dir / "hls_operator_dataset.json"
        calibrated_model_json = calibration_dir / "calibrated_model.json"
        estimate_vs_hls_json = calibration_dir / "estimate_vs_hls.json"
        summary_txt = calibration_dir / "summary.txt"

        dataset = build_calibration_dataset(
            compile_plan_path=compile_plan_path,
            hls_report_dir=report_root,
            output_path=dataset_json,
            clock_target_mhz=clock_mhz,
        )

        # Sprint 5.1: real FPGAI benchmark runs may already produce JSON
        # validation artifacts with predicted-vs-actual layer data, while the
        # compile_plan object passed here can be too high-level to contain layer
        # estimates. If the direct compile-plan path produced zero samples, scan
        # existing JSON artifacts under out_dir and convert any validation rows
        # into calibration samples.
        if not dataset.get("samples"):
            fallback = _try_build_dataset_from_validation_jsons(
                out_dir=out_dir,
                report_root=report_root,
                compile_plan_path=compile_plan_path,
                clock_mhz=clock_mhz,
            )
            if fallback.get("samples"):
                dataset = fallback
                dataset_json.write_text(json.dumps(dataset, indent=2, sort_keys=True), encoding="utf-8")

        model = fit_calibration_model(dataset)
        calibrated_model_json.write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
        report = write_estimate_vs_hls_report(dataset, model, estimate_vs_hls_json, summary_txt)

        sample_count = int(report.get("summary", {}).get("sample_count", len(dataset.get("samples", []))))
        if verbose:
            print(f"[FPGAI] hls_calibration: {summary_txt} ({sample_count} samples)")

        if sample_count == 0 and fail_on_empty:
            raise RuntimeError(
                "HLS calibration produced zero matched samples. "
                f"Compile plan: {compile_plan_path}; report root: {report_root}"
            )

        return HLSCalibrationResult(
            out_dir=calibration_dir,
            dataset_json=dataset_json,
            calibrated_model_json=calibrated_model_json,
            estimate_vs_hls_json=estimate_vs_hls_json,
            summary_txt=summary_txt,
            compile_plan_json=compile_plan_path,
            sample_count=sample_count,
            skipped=False,
        )

    except Exception as exc:
        error_path = calibration_dir / "error.txt"
        error_path.write_text(
            f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        if fail_on_error or fail_on_empty:
            raise
        if verbose:
            print(f"[FPGAI] hls_calibration skipped after error: {error_path}")
        return HLSCalibrationResult(
            out_dir=calibration_dir,
            skipped=True,
            reason=f"{type(exc).__name__}: {exc}",
        )



def _try_build_dataset_from_validation_jsons(
    *,
    out_dir: Path,
    report_root: Path,
    compile_plan_path: Path,
    clock_mhz: float | None,
) -> dict[str, Any]:
    """Build samples from existing FPGAI validation JSONs if available.

    This is a compatibility bridge for the current Sprint 4/Sprint 5 branch.
    Some runs print a good Layer-vs-HLS validation table but do not expose that
    same information under compile_plan.layers. This scanner accepts many JSON
    shapes and extracts rows containing both estimated/predicted and
    actual/HLS metrics.
    """
    from fpgai.analysis.hls_calibration_dataset import parse_hls_csynth_report

    samples: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    seen: set[str] = set()
    json_paths = []
    for root in (out_dir, report_root):
        if root.exists():
            json_paths.extend(root.rglob("*.json"))
    json_paths = sorted(set(p for p in json_paths if "calibration" not in p.parts))

    report_paths = _discover_report_paths(report_root)

    for path in json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in _walk_dict_rows(payload):
            sample = _row_to_sample(row, report_paths)
            if not sample:
                continue
            key = json.dumps({"layer": sample.get("layer_name"), "op": sample.get("operator"), "report": sample.get("hls_report")}, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            sample.setdefault("source_json", str(path))
            # If the row only carried actual metrics partially, enrich from the
            # matched HLS report.
            hls_report = sample.get("hls_report")
            if hls_report:
                try:
                    actual = parse_hls_csynth_report(hls_report)
                    current = sample.get("hls_actual", {})
                    for metric in ("lut", "ff", "dsp", "bram", "latency_cycles"):
                        if not current.get(metric) and actual.get(metric):
                            current[metric] = actual[metric]
                    sample["hls_actual"] = current
                except Exception as exc:
                    warnings.append({"warning": f"fallback_hls_parse_failed: {exc}", "hls_report": str(hls_report)})
            if _has_any_metric(sample.get("estimated")) and _has_any_metric(sample.get("hls_actual")):
                samples.append(sample)

    return {
        "schema_version": 1,
        "project": out_dir.name,
        "board": None,
        "part": None,
        "clock_target_mhz": clock_mhz,
        "samples": samples,
        "warnings": warnings,
        "debug": {
            "fallback": "validation_json_scan",
            "compile_plan_path": str(compile_plan_path),
            "json_files_scanned": len(json_paths),
            "hls_report_count": len(report_paths),
        },
    }


def _walk_dict_rows(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dict_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dict_rows(child)


def _row_to_sample(row: dict[str, Any], report_paths: list[Path]) -> dict[str, Any] | None:
    layer_name = str(row.get("layer_name") or row.get("layer") or row.get("name") or row.get("id") or row.get("node") or row.get("module") or "")
    operator = _normalize_operator_compat(str(row.get("operator") or row.get("op") or row.get("type") or row.get("kind") or layer_name))
    estimated = _extract_metric_group(row, prefixes=("pred", "predicted", "est", "estimated", "fpgai"), group_keys=("estimated", "estimate", "predicted", "prediction", "resource_estimate", "predicted_resources"))
    actual = _extract_metric_group(row, prefixes=("actual", "hls", "measured"), group_keys=("actual", "hls_actual", "hls", "measured", "actual_resources"))
    if not _has_any_metric(estimated):
        return None
    hls_report = str(row.get("hls_report") or row.get("report") or row.get("module_report") or "")
    if not hls_report:
        matched = _match_report_compat(layer_name, operator, report_paths, module=str(row.get("module") or row.get("hls_module") or ""))
        hls_report = str(matched) if matched else ""
    return {
        "operator": operator or "Unknown",
        "layer_name": layer_name or operator or "unknown",
        "estimated": _normalize_metric_names(estimated),
        "hls_actual": _normalize_metric_names(actual),
        "features": {},
        "hls_report": hls_report,
    }


def _extract_metric_group(row: dict[str, Any], *, prefixes: tuple[str, ...], group_keys: tuple[str, ...]) -> dict[str, Any]:
    for key in group_keys:
        value = row.get(key)
        if isinstance(value, dict):
            normalized = _normalize_metric_names(value)
            if _has_any_metric(normalized):
                return normalized
    out: dict[str, Any] = {}
    for metric in ("lut", "ff", "dsp", "bram", "latency_cycles"):
        aliases = _metric_aliases_compat(metric)
        for alias in aliases:
            for prefix in prefixes:
                keys = (
                    f"{prefix}_{alias}",
                    f"{prefix}{alias}",
                    f"{alias}_{prefix}",
                )
                for key in keys:
                    if key in row:
                        out[metric] = row[key]
                        break
                if metric in out:
                    break
            if metric in out:
                break
    return out


def _normalize_metric_names(data: dict[str, Any] | None) -> dict[str, float]:
    data = data or {}
    return {
        "lut": _to_float_compat(_first_key(data, "lut", "LUT", "LUTs")),
        "ff": _to_float_compat(_first_key(data, "ff", "FF", "FFs")),
        "dsp": _to_float_compat(_first_key(data, "dsp", "DSP", "DSP48E")),
        "bram": _to_float_compat(_first_key(data, "bram", "bram18", "BRAM", "BRAM18", "BRAM_18K", "bram_18k")),
        "latency_cycles": _to_float_compat(_first_key(data, "latency_cycles", "cycles", "latency", "Latency")),
    }


def _discover_report_paths(root: Path) -> list[Path]:
    reports: list[Path] = []
    if root.exists():
        for pattern in ("*_csynth.rpt", "csynth.rpt", "*_csynth.xml", "csynth.xml", "*.xml"):
            reports.extend(root.rglob(pattern))
    def score(path: Path) -> tuple[int, str]:
        key = re.sub(r"[^a-z0-9]+", "", path.name.lower())
        penalty = 0
        if "pipeline" in key or "vitisloop" in key:
            penalty += 100
        if key in {"csynthrpt", "csynthxml"}:
            penalty += 80
        if "deeplearn" in key:
            penalty += 20
        return (penalty, str(path))
    return sorted(set(reports), key=score)


def _match_report_compat(layer_name: str, operator: str, reports: list[Path], *, module: str = "") -> Path | None:
    keys = []
    for value in (module, layer_name, operator):
        safe = re.sub(r"[^a-z0-9]+", "", str(value).lower())
        if safe:
            keys.append(safe)
    op_l = operator.lower()
    layer_l = layer_name.lower()
    if "conv" in op_l or layer_l.startswith("conv"):
        keys.extend(["conv2d", "conv"])
    if "dense" in op_l or layer_l.startswith("dense"):
        keys.extend(["denseoutintiled", "denseoutin", "dense"])
    if "pool" in op_l or "pool" in layer_l:
        keys.extend(["maxpool2d", "maxpool"])
    if "relu" in op_l:
        keys.append("relu")
    if "softmax" in op_l:
        keys.append("softmax")
    primary = [p for p in reports if "pipeline" not in p.name.lower() and "vitis" not in p.name.lower()]
    for collection in (primary, reports):
        for key in keys:
            for report in collection:
                if key and key in re.sub(r"[^a-z0-9]+", "", report.stem.lower()):
                    return report
    return None


def _normalize_operator_compat(value: str) -> str:
    v = str(value).lower()
    if "conv" in v:
        return "Conv"
    if "dense" in v or "linear" in v or "gemm" in v:
        return "Dense"
    if "pool" in v:
        return "MaxPool"
    if "relu" in v:
        return "ReLU"
    if "softmax" in v:
        return "Softmax"
    return str(value) if value else "Unknown"


def _metric_aliases_compat(metric: str) -> tuple[str, ...]:
    if metric == "bram":
        return ("bram", "bram18", "bram_18k", "BRAM", "BRAM18", "BRAM_18K")
    if metric == "latency_cycles":
        return ("latency_cycles", "cycles", "latency", "Latency")
    return (metric, metric.upper())


def _first_key(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    lower = {str(k).lower(): v for k, v in data.items()}
    for key in keys:
        if key.lower() in lower:
            return lower[key.lower()]
    return 0


def _to_float_compat(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?[0-9]+(?:\.[0-9]+)?", str(value).replace(",", ""))
    return float(m.group(0)) if m else 0.0


def _has_any_metric(data: dict[str, Any] | None) -> bool:
    data = data or {}
    return any(_to_float_compat(data.get(metric)) > 0 for metric in ("lut", "ff", "dsp", "bram", "latency_cycles"))


def _materialize_compile_plan(
    *,
    out_dir: Path,
    calibration_dir: Path,
    compile_plan: Any | None,
    raw_cfg: dict[str, Any],
    clock_mhz: float | None,
) -> Path | None:
    """Return a compile-plan JSON path usable by build_calibration_dataset."""
    if compile_plan is not None:
        payload = _compile_plan_to_payload(compile_plan)
        if isinstance(payload, dict):
            payload.setdefault("project", _get_path(raw_cfg, "project.name", out_dir.name))
            payload.setdefault("board", _get_path(raw_cfg, "targets.platform.board", None))
            payload.setdefault("part", _get_path(raw_cfg, "targets.platform.part", None))
            payload.setdefault("clock_target_mhz", clock_mhz)
            path = calibration_dir / "compile_plan_for_calibration.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            return path

    found = _find_existing_compile_plan(out_dir)
    return found


def _find_existing_compile_plan(out_dir: Path) -> Path | None:
    names = (
        "compile_plan.json",
        "plan.json",
        "manifest.json",
        "summary.json",
        "benchmark_summary.json",
        "hls_artifact_metadata.json",
        "ir_artifacts.json",
    )
    for name in names:
        direct = out_dir / name
        if direct.exists():
            return direct
    for name in names:
        matches = sorted(out_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def _compile_plan_to_payload(compile_plan: Any) -> dict[str, Any]:
    raw = _json_safe(compile_plan)
    if isinstance(raw, dict):
        payload = dict(raw)
    elif isinstance(raw, list):
        payload = {"layers": raw}
    else:
        payload = {"raw_compile_plan": raw}

    # The standalone Sprint 5 dataset builder expects one of these keys.
    if not any(isinstance(payload.get(k), list) for k in ("layers", "operators", "nodes", "estimates", "operator_estimates", "schedule")):
        extracted = _extract_entries_from_object(compile_plan)
        if extracted:
            payload["layers"] = extracted

    # Also support dataclass/object payloads that put layer plans under common keys.
    for key in ("layer_plans", "layer_plan", "plans", "schedule", "operators", "nodes", "estimates"):
        value = payload.get(key)
        if isinstance(value, list) and "layers" not in payload:
            payload["layers"] = value
            break

    return payload


def _extract_entries_from_object(obj: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for attr in ("layers", "layer_plans", "plans", "operators", "nodes", "estimates", "schedule"):
        value = getattr(obj, attr, None)
        if isinstance(value, (list, tuple)):
            for item in value:
                converted = _json_safe(item)
                if isinstance(converted, dict):
                    entries.append(converted)
            if entries:
                return entries
    return entries


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            str(k): _json_safe(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return repr(value)


def _get_path(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _write_skip(calibration_dir: Path, reason: str) -> None:
    calibration_dir.mkdir(parents=True, exist_ok=True)
    (calibration_dir / "skipped.txt").write_text(reason + os.linesep, encoding="utf-8")
