from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from fpgai.config.access import get_path
_cfg_get = get_path


@dataclass(frozen=True)
class PrecisionEffectArtifacts:
    precision_effect_json: Path
    precision_effect_md: Path

    def as_dict(self) -> Dict[str, Path]:
        return {
            "precision_effect_json": self.precision_effect_json,
            "precision_effect_md": self.precision_effect_md,
        }

    def items(self):
        return self.as_dict().items()


def _read_json(path: Path | str | None) -> Dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _artifact_path(artifacts: Any, key: str) -> Path | None:
    if artifacts is None:
        return None
    if isinstance(artifacts, Mapping):
        value = artifacts.get(key)
        return Path(value) if value is not None else None
    value = getattr(artifacts, key, None)
    return Path(value) if value is not None else None


def _manual_sources(raw_cfg: Mapping[str, Any]) -> list[str]:
    paths = []
    for path in (
        "numerics.precision_mode",
        "analysis.precision_sweep.selected_candidate",
        "numerics.defaults.activation",
        "numerics.defaults.weight",
        "numerics.defaults.bias",
        "numerics.defaults.accum",
        "numerics.layers",
    ):
        if _cfg_get(raw_cfg, path, None) is not None:
            paths.append(path)
    return paths


def _scan_source_types(hls_dir: Path | None, bits: Mapping[str, Any]) -> Dict[str, Any]:
    type_header = None if hls_dir is None else hls_dir / "include" / "fpgai_types.h"
    source_files: list[Path] = []
    if type_header is not None and type_header.is_file():
        source_files.append(type_header)
    if hls_dir is not None:
        src_dir = hls_dir / "src"
        source_files.extend(sorted(src_dir.glob("*.cpp")))

    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in source_files if p.is_file())
    expected = {
        "activation": f"ap_fixed<{int(bits.get('activation', 0))},",
        "weight": f"ap_fixed<{int(bits.get('weight', 0))},",
        "bias": f"ap_fixed<{int(bits.get('bias', 0))},",
        "accum": f"ap_fixed<{int(bits.get('accum', 0))},",
    }
    evidence = {role: (needle in text) for role, needle in expected.items() if needle != "ap_fixed<0,"}
    ap_fixed_typedefs = re.findall(r"typedef\s+ap_fixed<[^>]+>\s+\w+;", text)
    float_typedefs = re.findall(r"typedef\s+float\s+\w+;", text)
    return {
        "type_header": str(type_header) if type_header is not None else None,
        "type_header_exists": bool(type_header is not None and type_header.is_file()),
        "source_files_scanned": [str(p) for p in source_files if p.is_file()],
        "expected_ap_fixed_patterns": expected,
        "expected_patterns_present": evidence,
        "type_changed": any(evidence.values()) or bool(ap_fixed_typedefs),
        "ap_fixed_typedef_count": len(ap_fixed_typedefs),
        "float_typedef_count": len(float_typedefs),
    }


def _quant_metrics(quant_result: Any) -> Dict[str, Any]:
    if quant_result is None:
        return {
            "status": "not_requested",
            "metrics_available": False,
        }
    metrics_json = Path(getattr(quant_result, "metrics_json", ""))
    payload = _read_json(metrics_json)
    if payload is None:
        return {
            "status": "artifact_missing",
            "metrics_available": False,
            "metrics_json": str(metrics_json),
        }
    keys = [
        "output_cosine",
        "cosine",
        "output_mse",
        "mse",
        "output_mae",
        "mae",
        "output_max_abs",
        "max_abs",
    ]
    extracted = {k: payload.get(k) for k in keys if k in payload}
    return {
        "status": "available",
        "metrics_available": True,
        "metrics_json": str(metrics_json),
        "metrics": extracted,
    }


def _precision_sweep_status(sweep_result: Any) -> Dict[str, Any]:
    if sweep_result is None:
        return {"status": "not_requested", "available": False}
    return {
        "status": "available" if Path(getattr(sweep_result, "results_json", "")).is_file() else "artifact_missing",
        "available": Path(getattr(sweep_result, "results_json", "")).is_file(),
        "results_json": str(getattr(sweep_result, "results_json", "")),
        "results_csv": str(getattr(sweep_result, "results_csv", "")),
    }


def _hls_claim_status(hls_truth_artifacts: Any) -> Dict[str, Any]:
    estimate_path = _artifact_path(hls_truth_artifacts, "estimate_vs_hls_json")
    estimate = _read_json(estimate_path)
    if estimate is None:
        return {
            "estimate_vs_hls_status": "artifact_missing",
            "paper_safe_hls_claim": False,
            "estimate_vs_hls_json": str(estimate_path) if estimate_path else None,
        }
    return {
        "estimate_vs_hls_status": estimate.get("status"),
        "paper_safe_hls_claim": bool(estimate.get("paper_safe", False)) and estimate.get("status") == "compared",
        "estimate_vs_hls_json": str(estimate_path),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    precision = payload.get("precision", {})
    artifacts = payload.get("generated_artifacts", {})
    numeric = payload.get("numeric_effect", {})
    resource = payload.get("resource_effect", {})
    lines = [
        "# Precision effect report",
        "",
        f"Status: `{payload.get('status')}`",
        f"Paper-safe HLS precision claim: `{resource.get('paper_safe_hls_claim')}`",
        "",
        "## Resolved precision",
        "",
        f"- Requested/resolved mode: `{precision.get('resolved')}`",
        f"- Source: `{precision.get('source')}`",
        f"- Activation bits: `{precision.get('bits', {}).get('activation')}`",
        f"- Weight bits: `{precision.get('bits', {}).get('weight')}`",
        f"- Bias bits: `{precision.get('bits', {}).get('bias')}`",
        f"- Accumulator bits: `{precision.get('bits', {}).get('accum')}`",
        "",
        "## Generated C++ evidence",
        "",
        f"- Type header exists: `{artifacts.get('type_header_exists')}`",
        f"- Type changed/materialized: `{artifacts.get('type_changed')}`",
        f"- ap_fixed typedef count: `{artifacts.get('ap_fixed_typedef_count')}`",
        "",
        "## Numeric/resource evidence",
        "",
        f"- Numeric metrics status: `{numeric.get('status')}`",
        f"- Precision sweep status: `{payload.get('precision_sweep', {}).get('status')}`",
        f"- Estimate-vs-HLS status: `{resource.get('estimate_vs_hls_status')}`",
        "",
        "## Truth boundary",
        "",
        "Precision is considered generated/materialized when the resolved layout and generated C++ types agree. Real HLS resource/timing claims are paper-safe only when estimate_vs_hls is compared against parsed HLS reports.",
        "",
    ]
    return "\n".join(lines)


def emit_precision_effect_reports(
    *,
    out_dir: str | Path,
    raw_config: Mapping[str, Any],
    hls_dir: str | Path | None,
    precision_layout_artifacts: Mapping[str, Any] | None,
    quant_result: Any = None,
    sweep_result: Any = None,
    hls_truth_artifacts: Any = None,
) -> PrecisionEffectArtifacts:
    out = Path(out_dir)
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    layout_path = Path(precision_layout_artifacts.get("json")) if precision_layout_artifacts and precision_layout_artifacts.get("json") else reports / "precision_layout.json"
    layout = _read_json(layout_path) or {}
    bits = layout.get("bits", {}) if isinstance(layout.get("bits"), dict) else {}
    manual_sources = _manual_sources(raw_config)
    source = "manual_yaml" if manual_sources else "compiler_default"
    generated = _scan_source_types(Path(hls_dir) if hls_dir is not None else None, bits)
    numeric = _quant_metrics(quant_result)
    sweep = _precision_sweep_status(sweep_result)
    resource = _hls_claim_status(hls_truth_artifacts)

    status = "validated" if layout and generated.get("type_changed") else "artifact_missing"
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "precision": {
            "requested": _cfg_get(raw_config, "numerics.precision_mode", _cfg_get(raw_config, "analysis.precision_sweep.selected_candidate", None)),
            "resolved": layout.get("precision_mode"),
            "source": source,
            "manual_yaml_sources": manual_sources,
            "bits": bits,
            "roles": layout.get("roles", {}),
        },
        "generated_artifacts": generated,
        "numeric_effect": numeric,
        "precision_sweep": sweep,
        "resource_effect": {
            "estimate_available": True,
            "hls_available": resource.get("estimate_vs_hls_status") == "compared",
            **resource,
        },
        "truth_boundary": {
            "generated_precision_materialized": bool(generated.get("type_changed")),
            "numeric_metrics_required_for_accuracy_claim": numeric.get("status") == "available",
            "paper_safe_hls_claim_requires_estimate_vs_hls_compared": True,
        },
    }

    json_path = reports / "precision_effect.json"
    md_path = reports / "precision_effect.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return PrecisionEffectArtifacts(json_path, md_path)
