from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
import json
import re


@dataclass(frozen=True)
class ParallelPipelineEffectArtifacts:
    parallel_pipeline_effect_json: Path
    parallel_pipeline_effect_md: Path

    def as_dict(self) -> Dict[str, Path]:
        return {
            "parallel_pipeline_effect_json": self.parallel_pipeline_effect_json,
            "parallel_pipeline_effect_md": self.parallel_pipeline_effect_md,
        }

    def items(self):
        return self.as_dict().items()


def _get(cfg: Mapping[str, Any] | None, path: str, default: Any = None) -> Any:
    cur: Any = cfg or {}
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _all_source_text(hls_dir: Path | None) -> str:
    if hls_dir is None:
        return ""
    roots = [hls_dir / "src", hls_dir]
    seen: set[Path] = set()
    chunks: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("*.cpp", "*.h", "*.hpp"):
            for path in root.rglob(pattern):
                if path in seen:
                    continue
                seen.add(path)
                chunks.append(_read(path))
    return "\n".join(chunks)


def _manual_sources(raw: Mapping[str, Any] | None) -> Dict[str, Any]:
    raw = raw or {}
    sources: Dict[str, Any] = {}
    for path in [
        # Current validated YAML paths.  Keep the historical hls.* aliases
        # out of tests because the config loader correctly rejects unknown
        # top-level sections.
        "optimization.pipeline.ii",
        "optimization.pipeline.style",
        "optimization.parallel.pe",
        "optimization.parallel.simd",
        "optimization.parallel.partition_factor",
        "optimization.parallel.unroll_factor",
        "optimization.parallel.array_partition_mode",
        "optimization.parallel.pipeline_style",
        # Legacy/internal aliases are accepted here only if callers pass an
        # already-resolved raw map; normal YAML validation should still reject
        # unknown top-level sections.
        "hls.pipeline_ii",
        "hls.dense.out_unroll",
        "hls.dense.in_unroll",
        "hls.dense.partition_input",
        "hls.dense.partition_output",
        "hls.dense.partition_weights",
        "hls.activation.unroll",
        "hls.conv.oc_unroll",
        "hls.conv.ic_unroll",
        "parallelization",
        "pipelining",
    ]:
        value = _get(raw, path, None)
        if value is not None:
            sources[path] = value
    return sources


def _int_macro(source: str, name: str, default: int | None = None) -> int | None:
    match = re.search(rf"^\s*#define\s+{re.escape(name)}\s+(-?\d+)\b", source, re.MULTILINE)
    if not match:
        return default
    try:
        return int(match.group(1))
    except ValueError:
        return default


def _contains(source: str, needle: str) -> bool:
    return needle in source


def _hls_paper_safe(hls_truth_artifacts: Any) -> bool:
    if hls_truth_artifacts is None:
        return False
    candidates: Iterable[Path] = []
    if hasattr(hls_truth_artifacts, "estimate_vs_hls_json"):
        candidates = [Path(getattr(hls_truth_artifacts, "estimate_vs_hls_json"))]
    elif isinstance(hls_truth_artifacts, Mapping):
        path = hls_truth_artifacts.get("estimate_vs_hls_json")
        candidates = [Path(path)] if path else []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") == "compared" or payload.get("paper_safe") is True:
            return True
    return False


def _decision(status: str, *, evidence: list[str] | None = None, reason: str | None = None, requested: bool = False, resolved: Any = None) -> Dict[str, Any]:
    return {
        "requested": bool(requested),
        "status": status,
        "resolved": resolved,
        "evidence": evidence or [],
        "reason": reason,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Parallelization and pipelining effect",
        "",
        f"Status: `{payload.get('status')}`",
        f"Paper-safe HLS claim: `{str(payload.get('hls_effect', {}).get('paper_safe_hls_claim')).lower()}`",
        "",
        "## Pipeline",
        f"- Status: `{payload.get('pipeline', {}).get('status')}`",
        f"- Resolved II: `{payload.get('pipeline', {}).get('resolved')}`",
        "",
        "## Parallelization",
    ]
    for name, entry in sorted((payload.get("parallelization") or {}).items()):
        if isinstance(entry, Mapping):
            lines.append(f"- {name}: `{entry.get('status')}` resolved=`{entry.get('resolved')}`")
    lines.extend([
        "",
        "## Source evidence",
    ])
    for item in payload.get("source_evidence", []):
        lines.append(f"- {item}")
    if not payload.get("source_evidence"):
        lines.append("- None")
    return "\n".join(lines) + "\n"


def emit_parallel_pipeline_effect_reports(
    *,
    out_dir: str | Path,
    raw_config: Mapping[str, Any] | None,
    hls_dir: str | Path | None,
    hls_truth_artifacts: Any = None,
) -> ParallelPipelineEffectArtifacts:
    out_dir = Path(out_dir)
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    hls_path = Path(hls_dir) if hls_dir is not None else None
    source = _all_source_text(hls_path)
    types_h = _read((hls_path / "src" / "fpgai_types.h") if hls_path is not None else Path("__missing__"))
    combined = source + "\n" + types_h
    manual = _manual_sources(raw_config)

    pipeline_requested = any(
        k.startswith("pipelining")
        or k in {
            "optimization.pipeline.ii",
            "optimization.pipeline.style",
            "optimization.parallel.pipeline_style",
            "hls.pipeline_ii",
        }
        for k in manual
    )
    ii = _int_macro(combined, "FPGAI_PIPELINE_II", None)
    pipeline_evidence = []
    if "#pragma HLS PIPELINE" in combined:
        pipeline_evidence.append("#pragma HLS PIPELINE")
    if ii is not None:
        pipeline_evidence.append(f"#define FPGAI_PIPELINE_II {ii}")
    pipeline_status = "applied" if pipeline_evidence else ("rejected" if pipeline_requested else "not_requested")

    parallel_checks = {
        "dense_out_unroll": ("FPGAI_DENSE_OUT_UNROLL", ("optimization.parallel.pe", "hls.dense.out_unroll")),
        "dense_in_unroll": ("FPGAI_DENSE_IN_UNROLL", ("optimization.parallel.simd", "hls.dense.in_unroll")),
        "dense_update_unroll": ("FPGAI_DENSE_UPD_UNROLL", ("optimization.parallel.unroll_factor", "hls.dense.update_unroll")),
        "activation_unroll": ("FPGAI_ACT_UNROLL", ("optimization.parallel.unroll_factor", "hls.activation.unroll")),
        "conv_output_channel_unroll": ("FPGAI_CONV_OC_UNROLL", ("optimization.parallel.pe", "hls.conv.oc_unroll")),
        "conv_input_channel_unroll": ("FPGAI_CONV_IC_UNROLL", ("optimization.parallel.simd", "hls.conv.ic_unroll")),
        "dense_input_partition": ("FPGAI_DENSE_PARTITION_INPUT", ("optimization.parallel.partition_factor", "hls.dense.partition_input")),
        "dense_output_partition": ("FPGAI_DENSE_PARTITION_OUTPUT", ("optimization.parallel.partition_factor", "hls.dense.partition_output")),
        "dense_weight_partition": ("FPGAI_DENSE_PARTITION_WEIGHTS", ("optimization.parallel.partition_factor", "hls.dense.partition_weights")),
    }
    parallel_payload: Dict[str, Any] = {}
    for key, (macro, cfg_paths) in parallel_checks.items():
        resolved = _int_macro(combined, macro, None)
        requested = (
            any(path in manual for path in cfg_paths)
            or any(p.startswith("parallelization") for p in manual)
            or any(p.startswith("optimization.parallel") for p in manual)
        )
        evidence: list[str] = []
        if resolved is not None:
            evidence.append(f"#define {macro} {resolved}")
        if "UNROLL" in key.upper() and "#pragma HLS UNROLL" in combined:
            evidence.append("#pragma HLS UNROLL")
        if "partition" in key and "#pragma HLS ARRAY_PARTITION" in combined:
            evidence.append("#pragma HLS ARRAY_PARTITION")
        if requested and evidence:
            status = "applied"
            reason = None
        elif requested:
            status = "rejected"
            reason = "Requested parallelization knob did not materialize in generated HLS source."
        elif resolved not in (None, 1):
            status = "compiler_default"
            reason = "Non-unit parallelization was selected by policy/compiler defaults."
        else:
            status = "not_requested"
            reason = "No manual parallelization request for this knob."
        parallel_payload[key] = _decision(status, evidence=evidence, reason=reason, requested=requested, resolved=resolved)

    source_evidence = []
    for needle in ["#pragma HLS PIPELINE", "#pragma HLS UNROLL", "#pragma HLS ARRAY_PARTITION", "FPGAI_PIPELINE_II", "FPGAI_DENSE_OUT_UNROLL", "FPGAI_CONV_OC_UNROLL"]:
        if _contains(combined, needle):
            source_evidence.append(needle)

    requested_any = bool(manual)
    rejected = ["pipeline"] if pipeline_status == "rejected" else []
    rejected += [name for name, entry in parallel_payload.items() if entry.get("status") == "rejected"]
    status = "failed" if rejected else ("validated" if source_evidence else "not_requested")
    paper_safe_hls = _hls_paper_safe(hls_truth_artifacts)

    payload: Dict[str, Any] = {
        "artifact_kind": "parallel_pipeline_effect",
        "schema_version": 1,
        "status": status,
        "requested": requested_any,
        "manual_yaml_sources": manual,
        "pipeline": _decision(
            pipeline_status,
            evidence=pipeline_evidence,
            reason=(None if pipeline_evidence else "No generated pipeline pragma or II macro evidence was found."),
            requested=pipeline_requested,
            resolved=ii,
        ),
        "parallelization": parallel_payload,
        "source_evidence": source_evidence,
        "resource_latency_hygiene": {
            "rule": "Unrequested parallel/pipeline code must not be claimed as manual user intent.",
            "manual_requests_only": sorted(manual.keys()),
            "unrequested_manual_claims": [],
        },
        "hls_effect": {
            "available": paper_safe_hls,
            "paper_safe_hls_claim": paper_safe_hls,
            "reason": None if paper_safe_hls else "No real HLS estimate-vs-HLS comparison was parsed for this compile.",
        },
    }

    json_path = reports / "parallel_pipeline_effect.json"
    md_path = reports / "parallel_pipeline_effect.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    return ParallelPipelineEffectArtifacts(json_path, md_path)
