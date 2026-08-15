from __future__ import annotations

"""Normalize Vitis HLS scheduling bottlenecks into compiler-facing diagnostics."""

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

_WARNING_RE = re.compile(
    r"WARNING:\s*\[HLS\s+200-885\].*?module\s+'(?P<module>[^']+)'.*?"
    r"\(loop\s+'(?P<loop>[^']+)'\):\s*(?P<message>.*?)(?=\n(?:INFO|WARNING|ERROR):|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_PIPELINE_RE = re.compile(
    r"Pipelining result\s*:\s*Target II\s*=\s*(?P<requested>\d+),\s*"
    r"Final II\s*=\s*(?P<achieved>\d+),.*?loop\s+'(?P<loop>[^']+)'",
    re.IGNORECASE,
)
_SOURCE_RE = re.compile(r"(?P<source>(?:\./)?[^\s,'()]+\.(?:cpp|cc|c|h|hpp)):(?P<line>\d+)")
_ARRAY_RE = re.compile(r"array\s+'(?P<array>[^']+)'", re.IGNORECASE)


@dataclass(frozen=True)
class HLSBottleneck:
    loop: str
    module: str
    requested_ii: int | None
    achieved_ii: int | None
    category: str
    resource: str | None
    source_file: str | None
    source_line: int | None
    message: str
    affected_tensor: str | None = None
    producer: str | None = None
    consumers: tuple[str, ...] = ()
    applicable_yaml_mechanisms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop": self.loop,
            "module": self.module,
            "requested_ii": self.requested_ii,
            "achieved_ii": self.achieved_ii,
            "ii_met": None if self.requested_ii is None or self.achieved_ii is None else self.achieved_ii <= self.requested_ii,
            "category": self.category,
            "resource": self.resource,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "message": self.message,
            "affected_tensor": self.affected_tensor,
            "producer": self.producer,
            "consumers": list(self.consumers),
            "applicable_yaml_mechanisms": list(self.applicable_yaml_mechanisms),
        }


def _classify(message: str) -> tuple[str, tuple[str, ...]]:
    lower = message.lower()
    if "limited memory ports" in lower or "memory ports" in lower:
        return "memory_port_contention", (
            "optimization.parallel.partition_factor",
            "hls.activation.unroll",
            "memory.activation_storage",
        )
    if "carried dependency" in lower or "dependence" in lower:
        return "loop_carried_dependency", ("optimization.pipeline.ii",)
    if "resource constraint" in lower or "resource limitation" in lower:
        return "resource_contention", ("optimization.parallel.pe", "optimization.parallel.simd")
    return "scheduling_constraint", ("optimization.pipeline.ii",)


def _tensor_context(resource: str | None, tensor_liveness: Mapping[str, Any] | None, resource_provenance: Mapping[str, Any] | None = None) -> tuple[str | None, str | None, tuple[str, ...]]:
    if not resource:
        return None, None, ()
    if resource_provenance and isinstance(resource_provenance, Mapping):
        entry = resource_provenance.get(resource)
        if isinstance(entry, Mapping):
            names = [str(x) for x in entry.get("tensors", []) or []]
            if len(names) == 1 and tensor_liveness and isinstance(tensor_liveness, Mapping):
                tensor_entry = (tensor_liveness.get("tensors", {}) or {}).get(names[0], {})
                if isinstance(tensor_entry, Mapping):
                    return names[0], tensor_entry.get("producer"), tuple(str(x) for x in tensor_entry.get("consumers", []) or [])
            if names:
                return names[-1], None, ()
    if not tensor_liveness:
        return None, None, ()
    tensors = tensor_liveness.get("tensors", {}) if isinstance(tensor_liveness, Mapping) else {}
    if not isinstance(tensors, Mapping):
        return None, None, ()
    candidates = [resource]
    if resource.startswith("layer_") and resource.endswith("_out"):
        candidates.append(resource)
    for name, entry in tensors.items():
        if name == resource or str(name).replace(".", "_") in resource or resource in str(name).replace(".", "_"):
            if isinstance(entry, Mapping):
                return str(name), entry.get("producer"), tuple(str(x) for x in entry.get("consumers", []) or [])
    return None, None, ()


def parse_hls_bottlenecks_text(text: str, *, tensor_liveness: Mapping[str, Any] | None = None, resource_provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    ii_by_loop: dict[str, tuple[int, int]] = {}
    for match in _PIPELINE_RE.finditer(text):
        ii_by_loop[match.group("loop")] = (int(match.group("requested")), int(match.group("achieved")))

    bottlenecks: list[HLSBottleneck] = []
    for match in _WARNING_RE.finditer(text):
        message = " ".join(match.group("message").split())
        loop = match.group("loop")
        requested, achieved = ii_by_loop.get(loop, (None, None))
        resource_match = _ARRAY_RE.search(message)
        resource = resource_match.group("array") if resource_match else None
        source_match = _SOURCE_RE.search(message)
        category, mechanisms = _classify(message)
        tensor, producer, consumers = _tensor_context(resource, tensor_liveness, resource_provenance)
        bottlenecks.append(HLSBottleneck(
            loop=loop,
            module=match.group("module"),
            requested_ii=requested,
            achieved_ii=achieved,
            category=category,
            resource=resource,
            source_file=source_match.group("source") if source_match else None,
            source_line=int(source_match.group("line")) if source_match else None,
            message=message,
            affected_tensor=tensor,
            producer=producer,
            consumers=consumers,
            applicable_yaml_mechanisms=mechanisms,
        ))

    return {
        "schema": "fpgai.hls-bottleneck-diagnostics/v1",
        "status": "passed",
        "warning_count": len(bottlenecks),
        "ii_violation_count": sum(1 for item in bottlenecks if item.requested_ii is not None and item.achieved_ii is not None and item.achieved_ii > item.requested_ii),
        "categories": sorted({item.category for item in bottlenecks}),
        "bottlenecks": [item.to_dict() for item in bottlenecks],
        "policy": "Diagnostics report applicable YAML-selectable mechanisms but never enable them automatically.",
    }


def analyze_hls_bottlenecks(stdout_log: str | Path | None, *, tensor_liveness: Mapping[str, Any] | None = None, resource_provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if stdout_log is None:
        return {"schema": "fpgai.hls-bottleneck-diagnostics/v1", "status": "not_run", "warning_count": 0, "ii_violation_count": 0, "categories": [], "bottlenecks": []}
    path = Path(stdout_log)
    if not path.is_file():
        return {"schema": "fpgai.hls-bottleneck-diagnostics/v1", "status": "log_missing", "source": str(path), "warning_count": 0, "ii_violation_count": 0, "categories": [], "bottlenecks": []}
    payload = parse_hls_bottlenecks_text(path.read_text(encoding="utf-8", errors="replace"), tensor_liveness=tensor_liveness, resource_provenance=resource_provenance)
    payload["source"] = str(path.resolve())
    return payload


def write_hls_bottleneck_diagnostics(payload: Mapping[str, Any], reports_dir: str | Path) -> tuple[Path, Path]:
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "hls_bottleneck_diagnostics.json"
    md_path = root / "hls_bottleneck_diagnostics.md"
    json_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# HLS bottleneck diagnostics", "", f"- Status: `{payload.get('status')}`", f"- HLS 200-885 warnings: {payload.get('warning_count', 0)}", f"- II violations: {payload.get('ii_violation_count', 0)}"]
    for item in payload.get("bottlenecks", []) or []:
        lines.extend(["", f"## {item.get('loop')}", f"- Category: `{item.get('category')}`", f"- Requested / achieved II: {item.get('requested_ii')} / {item.get('achieved_ii')}", f"- Resource: `{item.get('resource')}`", f"- Tensor: `{item.get('affected_tensor')}`", f"- Source: `{item.get('source_file')}:{item.get('source_line')}`", "- Applicable YAML mechanisms: " + ", ".join(f"`{x}`" for x in item.get("applicable_yaml_mechanisms", []))])
    lines.extend(["", "FPGAI reports these mechanisms as choices; it does not silently change architecture settings."])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
