from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

_SCHEMA = "fpgai.reference-model-profile/v1"


def load_reference_model_profile(path: str | Path) -> Dict[str, Any]:
    """Load evaluation/reference model metadata.

    These profiles are benchmark metadata only. They never participate in model
    import, canonicalization, lowering, or backend selection. FPGAI always
    compiles the actual imported graph layer-by-layer.
    """
    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("REFMODEL001: reference model profile must be a mapping")
    if data.get("schema") != _SCHEMA:
        raise ValueError(f"REFMODEL002: unsupported reference model profile schema {data.get('schema')!r}")
    if bool(data.get("compiler_special_case", True)):
        raise ValueError("REFMODEL003: evaluation model profiles must explicitly disable compiler special-casing")
    policy = data.get("policy") or {}
    if not bool(policy.get("compile_from_actual_exported_graph", False)):
        raise ValueError("REFMODEL004: reference profile must require compilation from the actual exported graph")
    if not bool(policy.get("layerwise_gap_audit_required", False)):
        raise ValueError("REFMODEL005: reference profile must require a layerwise gap audit")
    return data


__all__ = ["load_reference_model_profile"]
