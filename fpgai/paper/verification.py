"""Paper-safety verification artifacts.

Paper tables must be derived from real generated artifacts. This module writes
small, conservative per-compile verification summaries that later experiment
and paper-table pipelines can consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def emit_paper_verification_artifacts(
    out_dir: str | Path,
    *,
    pipeline_mode: str,
    source_generated: bool,
    numeric_validation_json: str | Path | None = None,
    hls_ran: bool = False,
    hls_ok: bool | None = None,
    vivado_implemented: bool = False,
    bitstream_generated: bool = False,
    fpga_executed: bool = False,
    build_stages: dict[str, Any] | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    numeric_payload: dict[str, Any] = {}
    if numeric_validation_json is not None:
        numeric_payload = _read_json(Path(numeric_validation_json))

    numeric_validated = bool(numeric_payload.get("passed", False))
    hls_synthesized = bool(hls_ran and hls_ok is not False)

    flags = {
        "source_generated": bool(source_generated),
        "numeric_validated": numeric_validated,
        "hls_synthesized": hls_synthesized,
        "vivado_implemented": bool(vivado_implemented),
        "bitstream_generated": bool(bitstream_generated),
        "fpga_executed": bool(fpga_executed),
    }

    allowed_claims = {
        "source_generation": flags["source_generated"],
        "numeric_correctness": flags["numeric_validated"],
        "hls_resource_timing": flags["hls_synthesized"],
        "vivado_implementation": flags["vivado_implemented"],
        "bitstream": flags["bitstream_generated"],
        "real_fpga_runtime": flags["fpga_executed"],
    }

    paper_safe = bool(
        flags["source_generated"]
        and (
            flags["numeric_validated"]
            or pipeline_mode not in {"training_on_device", "inference"}
        )
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "paper_verification",
        "pipeline_mode": str(pipeline_mode or "inference"),
        "build_stages": build_stages or {},
        "verification_flags": flags,
        "allowed_claims": allowed_claims,
        "paper_safe": paper_safe,
        "numeric_validation_json": str(numeric_validation_json) if numeric_validation_json is not None else None,
    }

    verification_json = reports / "paper_verification.json"
    verification_md = reports / "paper_verification.md"
    paper_row_json = reports / "paper_row.json"

    verification_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    row = {
        "pipeline_mode": payload["pipeline_mode"],
        **flags,
        "paper_safe": paper_safe,
        "numeric_validation_status": numeric_payload.get("status"),
    }
    paper_row_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Paper verification",
        "",
        f"- Pipeline mode: `{payload['pipeline_mode']}`",
        f"- Paper safe: `{str(paper_safe).lower()}`",
        "",
        "## Verification flags",
    ]
    for key, value in flags.items():
        lines.append(f"- {key}: `{str(bool(value)).lower()}`")
    lines += ["", "## Allowed claims"]
    for key, value in allowed_claims.items():
        lines.append(f"- {key}: `{str(bool(value)).lower()}`")
    verification_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "paper_verification_json": verification_json,
        "paper_verification_md": verification_md,
        "paper_row_json": paper_row_json,
    }
