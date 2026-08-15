"""Conservative per-compile validation summary artifacts for benchmark workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def emit_validation_summary_artifacts(
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

    validation_flags = {
        "source_generated": bool(source_generated),
        "numeric_validated": numeric_validated,
        "hls_synthesized": hls_synthesized,
        "vivado_implemented": bool(vivado_implemented),
        "bitstream_generated": bool(bitstream_generated),
        "fpga_executed": bool(fpga_executed),
    }

    validated_capabilities = {
        "source_generation": validation_flags["source_generated"],
        "numeric_correctness": validation_flags["numeric_validated"],
        "hls_resource_timing": validation_flags["hls_synthesized"],
        "vivado_implementation": validation_flags["vivado_implemented"],
        "bitstream": validation_flags["bitstream_generated"],
        "real_fpga_runtime": validation_flags["fpga_executed"],
    }

    validation_ready = bool(
        validation_flags["source_generated"]
        and (
            validation_flags["numeric_validated"]
            or pipeline_mode not in {"training_on_device", "inference"}
        )
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "validation_summary",
        "pipeline_mode": str(pipeline_mode or "inference"),
        "build_stages": build_stages or {},
        "validation_flags": validation_flags,
        "validated_capabilities": validated_capabilities,
        "validation_ready": validation_ready,
        "numeric_validation_json": str(numeric_validation_json) if numeric_validation_json is not None else None,
    }

    summary_json = reports / "validation_summary.json"
    summary_md = reports / "validation_summary.md"
    benchmark_row_json = reports / "benchmark_row.json"

    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    row = {
        "pipeline_mode": payload["pipeline_mode"],
        **validation_flags,
        "validation_ready": validation_ready,
        "numeric_validation_status": numeric_payload.get("status"),
    }
    benchmark_row_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Validation summary",
        "",
        f"- Pipeline mode: `{payload['pipeline_mode']}`",
        f"- Validation ready: `{str(validation_ready).lower()}`",
        "",
        "## Validation flags",
    ]
    for key, value in validation_flags.items():
        lines.append(f"- {key}: `{str(bool(value)).lower()}`")
    lines += ["", "## Validated capabilities"]
    for key, value in validated_capabilities.items():
        lines.append(f"- {key}: `{str(bool(value)).lower()}`")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "validation_summary_json": summary_json,
        "validation_summary_md": summary_md,
        "benchmark_row_json": benchmark_row_json,
    }
