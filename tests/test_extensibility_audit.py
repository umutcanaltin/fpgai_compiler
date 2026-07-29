from __future__ import annotations

import json
from pathlib import Path

from fpgai.devtools.extensibility_audit import (
    audit_extensibility,
    report_as_dict,
    write_reports,
)


def test_extensibility_audit_is_deterministic_and_metadata_only() -> None:
    root = Path(__file__).resolve().parents[1]

    first = report_as_dict(audit_extensibility(root))
    second = report_as_dict(audit_extensibility(root))

    assert first == second
    assert first["summary"]["extension_families"] >= 18
    assert first["summary"]["requires_core_edit"] > 0


def test_extensibility_audit_detects_existing_layer_registry_and_core_dispatch() -> None:
    root = Path(__file__).resolve().parents[1]
    report = audit_extensibility(root)
    findings = {item.capability: item for item in report.findings}

    assert "fpgai/layers/registry.py" in findings["ir_operator"].owner_files
    assert findings["ir_operator"].current_mechanism == "hard_coded_metadata_and_generic_op"
    assert findings["hls_implementation"].central_dispatch_occurrences > 0
    assert findings["training_reference"].central_dispatch_occurrences > 0
    assert findings["onnx_import"].requires_core_edit is True


def test_extensibility_audit_records_research_and_product_ownership() -> None:
    root = Path(__file__).resolve().parents[1]
    report = audit_extensibility(root)

    assert all(item.research_platform_owner == "fpgai" for item in report.findings)
    assert all(item.production_platform_owner == "morfics" for item in report.findings)
    assert "research" in report.contract_scope.lower()
    assert "production" in report.contract_scope.lower()


def test_extensibility_audit_writes_machine_readable_and_markdown_reports(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    report = audit_extensibility(root)

    json_path, markdown_path = write_reports(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["summary"]["critical_migrations"]
    assert payload["findings"]
    assert "FPGAI Extensibility Audit" in markdown_path.read_text(encoding="utf-8")
