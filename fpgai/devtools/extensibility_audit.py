"""Audit FPGAI extension points without importing or executing plugin code.

The audit is intentionally metadata-only. It reads source files as text, classifies
current extension mechanisms, and writes deterministic JSON and Markdown reports.

Run with::

    python -m fpgai.devtools.extensibility_audit

The command does not modify compiler sources or generated compiler artifacts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ExtensionFamilySpec:
    """Declarative description of one contributor-facing extension family."""

    capability: str
    owner_candidates: tuple[str, ...]
    mechanism: str
    recommended_contract: str
    migration_priority: str
    inference_support: str
    training_support: str
    automation_requirement: str
    search_markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtensionFamilyFinding:
    """Current repository status for one extension family."""

    capability: str
    owner_files: tuple[str, ...]
    current_mechanism: str
    external_contribution_possible: bool
    requires_core_edit: bool
    inference_support: str
    training_support: str
    research_platform_owner: str
    production_platform_owner: str
    morfics_backend_readiness: str
    migration_priority: str
    recommended_contract: str
    central_dispatch_occurrences: int


@dataclass(frozen=True)
class ExtensibilityAuditReport:
    """Serializable extensibility and ownership audit result."""

    repository_root: str
    contract_scope: str
    findings: tuple[ExtensionFamilyFinding, ...]

    @property
    def extension_family_count(self) -> int:
        return len(self.findings)

    @property
    def externally_extensible_count(self) -> int:
        return sum(item.external_contribution_possible for item in self.findings)

    @property
    def requires_core_edit_count(self) -> int:
        return sum(item.requires_core_edit for item in self.findings)

    @property
    def critical_migrations(self) -> tuple[str, ...]:
        return tuple(
            item.capability
            for item in self.findings
            if item.migration_priority == "critical"
        )


EXTENSION_FAMILIES: tuple[ExtensionFamilySpec, ...] = (
    ExtensionFamilySpec(
        "onnx_import",
        ("fpgai/frontend/onnx/importer.py", "fpgai/frontend/onnx/canonicalize.py"),
        "central_frontend_pipeline",
        "OnnxImporterRegistry",
        "critical",
        "supported",
        "indirect",
        "stable preflight and import contract",
        ("node.op_type", "canonicalize_op"),
    ),
    ExtensionFamilySpec(
        "ir_operator",
        ("fpgai/ir/ops.py", "fpgai/layers/registry.py"),
        "hard_coded_metadata_and_generic_op",
        "OperatorRegistry",
        "critical",
        "supported",
        "partial",
        "versioned operator contract",
        ("_LAYER_METADATA", "op_type"),
    ),
    ExtensionFamilySpec(
        "shape_and_type_inference",
        ("fpgai/frontend/onnx/parsing.py", "fpgai/frontend/onnx/annotate.py"),
        "frontend_helpers",
        "ShapeInferenceRegistry",
        "high",
        "partial",
        "partial",
        "deterministic capability inspection",
        ("shape", "dtype"),
    ),
    ExtensionFamilySpec(
        "canonicalization",
        ("fpgai/frontend/onnx/canonicalize.py", "fpgai/frontend/onnx/patterns.py"),
        "central_functions",
        "CanonicalizationRegistry",
        "high",
        "supported",
        "shared",
        "versioned lowering behavior",
        ("if ", "op_type"),
    ),
    ExtensionFamilySpec(
        "hls_implementation",
        ("fpgai/backends/hls/emit/top_cpp.py", "fpgai/backends/hls/emit/top_train_cpp.py"),
        "central_codegen_dispatch",
        "ImplementationRegistry",
        "critical",
        "supported",
        "partial",
        "stable source and artifact contract",
        ("op.op_type", "if op_type"),
    ),
    ExtensionFamilySpec(
        "vhdl_and_rtl_implementation",
        ("fpgai/backends",),
        "backend_not_yet_first_class",
        "RtlImplementationRegistry",
        "critical",
        "not_first_class",
        "not_first_class",
        "stable backend and wrapper contract",
        ("vhdl", "systemverilog", "verilog"),
    ),
    ExtensionFamilySpec(
        "training_reference",
        ("fpgai/benchmark/training_reference.py", "fpgai/benchmark/training_dataset_reference.py"),
        "central_operator_dispatch",
        "TrainingSemanticsRegistry",
        "critical",
        "not_applicable",
        "supported_for_built_ins",
        "deterministic numeric validation contract",
        ("op.op_type ==", "elif op.op_type"),
    ),
    ExtensionFamilySpec(
        "optimizer",
        ("fpgai/benchmark/training_reference.py", "fpgai/backends/hls/emit/top_train_cpp.py"),
        "string_conditionals",
        "OptimizerRegistry",
        "high",
        "not_applicable",
        "built_ins_only",
        "versioned optimizer-state contract",
        ("optimizer_type ==",),
    ),
    ExtensionFamilySpec(
        "loss",
        ("fpgai/benchmark/training_dataset_reference.py", "fpgai/backends/hls/emit/top_train_cpp.py"),
        "string_conditionals",
        "LossRegistry",
        "high",
        "not_applicable",
        "built_ins_only",
        "versioned target and loss contract",
        ("loss_type ==",),
    ),
    ExtensionFamilySpec(
        "board",
        ("fpgai/backends/vivado/boards.py",),
        "central_board_database",
        "BoardRegistry",
        "high",
        "shared",
        "shared",
        "stable target capability contract",
        ("BOARD", "board"),
    ),
    ExtensionFamilySpec(
        "backend_and_toolchain",
        ("fpgai/backends", "fpgai/engine/vivado_pipeline.py"),
        "direct_backend_imports",
        "BackendRegistry",
        "high",
        "shared",
        "shared",
        "service-safe backend result contract",
        ("run_vivado_bridge_flow", "vitis_hls"),
    ),
    ExtensionFamilySpec(
        "memory_policy",
        ("fpgai/engine/memory_semantics.py", "fpgai/analysis"),
        "central_policy_logic",
        "MemoryPolicyRegistry",
        "high",
        "supported",
        "supported",
        "deterministic memory-plan contract",
        ("bram", "uram", "ddr"),
    ),
    ExtensionFamilySpec(
        "transport",
        ("fpgai/engine/training_contracts.py", "fpgai/engine/compiler.py"),
        "central_contract_logic",
        "TransportRegistry",
        "high",
        "supported",
        "supported",
        "stable interface and movement contract",
        ("axi_stream", "m_axi", "transport"),
    ),
    ExtensionFamilySpec(
        "dataset",
        ("fpgai/benchmark", "fpgai/engine/hls_project_generation.py"),
        "direct_dataset_helpers",
        "DatasetRegistry",
        "medium",
        "supported",
        "supported",
        "stable dataset artifact contract",
        ("emit_dataset_artifacts", "dataset"),
    ),
    ExtensionFamilySpec(
        "validation",
        ("fpgai/validation",),
        "direct_report_calls",
        "ValidationRegistry",
        "medium",
        "supported",
        "supported",
        "stable validation result contract",
        ("emit_", "validation"),
    ),
    ExtensionFamilySpec(
        "reporter",
        ("fpgai/reporting", "fpgai/reports", "fpgai/benchmark"),
        "direct_report_calls",
        "ReporterRegistry",
        "medium",
        "shared",
        "shared",
        "machine-readable report contract",
        ("report", "emit_"),
    ),
    ExtensionFamilySpec(
        "runtime_package",
        ("fpgai/runtime",),
        "stable_public_entry_with_internal_modules",
        "RuntimeRegistry",
        "medium",
        "supported",
        "supported",
        "versioned artifact and invocation contract",
        ("emit_runtime_package", "manifest"),
    ),
    ExtensionFamilySpec(
        "model_package",
        ("fpgai/frontend", "models"),
        "path_based_model_loading",
        "ModelRegistry",
        "high",
        "supported",
        "supported_when_graph_is_legal",
        "versioned model package contract",
        ("model", "onnx"),
    ),
)


def _iter_candidate_files(root: Path, candidates: Sequence[str]) -> Iterable[Path]:
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from sorted(item for item in path.rglob("*.py") if item.is_file())


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _count_markers(paths: Sequence[Path], markers: Sequence[str]) -> int:
    count = 0
    for path in paths:
        source = path.read_text(encoding="utf-8", errors="replace")
        for marker in markers:
            count += source.count(marker)
    return count


def _is_externally_extensible(spec: ExtensionFamilySpec, owner_paths: Sequence[Path]) -> bool:
    """Conservatively classify only explicit plugin/entry-point systems as external."""

    marker_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in owner_paths
    ).lower()
    explicit_markers = (
        "importlib.metadata.entry_points",
        "entry_points(",
        "plugin_directory",
        "package_registry",
    )
    return any(marker in marker_text for marker in explicit_markers)


def audit_extensibility(repository_root: str | Path) -> ExtensibilityAuditReport:
    """Inspect extension points without importing repository modules."""

    root = Path(repository_root).resolve()
    findings: list[ExtensionFamilyFinding] = []

    for spec in EXTENSION_FAMILIES:
        owner_paths = tuple(_iter_candidate_files(root, spec.owner_candidates))
        externally_extensible = _is_externally_extensible(spec, owner_paths)
        dispatch_count = _count_markers(owner_paths, spec.search_markers)
        findings.append(
            ExtensionFamilyFinding(
                capability=spec.capability,
                owner_files=tuple(_relative(path, root) for path in owner_paths),
                current_mechanism=spec.mechanism,
                external_contribution_possible=externally_extensible,
                requires_core_edit=not externally_extensible,
                inference_support=spec.inference_support,
                training_support=spec.training_support,
                research_platform_owner="fpgai",
                production_platform_owner="morfics",
                morfics_backend_readiness=(
                    "partial_requires_versioned_contract"
                    if spec.automation_requirement
                    else "not_assessed"
                ),
                migration_priority=spec.migration_priority,
                recommended_contract=spec.recommended_contract,
                central_dispatch_occurrences=dispatch_count,
            )
        )

    return ExtensibilityAuditReport(
        repository_root=root.as_posix(),
        contract_scope=(
            "FPGAI is the open research, validation, and benchmarking compiler platform. "
            "Morfics owns commercial productization, managed builds, deployment, hosted "
            "inference/training, operations, security, billing, and production support."
        ),
        findings=tuple(findings),
    )


def report_as_dict(report: ExtensibilityAuditReport) -> dict[str, object]:
    """Return a deterministic JSON-serializable representation."""

    return {
        "repository_root": report.repository_root,
        "contract_scope": report.contract_scope,
        "summary": {
            "extension_families": report.extension_family_count,
            "externally_extensible": report.externally_extensible_count,
            "requires_core_edit": report.requires_core_edit_count,
            "critical_migrations": list(report.critical_migrations),
        },
        "findings": [asdict(item) for item in report.findings],
    }


def write_reports(
    report: ExtensibilityAuditReport,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown reports without touching compiler artifacts."""

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "extensibility_audit.json"
    markdown_path = output_dir / "extensibility_audit.md"
    payload = report_as_dict(report)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# FPGAI Extensibility Audit",
        "",
        f"Repository: `{report.repository_root}`",
        "",
        report.contract_scope,
        "",
        "## Summary",
        "",
        f"- Extension families: {report.extension_family_count}",
        f"- Externally extensible today: {report.externally_extensible_count}",
        f"- Requiring core edits today: {report.requires_core_edit_count}",
        "",
        "## Findings",
        "",
        "| Capability | Mechanism | Core edit | Priority | Recommended contract |",
        "|---|---|---:|---|---|",
    ]
    for item in report.findings:
        lines.append(
            f"| `{item.capability}` | `{item.current_mechanism}` | "
            f"`{'yes' if item.requires_core_edit else 'no'}` | "
            f"`{item.migration_priority}` | `{item.recommended_contract}` |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository_root", nargs="?", default=".")
    parser.add_argument(
        "--output-dir",
        default="dev_audits/extensibility",
        help="Generated report directory (default: dev_audits/extensibility).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = audit_extensibility(args.repository_root)
    json_path, markdown_path = write_reports(report, args.output_dir)
    print(f"extension_families: {report.extension_family_count}")
    print(f"externally_extensible: {report.externally_extensible_count}")
    print(f"requires_core_edit: {report.requires_core_edit_count}")
    print(f"json_report: {json_path}")
    print(f"markdown_report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
