from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .package_errors import PackageIssue
from .package_dependency import parse_dependencies
from .package_manifest import PackageManifest, load_package_manifest
from .package_types import (
    AssetType,
    ImplementationLanguage,
    InterfaceProtocol,
    LicenseCategory,
    PACKAGE_SCHEMA_V1,
    ValidationLevel,
)

_PACKAGE_ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)+$")
_REQUIRED_RESEARCH_USES = {"research", "experimentation", "validation", "benchmarking"}
_FORBIDDEN_SCOPE_TERMS = {
    "production",
    "certified_production",
    "commercial_hosting",
    "managed_runtime",
    "fleet_operation",
}
_PATH_KEYS = {"path", "sources", "headers", "source_order", "testbench", "reference", "python_module"}


@dataclass(frozen=True)
class PackageValidationResult:
    status: str
    schema: str
    package_id: str
    errors: tuple[PackageIssue, ...]
    warnings: tuple[PackageIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "schema": self.schema,
            "package_id": self.package_id,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _issue(code: str, field: str, message: str, *, warning: bool = False) -> PackageIssue:
    return PackageIssue(code, field, message, "warning" if warning else "error")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_safe_relative_path(value: str) -> bool:
    if not value or "${" in value or value.startswith(("http://", "https://")):
        return False
    path = PurePosixPath(value.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def _iter_declared_paths(value: Any, field: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_field = f"{field}.{key}" if field else str(key)
            if key in _PATH_KEYS:
                if isinstance(child, str):
                    yield child_field, child
                elif isinstance(child, list):
                    for index, item in enumerate(child):
                        if isinstance(item, str):
                            yield f"{child_field}[{index}]", item
                elif isinstance(child, Mapping):
                    yield from _iter_declared_paths(child, child_field)
            else:
                yield from _iter_declared_paths(child, child_field)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_declared_paths(child, f"{field}[{index}]")


def _validate_entrypoints(raw: Mapping[str, Any], asset_type: str, errors: list[PackageIssue]) -> None:
    entrypoints = _mapping(raw.get("entrypoints"))
    expected = {
        "model": "model",
        "operator": "operator",
        "implementation": "implementation",
        "system_block": "implementation",
        "adapter": "implementation",
    }.get(asset_type)
    if expected and not isinstance(entrypoints.get(expected), Mapping):
        errors.append(_issue("PKG008", f"entrypoints.{expected}", f"Missing {expected} entrypoint"))
        return
    if expected == "implementation":
        implementation = _mapping(entrypoints.get("implementation"))
        language = str(implementation.get("language", ""))
        if language not in {item.value for item in ImplementationLanguage}:
            errors.append(_issue("PKG008", "entrypoints.implementation.language", "Unsupported or missing implementation language"))
        if not implementation.get("top"):
            errors.append(_issue("PKG008", "entrypoints.implementation.top", "Implementation top is required"))
        if not implementation.get("sources"):
            errors.append(_issue("PKG008", "entrypoints.implementation.sources", "At least one source is required"))


def validate_package_manifest(package_root: str | Path) -> PackageValidationResult:
    root = Path(package_root).expanduser().resolve()
    errors: list[PackageIssue] = []
    warnings: list[PackageIssue] = []
    try:
        manifest = load_package_manifest(root)
    except FileNotFoundError as exc:
        return PackageValidationResult("failed", "", "", (_issue("PKG001", "fpgai.yaml", str(exc)),), ())
    except (ValueError, OSError) as exc:
        return PackageValidationResult("failed", "", "", (_issue("PKG012", "fpgai.yaml", str(exc)),), ())

    raw = manifest.raw
    if manifest.schema != PACKAGE_SCHEMA_V1:
        errors.append(_issue("PKG002", "schema", f"Expected {PACKAGE_SCHEMA_V1}"))

    package = _mapping(raw.get("package"))
    package_id = str(package.get("id", ""))
    if not _PACKAGE_ID_RE.fullmatch(package_id):
        errors.append(_issue("PKG003", "package.id", "Package ID must be lowercase and namespace-qualified"))
    try:
        Version(str(package.get("version", "")))
    except InvalidVersion:
        errors.append(_issue("PKG004", "package.version", "Package version must use semantic version syntax"))
    asset_type = str(package.get("asset_type", ""))
    if asset_type not in {item.value for item in AssetType}:
        errors.append(_issue("PKG005", "package.asset_type", "Unsupported asset type"))

    license_cfg = _mapping(raw.get("license"))
    license_category = str(license_cfg.get("category", ""))
    if license_category not in {item.value for item in LicenseCategory}:
        errors.append(_issue("PKG006", "license.category", "A supported license category is required"))
    if license_category == LicenseCategory.OPEN_SOURCE.value and not license_cfg.get("identifier"):
        errors.append(_issue("PKG006", "license.identifier", "Open-source packages require an SPDX identifier"))

    usage = _mapping(raw.get("usage"))
    permitted = {str(item) for item in usage.get("permitted_uses", []) if isinstance(item, str)}
    if usage.get("platform_scope") != "research" or usage.get("production_path") != "morfics":
        errors.append(_issue("PKG011", "usage", "FPGAI packages must declare research scope and Morfics as the production path"))
    if not _REQUIRED_RESEARCH_USES.issubset(permitted) or permitted.intersection(_FORBIDDEN_SCOPE_TERMS):
        errors.append(_issue("PKG011", "usage.permitted_uses", "Permitted uses must remain research, experimentation, validation, benchmarking, or education"))

    compatibility = _mapping(raw.get("compatibility"))
    contract_range = str(compatibility.get("fpgai_contract", ""))
    try:
        if not contract_range:
            raise InvalidSpecifier(contract_range)
        SpecifierSet(contract_range)
    except InvalidSpecifier:
        errors.append(_issue("PKG014", "compatibility.fpgai_contract", "Invalid FPGAI contract version range"))

    try:
        dependencies = parse_dependencies(raw)
        dependency_ids: set[str] = set()
        for index, dependency in enumerate(dependencies):
            if not _PACKAGE_ID_RE.fullmatch(dependency.package_id):
                errors.append(_issue("PKG015", f"dependencies[{index}].package", "Dependency package ID must be lowercase and namespace-qualified"))
            if dependency.package_id in dependency_ids:
                errors.append(_issue("PKG015", f"dependencies[{index}].package", "Duplicate dependency declaration"))
            dependency_ids.add(dependency.package_id)
    except (InvalidSpecifier, InvalidVersion, ValueError) as exc:
        errors.append(_issue("PKG015", "dependencies", str(exc)))

    validation = _mapping(raw.get("validation"))
    level = str(validation.get("declared_level", ""))
    if level not in {item.value for item in ValidationLevel}:
        errors.append(_issue("PKG010", "validation.declared_level", "Unsupported validation level"))

    capabilities = _mapping(raw.get("capabilities"))
    training = _mapping(capabilities.get("training"))
    if training.get("optimizer_update") and not training.get("parameter_gradients"):
        errors.append(_issue("PKG009", "capabilities.training", "Optimizer update requires parameter gradients"))
    if any(training.get(key) for key in ("backward_input", "parameter_gradients", "bias_gradients", "optimizer_update")) and not training.get("forward"):
        errors.append(_issue("PKG009", "capabilities.training", "Training backward or update support requires training forward support"))

    _validate_entrypoints(raw, asset_type, errors)

    for field, declared_path in _iter_declared_paths(raw.get("entrypoints", {}), "entrypoints"):
        if not _is_safe_relative_path(declared_path):
            errors.append(_issue("PKG007", field, "Path must remain inside the package root"))
            continue
        if not (root / declared_path).is_file():
            errors.append(_issue("PKG013", field, f"Declared package file does not exist: {declared_path}"))

    interfaces = _mapping(raw.get("interfaces"))
    valid_protocols = {item.value for item in InterfaceProtocol}
    for name, interface in interfaces.items():
        protocol = str(_mapping(interface).get("protocol", ""))
        if protocol and protocol not in valid_protocols:
            errors.append(_issue("PKG012", f"interfaces.{name}.protocol", "Unsupported interface protocol"))

    if not (root / "README.md").is_file():
        warnings.append(_issue("PKGW001", "README.md", "Package README.md is missing", warning=True))

    return PackageValidationResult(
        "passed" if not errors else "failed",
        manifest.schema,
        package_id,
        tuple(errors),
        tuple(warnings),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an FPGAI research package manifest")
    parser.add_argument("package_root", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    result = validate_package_manifest(args.package_root)
    if args.as_json:
        print(result.to_json())
    else:
        print(f"status: {result.status}")
        print(f"package_id: {result.package_id or 'unknown'}")
        print(f"errors: {len(result.errors)}")
        print(f"warnings: {len(result.warnings)}")
        for issue in (*result.errors, *result.warnings):
            print(f"{issue.code} {issue.field}: {issue.message}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
