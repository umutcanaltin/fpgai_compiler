from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PackageIssue:
    code: str
    field: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "field": self.field,
            "message": self.message,
            "severity": self.severity,
        }


ERROR_CODES = {
    "PKG001": "missing_manifest",
    "PKG002": "unsupported_schema",
    "PKG003": "invalid_package_id",
    "PKG004": "invalid_semantic_version",
    "PKG005": "unsupported_asset_type",
    "PKG006": "missing_license",
    "PKG007": "unsafe_path",
    "PKG008": "missing_entrypoint",
    "PKG009": "contradictory_capability",
    "PKG010": "invalid_validation_level",
    "PKG011": "invalid_research_scope",
    "PKG012": "invalid_manifest",
    "PKG013": "missing_required_file",
    "PKG014": "invalid_compatibility_range",
    "PKG015": "invalid_dependency",
    "PKG016": "invalid_ecosystem_role",
}

RESOLUTION_ERROR_CODES = {
    "PKGR001": "invalid_candidate",
    "PKGR002": "duplicate_identity_conflict",
    "PKGR003": "unsatisfied_version_range",
    "PKGR004": "missing_required_dependency",
    "PKGR005": "dependency_cycle",
    "PKGR006": "selected_version_conflict",
}

RESOLUTION_WARNING_CODES = {
    "PKGRW001": "missing_optional_dependency",
}

WARNING_CODES = {
    "PKGW001": "missing_readme",
    "PKGW002": "missing_validation_artifacts",
    "PKGW003": "broad_toolchain_claim",
    "PKGW004": "deprecated_manifest_field",
}
