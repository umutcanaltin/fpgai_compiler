from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import shutil

import yaml

from fpgai.contracts.package_types import AssetType

_SUPPORTED = {item.value for item in AssetType}
_TYPE_ALIASES = {
    "layer": ("operator", None),
    "hls": ("implementation", "hls_cpp"),
    "hls_cpp": ("implementation", "hls_cpp"),
    "vhdl": ("implementation", "vhdl"),
    "verilog": ("implementation", "verilog"),
    "systemverilog": ("implementation", "systemverilog"),
}
_LANGUAGE_EXTENSIONS = {
    "hls_cpp": "cpp",
    "vhdl": "vhd",
    "verilog": "v",
    "systemverilog": "sv",
}


def supported_contribution_types() -> tuple[str, ...]:
    """Return canonical FPGAI Ecosystem asset types accepted by the scaffold."""
    return tuple(sorted(_SUPPORTED))


def _manifest(asset_type: str, package_id: str, name: str, language: str | None) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "schema": "fpgai.package/v1",
        "package": {
            "id": package_id,
            "name": name,
            "version": "0.1.0",
            "asset_type": asset_type,
            "provider": "external",
            "description": f"FPGAI Ecosystem {asset_type} contribution.",
        },
        "usage": {
            "platform_scope": "research",
            "permitted_uses": ["research", "experimentation", "validation", "benchmarking"],
            "production_path": "morfics",
        },
        "license": {"category": "open_source", "identifier": "Apache-2.0"},
        "compatibility": {"fpgai_contract": ">=1.0,<2.0"},
        "capabilities": {
            "inference": True,
            "training": {
                "forward": False,
                "backward_input": False,
                "parameter_gradients": False,
                "bias_gradients": False,
                "optimizer_update": False,
            },
        },
        "validation": {"declared_level": "unvalidated"},
        "ecosystem": {"role": ("model" if asset_type == "model" else "operator_semantics" if asset_type == "operator" else "operator_implementation" if asset_type == "implementation" else asset_type)},
    }

    if asset_type == "model":
        raw["entrypoints"] = {"model": {"path": "model/model.onnx", "format": "onnx"}}
        raw["validation"]["numeric"] = {
            "required": True,
            "reference": "source_model",
            "compare": ["fpgai_ir", "generated_backend"],
            "levels": ["model", "layer", "intermediate"],
        }
    elif asset_type == "operator":
        raw["entrypoints"] = {"operator": {"python_module": "python/operator.py"}}
        raw["validation"]["numeric"] = {"required": True, "reference": "operator_numeric_reference"}
    elif asset_type in {"implementation", "system_block", "adapter"}:
        lang = language or "hls_cpp"
        ext = _LANGUAGE_EXTENSIONS.get(lang, "cpp")
        backend = "vitis_hls" if lang == "hls_cpp" else lang
        raw["entrypoints"] = {
            "implementation": {
                "language": lang,
                "backend": backend,
                "top": "fpgai_top",
                "sources": [f"src/fpgai_top.{ext}"],
            }
        }
        raw["implementation"] = {
            "implements": {"operator_id": f"{package_id}.operator", "version": 1},
            "backend": backend,
        }
        raw["validation"]["numeric"] = {
            "required": True,
            "reference": "operator_semantics",
            "levels": ["layer"],
        }
        raw["export"] = {
            "standalone_source": True,
            "requires_hls_synthesis": False,
            "requires_vivado": False,
            "requires_bitstream": False,
        }
        raw["interfaces"] = {
            "input": {"protocol": "memory"},
            "output": {"protocol": "memory"},
        }
    return raw


def _write_readme(root: Path, *, asset_type: str, package_id: str, display_name: str) -> None:
    root.joinpath("README.md").write_text(
        f"# {display_name}\n\n"
        f"FPGAI Ecosystem `{asset_type}` contribution (`{package_id}`).\n\n"
        "This contribution is discovered through the existing FPGAI Ecosystem registry and uses "
        "`fpgai.package/v1` as its distribution manifest.\n\n"
        "Validate without executing contributor code:\n\n"
        "```bash\n"
        "fpgai ecosystem validate .\n"
        "```\n\n"
        "Discover it from a project root:\n\n"
        "```bash\n"
        "fpgai ecosystem discover --project-root /path/to/project\n"
        "```\n",
        encoding="utf-8",
    )


def scaffold_contribution(
    out_dir: str | Path,
    *,
    asset_type: str,
    package_id: str,
    name: str | None = None,
    language: str | None = None,
    force: bool = False,
) -> Path:
    """Create a contribution skeleton for the existing FPGAI Ecosystem.

    `fpgai.package/v1` remains the low-level package format; this API is the
    contributor-facing Ecosystem entry point.
    """
    requested_type = str(asset_type).strip().lower()
    alias = _TYPE_ALIASES.get(requested_type)
    if alias is not None:
        asset_type, alias_language = alias
        if language is None:
            language = alias_language
    else:
        asset_type = requested_type

    if asset_type not in _SUPPORTED:
        raise ValueError(f"ECO001: unsupported ecosystem contribution type {requested_type!r}")
    if language is not None and language not in _LANGUAGE_EXTENSIONS:
        raise ValueError(f"ECO003: unsupported implementation language {language!r}")

    root = Path(out_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(f"ECO002: output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    display_name = name or package_id
    manifest = _manifest(asset_type, package_id, display_name, language)
    root.joinpath("fpgai.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    _write_readme(root, asset_type=asset_type, package_id=package_id, display_name=display_name)

    if asset_type == "model":
        root.joinpath("model").mkdir(exist_ok=True)
        root.joinpath("model", "model.onnx").write_bytes(b"")
    elif asset_type == "operator":
        root.joinpath("python").mkdir(exist_ok=True)
        root.joinpath("python", "operator.py").write_text(
            '"""FPGAI Ecosystem external operator entrypoint scaffold."""\n\n'
            "# Implement the operator contract/import callbacks documented in\n"
            "# docs/ecosystem/adding_an_operator.md.\n",
            encoding="utf-8",
        )
    elif asset_type in {"implementation", "system_block", "adapter"}:
        lang = language or "hls_cpp"
        ext = _LANGUAGE_EXTENSIONS[lang]
        root.joinpath("src").mkdir(exist_ok=True)
        source = root.joinpath("src", f"fpgai_top.{ext}")
        if lang == "hls_cpp":
            source.write_text("void fpgai_top(const float *in, float *out) { out[0] = in[0]; }\n", encoding="utf-8")
        elif lang == "vhdl":
            source.write_text(
                "library ieee; use ieee.std_logic_1164.all;\n"
                "entity fpgai_top is end; architecture rtl of fpgai_top is begin end;\n",
                encoding="utf-8",
            )
        else:
            source.write_text("module fpgai_top(); endmodule\n", encoding="utf-8")

    return root


def export_implementation_artifact(
    package_id: str,
    out_dir: str | Path,
    *,
    project_root: str | Path = ".",
    directories: Iterable[str | Path] = (),
    force: bool = False,
) -> Path:
    """Export one discovered HLS/VHDL/RTL implementation without running tools.

    The export is intentionally source-level: it copies only files declared by
    the existing implementation contract and writes reusable contract/provenance
    manifests.  It never invokes Vitis HLS, Vivado, implementation, or bitstream
    generation.
    """
    from fpgai.discovery import DiscoveryRequest, discover_packages
    from fpgai.implementations.implementation_contract import implementation_contract_from_manifest

    result = discover_packages(DiscoveryRequest(
        project_root=project_root,
        configured_directories=tuple(directories),
        include_builtin=True,
        strict=False,
    ))
    if not result.ok:
        raise RuntimeError("ECOEXP001: ecosystem package discovery failed")
    entries = [entry for entry in result.catalogue.find_by_package_id(str(package_id)) if entry.asset_type == "implementation"]
    if not entries:
        raise FileNotFoundError(f"ECOEXP002: implementation package {package_id!r} was not discovered")
    entries.sort(key=lambda entry: (entry.priority, entry.version), reverse=True)
    entry = entries[0]
    if entry.source_path is None:
        raise RuntimeError(f"ECOEXP003: {package_id!r} has no exportable source package")

    contract = implementation_contract_from_manifest(entry.source_path, manifest_hash=entry.manifest_hash)
    root = Path(out_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(f"ECOEXP004: output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    package_root = Path(entry.source_path).resolve()
    declared = tuple(dict.fromkeys((*contract.sources, *contract.headers)))
    copied: list[str] = []
    for rel in declared:
        src = (package_root / rel).resolve()
        try:
            src.relative_to(package_root)
        except ValueError as exc:
            raise RuntimeError(f"ECOEXP005: declared source escapes package root: {rel}") from exc
        if not src.is_file():
            raise FileNotFoundError(f"ECOEXP006: declared implementation file is missing: {rel}")
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)

    # Keep the package manifest so the export remains independently inspectable.
    shutil.copy2(package_root / "fpgai.yaml", root / "fpgai.yaml")
    (root / "implementation_contract.json").write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    artifact_manifest = {
        "schema": "fpgai.artifact-export/v1",
        "status": "exported",
        "package_id": contract.package_id,
        "version": contract.version,
        "operator_id": contract.operator_id,
        "contribution_role": "operator_implementation",
        "implements": {"operator_id": contract.operator_id, "version": contract.semantics_version},
        "architecture_mapping": {key: dict(value) for key, value in contract.architecture_mapping.items()},
        "numeric_validation_required": bool((contract.metadata.get("validation", {}) or {}).get("numeric", {}).get("required", False)) if isinstance(contract.metadata.get("validation", {}), Mapping) else False,
        "language": contract.language,
        "backend": contract.backend,
        "top": contract.top,
        "files": copied,
        "granularity": "implementation_block",
        "build_stage": "source_export",
        "tool_execution": {"vitis_hls": False, "vivado": False, "bitstream": False},
        "validation_level": contract.validation_level,
        "capabilities": contract.to_dict().get("capabilities", {}),
        "source_package": str(package_root),
        "manifest_hash": contract.manifest_hash,
    }
    (root / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


__all__ = ["scaffold_contribution", "supported_contribution_types", "export_implementation_artifact"]
