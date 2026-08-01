from __future__ import annotations
import importlib.util, inspect, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from fpgai.contracts.package_manifest import load_package_manifest
from .external_definition import ExternalOperatorDefinition
from .load_request import OperatorLoadRequest
from .loading_errors import OperatorLoadIssue
from .onnx_binding_registry import OnnxBindingRegistry, RegisteredOnnxBinding

@dataclass
class ExternalOperatorContext:
    binding_registry: OnnxBindingRegistry = field(default_factory=OnnxBindingRegistry)
    definitions: dict[str,ExternalOperatorDefinition] = field(default_factory=dict)
    provenance: dict[str,dict[str,Any]] = field(default_factory=dict)
    def reference_for(self, operator_id: str):
        d=self.definitions.get(operator_id); return d.numeric_reference if d else None

@dataclass(frozen=True)
class OperatorLoadResult:
    context: ExternalOperatorContext
    loaded: tuple[str,...]=()
    errors: tuple[OperatorLoadIssue,...]=()
    warnings: tuple[OperatorLoadIssue,...]=()
    @property
    def ok(self): return not self.errors
    def to_dict(self):
        return {"status":"passed" if self.ok else "failed","loaded":list(self.loaded),"errors":[x.to_dict() for x in self.errors],"warnings":[x.to_dict() for x in self.warnings]}

def _safe_module_path(root: Path, relative: str) -> Path:
    rel=Path(relative)
    if rel.is_absolute() or ".." in rel.parts: raise ValueError("Unsafe module path")
    path=(root/rel).resolve()
    if root.resolve() not in path.parents or not path.is_file(): raise ValueError("Module path escapes package root or does not exist")
    return path

def _load_symbol(path: Path, symbol: str, unique: str):
    spec=importlib.util.spec_from_file_location(unique,path)
    if spec is None or spec.loader is None: raise ImportError(f"Unable to load {path}")
    mod=importlib.util.module_from_spec(spec); sys.modules[unique]=mod
    try: spec.loader.exec_module(mod)
    finally: sys.modules.pop(unique,None)
    if not hasattr(mod,symbol): raise AttributeError(symbol)
    return getattr(mod,symbol)

def load_operator_packages(request: OperatorLoadRequest) -> OperatorLoadResult:
    ctx=ExternalOperatorContext(); errors=[]; loaded=[]
    if request.trust_level!="approved_for_reference":
        return OperatorLoadResult(ctx,errors=(OperatorLoadIssue("OPLOAD002","trust_level","Package code requires approved_for_reference"),))
    for package_id in request.package_ids:
        candidates=request.catalogue.find_by_package_id(package_id)
        if not candidates:
            errors.append(OperatorLoadIssue("OPLOAD001",package_id,"Package not found")); continue
        entry=sorted(candidates,key=lambda e:(e.priority,e.version),reverse=True)[0]
        if entry.asset_type!="operator" or entry.source_path is None:
            errors.append(OperatorLoadIssue("OPLOAD003",package_id,"Operator package entrypoint is unavailable")); continue
        manifest=load_package_manifest(entry.source_path)
        op_ep=dict(manifest.raw.get("entrypoints",{})).get("operator",{})
        if not isinstance(op_ep,dict) or not op_ep.get("python_module") or not op_ep.get("symbol"):
            errors.append(OperatorLoadIssue("OPLOAD003",package_id,"Missing operator python_module or symbol")); continue
        try:
            path=_safe_module_path(entry.source_path,str(op_ep["python_module"]))
            factory=_load_symbol(path,str(op_ep["symbol"]),f"_fpgai_plugin_{entry.manifest_hash.replace(':','_')}")
            if not callable(factory) or len(inspect.signature(factory).parameters)!=0: raise TypeError("Operator factory must accept no arguments")
            definition=factory()
            if not isinstance(definition,ExternalOperatorDefinition): raise TypeError("Factory did not return ExternalOperatorDefinition")
            binding_errors = []
            for binding in definition.contract.onnx_bindings:
                binding_errors.extend(ctx.binding_registry.register(RegisteredOnnxBinding(package_id,entry.version,entry.manifest_hash,binding,definition)))
            if binding_errors:
                errors.extend(binding_errors)
                continue
            ctx.definitions[definition.contract.operator_id]=definition
            ctx.provenance[definition.contract.operator_id]={"package_id":package_id,"version":entry.version,"manifest_hash":entry.manifest_hash}
            loaded.append(package_id)
        except ValueError as exc: errors.append(OperatorLoadIssue("OPLOAD004",package_id,str(exc)))
        except AttributeError as exc: errors.append(OperatorLoadIssue("OPLOAD006",package_id,f"Symbol not found: {exc}"))
        except Exception as exc: errors.append(OperatorLoadIssue("OPLOAD005",package_id,str(exc)))
    return OperatorLoadResult(ctx,tuple(loaded),tuple(errors))
