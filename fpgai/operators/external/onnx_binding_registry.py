from __future__ import annotations
from dataclasses import dataclass
from fpgai.operators import OnnxBinding
from .external_definition import ExternalOperatorDefinition
from .loading_errors import OperatorLoadIssue

@dataclass(frozen=True)
class RegisteredOnnxBinding:
    package_id: str
    package_version: str
    manifest_hash: str
    binding: OnnxBinding
    definition: ExternalOperatorDefinition
    def supports(self, domain: str, op_type: str, opset: int) -> bool:
        b=self.binding
        return b.domain==domain and b.op_type==op_type and (b.opset_min is None or opset>=b.opset_min) and (b.opset_max is None or opset<=b.opset_max)

class OnnxBindingRegistry:
    def __init__(self): self._items: list[RegisteredOnnxBinding]=[]
    def register(self, item: RegisteredOnnxBinding) -> tuple[OperatorLoadIssue,...]:
        for other in self._items:
            if other.binding.domain!=item.binding.domain or other.binding.op_type!=item.binding.op_type: continue
            lo=max(other.binding.opset_min or 1,item.binding.opset_min or 1)
            hi=min(other.binding.opset_max or 10**9,item.binding.opset_max or 10**9)
            if lo<=hi and other.package_id!=item.package_id:
                return (OperatorLoadIssue("OPLOAD009","onnx_bindings",f"Overlapping ONNX binding with {other.package_id}"),)
        self._items.append(item); return ()
    def resolve(self, domain: str, op_type: str, opset: int, package_id: str|None=None) -> RegisteredOnnxBinding|None:
        found=[x for x in self._items if x.supports(domain,op_type,opset) and (package_id is None or x.package_id==package_id)]
        if len(found)>1: raise ValueError("Ambiguous external ONNX binding")
        return found[0] if found else None
    def inventory(self): return tuple(sorted(self._items,key=lambda x:(x.binding.domain,x.binding.op_type,x.package_id)))
