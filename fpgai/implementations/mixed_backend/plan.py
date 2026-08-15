from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True)
class BackendSegment:
    backend: str
    nodes: tuple[str, ...]


def build_mixed_backend_plan(graph: Any, selected_contracts: Mapping[str, Any]) -> dict:
    segments=[]; current_backend=None; current=[]; bridges=[]
    for op in getattr(graph,'ops',()):
        contract=selected_contracts.get(op.name)
        backend=(contract.backend if contract is not None else 'vitis_hls')
        if current_backend is None or backend==current_backend:
            current_backend=backend; current.append(op.name)
        else:
            segments.append(BackendSegment(current_backend,tuple(current)))
            prev=segments[-1]
            bridges.append({'from_backend':prev.backend,'to_backend':backend,'after_node':prev.nodes[-1],'before_node':op.name,'status':'bridge_required'})
            current_backend=backend; current=[op.name]
    if current: segments.append(BackendSegment(current_backend,tuple(current)))
    return {
      'schema':'fpgai.mixed-backend-plan/v1',
      'status':'planned',
      'segments':[{'backend':s.backend,'nodes':list(s.nodes)} for s in segments],
      'bridges':bridges,
      'direct_mixed_rtl_emission_supported':False,
      'policy':'Plan backend boundaries explicitly; direct HLS/VHDL RTL stitching is a later implementation stage.',
    }
