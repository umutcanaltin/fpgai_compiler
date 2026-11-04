"""Property-wise model, region, operator and supported loop-scoped requests.

Loop tile extents lower into concrete operator tile loops. SIMD/pipeline
requests are supported only for the reduction loops that HLS materializes.
Unknown or conflicting scopes produce explicit blockers.
"""
from __future__ import annotations
from dataclasses import fields, replace
from typing import Mapping
from fpgai_research.model_ir.contracts import ArchitectureRequest, ArchitecturalModel, FunctionalModel
from fpgai_research.architecture.resolver import ArchitectureError, TargetLimits, iteration_shape, resolve

_KEYS={f.name for f in fields(ArchitectureRequest)}
_LOOPS={"MatMul":("m","n","k"),"Gemm":("m","n","k"),
        "Conv":("batch","oc","oh","ow","ic","ky","kx")}
_TILE_LOOPS={"MatMul":("m","n","k"),"Gemm":("m","n","k"),
             "Conv":("oc","oh","ow")}
_REDUCTION_LOOP={"MatMul":"k","Gemm":"k","Conv":"ic"}

def _settings(name: str, settings: Mapping) -> dict:
    if not isinstance(settings,Mapping):
        raise ArchitectureError(name,"SCOPE","Settings must be an object")
    unknown=set(settings)-_KEYS
    if unknown:
        raise ArchitectureError(name,"SCOPE_KEYS",str(sorted(unknown)))
    return {key:(tuple(value) if key=="tile" else value)
            for key,value in settings.items()}

def resolve_scoped(model: FunctionalModel, default: Mapping | ArchitectureRequest,
                   *, regions: Mapping[str,Mapping] | None=None,
                   operators: Mapping[str,Mapping] | None=None,
                   loops: Mapping[str,Mapping] | None=None,
                   limits: TargetLimits=TargetLimits()) -> ArchitecturalModel:
    """Resolve model < region < operator < compatible named loop scope.

    regions={"stem":{"operators":["mm"],"architecture":{"pes":2}}}
    operators={"mm":{"simd":4}}
    loops={"mm.k":{"pipeline_ii":2,"tile_extent":8}}
    """
    model.validate()
    model_default=(default if isinstance(default,ArchitectureRequest)
                   else ArchitectureRequest(**_settings("model",default)))
    groups=regions or {}
    operator_settings=operators or {}
    loop_settings=loops or {}
    op_by_name={op.name:op for op in model.operators}
    if set(operator_settings)-set(op_by_name):
        raise ArchitectureError("<model>","UNKNOWN_SCOPE","Unknown operator override")
    memberships={}
    for region,declaration in groups.items():
        if set(declaration)-{"operators","architecture"}:
            raise ArchitectureError(region,"SCOPE_KEYS","Region only accepts operators/architecture")
        names=declaration.get("operators",())
        if not isinstance(names,(list,tuple)) or len(names)!=len(set(names)):
            raise ArchitectureError(region,"SCOPE","Invalid operator region membership")
        for name in names:
            if name not in op_by_name:
                raise ArchitectureError(region,"UNKNOWN_SCOPE",str(name))
            if name in memberships:
                raise ArchitectureError(name,"AMBIGUOUS_REGION","Operator is in multiple regions")
            memberships[name]=region
        _settings(region,declaration.get("architecture",{}))
    resolved={}
    decisions={}
    for name,op in op_by_name.items():
        values={key:getattr(model_default,key) for key in _KEYS}
        sources={key:"model" for key in _KEYS}
        if name in memberships:
            region=memberships[name]
            for key,value in _settings(region,groups[region].get("architecture",{})).items():
                values[key]=value
                sources[key]="region:"+region
        for key,value in _settings(name,operator_settings.get(name,{})).items():
            values[key]=value
            sources[key]="operator:"+name
        resolved[name]=values
        decisions[name]=sources
    for identity,patch in loop_settings.items():
        if not isinstance(identity,str) or "." not in identity:
            raise ArchitectureError(str(identity),"LOOP_SCOPE","Expected operator.loop")
        name,loop=identity.rsplit(".",1)
        if name not in op_by_name or loop not in _LOOPS.get(op_by_name[name].kind,()):
            raise ArchitectureError(identity,"LOOP_SCOPE","Unsupported operator/loop")
        if not isinstance(patch,Mapping) or set(patch)-{"tile_extent","simd","pipeline_ii"}:
            raise ArchitectureError(identity,"LOOP_SCOPE","Unsupported loop property")
        if "tile_extent" in patch:
            if loop not in _TILE_LOOPS[op_by_name[name].kind]:
                raise ArchitectureError(identity,"LOOP_CAPABILITY","Cannot tile this loop")
            current=resolved[name]["tile"] or iteration_shape(model,op_by_name[name])
            tile=list(current)
            tile[_LOOPS[op_by_name[name].kind].index(loop)]=patch["tile_extent"]
            resolved[name]["tile"]=tuple(tile)
            decisions[name]["tile["+loop+"]"]="loop:"+identity
            decisions[name]["tile"]="loop:"+identity
        for field in ("simd","pipeline_ii"):
            if field in patch:
                if loop!=_REDUCTION_LOOP[op_by_name[name].kind]:
                    raise ArchitectureError(identity,"LOOP_CAPABILITY",
                                            "Only reduction loop accepts SIMD or pipeline II")
                resolved[name][field]=patch[field]
                decisions[name][field]="loop:"+identity
    selected={name:ArchitectureRequest(**values) for name,values in resolved.items()}
    arch=resolve(model,model_default,selected,limits)
    updated=[]
    for point in arch.points:
        sources=decisions[point.operator]
        winner=next((p for p in ("loop:","operator:","region:")
                     if any(v.startswith(p) for v in sources.values())),"model")
        updated.append(replace(point,source_scope=winner.removesuffix(":") if winner!="model" else "model",
                               decision_sources=tuple(sorted(sources.items()))))
    return ArchitecturalModel(model,tuple(updated))
