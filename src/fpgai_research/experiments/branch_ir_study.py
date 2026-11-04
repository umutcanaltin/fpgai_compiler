"""Representation-level generality check on a branched Conv→Add→Pool graph.

Shows stage identity, operator-local mapping, and HLS host-float agreement in
an actual nonsequential tensor dependency graph. Does not claim synthesis.
"""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

from fpgai_research.architecture.phi import iteration_mapping_receipt
from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.model_ir.contracts import ArchitectureRequest, FunctionalModel, Operator, Tensor
from fpgai_research.model_ir.explain import explain_progressive_ir
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import functional_fingerprint
from fpgai_research.validation.host import validate_host


def make_branched_conv_model() -> tuple[FunctionalModel, np.ndarray]:
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4) / 32
    a = (.1, -.2)
    b = (-.3, .2)
    tensors = (
        Tensor('x',(1,1,4,4)), Tensor('w_a',(2,1,1,1),data=a),
        Tensor('a',(1,2,4,4)), Tensor('w_b',(2,1,1,1),data=b),
        Tensor('b',(1,2,4,4)), Tensor('sum',(1,2,4,4)),
        Tensor('act',(1,2,4,4)), Tensor('output',(1,2,1,1)),
    )
    operations = (
        Operator('conv_a','Conv',('x','w_a'),('a',)),
        Operator('conv_b','Conv',('x','w_b'),('b',)),
        Operator('join','Add',('a','b'),('sum',)),
        Operator('relu','Relu',('sum',),('act',)),
        Operator('pool','GlobalAveragePool',('act',),('output',)),
    )
    return FunctionalModel(tensors,operations,('x',),('output',)), x


def run_branch_ir_study(*, output: str | Path | None = None, host_cpp: bool = False) -> dict:
    model, x = make_branched_conv_model()
    base = ArchitectureRequest()
    a = resolve(model,base)
    requested = replace(base,pes=2,simd=2,serialization=2,tile=(1,2,2,3,1,1,1),pipeline_ii=2)
    b = resolve(model,base,{'conv_a':requested})
    left = explain_progressive_ir(lower(a),mapping_bound=100_000)
    right = explain_progressive_ir(lower(b),mapping_bound=100_000)
    per_op_a={p.operator:iteration_mapping_receipt(p)['complete_mapping_sha256'] for p in a.points}
    per_op_b={p.operator:iteration_mapping_receipt(p)['complete_mapping_sha256'] for p in b.points}
    changed=sorted(k for k in per_op_a if per_op_a[k]!=per_op_b[k])
    expected=['conv_a']
    if changed!=expected:
        raise AssertionError('Operator-local mapping unexpectedly modified another graph node')
    for record in (left,right):
        assert [x['functional']['id'] for x in record['operator_stage_traces']] == [
            op.name for op in model.operators]
        assert record['invariants']['stage_invariants_verified']
    joins=[t for t in right['graph']['tensor_identities'] if t['id'] in ('a','b')]
    if sorted(t['consumers'] for t in joins)!=[['join'],['join']]:
        raise AssertionError('Branched tensor consumer identities were lost')
    result={
        'schema':'fpgai-research/branch-ir-study/v1',
        'evidence':'E0/E1-plus-optional-gcc-host-float',
        'model':'two Conv branches→Add→Relu→GlobalAveragePool',
        'functional_sha256':functional_fingerprint(model),
        'input_sha256':sha256(x.tobytes()).hexdigest(),
        'model_operators': [op.name for op in model.operators],
        'operators_preserved_in_both_stages': True,
        'only_changed_operator':changed,
        'other_operator_mapping_unchanged':all(per_op_a[k]==per_op_b[k] for k in per_op_a if k!='conv_a'),
        'baseline_cpp_sha256':emit_hls(lower(a)).manifest['cpp_sha256'],
        'specialized_cpp_sha256':emit_hls(lower(b)).manifest['cpp_sha256'],
        'vendor_tools_run':False,
        'fixed_point_bit_exact':False,
    }
    if host_cpp:
        reference=validate_host(model,emit_hls(lower(a)),x)
        specialized=validate_host(model,emit_hls(lower(b)),x)
        result['baseline_host_float_passed']=bool(reference['accepted'])
        result['specialized_host_float_passed']=bool(specialized['accepted'])
        result['baseline_max_abs_error']=reference['max_absolute_error']
        result['specialized_max_abs_error']=specialized['max_absolute_error']
        if not (reference['accepted'] and specialized['accepted']):
            raise AssertionError('The executed branched graph did not pass host-float agreement')
    if output is not None:
        path=Path(output); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result
