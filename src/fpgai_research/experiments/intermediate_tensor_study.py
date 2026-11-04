"""Evidence about Model IR refinement preserving *every* operator's tensor semantics.

This is stronger than final-output accuracy alone, but intentionally limited
to GCC float mode and deterministic, static-shape research fixtures.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from fpgai_research.architecture.resolver import resolve
from fpgai_research.experiments.branch_ir_study import make_branched_conv_model
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import verify_refinement
from fpgai_research.validation.tensor_trace import check_tensor_trace


def run_intermediate_tensor_study(*, seed: int = 2026, samples: int = 3,
                                  output: str | Path | None = None) -> dict:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    if type(samples) is not int or not 1 <= samples <= 20:
        raise ValueError('samples must be 1..20')
    model, fixture = make_branched_conv_model()
    rng = np.random.default_rng(seed)
    inputs = [fixture, *(rng.uniform(-.5, .5, size=fixture.shape).astype(np.float32)
                         for _ in range(samples - 1))]
    base = ArchitectureRequest()
    mapped = ArchitectureRequest(pes=2, parallelism=2, simd=2,
                                 serialization=2, tile=(1, 2, 2, 3, 1, 1, 1))
    options = (
        ('baseline', base, {}),
        ('branch_a_spatial', base, {'conv_a':mapped}),
        ('branch_b_spatial', base, {'conv_b':mapped}),
        ('both_branches', base, {'conv_a':mapped,'conv_b':mapped}),
        ('branch_a_pipeline', base, {'conv_a':replace(base,pipeline_ii=2)}),
        ('branch_a_serial', base, {'conv_a':replace(base,serialization=3)}),
        ('precision_12', replace(base,precision_bits=12), {}),
    )
    cases = []
    baseline_functional_hash = None
    for label, default, overrides in options:
        arch = resolve(model, default, overrides)
        lowered = lower(arch)
        stage = verify_refinement(lowered)
        if baseline_functional_hash is None:
            baseline_functional_hash = stage['functional_sha256']
        if stage['functional_sha256'] != baseline_functional_hash:
            raise AssertionError(f'Functional graph changed in {label}')
        attempts = [check_tensor_trace(lowered, x) for x in inputs]
        if any(not item['accepted'] for item in attempts):
            raise AssertionError(f'An operator tensor changed under refinement {label}')
        cases.append({
            'architecture_label': label,
            'functional_sha256': stage['functional_sha256'],
            'effective_architecture_sha256': stage['effective_architecture_sha256'],
            'source_sha256': attempts[0]['cpp_sha256'],
            'operator_tensors_checked': sum(item['operator_count'] for item in attempts),
            'tensor_elements_checked': sum(item['total_values'] for item in attempts),
            'all_operator_tensors_accepted': True,
            'maximum_absolute_error': max(
                record['max_absolute_error']
                for sample in attempts for record in sample['operators']),
            'sample_trace_sha256': [item['raw_trace_sha256'] for item in attempts],
        })
    out = {
        'schema':'fpgai-research/intermediate-tensor-refinement-study/v1',
        'seed':seed, 'sample_count':len(inputs),
        'graph_kind':'branched-two-Conv-Add-Relu-GlobalAveragePool',
        'operator_count':len(model.operators),
        'functional_sha256':baseline_functional_hash,
        'case_count':len(cases),
        'total_operator_boundaries_checked':sum(case['operator_tensors_checked'] for case in cases),
        'all_cases_passed':all(case['all_operator_tensors_accepted'] for case in cases),
        'case_reports':cases,
        'evidence':'E0-E1-and-GCC-host-float-intermediate-tensor-check',
        'vitis_executed':False,'fixed_point_bit_exact':False,
        'limitations':'Finite test cases; host-floating C++ and NumPy only; not fixed-point, RTL or physical evidence',
    }
    if output is not None:
        path=Path(output)
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite scientific result: {path}')
        path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    return out
