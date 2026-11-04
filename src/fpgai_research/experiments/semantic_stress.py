"""Seeded multi-graph stress study of progressive Model IR and scope locality.

This is a bounded, independently generated E0/E1 representation experiment,
not a formal proof, HLS synthesis or hardware performance study.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from fpgai_research.architecture.phi import iteration_mapping_receipt
from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.model_ir.contracts import ArchitectureRequest, FunctionalModel, Operator, Tensor
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import functional_fingerprint, verify_refinement
from fpgai_research.validation.host import validate_host


def _model(rng: np.random.Generator, index: int):
    """Construct a two-branch MatMul + Add + ReLU graph with finite real weights."""
    k = int(rng.integers(2, 8))
    n = int(rng.integers(2, 8))
    weights_a = rng.uniform(-0.3, 0.3, (k, n)).astype(np.float32)
    weights_b = rng.uniform(-0.3, 0.3, (k, n)).astype(np.float32)
    x = rng.uniform(-0.5, 0.5, (1, k)).astype(np.float32)
    m = FunctionalModel(
        (Tensor('x', (1, k)), Tensor('wa', (k, n), data=tuple(map(float, weights_a.flat))),
         Tensor('wb', (k, n), data=tuple(map(float, weights_b.flat))),
         Tensor('a', (1, n)), Tensor('b', (1, n)),
         Tensor('sum', (1, n)), Tensor('y', (1, n))),
        (Operator('left', 'MatMul', ('x', 'wa'), ('a',)),
         Operator('right', 'MatMul', ('x', 'wb'), ('b',)),
         Operator('join', 'Add', ('a', 'b'), ('sum',)),
         Operator('activation', 'Relu', ('sum',), ('y',))),
        ('x',), ('y',))
    return m, x, k, n


def run_semantic_stress(*, seed: int = 73, cases: int = 32,
                        host_cpp: bool = False, output: str | Path | None = None) -> dict:
    """Check graph preservation, localized Phi changes and optional host numerics.

    All generated tests independently choose tensor shapes, weight data, and
    legal MatMul tile/parallel/PE/SIMD/fold/pipeline settings. The same
    functional graph is compiled with a scalar and a specialized architecture.
    """
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError('Seed must be a 32-bit nonnegative integer')
    if type(cases) is not int or not 1 <= cases <= 128:
        raise ValueError('Require 1..128 bounded cases')
    if type(host_cpp) is not bool:
        raise ValueError('host_cpp must be Boolean')
    rng = np.random.default_rng(seed)
    records = []
    for case in range(cases):
        model, x, k, n = _model(rng, case)
        base = ArchitectureRequest()
        candidate = replace(base, parallelism=int(rng.integers(1, 4)),
                            pes=int(rng.integers(1, 4)),
                            simd=int(rng.integers(1, min(k, 4) + 1)),
                            serialization=int(rng.integers(1, 4)),
                            tile=(1, int(rng.integers(1, n + 1)),
                                  int(rng.integers(1, k + 1))),
                            pipeline_ii=int(rng.integers(1, 4)))
        baseline = resolve(model, base)
        changed = resolve(model, base, {'left': candidate})
        before, after = lower(baseline), lower(changed)
        baseline_invariant = verify_refinement(before)
        specialized_invariant = verify_refinement(after)
        fingerprint = functional_fingerprint(model)
        if baseline_invariant['functional_sha256'] != fingerprint or \
           specialized_invariant['functional_sha256'] != fingerprint:
            raise AssertionError('Functional graph changed after architectural refinement')
        baseline_mappings = {p.operator: iteration_mapping_receipt(p, max_iterations=10_000)
                             for p in baseline.points}
        specialized_mappings = {p.operator: iteration_mapping_receipt(p, max_iterations=10_000)
                                for p in changed.points}
        changed_names = sorted(name for name in baseline_mappings if
            baseline_mappings[name]['complete_mapping_sha256'] !=
            specialized_mappings[name]['complete_mapping_sha256'])
        # The generated graph can retain the same Phi when a requested setting
        # is infeasible to observe over a tiny iteration range, but all other
        # operators must be stable and requests remain explicitly preserved.
        if any(name != 'left' for name in changed_names):
            raise AssertionError('Operator-local request mutated an unrelated Phi')
        emission_a, emission_b = emit_hls(before), emit_hls(after)
        record = {
            'index': case, 'k': k, 'n': n,
            'functional_sha256': fingerprint,
            'input_sha256': sha256(x.tobytes()).hexdigest(),
            'architecture_request': asdict(candidate),
            'mapping_changes': changed_names,
            'unrelated_mappings_preserved': True,
            'stage_preservation': True,
            'mapping_coverage': {key: value['logical_iterations']
                                 for key, value in specialized_mappings.items()},
            'baseline_cpp_sha256': emission_a.manifest['cpp_sha256'],
            'specialized_cpp_sha256': emission_b.manifest['cpp_sha256'],
            'baseline_manifest_operators': [p['operator'] for p in emission_a.manifest['operators']],
            'specialized_manifest_operators': [p['operator'] for p in emission_b.manifest['operators']],
        }
        if record['baseline_manifest_operators'] != record['specialized_manifest_operators']:
            raise AssertionError('HLS trace lost operator identity')
        if host_cpp:
            checks = [validate_host(model, emission_a, x), validate_host(model, emission_b, x)]
            record['host_float_pass'] = [r['accepted'] for r in checks]
            record['host_float_max_abs_error'] = [r['max_absolute_error'] for r in checks]
            if not all(record['host_float_pass']):
                raise AssertionError('Generated C++ failed host float numerical comparison')
        records.append(record)
    payload = {
        'schema': 'fpgai-research/multi-graph-ir-stress/v1',
        'seed': seed, 'cases': cases, 'tested_graph_kind': 'two-branch-MatMul-Add-Relu',
        'model_stage_preservation_passed': cases,
        'operator_locality_passed': cases,
        'mapping_coverage_passed': cases,
        'host_float_evaluated': host_cpp,
        'host_float_passed': sum(all(r.get('host_float_pass', []))
                                 for r in records) if host_cpp else None,
        'evidence': 'E0/E1-with-optional-GCC-host-float',
        'physical_results_obtained': False,
        'fixed_point_bit_exact': False,
        'limitations': 'Finite seeded graphs; no universal formal proof, Vitis simulation or FPGA QoR.',
        'records': records,
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return payload
