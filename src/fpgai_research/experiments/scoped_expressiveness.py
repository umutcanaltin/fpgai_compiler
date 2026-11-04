"""Bounded empirical expressiveness study for independent operator-local Model IR mappings.

The experiment counts *actual Phi mapping identities*, not merely distinct
parameter tuples. It does not measure latency, resources or synthesis QoR.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np

from fpgai_research.architecture.phi import iteration_mapping_receipt
from fpgai_research.architecture.resolver import resolve
from fpgai_research.model_ir.contracts import ArchitectureRequest, FunctionalModel, Operator, Tensor
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import functional_fingerprint, verify_refinement


def _chain(operators: int, *, seed: int) -> FunctionalModel:
    rng = np.random.default_rng(seed)
    tensors = [Tensor('x',(1,3))]
    nodes = []
    last = 'x'
    for i in range(operators):
        weights=f'w{i}'
        output=f'y{i}'
        tensors.append(Tensor(weights,(3,3),data=tuple(map(float,rng.uniform(-.5,.5,9)))))
        tensors.append(Tensor(output,(1,3)))
        nodes.append(Operator(f'mm{i}','MatMul',(last,weights),(output,)))
        last=output
    return FunctionalModel(tuple(tensors),tuple(nodes),('x',),(last,))


def run_scoped_expressiveness_study(*, seed=2026, max_operators=4,
                                    output: str | Path | None = None) -> dict:
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('Seed must be a 32-bit unsigned integer')
    if type(max_operators) is not int or not 1<=max_operators<=5:
        raise ValueError('max_operators must be 1..5')
    pe_choices=(1,2,3)
    results=[]
    for length in range(1,max_operators+1):
        model=_chain(length,seed=seed)
        functional_hash=functional_fingerprint(model)
        identities=set()
        global_identifiers=set()
        for values in product(pe_choices, repeat=length):
            overrides={f'mm{i}':ArchitectureRequest(pes=values[i]) for i in range(length)}
            architectural=resolve(model,ArchitectureRequest(),overrides)
            refinement=verify_refinement(lower(architectural))
            if refinement['functional_sha256']!=functional_hash:
                raise AssertionError('Functional graph changed during operator-local resolution')
            signature=tuple(iteration_mapping_receipt(point,max_iterations=1000)
                            ['complete_mapping_sha256'] for point in architectural.points)
            if signature in identities:
                raise AssertionError('Distinct PE settings were not distinguishable in the studied Phi mappings')
            identities.add(signature)
            if len(set(values))==1:
                global_identifiers.add(signature)
        expected_local=len(pe_choices)**length
        if len(identities)!=expected_local or len(global_identifiers)!=len(pe_choices):
            raise AssertionError('Mapping cardinality differs from operator-local selection hypothesis')
        results.append({
            'operators':length,
            'independent_pe_options':list(pe_choices),
            'global_only_observed_profiles':len(global_identifiers),
            'operator_local_observed_profiles':len(identities),
            'added_profiles':len(identities)-len(global_identifiers),
            'local_to_global_ratio':len(identities)/len(global_identifiers),
            'functional_sha256':functional_hash,
            'verified_source':'E0-bounded-Phi-mapping-enumeration',
        })
    outcome={
        'schema':'fpgai-research/scoped-expressiveness-study/v1',
        'evidence':'E0-actual-Phi-mapping-identities',
        'seed':seed,'choices_per_operator':list(pe_choices),
        'max_operators':max_operators,'cases':results,
        'total_operator_local_profiles_checked':sum(v['operator_local_observed_profiles'] for v in results),
        'native_vendor_tools_executed':False,
        'limitations':'Small chain fixtures and PE 1/2/3; representation count only, not FPGA designs physically measured',
    }
    if output is not None:
        target=Path(output)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            raise FileExistsError('Refusing to overwrite research experiment output')
        target.write_text(json.dumps(outcome,indent=2,sort_keys=True)+'\n')
    return outcome
