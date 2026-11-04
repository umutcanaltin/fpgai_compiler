"""Controlled E0 ablations of Phi-conditioned multi-read boundary certificates.

Comparator modes are *explicitly weaker, intentionally incomplete proxies*;
they are NOT reported as implementations of Allo, HIDA, ST-Flow or others.
No mode other than FULL can certify the research target specification.
Measured 'false accepts' are RELATIVE TO the full bounded typed checker,
not claims of runtime failure or RTL proof.
"""
from __future__ import annotations

from dataclasses import replace
from fpgai_research.architecture.resolver import coordinate
from fpgai_research.model_ir.branch_matmul_contract import (
    BranchEvent,admit_branch,validate_branch_events,
)
from fpgai_research.model_ir.fanout_lifetime import (
    LifetimeEvent,verify_fanout_events,
)

MODES=("metadata_only","first_read_ack","read_cardinality_no_phi","full_phi_and_last_use")


def _first_read_ack(arch,offsets,events):
    # Deliberately unsound proxy: first read is mistaken for the
    # consumer's last read; extra reduction contributions are hidden.
    seen=set()
    collapsed=[]
    for e in events:
        if e.kind=="COMMIT":
            collapsed.append(LifetimeEvent("COMMIT",e.tensor,e.element))
        elif e.kind=="READ":
            key=(e.tensor,e.element,e.consumer)
            if key not in seen:
                seen.add(key)
                collapsed.append(LifetimeEvent("CONSUME",e.tensor,e.element,e.consumer))
        else:
            return False
    return verify_fanout_events(
        arch.functional,offsets,tuple(collapsed),
        max_events=max(1,len(collapsed)))["status"]=="VERIFIED"


def _normalize_owner(arch,events):
    (m,k,n,q),(h,a,b,g),_,ops=admit_branch(arch.functional)
    normalized=[]
    for e in events:
        if e.kind=="READ" and e.tensor==h and e.consumer in (ops[1].name,ops[2].name):
            mi,kk=divmod(e.element,n)
            row,ni=divmod(e.output_element,q)
            if row==mi and 0<=ni<q:
                stage=1 if e.consumer==ops[1].name else 2
                point=coordinate(arch.points[stage],(mi,ni,kk))
                owner=(point["replica"],point["pe"],point["lane"],
                       point["reuse_step"][2])
                normalized.append(replace(e,phi_owner=owner))
                continue
        normalized.append(e)
    return tuple(normalized)


def ablation_admission(arch,offsets,events,*,mode:str):
    if mode not in MODES:
        raise ValueError("Unsupported scientific ablation comparator")
    if mode=="metadata_only":
        # A graph hash cannot verify read ownership or scratch correctness.
        # We still require the public argument types; no fabricated
        # certificate is produced for this proxy.
        return type(offsets) is dict and type(events) is tuple
    if mode=="first_read_ack":
        return _first_read_ack(arch,offsets,events)
    if mode=="read_cardinality_no_phi":
        events=_normalize_owner(arch,events)
    return validate_branch_events(arch,offsets,events)["status"]=="VERIFIED_E0"
