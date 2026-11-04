"""Independent Conv2D Phi, exact NCHW reduction read lifetime and GCC numeric."""
from dataclasses import replace
import shutil

import numpy as np
import pytest

from fpgai_research.architecture.resolver import resolve
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.experiments.cst_conv_boundary_fixtures import conv_diamond
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.conv_boundary import (
    admissible_conv_diamond,conv_contract,conv_logical_events,
    disjoint_conv_offsets,verify_conv_logical,
)
from fpgai_research.validation.conv_access_oracle import independent_conv_access
from fpgai_research.validation.host import validate_host


def mapped(model,*,different=True):
    base=ArchitectureRequest()
    overrides={}
    if different:
        overrides={
            "producer":replace(base,pes=2,simd=2,serialization=2),
            "reader1":replace(base,pes=2,simd=2,serialization=1),
        }
    return resolve(model,base,overrides)


@pytest.mark.parametrize("geometry",[
    (4,4,3,1,1),(5,6,3,0,1),(5,6,3,1,2),(3,4,1,0,1),
])
@pytest.mark.skipif(shutil.which("g++") is None,reason="Actual GCC required")
def test_conv_diamond_e0_and_real_vendor_style_generated_cpp_numeric(geometry):
    model=conv_diamond(height=geometry[0],width=geometry[1],
                       kernel=geometry[2],pad=geometry[3],stride=geometry[4])
    arch=mapped(model)
    offsets=disjoint_conv_offsets(arch)
    certified=conv_contract(arch,offsets)
    events=conv_logical_events(arch)
    independent=independent_conv_access(arch,events)
    assert certified["status"]=="VERIFIED_E0"
    assert independent["status"]=="VERIFIED_INDEPENDENT_CONV_READS_E0"
    assert independent["read_count"]==certified["logical_reads"]
    assert certified["conv_branch_intermediate_reads"]>0
    # Reuse is logically legal after all consumers finish, but the existing
    # general E1 emitter allocates its OWN arrays. No claim of emitted reuse.
    reused={**offsets,"out":offsets["h"]}
    assert verify_conv_logical(arch,reused,events)["status"]=="VERIFIED_E0"
    source=emit_hls(lower(arch))
    x=np.asarray([(-1)**i*(i+1)/17 for i in
                  range(geometry[0]*geometry[1])],
                 dtype=np.float32).reshape(1,1,geometry[0],geometry[1])
    assert validate_host(model,source,x)["accepted"]


def test_pad_skips_are_not_false_reads_and_stride_is_reflected():
    model=conv_diamond(height=5,width=6,kernel=3,pad=1,stride=2)
    arch=mapped(model, different=False)
    events=conv_logical_events(arch)
    independent=independent_conv_access(arch,events)
    assert independent["status"]=="VERIFIED_INDEPENDENT_CONV_READS_E0"
    h=model.tensor("h").shape
    assert h[2:]==(3,3)
    # Second 1x1 Conv should read each intermediate element once per output
    # channel; the other independent ReLU reader needs one more read.
    assert sum(e.kind=="READ" and e.tensor=="h" for e in events)==(
        3*len(range(h[1]*h[2]*h[3])))
    # Exact first-reader-only reuse is unsafe while the second Conv is pending.
    bad={**disjoint_conv_offsets(arch),"a":0}
    result=verify_conv_logical(arch,bad,events)
    assert result["status"]=="REJECTED"
    assert result["reason"]=="OVERWRITE_BEFORE_ALL_READS"


def test_independent_numpy_oracle_catches_in_bounds_wrong_read_and_e0_rejects_phi():
    model=conv_diamond(height=4,width=4,kernel=3,pad=1)
    arch=mapped(model)
    events=conv_logical_events(arch)
    offsets=disjoint_conv_offsets(arch)
    pos=next(i for i,e in enumerate(events)
             if e.kind=="READ" and e.consumer=="reader1")
    event=events[pos]
    faulty=events[:pos]+(replace(event,element=(event.element+1)%(
        model.tensor("h").shape[1]*model.tensor("h").shape[2]*model.tensor("h").shape[3])),)+events[pos+1:]
    assert independent_conv_access(arch,faulty)["status"]=="REJECTED_INDEPENDENT_CONV_READ_MAP"
    assert verify_conv_logical(arch,offsets,faulty)["reason"]=="INVALID_CONV_READ_OR_PHI"
    wrong_phi=events[:pos]+(replace(event,phi_owner=(99,0,0,0)),)+events[pos+1:]
    assert verify_conv_logical(arch,offsets,wrong_phi)["reason"]=="INVALID_CONV_READ_OR_PHI"
    missing=events[:pos]+events[pos+1:]
    assert verify_conv_logical(arch,offsets,missing)["reason"]=="INCOMPLETE_CONV_REDUCTION_OR_LAST_USE"


def test_conv_contract_refuses_group_dilation_invalid_source_and_event_budget():
    model=conv_diamond()
    arch=mapped(model,different=False)
    with pytest.raises(ValueError,match="event budget"):
        conv_logical_events(arch,max_events=1)
    with pytest.raises(ValueError,match="Bounded"):
        conv_diamond(kernel=0)
    for changed in ((("group",2),),(("dilations",(2,2)),)):
        orig=model.operators[2]
        r=replace(orig,attributes=changed)
        corrupted=replace(model,operators=model.operators[:2]+(r,)+model.operators[3:])
        with pytest.raises(ValueError):
            # The canonical resolver or the CST admission must refuse.
            ac=resolve(corrupted,ArchitectureRequest())
            admissible_conv_diamond(ac)


def test_mapping_change_changes_architecture_not_conv_math_or_read_cardinality():
    model=conv_diamond(height=4,width=5,kernel=3,pad=1)
    a=mapped(model,different=False)
    b=mapped(model,different=True)
    p=conv_contract(a,disjoint_conv_offsets(a))
    q=conv_contract(b,disjoint_conv_offsets(b))
    assert p["functional_sha256"]==q["functional_sha256"]
    assert p["architecture_sha256"]!=q["architecture_sha256"]
    assert p["logical_reads"]==q["logical_reads"]
    assert p["logical_commits"]==q["logical_commits"]
    assert p["source_event_sha256"]!=q["source_event_sha256"]
