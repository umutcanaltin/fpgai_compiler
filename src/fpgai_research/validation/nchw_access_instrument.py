"""Opt-in C++ proxy-backed observation of *actual* NCHW array accesses.

Original HLS source remains unchanged. The returned source wraps the
real operator subscripts, not a separately emitted intended event list.
Active only under GCC FPGAI_HOST_FLOAT+FPGAI_TRACE_NCHW_EVENTS.
"""
from __future__ import annotations
from hashlib import sha256
import re
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.model_ir.nchw_graph_contract import admit_nchw_graph
from fpgai_research.model_ir.semantics import functional_fingerprint

SCHEMA="fpgai-research/actual-NCHW-GCC-access-source/v1"

PROXY=r"""
#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_NCHW_EVENTS)
struct __cst_ref {
  data_t* ptr; int tensor, index, stage;
  operator data_t() const {
    std::printf("CST_ACCESS R %d %d %d%c",stage,tensor,index,10);
    return ptr[index];
  }
  __cst_ref& operator=(data_t value) {
    ptr[index]=value;
    std::printf("CST_ACCESS W %d %d %d%c",stage,tensor,index,10);
    return *this;
  }
  __cst_ref& operator=(const __cst_ref& value) {
    return (*this)=static_cast<data_t>(value);
  }
};
struct __cst_view {
  data_t* ptr; int tensor; int* stage;
  __cst_ref operator[](int index) const {
    return __cst_ref{ptr,tensor,index,*stage};
  }
};
#endif
"""

def instrument_nchw_source(arch,source:Emission):
    model,produced=admit_nchw_graph(arch)
    if (source.manifest.get("cpp_sha256")!=sha256(source.cpp.encode()).hexdigest()
            or source.manifest.get("functional_sha256")!=functional_fingerprint(model)
            or len(source.manifest.get("operators",()))!=len(model.operators)):
        raise ValueError("Original source-provenanced NCHW graph required")
    for op,point,receipt in zip(model.operators,arch.points,source.manifest["operators"]):
        if (receipt["operator"]!=op.name
                or receipt["effective"]!=vars(point.effective)):
            raise ValueError("Original operator Phi receipt mismatch")
    head='extern "C" void fpgai_model('
    if source.cpp.count(head)!=1:
        raise ValueError("Missing unique original emitted top")
    cpp=source.cpp.replace(head,PROXY+"\n"+head,1)
    lines=cpp.splitlines()
    indices={t.name:i for i,t in enumerate(model.tensors)}
    first=next((i for i,s in enumerate(lines) if s.startswith("  // OP ")),None)
    if first is None:
        raise ValueError("Generated source has no operator marker")
    views=["  int __cst_stage=-1;"]
    for name in sorted(produced):
        i=indices[name]
        symbol=f"tensor_{i}"
        views.extend([
            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_NCHW_EVENTS)",
            f"  __cst_view __cst_view_{i}{{{symbol},{i},&__cst_stage}};",
            "#else",
            f"  data_t* __cst_view_{i}={symbol};",
            "#endif",
        ])
    lines[first:first]=views
    out=[]
    ordinal=-1
    inside=False
    substitutions=0
    for line in lines:
        if line.startswith("  // OP "):
            ordinal+=1
            out.append(f"  __cst_stage={ordinal};")
            inside=True
        if line=="#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_TENSORS)":
            inside=False
        if inside:
            for name in produced:
                i=indices[name]
                line,n=re.subn(r"\btensor_"+str(i)+r"(?=\[)",
                               f"__cst_view_{i}",line)
                substitutions+=n
        out.append(line)
    if ordinal+1!=len(model.operators) or substitutions<2*len(model.operators):
        raise ValueError("Not all source operator array access sites instrumented")
    transformed="\n".join(out)+"\n"
    return Emission(transformed,{
        **source.manifest,
        "schema":SCHEMA,
        "cpp_sha256":sha256(transformed.encode()).hexdigest(),
        "original_cpp_sha256":source.manifest["cpp_sha256"],
        "real_nchw_access_opt_in":True,
        "instrumented_array_access_sites":substitutions,
        "native_vitis_calls":0,
        "hardware_qualified":False,
    })
