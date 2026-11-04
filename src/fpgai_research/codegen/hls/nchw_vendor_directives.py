"""Explicit AMD Vitis HLS 2023.2 native scratch binding requests.

Architecture intent and source-level policy are not physical facts. Use
identical requested vendor directives in disjoint/shared source pairs.
No GCC or source parser can establish that Vitis honored a pragma:
original CSynth, RTL and physical evidence are separate.
"""
from __future__ import annotations

from hashlib import sha256
import re

from fpgai_research.codegen.hls.emitter import Emission

POLICIES=("auto","bram_2p")
PRAGMA="#pragma HLS BIND_STORAGE variable=__cst_scratch type=ram_2p impl=bram"


def apply_native_scratch_binding(emission:Emission,*,policy:str="auto")->Emission:
    if policy not in POLICIES or type(policy) is not str:
        raise ValueError("Only supported Vitis source memory binding policies admitted")
    if (emission.manifest.get("shared_source_memory")!=
            "actual-local-C++-scratch-pointer-aliases" or
            emission.manifest.get("cpp_sha256")!=sha256(emission.cpp.encode()).hexdigest()):
        raise ValueError("Vendor pragma needs source-attested generated local shared scratch")
    cpp=emission.cpp
    matched=list(re.finditer(
        r"(?m)^  data_t __cst_scratch\[([0-9]+)\];$",cpp))
    if len(matched)!=1:
        raise ValueError("Expected exactly one concrete C++ scratch declaration")
    reported=emission.manifest.get("source_scratch_slots")
    if type(reported) is not int or int(matched[0].group(1))!=reported:
        raise ValueError("C++ source scratch geometry and source manifest disagree")
    if PRAGMA in cpp:
        raise ValueError("Existing Vitis scratch pragma must not be duplicated")
    if policy=="bram_2p":
        anchor=matched[0].group(0)
        cpp=cpp.replace(anchor,anchor+"\n  "+PRAGMA,1)
    return Emission(cpp,{
        **emission.manifest,
        "cpp_sha256":sha256(cpp.encode()).hexdigest(),
        "pre_memory_directive_cpp_sha256":emission.manifest["cpp_sha256"],
        "requested_native_memory_binding":policy,
        "requested_native_pragma":PRAGMA if policy=="bram_2p" else None,
        "binding_physically_verified":False,
        "native_vitis_calls":0,
    })
