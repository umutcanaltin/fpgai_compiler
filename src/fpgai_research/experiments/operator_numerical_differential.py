"""Read-only operator-local numerical differential of aligned private traces.

The ordered trace contract is explicit: equal semantic tensors must be
aligned by the researcher *before* auditing (same axes/layout, stage and
numeric interpretation). A first threshold-exceeding node is a diagnostic
candidate, not proof of the causal first erroneous arithmetic operator.
No vendor tools run, no intermediate traces are leaked to public outputs.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

SCHEMA="fpgai-research/three-way-operator-trace-manifest/v1"
OUT="fpgai-research/three-way-operator-divergence/v1"


def _regular(path,limit=128*1024*1024):
    p=Path(path)
    if p.is_symlink() or not p.is_file() or not 0<p.stat().st_size<=limit:
        raise ValueError("Missing or unsafe private trace source")
    return p


def _hex(s):
    return isinstance(s,str) and len(s)==64 and all(c in "0123456789abcdef" for c in s)


def _read_npz(file, expected_hash):
    path=_regular(file)
    if not _hex(expected_hash) or sha256(path.read_bytes()).hexdigest()!=expected_hash:
        raise ValueError("Source-bound intermediate trace SHA-256 mismatch")
    with np.load(path,allow_pickle=False) as source:
        if not source.files or len(source.files)>256:
            raise ValueError("Invalid trace tensor count")
        result={}
        for key in source.files:
            value=source[key]
            if value.dtype.kind not in "fiu" or not np.all(np.isfinite(value)):
                raise ValueError("Non-numeric or nonfinite intermediate trace")
            result[key]=np.asarray(value,dtype=np.float64)
    return result


def analyze_operator_traces(manifest_path):
    path=_regular(manifest_path,1024*1024)
    m=json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(m,dict) or m.get("schema")!=SCHEMA
            or m.get("tensor_contract")!="aligned_logical_values"
            or m.get("layout_and_shape_alignment_externally_qualified") is not False
            or m.get("native_execution_independently_attested") is not False
            or not _hex(m.get("original_onnx_sha256"))
            or not _hex(m.get("ordered_sample_cohort_sha256"))):
        raise ValueError("Source and tensor semantics missing in trace manifest")
    src=m.get("traces")
    if not isinstance(src,dict) or set(src)!={"reference","fpgai","hls4ml"}:
        raise ValueError("Exactly three comparator trace archives required")
    traces={}
    hashes={}
    for who,spec in src.items():
        if not isinstance(spec,dict) or set(spec)!={"file","sha256"}:
            raise ValueError("Missing complete trace source data")
        name=spec["file"]
        if (not isinstance(name,str) or Path(name).is_absolute()
                or len(Path(name).parts)!=1 or ".." in Path(name).parts
                or Path(name).suffix!=".npz"):
            raise ValueError("Trace source must be a local ordinary NPZ filename")
        traces[who]=_read_npz(path.parent/name,spec["sha256"])
        hashes[who]=spec["sha256"]
    steps=m.get("ordered_logical_tensors")
    if (not isinstance(steps,list) or not 1<=len(steps)<=128
            or not all(isinstance(row,dict) and
                       set(row)=={"operator","reference","fpgai","hls4ml"}
                       for row in steps)):
        raise ValueError("Incomplete strict logical operator mapping")
    tolerance=m.get("diagnostic_abs_tolerance")
    if type(tolerance) not in (float,int) or not isfinite(tolerance) or tolerance<=0:
        raise ValueError("Invalid explicitly preregistered trace diagnostic tolerance")
    cohort=m.get("sample_count")
    if type(cohort) is not int or not 1<=cohort<=10000:
        raise ValueError("Invalid common selected cohort count")
    seen=set()
    report=[]
    for step in steps:
        identity=step["operator"]
        if not isinstance(identity,str) or not identity or identity in seen:
            raise ValueError("Duplicate/invalid ordered operator identity")
        seen.add(identity)
        values=[]
        for who in ("reference","fpgai","hls4ml"):
            key=step[who]
            if not isinstance(key,str) or key not in traces[who]:
                raise ValueError("Logical operator missing corresponding trace tensor")
            values.append(traces[who][key])
        r,a,b=values
        if (r.shape!=a.shape or r.shape!=b.shape
                or r.ndim<2 or r.shape[0]!=cohort):
            raise ValueError("Unmatched operator tensor shape or sample cohort")
        if (r.size==0 or a.size==0 or b.size==0):
            raise ValueError("Empty operator tensor")
        max_a=float(np.max(np.abs(a-r)))
        max_b=float(np.max(np.abs(b-r)))
        report.append({
            "operator":identity,
            "logical_tensor_shape":list(r.shape),
            "fpgaI_max_absolute_difference":max_a,
            "hls4ml_max_absolute_difference":max_b,
            "fpgaI_mean_absolute_difference":float(np.mean(np.abs(a-r))),
            "hls4ml_mean_absolute_difference":float(np.mean(np.abs(b-r))),
            "fpgaI_diagnostic_tolerance_exceeded":max_a>tolerance,
            "hls4ml_diagnostic_tolerance_exceeded":max_b>tolerance,
        })
    first={}
    for who in ("fpgaI","hls4ml"):
        first[who]=next(
            (x["operator"] for x in report
             if x[who+"_diagnostic_tolerance_exceeded"]),None)
    return {
        "schema":OUT,
        "original_onnx_sha256":m["original_onnx_sha256"],
        "ordered_sample_cohort_sha256":m["ordered_sample_cohort_sha256"],
        "input_manifest_sha256":sha256(path.read_bytes()).hexdigest(),
        "trace_source_sha256":hashes,
        "operator_count":len(steps),
        "sample_count":cohort,
        "diagnostic_abs_tolerance":float(tolerance),
        "operator_observations":report,
        "first_threshold_exceeding_operator_candidate":first,
        "trace_semantic_alignment_verified_by_independent_vendor_receipts":False,
        "operator_level_root_cause_proven":False,
        "native_e2_qualification_observed":False,
        "vendor_tools_executed_by_analysis":False,
        "cross_compiler_hardware_superiority_proven":False,
    }


def export_operator_differential(manifest,output):
    destination=Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Operator trace diagnostic must use a new private destination")
    result=analyze_operator_traces(manifest)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".operator-differential-",dir=destination.parent) as tmp:
        root=Path(tmp)/"result"
        root.mkdir()
        (root/"operator_differential.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",
            encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Operator differential appeared concurrently")
        root.rename(destination)
    return result
