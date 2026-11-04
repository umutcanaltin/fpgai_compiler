"""Explain equal E3 HLS-estimate pairs using only attested private E1 sources.

A privacy-minimizing, observational tool. Structural counts and changed
architecture-field NAMES are exploratory; they do not prove schedule causes,
RTL equivalence or physical performance. Raw source and trained weights never
leave the local replay workspace.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from fpgai_research.experiments.isolated_hls_e3_ablation import (
    collect_isolated_hls_e3_ablation,
)
from fpgai_research.experiments.isolated_hls_e3_interpretation import (
    interpret_isolated_hls_e3,
)
from fpgai_research.experiments.isolated_hls_replay import (
    AUDIT_SCHEMA, _regular_bytes, audit_isolated_hls_replay,
)
from fpgai_research.experiments.native_hls_rerun_plan import _read_json

SCHEMA="fpgai-research/isolated-e3-tied-source-diagnostics/v1"
_SOURCE_LIMIT=64*1024*1024
_PRAGMA=re.compile(r"^\s*#pragma\s+HLS\s+([A-Za-z_][A-Za-z_0-9]*)")
_LOOP=re.compile(r"\bfor\s*\(")
_CONDITIONAL=re.compile(r"\bif\s*\(")
_CONST=re.compile(r"^\s*static\s+const\s+data_t\s+tensor_[0-9]+\s*\[\s*[0-9]+\s*\]\s*=")
_MECHANISMS=frozenset((
    "tile_shape","replicas","pe_count","simd_lanes","fold_factor",
    "pipeline_ii","tail_policy","numerical_type",
))
# Public ArchitectureRequest uses different canonical names than LoweredOperator.
_ARCH_FIELDS=frozenset((
    "precision_bits","integer_bits","parallelism","pes","simd",
    "serialization","tile","pipeline_ii",
))
_PRAGMA_KINDS=frozenset((
    "PIPELINE","UNROLL","ARRAY_PARTITION","DATAFLOW","INLINE",
    "RESOURCE","BIND_STORAGE","BIND_OP","LOOP_TRIPCOUNT","INTERFACE",
    "STREAM","DEPENDENCE","AGGREGATE",
))


def _safe_profile(raw:bytes)->dict:
    """Compare in memory; only closed-vocabulary counts and digests stay private."""
    if not raw or len(raw)>_SOURCE_LIMIT:
        raise ValueError("Invalid or unbounded HLS source")
    text=raw.decode("utf-8",errors="strict")
    pragmas=Counter()
    nonweight_lines=[]
    loops=conditionals=op_annotations=redacted_constants=0
    for line in text.splitlines():
        stripped=line.strip()
        if _CONST.match(line):
            redacted_constants+=1
            continue
        if stripped.startswith("//"):
            if stripped.startswith("// OP "):
                op_annotations+=1
            continue
        match=_PRAGMA.match(line)
        if match:
            kind=match.group(1).upper()
            pragmas[kind if kind in _PRAGMA_KINDS else "OTHER"]+=1
        loops+=len(_LOOP.findall(line))
        conditionals+=len(_CONDITIONAL.findall(line))
        nonweight_lines.append(line)
    # Fingerprint is NOT exported: it is useful only to answer whether
    # source text outside generated weight declarations/comments matches.
    body_hash=sha256(("\n".join(nonweight_lines)+"\n").encode()).hexdigest()
    return {
        "_body_hash":body_hash,
        "generated_constant_declaration_count":redacted_constants,
        "operator_annotation_line_count":op_annotations,
        "nonweight_noncomment_line_count":len(nonweight_lines),
        "for_loop_text_count":loops,
        "if_text_count":conditionals,
        "hls_pragma_kind_counts":dict(sorted(pragmas.items())),
    }


def _positive_shape(raw, length):
    if (not isinstance(raw,(list,tuple)) or len(raw)!=length
            or any(type(v) is not int or v<=0 for v in raw)):
        raise ValueError("Missing or nonpositive static architecture extent")
    return tuple(raw)


def _static_geometry(operator:dict)->dict:
    """Upper bounds from emitter loop coordinates, NOT achieved HLS lanes.

    MatMul/Gemm: output channel n, tiled output tn; reduction k, tile tk.
    Conv: output channel oc, tiled output toc; input channels ic. The
    emitter's conv SIMD loop traverses ic, not kernel ky/kx.
    No numeric extents or widths are exported to the publication report.
    """
    kind=operator.get("kind")
    if kind not in ("MatMul","Gemm","Conv"):
        return {"status":"UNSUPPORTED_NONCOMPUTE_GEOMETRY"}
    length=3 if kind in ("MatMul","Gemm") else 7
    shape=_positive_shape(operator.get("iteration_shape"),length)
    lowered=operator.get("lowered_mechanisms")
    if not isinstance(lowered,dict):
        raise ValueError("Lowered architecture mechanisms missing")
    tile=_positive_shape(lowered.get("tile_shape"),length)
    replica=lowered.get("replicas")
    pes=lowered.get("pe_count")
    simd=lowered.get("simd_lanes")
    if any(type(x) is not int or x<1 for x in (replica,pes,simd)):
        raise ValueError("Invalid static spatial/reduction declaration")
    if kind in ("MatMul","Gemm"):
        output_extent=min(shape[1],tile[1])
        reduction_extent=min(shape[2],tile[2])
    else:
        output_extent=min(shape[1],tile[1])
        reduction_extent=shape[4]
    capacity=replica*pes
    return {
        "status":"SUPPORTED_STATIC_EMITTER_COORDINATES",
        "_declared_spatial_slots":capacity,
        "_static_spatial_upper_bound":min(capacity,output_extent),
        "_spatial_exceeds_tile_extent":capacity>output_extent,
        "_declared_simd_lanes":simd,
        "_static_simd_upper_bound":min(simd,reduction_extent),
        "_simd_exceeds_reduction_extent":simd>reduction_extent,
    }


def _geometry_contrast(left:dict,right:dict)->dict:
    a,b=_static_geometry(left),_static_geometry(right)
    if a["status"]!=b["status"]:
        raise ValueError("Changed operator geometry family")
    if a["status"]!="SUPPORTED_STATIC_EMITTER_COORDINATES":
        return {"geometry_status":a["status"],
                "operator_loop_ii_verified":False,
                "physical_lane_utilization_verified":False}
    return {
        "geometry_status":"SUPPORTED_STATIC_EMITTER_COORDINATES",
        "declared_spatial_slot_product_equal":
            a["_declared_spatial_slots"]==b["_declared_spatial_slots"],
        "spatial_slots_static_upper_bound_equal":
            a["_static_spatial_upper_bound"]==b["_static_spatial_upper_bound"],
        "spatial_capacity_exceeds_local_extent_a":a["_spatial_exceeds_tile_extent"],
        "spatial_capacity_exceeds_local_extent_b":b["_spatial_exceeds_tile_extent"],
        "declared_simd_lane_count_equal":
            a["_declared_simd_lanes"]==b["_declared_simd_lanes"],
        "simd_lanes_static_upper_bound_equal":
            a["_static_simd_upper_bound"]==b["_static_simd_upper_bound"],
        "simd_capacity_exceeds_reduction_extent_a":a["_simd_exceeds_reduction_extent"],
        "simd_capacity_exceeds_reduction_extent_b":b["_simd_exceeds_reduction_extent"],
        "operator_loop_ii_verified":False,
        "physical_lane_utilization_verified":False,
    }


def _architecture_difference(left:dict,right:dict)->list[dict]:
    a,b=left.get("operators"),right.get("operators")
    if not isinstance(a,list) or not isinstance(b,list) or len(a)!=len(b):
        raise ValueError("Missing or mismatched architecture operator inventory")
    changed=[]
    for index,(lhs,rhs) in enumerate(zip(a,b)):
        if not isinstance(lhs,dict) or not isinstance(rhs,dict):
            raise ValueError("Invalid architecture operator entry")
        if (lhs.get("operator")!=rhs.get("operator")
                or lhs.get("kind")!=rhs.get("kind")
                or lhs.get("iteration_shape")!=rhs.get("iteration_shape")):
            raise ValueError("Functional operator alignment differs across tied sources")
        lowered_a=lhs.get("lowered_mechanisms")
        lowered_b=rhs.get("lowered_mechanisms")
        if not isinstance(lowered_a,dict) or not isinstance(lowered_b,dict):
            raise ValueError("Missing lowered architecture mechanism records")
        # Only expose a fixed whitelist of mechanism keys; never operator
        # identifiers, raw values, tensor data or confidential scope labels.
        mechanisms=sorted(k for k in _MECHANISMS if lowered_a.get(k)!=lowered_b.get(k))
        requested=sorted(k for k in _ARCH_FIELDS if
                         (lhs.get("requested") or {}).get(k)!=(rhs.get("requested") or {}).get(k))
        effective=sorted(k for k in _ARCH_FIELDS if
                         (lhs.get("effective") or {}).get(k)!=(rhs.get("effective") or {}).get(k))
        if mechanisms or requested or effective:
            changed.append({
                "operator_ordinal":index,
                "changed_lowered_mechanism_keys":mechanisms,
                "changed_requested_architecture_keys":requested,
                "changed_effective_architecture_keys":effective,
                "static_workload_geometry":_geometry_contrast(lhs,rhs),
            })
    return changed


def diagnose_equal_e3_sources(directory,e3_document:dict,audit:dict)->dict:
    """Require real E3-attested sources before explaining any equal estimates."""
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Normal isolated replay directory required")
    if audit.get("schema")!=AUDIT_SCHEMA or audit.get("per_loop_schedule_evidence_available") is not False:
        raise ValueError("Expected independently audited non-attributing E3 replay")
    interpretation=interpret_isolated_hls_e3(e3_document)
    if interpretation["case_count"]!=audit.get("case_count"):
        raise ValueError("Audit and E3 interpretation counts disagree")
    audited={}
    for row in audit.get("cases",[]):
        key=(row["model"],row["case"])
        if key in audited:
            raise ValueError("Duplicate audited native case")
        audited[key]=row
    mapped={}
    for row in interpretation["cases"]:
        key=row["model"],row["case"]
        record=audited.get(key)
        if (not isinstance(record,dict) or record.get("axis")!=row["axis"]
                or record.get("operator_loop_ii_verified") is not False):
            raise ValueError("Source inventory differs from independent audit")
        mapped[key]=row
    if len(mapped)!=len(audited):
        raise ValueError("Missing E3 audited cases")
    pairs=set()
    for row in interpretation["cases"]:
        for partner in row["equal_estimate_cases"]:
            pair=(row["model"],*sorted((row["case"],partner)))
            pairs.add(pair)
    results=[]
    cache={}
    for model,left,right in sorted(pairs):
        a,b=audited[(model,left)],audited[(model,right)]
        for row in (a,b):
            if (row.get("evidence")!="E3-HLS-synthesis-report"
                    or row.get("fixed_point_csim_passed") is not True
                    or row.get("fresh_hls_report_attested") is not True):
                raise ValueError("Equal E3 comparison lacks source-attested CSim/HLS proof")
        for name,row in ((left,a),(right,b)):
            key=model,name
            if key in cache:
                continue
            folder=root/model/name
            if folder.is_symlink() or not folder.is_dir():
                raise ValueError("Linked or missing replay case")
            cpp=folder/"model.cpp"
            raw=_regular_bytes(cpp,_SOURCE_LIMIT)
            if sha256(raw).hexdigest()!=row["source_sha256"]:
                raise ValueError("Generated HLS code changed since E3 attestation")
            manifest_path=folder/"manifest.json"
            if manifest_path.is_symlink():
                raise ValueError("Linked HLS manifest")
            manifest=_read_json(manifest_path)
            if manifest.get("cpp_sha256")!=row["source_sha256"]:
                raise ValueError("HLS manifest does not match attested generated code")
            cache[key]=(_safe_profile(raw),manifest)
        p1,m1=cache[(model,left)]
        p2,m2=cache[(model,right)]
        pragmas=sorted(set(p1["hls_pragma_kind_counts"])|set(p2["hls_pragma_kind_counts"]))
        delta={kind:p2["hls_pragma_kind_counts"].get(kind,0)
                    -p1["hls_pragma_kind_counts"].get(kind,0)
               for kind in pragmas}
        delta={k:v for k,v in delta.items() if v}
        changed=_architecture_difference(m1,m2)
        identical_source=a["source_sha256"]==b["source_sha256"]
        equal_body=p1["_body_hash"]==p2["_body_hash"]
        results.append({
            "model":model,"case_a":left,"case_b":right,
            "axis_a":mapped[(model,left)]["axis"],
            "axis_b":mapped[(model,right)]["axis"],
            "e3_hls_estimate_vector_equal":True,
            "source_cpp_sha256_equal":identical_source,
            "nonweight_noncomment_text_equal":equal_body,
            "source_difference_class":(
                "BYTE_IDENTICAL_CPP" if identical_source else
                "SAME_NONWEIGHT_NONCOMMENT_TEXT" if equal_body else
                "DIFFERENT_NONWEIGHT_CPP_TEXT"
            ),
            "changed_operator_mechanisms":changed,
            "changed_operator_count":len(changed),
            "pragma_kind_count_delta_b_minus_a":delta,
            "for_loop_text_count_delta_b_minus_a":
                p2["for_loop_text_count"]-p1["for_loop_text_count"],
            "if_text_count_delta_b_minus_a":
                p2["if_text_count"]-p1["if_text_count"],
            "nonweight_noncomment_line_delta_b_minus_a":
                p2["nonweight_noncomment_line_count"]-p1["nonweight_noncomment_line_count"],
            "operator_loop_ii_verified":False,
            "causal_mechanism_verified":False,
        })
    if len(results)!=interpretation["equal_hls_estimate_pair_count"]:
        raise ValueError("E3 estimate pair inventory inconsistent")
    return {
        "schema":SCHEMA,
        "source_suite_sha256":interpretation["source_suite_sha256"],
        "equal_e3_estimate_pair_count":len(results),
        "byte_identical_cpp_pair_count":sum(x["source_cpp_sha256_equal"] for x in results),
        "nonweight_noncomment_text_equal_pair_count":
            sum(x["nonweight_noncomment_text_equal"] for x in results),
        "pairs":results,
        "private_source_exported":False,
        "operator_names_exported":False,
        "weight_values_exported":False,
        "native_tools_executed":False,
        "e4_physical_comparison_performed":False,
        "operator_schedule_attribution_verified":False,
        "limitations":(
            "Observational comparison of fresh E3 pairs tied across HLS "
            "latency and resource estimates. Changed mechanism key names "
            "derive from existing E1 manifests; lexical loop/pragma counts "
            "are heuristics, not synthesis trace or causal evidence. "
            "Unchanged non-weight non-comment C++ text does NOT prove "
            "functional equivalence: constants and comments are excluded. "
            "Static coordinate bounds from emitter MatMul/Gemm/Conv loops "
            "describe only declared spatial/reduction capacities; they do "
            "not prove actual active lanes, schedule or bottlenecks. "
            "No operator achieved II, Verilog/RTL identity, E4 PPA, power, "
            "timing or board-level performance is established."
        ),
    }


def export_equal_e3_source_diagnostics(directory,comparison_path,output)->dict:
    root=Path(directory)
    saved=_read_json(Path(comparison_path))
    # Independently reconstruct the current source-attested E3 comparisons.
    # A copied/tampered snapshot cannot drive a private-source disclosure.
    if saved!=collect_isolated_hls_e3_ablation(root):
        raise ValueError("E3 snapshot no longer matches independently audited replay")
    audit=audit_isolated_hls_replay(root)
    result=diagnose_equal_e3_sources(root,saved,audit)
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Immutable tied-source diagnostic already exists")
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-e3-source-",dir=target.parent) as tmp:
        stage=Path(tmp)
        (stage/"equal_e3_source_diagnostics.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        lines=[
            "# Architecture mechanism contrasts among tied E3 HLS estimates","",
            "**Source-attested HLS estimates only. No achieved operator II, "
            "RTL equivalence or inferred synthesis cause.**","",
            f"Equal-estimate pairs: {result['equal_e3_estimate_pair_count']}; "
            f"byte-identical source pairs: {result['byte_identical_cpp_pair_count']}; "
            f"nonweight text-equal pairs: "
            f"{result['nonweight_noncomment_text_equal_pair_count']}.","",
            "| Model | Cases | Axes | CPP byte equality | Nonweight code text equal | Changed operator entries |",
            "| --- | --- | --- | --- | --- | ---: |",
        ]
        for pair in result["pairs"]:
            lines.append("| "+" | ".join((
                pair["model"],pair["case_a"]+" / "+pair["case_b"],
                pair["axis_a"]+" / "+pair["axis_b"],
                str(pair["source_cpp_sha256_equal"]),
                str(pair["nonweight_noncomment_text_equal"]),
                str(pair["changed_operator_count"]),
            ))+" |")
        lines.extend(["","## Changed architecture mechanism names (no values)",""])
        for pair in result["pairs"]:
            lines.append("### "+pair["model"]+" / "+pair["case_a"]+" vs "+pair["case_b"])
            for op in pair["changed_operator_mechanisms"]:
                lines.append(
                    f"Operator index {op['operator_ordinal']}: lowered "
                    +", ".join(op["changed_lowered_mechanism_keys"])
                    +"; requested "+", ".join(op["changed_requested_architecture_keys"])
                    +"; effective "+", ".join(op["changed_effective_architecture_keys"])
                )
            if not pair["changed_operator_mechanisms"]:
                lines.append("No whitelisted architecture mechanism changes detected")
            for op in pair["changed_operator_mechanisms"]:
                geometry=op["static_workload_geometry"]
                lines.append("Operator index "+str(op["operator_ordinal"])
                             +" static capacity geometry (no extents/widths): "
                             +json.dumps(geometry,sort_keys=True))
            lines.append("Pragma-kind count change (case B - A): "+
                         json.dumps(pair["pragma_kind_count_delta_b_minus_a"],sort_keys=True))
            lines.append("For-loop text count change (B - A): "+
                         str(pair["for_loop_text_count_delta_b_minus_a"]))
            lines.append("")
        lines.extend(["## Evidence boundary","",result["limitations"],""])
        (stage/"equal_e3_source_diagnostics.md").write_text("\n".join(lines),encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Tied-source destination appeared")
        stage.rename(target)
    return result
