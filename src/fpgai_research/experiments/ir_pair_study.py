"""Weight-free pairwise IR-mechanism and empirical QoR comparisons.

Distinct operator-local lowering contracts may have the same *nominal*
spatial capacity and measured HLS latency. Neither proves equivalent RTL.
All comparisons are made within one Functional-IR-identical trained model.
No vendor tools or sources are executed.
"""
from __future__ import annotations

from itertools import combinations
from hashlib import sha256
import csv
import json
from math import isfinite
from pathlib import Path
from fpgai_research.model_ir.output_ownership import enumerate_output_ownership

SCHEMA="fpgai-research/ir-mechanism-pairs/v1"
NATIVE_RESOURCES=("lut","ff","dsp","bram_tiles","uram")
CSV_FIELDS=(
    "model","case_a","case_b","axis_a","axis_b",
    "same_functional_ir","identical_lowered_contract","same_nominal_spatial_capacity",
    "identical_hls_source","normalized_output_ownership_equal",
    "output_ownership_scope","both_e4_qualified",
    "same_estimated_hls_latency","same_routed_lut","same_routed_resource_vector",
    "qualified_hls_latency_difference_cycles","qualified_lut_difference",
    "relationship","evidence_boundary",
)


def _canonical(value)->str:
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def _mechanisms(row:dict)->dict:
    return {op["operator"]:{
        "kind":op["kind"],
        "iteration_shape":op["iteration_shape"],
        **op["lowered_mechanisms"]
    } for op in row["operators"]}


def _nominal_shape(row:dict)->dict:
    """Nominal spatial slots; not structural/RTL equivalence.

    P×PE expresses a loop-mapping bound, not achieved multiply parallelism
    or a proven throughput/resource law.
    """
    return {op["operator"]:{
        "kind":op["kind"],
        "iteration_shape":op["iteration_shape"],
        "spatial_slots":int(op["lowered_mechanisms"]["replica_count"])
                         *int(op["lowered_mechanisms"]["pe_count"]),
        **{k:op["lowered_mechanisms"][k] for k in
           ("tile_shape","simd_lanes","fold_factor","pipeline_ii",
            "tail_policy","numerical_type")},
    } for op in row["operators"]}


def _number(value)->bool:
    return type(value) in (int,float) and isfinite(value)


def _pair(first:dict, second:dict, ownership:dict)->dict:
    if (first["model"]!=second["model"]
            or first["functional_sha256"]!=second["functional_sha256"]):
        raise ValueError("Only same-model, same-Functional-IR comparisons are scientific ablations")
    left=_mechanisms(first)
    right=_mechanisms(second)
    if set(left)!=set(right):
        raise ValueError("IR operator identity mismatch within model")
    same_lowered=_canonical(left)==_canonical(right)
    same_slots=_canonical(_nominal_shape(first))==_canonical(_nominal_shape(second))
    same_source=first["source_sha256"]==second["source_sha256"]
    a_ownership=ownership[(first["model"],first["case"])]
    b_ownership=ownership[(second["model"],second["case"])]
    checked=[]
    unverified=[]
    equal=[]
    for op in sorted(left):
        a=a_ownership[op]
        b=b_ownership[op]
        if a["status"]=="UNSUPPORTED_KIND" or b["status"]=="UNSUPPORTED_KIND":
            unverified.append(op)
        elif a["status"]!="EXHAUSTIVE_COVERAGE" or b["status"]!="EXHAUSTIVE_COVERAGE":
            unverified.append(op)
        else:
            checked.append(op)
            equal.append(a["normalized_ownership_sha256"]==
                         b["normalized_ownership_sha256"])
    ownership_equal=all(equal) if checked and not any(
        a_ownership[op]["status"]=="BOUND_EXCEEDED" or
        b_ownership[op]["status"]=="BOUND_EXCEEDED" for op in unverified
    ) else None
    qualified=first["outcome"]=="QUALIFIED" and second["outcome"]=="QUALIFIED"
    latency_equal=None
    lut_equal=None
    vector_equal=None
    latency_delta=None
    lut_delta=None
    if qualified:
        a,b=first["hls_latency_cycles"],second["hls_latency_cycles"]
        x,y=first["routed_lut"],second["routed_lut"]
        if not all(_number(v) for v in (a,b,x,y)):
            raise ValueError("Qualified pair missing native HLS or routed LUT measurements")
        latency_equal=a==b
        lut_equal=x==y
        latency_delta=round(b-a,9)
        lut_delta=round(y-x,9)
        ra,rb=first.get("routed_resources"),second.get("routed_resources")
        if (not isinstance(ra,dict) or not isinstance(rb,dict)
                or any(not _number(ra.get(k)) or not _number(rb.get(k))
                       for k in NATIVE_RESOURCES)):
            raise ValueError("Qualified pair missing complete source-attested native resources")
        vector_equal=all(ra[k]==rb[k] for k in NATIVE_RESOURCES)
    if same_lowered and same_source:
        relationship="IDENTICAL_SOURCE_AND_LOWERED_CONTRACT"
    elif same_lowered:
        relationship="SAME_LOWERED_CONTRACT_DISTINCT_SOURCE"
    elif same_slots:
        relationship="SAME_NOMINAL_SPATIAL_CAPACITY_DISTINCT_LOWERING"
    else:
        relationship="DISTINCT_NOMINAL_MAPPING"
    if same_slots and not same_lowered:
        explanatory=(
            "The same P×PE bound and other nominal controls do not imply "
            "equivalent loop structure, scheduling, RTL or throughput."
        )
    elif latency_equal is True and not same_lowered:
        explanatory=(
            "Distinct Lowered IRs produced the same estimated HLS latency; "
            "this does not establish equivalent RTL or performance beyond this study."
        )
    else:
        explanatory=(
            "IR similarity and native QoR similarity are separate observations. "
            "Neither alone establishes physical equivalence."
        )
    diffs={}
    for op in sorted(left):
        changes={k:{"a":left[op].get(k),"b":right[op].get(k)}
                 for k in sorted(set(left[op]) | set(right[op]))
                 if left[op].get(k)!=right[op].get(k)}
        if changes:
            diffs[op]=changes
    return {
        "model":first["model"],"case_a":first["case"],"case_b":second["case"],
        "axis_a":first["axis"],"axis_b":second["axis"],
        "functional_sha256":first["functional_sha256"],
        "same_functional_ir":True,
        "identical_lowered_contract":same_lowered,
        "same_nominal_spatial_capacity":same_slots,
        "identical_hls_source":same_source,
        "checked_output_ownership_operators":checked,
        "unverified_output_ownership_operators":unverified,
        "normalized_output_ownership_equal":ownership_equal,
        "output_ownership_evidence":"E0-exhaustive-lowered-loop-enumeration" if checked else None,
        "output_ownership_scope":"SUPPORTED_MATMUL_GEMM_CONV_OPERATORS_ONLY" if checked else "NONE",
        "lowered_mechanism_changes":diffs,
        "lowered_contract_sha256_a":sha256(_canonical(left).encode()).hexdigest(),
        "lowered_contract_sha256_b":sha256(_canonical(right).encode()).hexdigest(),
        "both_e4_qualified":qualified,
        "same_estimated_hls_latency":latency_equal,
        "same_routed_lut":lut_equal,
        "same_routed_resource_vector":vector_equal,
        "qualified_hls_latency_difference_cycles":latency_delta,
        "qualified_lut_difference":lut_delta,
        "relationship":relationship,
        "evidence_boundary":explanatory,
    }


def analyze_ir_pairs(study:dict)->dict:
    if study.get("schema")!="fpgai-research/scientific-ir-mechanism-study/v1":
        raise ValueError("Expected a verified scientific IR mechanism study")
    rows=study.get("cases")
    models=study.get("models")
    if not isinstance(rows,list) or not isinstance(models,list) or not rows:
        raise ValueError("Scientific IR case inventory is incomplete")
    groups={}
    ids=set()
    for row in rows:
        key=(row["model"],row["case"])
        if key in ids:
            raise ValueError("Duplicate model/case in scientific IR evidence")
        ids.add(key)
        groups.setdefault(row["model"],[]).append(row)
    if set(groups)!={item["model"] for item in models}:
        raise ValueError("Scientific IR model inventory changed")
    ownership={}
    for row in rows:
        proof={}
        for op in row["operators"]:
            identity=op["operator"]
            if identity in proof:
                raise ValueError("Duplicate Lowered Model IR operator")
            proof[identity]=enumerate_output_ownership(op)
        ownership[(row["model"],row["case"])]=proof
    pairs=[]
    for model, group in groups.items():
        if len({item["functional_sha256"] for item in group})!=1:
            raise ValueError("Functional IR not preserved within a model")
        for a,b in combinations(group,2):
            pairs.append(_pair(a,b,ownership))
    interesting=[p for p in pairs if (
        p["axis_a"]=="baseline" or p["axis_b"]=="baseline"
        or p["same_nominal_spatial_capacity"]
        or (p["both_e4_qualified"] and p["same_estimated_hls_latency"])
    )]
    return {
        "schema":SCHEMA,
        "source_suite_sha256":study.get("source_suite_sha256"),
        "model_count":len(models),"all_pair_count":len(pairs),
        "interesting_pair_count":len(interesting),
        "same_nominal_capacity_distinct_lowering_count":sum(
            p["same_nominal_spatial_capacity"] and not p["identical_lowered_contract"]
            for p in pairs),
        "identical_source_distinct_lowered_count":sum(
            p["identical_hls_source"] and not p["identical_lowered_contract"]
            for p in pairs),
        "output_ownership_by_case":[
            {"model":model,"case":case,"operators":proof}
            for (model,case),proof in sorted(ownership.items())
        ],
        "pairs":interesting,
        "weight_values_exported":False,"vendor_tools_executed":False,
        "limitations":(
            "E0 output ownership enumeration checks output index coverage "
            "for emitted MatMul/Gemm/Conv loops only, NOT fixed-point "
            "accumulator arithmetic, compiler scheduling or RTL identity. "
            "Same nominal P×PE slots do not imply identical hardware. "
            "Matching estimated HLS latency, source hashes, or physical "
            "resource vectors alone cannot prove equivalence of RTL or "
            "achieved performance. Native contrasts require BOTH cases to "
            "meet fixed-point, OOC route, timing, and DRC acceptance. "
            "No board deployment or full-dataset accuracy is established."
        ),
    }


def export_ir_pairs(study:dict,stage:Path)->dict:
    result=analyze_ir_pairs(study)
    stage=Path(stage)
    (stage/"ir_pairs.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    with (stage/"ir_pairs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=CSV_FIELDS)
        writer.writeheader()
        for pair in result["pairs"]:
            writer.writerow({k:("TBD" if pair.get(k) is None else pair[k])
                             for k in CSV_FIELDS})
    md=["# Operator-local IR mapping comparisons","",
        "This is a within-model study. Different mappings can have equal "
        "nominal slot counts while remaining distinct Lowered IR contracts.","",
        "| Model | Pair | Relationship | Equal E0 ownership | Both E4-qualified | Equal HLS latency | Equal routed LUT |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for pair in result["pairs"]:
        if not (pair["same_nominal_spatial_capacity"] or
                (pair["both_e4_qualified"] and pair["same_estimated_hls_latency"])):
            continue
        md.append("| "+" | ".join([
            pair["model"],
            f'{pair["case_a"]} ({pair["axis_a"]}) vs {pair["case_b"]} ({pair["axis_b"]})',
            pair["relationship"],
            "TBD" if pair["normalized_output_ownership_equal"] is None else
                str(pair["normalized_output_ownership_equal"]),
            str(pair["both_e4_qualified"]),
            "TBD" if pair["same_estimated_hls_latency"] is None else
                str(pair["same_estimated_hls_latency"]),
            "TBD" if pair["same_routed_lut"] is None else
                str(pair["same_routed_lut"]),
        ])+" |")
    md.extend(["","## Limitations","",result["limitations"],""])
    (stage/"ir_pairs.md").write_text("\n".join(md))
    return result
