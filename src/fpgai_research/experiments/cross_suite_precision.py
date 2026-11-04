"""Compare a separately prepared numeric split with immutable seven-axis baselines.

This is a *within-trained-model* precision comparison across two experiment
suites, never a pooled seven-axis ablation or a board deployment result.
Native E2/E3/E4 measurements are independently recollected and qualified.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.results import summarize_case

SCHEMA = "fpgai-research/cross-suite-precision/v1"
RESOURCES = ("lut", "ff", "dsp", "bram", "uram")
ARCHITECTURE_EXCEPT_NUMERIC = (
    "parallelism", "pes", "simd", "serialization", "tile", "pipeline_ii"
)
IDENTITY = (
    "model_sha256", "inputs_sha256", "labels_sha256", "functional_sha256", "operator"
)


def _regular_json(path: Path) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular immutable experiment file missing: {path}")
    content = path.read_bytes()
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return value, sha256(content).hexdigest()


def _member(root: Path, identity: str) -> tuple[Path, dict]:
    suite, _ = _regular_json(root/"suite_results.json")
    if suite.get("schema") != "fpgai-research/full-paper-suite/v1":
        raise ValueError("Expected a prepared suite manifest")
    matches = [r for r in suite.get("studies", []) if r.get("id") == identity]
    if len(matches) != 1 or matches[0].get("status") != "E1-PREPARED":
        raise ValueError(f"Model {identity} not uniquely source-prepared in suite")
    study = root/identity
    if study.is_symlink() or not study.is_dir():
        raise ValueError("Prepared study must be a real directory")
    receipt, _ = _regular_json(study/"full_experiment.json")
    if receipt.get("schema") != "fpgai-research/full-progressive-experiment/v1":
        raise ValueError("Unexpected pinned research experiment schema")
    return study, receipt


def _valid_resources(resources: dict) -> bool:
    """Vivado Block RAM Tile use can be fractional (e.g. 12.5 tiles).

    The underlying report parser preserves fractional tile values. A
    verified E4 result must not be downgraded merely because these
    legitimate numeric resources are not all Python ints. Disallow bool,
    strings, NaN, infinity, negative values and missing resources.
    """
    return (isinstance(resources,dict) and
            all(type(resources.get(key)) in (int,float)
                and isfinite(resources[key]) and resources[key]>=0
                for key in RESOURCES))


def _case(study: Path, *, width: int, integer: int) -> dict:
    campaign, campaign_sha = _regular_json(study/"campaign.json")
    if campaign.get("schema") != "fpgai-research/paper-campaign/v1":
        raise ValueError("Unexpected campaign schema")
    rows = [r for r in campaign.get("cases", [])
            if r.get("status")=="READY" and
            (r.get("requested") or {}).get("precision_bits")==width and
            (r.get("requested") or {}).get("integer_bits")==integer and
            r.get("dimension") in ("baseline","precision_bits")]
    if len(rows)!=1:
        raise ValueError(f"Expected exactly one {width}/{integer} prepared case")
    row=rows[0]
    label=row.get("case")
    if not isinstance(label,str) or not label.startswith("case_") or not label[5:].isdigit():
        raise ValueError("Unsafe campaign case name")
    path=study/label
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Prepared case must be a real directory")
    manifest, manifest_sha = _regular_json(path/"manifest.json")
    requested=row["requested"]
    for key,val in (("precision_bits",width),("integer_bits",integer)):
        if manifest.get(key)!=val:
            raise ValueError(f"Case source manifest does not match {key}")
    if manifest.get("part")!=campaign.get("part") or manifest.get("clock_ns")!=campaign.get("clock_ns"):
        raise ValueError("HLS case target or clock differs from prepared campaign")
    if manifest.get("numerical_tolerance")!=campaign.get("tolerance"):
        raise ValueError("Case tolerance differs from prepared campaign")
    evidence=summarize_case(str(path))
    if evidence.get("precision_bits")!=width or evidence.get("integer_bits")!=integer:
        raise ValueError("Measured numeric format does not match prepared case")
    if evidence.get("source_sha256")!=row.get("cpp_sha256"):
        raise ValueError("Source hash differs from prepared campaign")
    physical=evidence.get("vivado_metrics") or {}
    drc=physical.get("drc") or {}
    timing=physical.get("timing") or {}
    qualifies=(evidence.get("evidence")=="E4-Vivado-implementation"
        and evidence.get("csim_passed") is True
        and evidence.get("vivado_implementation_mode")=="out_of_context"
        and physical.get("routed_nets_complete") is True
        and timing.get("timing_met") is True
        and drc.get("summary_verified") is True
        and drc.get("critical_warnings")==0
        and drc.get("unconstrained_io") is False
        and _valid_resources(physical.get("utilization") or {})
        and type(timing.get("wns_ns")) in (float,int)
        and isfinite(timing["wns_ns"]))
    blockers=[]
    if not qualifies:
        if evidence.get("evidence")!="E4-Vivado-implementation":
            blockers.append("No attested E4 routed evidence")
        if evidence.get("csim_passed") is not True:
            blockers.append("Native fixed-point C simulation not numerically accepted")
        if evidence.get("vivado_implementation_mode")!="out_of_context":
            blockers.append("Not a matched out-of-context Vivado implementation")
        if timing.get("timing_met") is not True:
            blockers.append("Post-route timing not met or absent")
        if not (type(timing.get("wns_ns")) in (float,int)
                and isfinite(timing["wns_ns"])):
            blockers.append("Post-route WNS missing or nonfinite")
        if (drc.get("summary_verified") is not True or
            drc.get("critical_warnings")!=0 or
            drc.get("unconstrained_io") is not False):
            blockers.append("Physical DRC qualification incomplete")
        if physical.get("routed_nets_complete") is not True:
            blockers.append("Complete routing not verified")
        if not _valid_resources(physical.get("utilization") or {}):
            problems=[key for key in RESOURCES
                      if not (type((physical.get("utilization") or {}).get(key)) in (float,int)
                              and isfinite(physical["utilization"][key])
                              and physical["utilization"][key]>=0)]
            blockers.append("Missing/invalid post-route resources: "+", ".join(problems))
    csim=evidence.get("csim_receipt") or {}
    metrics=evidence.get("metrics") or {}
    resources=physical.get("utilization") or {}
    return {
        "case":label,"width":width,"integer_bits":integer,
        "fractional_bits":width-integer,
        "case_source_sha256":evidence["source_sha256"],
        "campaign_sha256":campaign_sha,
        "case_manifest_sha256":manifest_sha,
        "requested_non_numeric":{k:requested.get(k) for k in ARCHITECTURE_EXCEPT_NUMERIC},
        "operator":row.get("operator"),
        "part":campaign.get("part"),"clock_ns":campaign.get("clock_ns"),
        "tolerance":campaign.get("tolerance"),
        "testbench_samples":manifest.get("csim_sample_count"),
        "testbench_outputs":manifest.get("csim_outputs_per_sample"),
        "evidence":evidence.get("evidence"),
        "csim_passed":evidence.get("csim_passed"),
        "csim_failures":csim.get("failures"),
        "csim_max_absolute_error":csim.get("max_absolute_error"),
        "csim_rmse":csim.get("rmse"),
        "classification":csim.get("classification"),
        "hls_latency_cycles":metrics.get("latency_best")
            if evidence.get("evidence") in ("E3-HLS-synthesis-report","E4-Vivado-implementation") else None,
        "hls_ii_cycles":metrics.get("ii_min")
            if evidence.get("evidence") in ("E3-HLS-synthesis-report","E4-Vivado-implementation") else None,
        "routed_utilization":resources if evidence.get("evidence")=="E4-Vivado-implementation" else None,
        "wns_ns":timing.get("wns_ns") if evidence.get("evidence")=="E4-Vivado-implementation" else None,
        "drc":drc if evidence.get("evidence")=="E4-Vivado-implementation" else None,
        "report_sha256":physical.get("raw_report_sha256") if evidence.get("evidence")=="E4-Vivado-implementation" else None,
        "warnings":evidence.get("warnings",[]),
        "qualified":qualifies,"blockers":blockers,
    }


def _same_identity(a:dict,b:dict):
    for key in (*ARCHITECTURE_EXCEPT_NUMERIC,"operator","part","clock_ns",
                "tolerance","testbench_samples","testbench_outputs"):
        k="requested_non_numeric" if key in ARCHITECTURE_EXCEPT_NUMERIC else key
        if key in ARCHITECTURE_EXCEPT_NUMERIC:
            va,vb=a[k].get(key),b[k].get(key)
        else:
            va,vb=a.get(key),b.get(key)
        if va != vb:
            raise ValueError(f"Unmatched {key} between precision experiments")


def _difference(base:dict, other:dict) -> dict:
    if not (base["qualified"] and other["qualified"]):
        details=[]
        if not base["qualified"]:
            details.append(f'Baseline {base["width"]}/{base["integer_bits"]}: '+
                           "; ".join(base["blockers"]))
        if not other["qualified"]:
            details.append(f'Variant {other["width"]}/{other["integer_bits"]}: '+
                           "; ".join(other["blockers"]))
        return {"qualified":False,"reason":" | ".join(details),
                "deltas":None}
    delta={}
    for resource in RESOURCES:
        start=base["routed_utilization"][resource]
        end=other["routed_utilization"][resource]
        delta[resource]={"baseline":start,"observed":end,"absolute":end-start,
            "reduction_percent":round((start-end)/start*100,4) if start else None}
    # WNS is a timing margin, not a performance throughput measurement.
    delta["wns_ns"]={"baseline":base["wns_ns"],"observed":other["wns_ns"],
                     "difference":round(other["wns_ns"]-base["wns_ns"],9)}
    for key in ("hls_latency_cycles","hls_ii_cycles"):
        try:
            x,y=int(base[key]),int(other[key])
        except (ValueError,TypeError):
            delta[key]=None
        else:
            delta[key]={"baseline":x,"observed":y,"difference":y-x}
    return {"qualified":True,"reason":None,"deltas":delta}


def compare_precision_suites(*, baseline_suite, precision_suite, model,
                             integer_bits=4) -> dict:
    if not isinstance(model,str) or not model or "/" in model or "\\" in model or ".." in model:
        raise ValueError("Unsafe model identity")
    if type(integer_bits) is not int or not 1<=integer_bits<=7:
        raise ValueError("Invalid 8-bit integer width")
    old,new=Path(baseline_suite),Path(precision_suite)
    for root in (old,new):
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Suite must be a regular directory")
    baseline_path,baseline=_member(old,model)
    trial_name=f"{model}_w8i{integer_bits}"
    trial_path,trial=_member(new,trial_name)
    for attr in IDENTITY:
        if not baseline.get(attr) or baseline[attr]!=trial.get(attr):
            raise ValueError(f"Source study identity mismatch: {attr}")
    original16=_case(baseline_path,width=16,integer=6)
    original12=_case(baseline_path,width=12,integer=6)
    trial8=_case(trial_path,width=8,integer=integer_bits)
    for point in (original12,trial8):
        _same_identity(original16,point)
    return {
        "schema":SCHEMA,"model":model,"precision_trial":trial_name,
        "source_provenance":{key:baseline[key] for key in IDENTITY},
        "original_suite_sha256":sha256((old/"suite_results.json").read_bytes()).hexdigest(),
        "precision_suite_sha256":sha256((new/"suite_results.json").read_bytes()).hexdigest(),
        "cases":{"16_6":original16,"12_6":original12,
                 f"8_{integer_bits}":trial8},
        "comparisons":{
            "12_vs_16":_difference(original16,original12),
            "8_vs_16":_difference(original16,trial8),
            "8_vs_12":_difference(original12,trial8)},
        "limitations":(
            "Different immutable campaigns, same trained ONNX, samples, labels, "
            "model architecture, target and clock. HLS cycles are synthesis "
            "estimates; WNS/resources are OOC Vivado measurements. "
            "No board throughput, full-dataset fixed-point accuracy or bitstream "
            "deployment is established by these 16-sample C simulations."
        )
    }


def export_precision_comparison(*, baseline_suite, precision_suite, model,
                                integer_bits=4, output) -> dict:
    result=compare_precision_suites(
        baseline_suite=baseline_suite,precision_suite=precision_suite,
        model=model,integer_bits=integer_bits)
    dest=Path(output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Cannot overwrite previous precision comparison")
    with TemporaryDirectory(prefix=".fpgai-precision-comparison-",dir=dest.parent) as tmp:
        stage=Path(tmp)
        (stage/"precision_comparison.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
        from fpgai_research.experiments.precision_publication import write_precision_publication
        write_precision_publication(result, stage)
        stage.rename(dest)
    return result
