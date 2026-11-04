"""Transactional E0/E1 experiment preparation across the three Model IR layers.

Produces Functional/Architectural/Lowered IR artifacts, HLS/Vivado *scripts*,
NumPy reference values and bounded real GCC C++ host checks. Vendor tools
are never invoked, and absence of E2/E3/E4 is recorded explicitly.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from fpgai_research.architecture.resolver import resolve, TargetLimits
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.experiments.campaign import generate_points, prepare_campaign
from fpgai_research.experiments.reference_evaluation import evaluate_reference
from fpgai_research.model_ir.artifacts import write_ir_artifact, read_ir_artifact
from fpgai_research.model_ir.contracts import ArchitectureRequest
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import functional_fingerprint, verify_refinement
from fpgai_research.validation.host import validate_host
from fpgai_research.validation.samples import normalize_samples

SCHEMA = "fpgai-research/full-progressive-experiment/v1"

def _hash(path: Path) -> str:
    d=sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):
            d.update(block)
    return d.hexdigest()

def parse_plan(value: dict) -> dict:
    valid={"schema","operator","default","variants","interactions","target_part","clock_ns","tolerance"}
    if not isinstance(value,dict) or set(value)-valid:
        raise ValueError("Invalid research experiment plan fields")
    if value.get("schema")!=SCHEMA or not isinstance(value.get("operator"),str):
        raise ValueError("Plan needs a versioned schema and exact Model IR operator ID")
    raw=value.get("default",{})
    if not isinstance(raw,dict) or set(raw)-set(ArchitectureRequest.__dataclass_fields__):
        raise ValueError("Unsupported default architecture knob")
    if "tile" in raw and (not isinstance(raw["tile"],list) or
                          not all(type(x) is int for x in raw["tile"])):
        raise ValueError("Architecture tile extents must be integer arrays")
    default=ArchitectureRequest(**{**raw,"tile":tuple(raw.get("tile",()))})
    variants=value.get("variants",{})
    interactions=value.get("interactions",())
    if not isinstance(variants,dict) or not all(isinstance(v,list) for v in variants.values()):
        raise ValueError("Each axis needs an array of requested settings")
    if not isinstance(interactions,(list,tuple)) or not all(isinstance(i,dict) for i in interactions):
        raise ValueError("Interactions must be an array of mappings")
    points=generate_points(default,variants,interactions)
    if len(points)>256:
        raise ValueError("At most 256 experiment architecture points")
    clock=value.get("clock_ns",5.0)
    tolerance=value.get("tolerance",0.1)
    if type(clock) not in (float,int) or not isfinite(clock) or clock<=0:
        raise ValueError("Clock period must be finite and positive")
    if type(tolerance) not in (float,int) or not isfinite(tolerance) or tolerance<0:
        raise ValueError("Numerical tolerance must be finite and nonnegative")
    target=value.get("target_part","xck26-sfvc784-2LV-c")
    if not isinstance(target,str) or not target:
        raise ValueError("Target part must be nonempty")
    return {"operator":value["operator"],"default":default,"variants":variants,
            "interactions":interactions,"points":points,"target_part":target,
            "clock_ns":clock,"tolerance":tolerance}

def prepare_full_experiment(*,model,model_path,inputs_path,plan,directory,
                            labels_path=None,host_samples=1,require_all_legal=True,
                            archive_inputs=False,crosscheck_onnxruntime=False,
                            trace_intermediates=False,limits=TargetLimits()):
    """Create one atomic paper experiment with all IR snapshots per READY case."""
    if type(host_samples) is not int or not 0<=host_samples<=8:
        raise ValueError("host_samples must be between 0 and 8")
    if any(type(v) is not bool for v in (require_all_legal,archive_inputs,crosscheck_onnxruntime,trace_intermediates)):
        raise ValueError("Experiment policy flags must be boolean")
    if trace_intermediates and not host_samples:
        raise ValueError("Intermediate tensor tracing requires at least one GCC host sample")
    spec=parse_plan(plan)
    model.validate()
    if len(model.inputs)!=1 or len(model.outputs)!=1:
        raise ValueError("Full experiment requires one input and one output")
    if spec["operator"] not in {op.name for op in model.operators}:
        raise ValueError("Unknown exact operator identity in study plan")
    files={"onnx":Path(model_path),"samples":Path(inputs_path)}
    if labels_path is not None:
        files["labels"]=Path(labels_path)
    for name,path in files.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or linked experiment {name}: {path}")
    raw=np.load(files["samples"],allow_pickle=False)
    if raw.dtype!=np.float32:
        raise ValueError("Sample input must be float32 without implicit conversion")
    samples=normalize_samples(raw,model.tensor(model.inputs[0]).shape)
    if host_samples>len(samples):
        raise ValueError("Requested GCC host samples exceed available input samples")
    original_hashes={role:_hash(path) for role,path in files.items()}
    labels=np.load(files["labels"],allow_pickle=False) if "labels" in files else None
    if host_samples and shutil.which("g++") is None:
        raise RuntimeError("GCC is required for requested host C++ verification")
    destination=Path(directory)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Experiment output must be a new directory")
    with TemporaryDirectory(prefix=".fpgai-study-",dir=destination.parent) as folder:
        root=Path(folder)
        reference=evaluate_reference(model=model,model_path=files["onnx"],
            inputs_path=files["samples"],labels_path=files.get("labels"),
            output_dir=root/"reference",crosscheck_onnxruntime=crosscheck_onnxruntime)
        provenance={"onnx_sha256":_hash(files["onnx"]),
                    "inputs_npy_sha256":_hash(files["samples"])}
        if "labels" in files:
            provenance["labels_npy_sha256"]=_hash(files["labels"])
        plan_bytes=(json.dumps(plan,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf-8")
        (root/"experiment_plan.json").write_bytes(plan_bytes)
        plan_digest=sha256(plan_bytes).hexdigest()
        rows=prepare_campaign(model=model,runtime_inputs=samples,
            runtime_labels=labels,directory=root,operator=spec["operator"],
            default=spec["default"],variants=spec["variants"],
            interactions=spec["interactions"],limits=limits,
            part=spec["target_part"],clock_ns=spec["clock_ns"],
            tolerance=spec["tolerance"],source_provenance=provenance)
        blocked=[(r["case"],r["reason"]) for r in rows if r["status"]!="READY"]
        if require_all_legal and blocked:
            raise ValueError(f"Architecture plan contains illegal variants: {blocked}")
        if len(blocked)==len(rows):
            raise ValueError("No legal experiment architectures")
        if archive_inputs:
            from fpgai_research.experiments.input_archive import archive_inputs as save_inputs, verify_case_provenance
            save_inputs(root,model=files["onnx"],samples=files["samples"],
                        labels=files.get("labels"))
            verify_case_provenance(root)
        cases=[]
        for row,(axis,requested) in zip(rows,spec["points"]):
            if row["dimension"]!=axis or row["requested"]["precision_bits"]!=requested.precision_bits:
                raise AssertionError("Campaign case index diverges from requested architecture")
            if row["status"]!="READY":
                cases.append({"case":row["case"],"status":"BLOCKED","reason":row["reason"]})
                continue
            numerical=replace(spec["default"],precision_bits=requested.precision_bits,
                              integer_bits=requested.integer_bits)
            architecture=resolve(model,numerical,{spec["operator"]:requested},limits=limits)
            lowered=lower(architecture)
            preserved=verify_refinement(lowered)
            emitted=emit_hls(lowered)
            if emitted.manifest["cpp_sha256"]!=row["cpp_sha256"]:
                raise ValueError("Scientific IR/HLS output disagrees with prepared experiment")
            case=root/row["case"]
            snapshot=write_ir_artifact(case/"progressive_ir.json",lowered)
            read_ir_artifact(case/"progressive_ir.json")
            host=[]
            for index in range(min(host_samples,len(samples))):
                receipt=validate_host(model,emitted,samples[index])
                if not receipt["accepted"]:
                    raise ValueError(f"Host C++ numeric mismatch: {row['case']} input {index}")
                host.append({"sample_index":index,**receipt})
            if host:
                (case/"host_validation.json").write_text(json.dumps(host[0],sort_keys=True,indent=2)+"\n")
            trace_record=None
            if trace_intermediates:
                from fpgai_research.validation.tensor_trace import check_tensor_trace
                trace_record=check_tensor_trace(lowered,samples[0])
                if not trace_record["accepted"] or trace_record["cpp_sha256"]!=emitted.manifest["cpp_sha256"]:
                    raise ValueError("Intermediate operator numerical semantics differ from the Functional Model IR")
                (case/"tensor_trace.json").write_text(json.dumps(trace_record,sort_keys=True,indent=2)+"\n")
            (case/"host_validation_samples.json").write_text(json.dumps({
                "schema":"fpgai-research/host-sample-list/v1",
                "source_sha256":emitted.manifest["cpp_sha256"],
                "evidence":"E1-GCC-host-float-only","results":host},sort_keys=True,indent=2)+"\n")
            host_receipt_sha256=_hash(case/"host_validation_samples.json")
            trace_receipt_sha256=_hash(case/"tensor_trace.json") if trace_record else None
            cases.append({"case":row["case"],"status":"E1-PREPARED",
                "host_receipt_sha256":host_receipt_sha256,
                "trace_receipt_sha256":trace_receipt_sha256,
                "functional_sha256":preserved["functional_sha256"],
                "architecture_sha256":preserved["effective_architecture_sha256"],
                "stage_sha256":snapshot["stage_sha256"],
                "cpp_sha256":emitted.manifest["cpp_sha256"],
                "host_samples_passed":len(host),
                "intermediate_tensor_trace_passed":bool(trace_record) if trace_intermediates else None,
                "evidence":"E1-GCC-host-float" if host else "E1-source-only"})
        from fpgai_research.experiments.representation_evidence import analyze_representation
        representation=analyze_representation(root,rows)
        (root/"representation_ablation.json").write_text(json.dumps(representation,sort_keys=True,indent=2)+"\n")
        receipt={"schema":SCHEMA,"operator":spec["operator"],
                 "model_sha256":_hash(files["onnx"]),
                 "inputs_sha256":_hash(files["samples"]),
                 "labels_sha256":_hash(files["labels"]) if "labels" in files else None,
                 "functional_sha256":functional_fingerprint(model),
                 "plan_sha256":plan_digest,
                 "representation_evidence_sha256":_hash(root/"representation_ablation.json"),
                 "requested_cases":len(rows),"ready_cases":len(rows)-len(blocked),
                 "blocked_cases":len(blocked),
                 "host_samples_requested":host_samples,
                 "intermediate_trace_requested":trace_intermediates,
                 "all_ready_cases_have_three_ir_stages":all(
                    (root/r["case"]/"progressive_ir.json").is_file() for r in rows if r["status"]=="READY"),
                 "reference_evaluation":reference,
                 "reference_receipt_sha256":_hash(root/"reference"/"reference_evaluation.json"),
                 "sample_value_sha256":[sha256(item.tobytes()).hexdigest()
                                        for item in samples[:host_samples]],
                 "cases":cases,
                 "evidence":"E0/E1-prepared-source-only",
                 "native_vitis_csim_executed":False,
                 "hls_synthesis_executed":False,"vivado_routed":False,
                 "not_a_physical_benchmark":True,
                 "next_actions":["run-campaign Vitis explicitly on AMD workstation",
                     "run-campaign Vivado only after verified E3",
                     "collect-campaign and audit-campaign for E2/E3/E4"]}
        (root/"full_experiment.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
        if any(_hash(path)!=original_hashes[role] for role,path in files.items()):
            raise ValueError("Experiment source model, samples or labels changed during preparation")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Experiment destination appeared during preparation")
        Path(folder).rename(destination)
    return receipt
