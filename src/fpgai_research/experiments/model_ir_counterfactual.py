"""Controlled counterfactual: same Functional Model IR, one operator-local PE change.

E0/E1 only. This is an executable witness to WHAT Model IR preserves and
adds, not an FPGA timing or speedup experiment. Nothing here runs Vitis,
synthesis, place/route or modifies the original trained MNIST artifacts.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.model_ir_study import _fixture
from fpgai_research.architecture.scopes import resolve_scoped
from fpgai_research.architecture.phi import iteration_mapping_receipt
from fpgai_research.architecture.resolver import coordinate
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.model_ir.lowering import lower
from fpgai_research.model_ir.semantics import (
    functional_fingerprint, architecture_fingerprint, verify_refinement,
)

SCHEMA = "fpgai-research/operator-local-model-ir-counterfactual/v1"


def _stage(model, operator_pes):
    arch=resolve_scoped(model, {"precision_bits":16,"integer_bits":6},
                        operators={"mm_a":{"pes":operator_pes}})
    low=lower(arch)
    invariants=verify_refinement(low)
    maps={p.operator:iteration_mapping_receipt(p,max_iterations=1000)
          for p in arch.points}
    generated=emit_hls(low)
    return arch,low,invariants,maps,generated


def counterfactual():
    """Verify a controlled source-level mapping intervention, fail-closed."""
    model,_=_fixture()
    model.validate()
    baseline=_stage(model,1)
    changed=_stage(model,2)
    a,lo_a,inv_a,phi_a,gen_a=baseline
    b,lo_b,inv_b,phi_b,gen_b=changed
    names=tuple(op.name for op in model.operators)
    if names!=("mm_a","mm_b"):
        raise ValueError("Counterfactual fixture operator identity unexpectedly changed")
    fsha=functional_fingerprint(model)
    if not (inv_a["functional_sha256"]==inv_b["functional_sha256"]==fsha):
        raise ValueError("Counterfactual changed the mathematical model")
    if (architecture_fingerprint(a)==architecture_fingerprint(b)
            or inv_a["stage_invariants_verified"] is not True
            or inv_b["stage_invariants_verified"] is not True):
        raise ValueError("Architecture variation or IR preservation not verified")
    differences=[]
    for index,op in enumerate(names):
        pa,pb=a.points[index],b.points[index]
        x,y=lo_a.operators[index],lo_b.operators[index]
        delta={
            "operator":op,
            "mapping_changed":phi_a[op]["complete_mapping_sha256"]!=
                              phi_b[op]["complete_mapping_sha256"],
            "lowered_mechanisms_changed":asdict(x)!=asdict(y),
            "baseline_requested_pes":pa.requested.pes,
            "counterfactual_requested_pes":pb.requested.pes,
            "baseline_lowered_pe_count":x.pe_count,
            "counterfactual_lowered_pe_count":y.pe_count,
            "logical_iterations":phi_a[op]["logical_iterations"],
            "iteration_assignments_verified":
                phi_a[op]["oracle"]["distinct_assignments"]==
                phi_b[op]["oracle"]["distinct_assignments"]==
                phi_a[op]["logical_iterations"],
            "decision_scope":dict(pb.decision_sources).get("pes"),
        }
        differences.append(delta)
    # One explicit logical iteration: the same semantic MatMul work at
    # (m,n,k)=(0,1,0) gets a different *declared* PE coordinate. This
    # is inspectable, unlike an opaque hash inequality, but remains E0.
    logical_index=(0,1,0)
    witnesses={}
    for index,name in enumerate(names):
        # coordinate() uses tuples for tile/reuse dimensions. The
        # immutable E0/E1 witness is an on-disk JSON contract, so return
        # exactly the types that the JSON writer and reader preserve.
        # This canonicalization keeps the existing exported JSON bytes
        # unchanged; it prevents Python tuple/list inequality on replay.
        ca=json.loads(json.dumps(coordinate(a.points[index],logical_index),
                                 sort_keys=True,allow_nan=False))
        cb=json.loads(json.dumps(coordinate(b.points[index],logical_index),
                                 sort_keys=True,allow_nan=False))
        witnesses[name]={
            "logical_iteration":list(logical_index),
            "baseline":ca,
            "counterfactual":cb,
        }
    if (witnesses["mm_a"]["baseline"]["pe"]!=0
            or witnesses["mm_a"]["counterfactual"]["pe"]!=1
            or witnesses["mm_b"]["baseline"]!=witnesses["mm_b"]["counterfactual"]):
        raise ValueError("Explicit logical-work PE witness does not match requested intervention")
    first,second=differences
    if (first["baseline_requested_pes"]!=1
            or first["counterfactual_requested_pes"]!=2
            or first["baseline_lowered_pe_count"]!=1
            or first["counterfactual_lowered_pe_count"]!=2
            or not first["mapping_changed"] or not first["lowered_mechanisms_changed"]
            or not first["iteration_assignments_verified"]
            or first["decision_scope"]!="operator:mm_a"):
        raise ValueError("Selected operator intervention not realized by Model IR")
    if (second["baseline_requested_pes"]!=1
            or second["counterfactual_requested_pes"]!=1
            or second["mapping_changed"] or second["lowered_mechanisms_changed"]
            or not second["iteration_assignments_verified"]):
        raise ValueError("Unselected operator mapping changed")
    if (gen_a.manifest["cpp_sha256"]==gen_b.manifest["cpp_sha256"]
            or sha256(gen_a.cpp.encode()).hexdigest()!=gen_a.manifest["cpp_sha256"]
            or sha256(gen_b.cpp.encode()).hexdigest()!=gen_b.manifest["cpp_sha256"]):
        raise ValueError("HLS output identity/change not proven")
    return {
        "schema":SCHEMA,
        "experiment":"fixed-seed-2026-two-4x4-MatMul",
        "evidence":"E0-typed-IR-and-E1-generated-source-only",
        "model_operator_chain":list(names),
        "tensor_dependencies_preserved":True,
        "functional_sha256_baseline":fsha,
        "functional_sha256_counterfactual":inv_b["functional_sha256"],
        "functional_graph_and_constant_bytes_equal":True,
        "effective_architecture_sha256_baseline":
            inv_a["effective_architecture_sha256"],
        "effective_architecture_sha256_counterfactual":
            inv_b["effective_architecture_sha256"],
        "generated_hls_cpp_sha256_baseline":gen_a.manifest["cpp_sha256"],
        "generated_hls_cpp_sha256_counterfactual":gen_b.manifest["cpp_sha256"],
        "generated_hls_cpp_changed":True,
        "operator_comparisons":differences,
        "explicit_iteration_coordinate_witnesses":witnesses,
        "changed_architecture_operator_count":1,
        "unchanged_architecture_operator_count":1,
        "illegal_substitutions_silently_accepted":False,
        "native_fixed_point_e2_qualified":False,
        "achieved_ii_or_cycles_measured":False,
        "physical_e3_e4_or_speedup_measured":False,
        "cross_tool_novelty_or_superiority_proven":False,
    }


def table_tex(result):
    if (result.get("schema")!=SCHEMA
            or result.get("changed_architecture_operator_count")!=1
            or result.get("unchanged_architecture_operator_count")!=1
            or result.get("native_fixed_point_e2_qualified") is not False
            or result.get("physical_e3_e4_or_speedup_measured") is not False):
        raise ValueError("Unqualified Model IR comparison table")
    rows=result["operator_comparisons"]
    if (len(rows)!=2 or [r["operator"] for r in rows]!=["mm_a","mm_b"]):
        raise ValueError("Missing Model IR operator contrasts")
    def flag(b):
        return "Yes" if b else "No"
    lines=[
        "% Reproducible E0/E1-only Model IR counterfactual. No vendor measurements.",
        r"\begin{table}[t]",r"\centering\small",
        r"\caption{One-operator mapping intervention with immutable functional"
        r" graph and constants. PE values are IR requests and lowering"
        r" contracts, not measured physical cells.}",
        r"\label{tab:model-ir-counterfactual}",
        r"\begin{tabular}{@{}lrrrr@{}}",r"\toprule",
        r"Operator & PE (A) & PE (B) & $\Phi$ changes? & Lowering changes? \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            r"\texttt{"+row["operator"].replace("_",r"\_")+"} & "+
            str(row["baseline_requested_pes"])+" & "+
            str(row["counterfactual_requested_pes"])+" & "+
            flag(row["mapping_changed"])+" & "+
            flag(row["lowered_mechanisms_changed"])+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}",""])
    return "\n".join(lines)


def verify_existing_counterfactual(output):
    """Independently recompute and validate a saved E0/E1 paper witness.

    Rejects missing/changed source, extra files and stale LaTeX.
    Does not touch original model weights or invoke FPGA vendor tools.
    """
    root=Path(output)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing Model IR counterfactual evidence directory")
    if {x.name for x in root.iterdir()} != {
            "model_ir_counterfactual.json","model_ir_counterfactual.tex"}:
        raise ValueError("Unexpected or incomplete Model IR counterfactual files")
    jsonfile=root/"model_ir_counterfactual.json"
    texfile=root/"model_ir_counterfactual.tex"
    for f in (jsonfile,texfile):
        if (f.is_symlink() or not f.is_file() or
                not 0 < f.stat().st_size < 1024*1024):
            raise ValueError("Unsafe Model IR counterfactual evidence")
    expected=counterfactual()
    if json.loads(jsonfile.read_text(encoding="utf-8"))!=expected:
        raise ValueError("Model IR JSON does not match independently recomputed witness")
    if texfile.read_text(encoding="utf-8")!=table_tex(expected):
        raise ValueError("Model IR paper table does not match independently recomputed witness")
    return expected


def export_counterfactual(output):
    target=Path(output)
    if target.is_symlink() or target.exists():
        raise FileExistsError("Counterfactual output must be a new private folder")
    result=counterfactual()
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".model-ir-difference-",dir=target.parent) as tmp:
        stage=Path(tmp)/"result"
        stage.mkdir()
        (stage/"model_ir_counterfactual.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",
            encoding="utf-8")
        (stage/"model_ir_counterfactual.tex").write_text(table_tex(result),
                                                        encoding="utf-8")
        if target.is_symlink() or target.exists():
            raise FileExistsError("Counterfactual output appeared concurrently")
        stage.rename(target)
    return result
