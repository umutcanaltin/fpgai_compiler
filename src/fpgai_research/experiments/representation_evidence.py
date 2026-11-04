"""Scientific representation ablation over progressive Model IR stages.

Compare actual per-operator architecture and lowered mechanisms to a
Functional-IR-identical baseline. These are E0/E1 representation claims,
never achieved Vitis scheduling, area or timing evidence.
"""
from __future__ import annotations
from pathlib import Path
from fpgai_research.model_ir.artifacts import read_ir_artifact

def analyze_representation(directory, rows):
    root=Path(directory)
    if not isinstance(rows,list) or not rows or rows[0].get("dimension")!="baseline":
        raise ValueError("Representation ablation requires a first baseline")
    if rows[0].get("status")!="READY":
        raise ValueError("Baseline architecture must be legal")
    baseline=read_ir_artifact(root/rows[0]["case"]/"progressive_ir.json")
    functional=baseline["functional_sha256"]
    base_points={p["operator"]:p for p in baseline["stages"]["architectural"]["points"]}
    base_lowered={op["name"]:op for op in baseline["stages"]["lowered"]["operators"]}
    base_source=baseline["stages"]["hls"]["manifest"]["cpp_sha256"]
    entries=[]
    arch_hashes=set()
    source_hashes=set()
    for row in rows:
        if row.get("status")!="READY":
            entries.append({"case":row["case"],"dimension":row["dimension"],
                            "status":"BLOCKED","reason":row.get("reason")})
            continue
        artifact=read_ir_artifact(root/row["case"]/"progressive_ir.json")
        if artifact["functional_sha256"]!=functional:
            raise ValueError("Functional Model IR changed under architectural specialization")
        points={p["operator"]:p for p in artifact["stages"]["architectural"]["points"]}
        lowered={o["name"]:o for o in artifact["stages"]["lowered"]["operators"]}
        if points.keys()!=base_points.keys() or lowered.keys()!=base_lowered.keys():
            raise ValueError("Architectural/Lowered operator identity changed from baseline")
        mapped=sorted(op for op in points
            if points[op]["effective"]!=base_points[op]["effective"])
        mechanisms=sorted(op for op in lowered
            if lowered[op]!=base_lowered[op])
        source_sha=artifact["stages"]["hls"]["manifest"]["cpp_sha256"]
        arch_sha=artifact["stage_sha256"]["architectural"]
        arch_hashes.add(arch_sha)
        source_hashes.add(source_sha)
        entries.append({"case":row["case"],"dimension":row["dimension"],
            "status":"PASS","functional_sha256":functional,
            "changed_architectural_operators":mapped,
            "changed_lowered_operators":mechanisms,
            "hls_source_changed_vs_baseline":source_sha!=base_source,
            "architecture_stage_sha256":arch_sha,
            "lowered_stage_sha256":artifact["stage_sha256"]["lowered"],
            "cpp_sha256":source_sha})
    passing=[item for item in entries if item["status"]=="PASS"]
    return {"schema":"fpgai-research/progressive-representation-ablation/v1",
        "functional_preserved_all_legal_cases":True,
        "functional_sha256":functional,
        "legal_cases":len(passing),"blocked_cases":len(entries)-len(passing),
        "unique_architectural_stage_artifacts":len(arch_hashes),
        "unique_generated_hls_sources":len(source_hashes),
        "operator_local_changes":sum(bool(item["changed_architectural_operators"]) for item in passing),
        "evidence":"E0/E1-progressive-Model-IR-source-representation-only",
        "physical_qor_measured":False,"cases":entries}
