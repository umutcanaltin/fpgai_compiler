"""Scientific IR-to-lowering-to-native outcome explanations for research suites.

This is *not* a new compiler stage or a QoR predictor. It derives
weight-free, hash-attested evidence from the saved Functional,
Architectural and Lowered Model IR, independently collected E1/E4 native
reports, and the immutable seven-axis campaign. No tools are executed.
"""
from __future__ import annotations

from hashlib import sha256
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.full_suite import audit_suite
from fpgai_research.experiments.results import summarize_case
from fpgai_research.experiments.paper_tables import make_rows
from fpgai_research.experiments.ablation_comparison import compare_suite_tables
from fpgai_research.model_ir.artifacts import (
    read_ir_artifact, compare_ir_artifacts, verify_ir_artifact
)
from fpgai_research.model_ir.phi_refinement import verify_hls_phi

SCHEMA="fpgai-research/scientific-ir-mechanism-study/v1"
ARCH_FIELDS=("precision_bits","integer_bits","parallelism","pes","simd",
             "serialization","tile","pipeline_ii")
LOWERED_FIELDS=("tile_shape","replica_count","pe_count","simd_lanes",
                "fold_factor","pipeline_ii","numerical_type","tail_policy",
                "source_scope","decision_sources")
COLUMNS=("model","family","case","axis","functional_preserved",
         "changed_architectural_operators","changed_lowered_operators",
         "changed_architecture_fields","changed_lowering_fields",
         "phi_verified_operators","phi_unverified_operators",
         "hls_source_changed","source_sha256","evidence","numerical_pass",
         "outcome","hls_latency_cycles","hls_latency_delta_cycles",
         "routed_lut","routed_wns_ns","qualified_lut_delta",
         "interpretation")


def _digest(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def _changes(old:dict,new:dict,fields:tuple[str,...])->dict:
    return {field:{"baseline":old.get(field),"variant":new.get(field)}
            for field in fields if old.get(field)!=new.get(field)}


def _op_maps(artifact:dict)->tuple[dict,dict]:
    points={p["operator"]:p for p in artifact["stages"]["architectural"]["points"]}
    lowered={lo["name"]:lo for lo in artifact["stages"]["lowered"]["operators"]}
    if set(points)!=set(lowered):
        raise ValueError("IR stages do not preserve operator identities")
    return points,lowered


def _classification(row:dict)->tuple[str,str]:
    ev=row.get("evidence")
    if ev!="E4-Vivado-implementation":
        if ev=="E3-HLS-synthesis-report":
            return "HLS_ONLY","HLS estimates exist; native routed impact is unmeasured"
        return "SOURCE_ONLY","Only E0/E1 compiler representation evidence exists"
    if row.get("numerical_pass") is not True:
        return "NUMERICALLY_REJECTED","Routed hardware failed the original fixed-point acceptance gate"
    if row.get("postroute_timing_met") is not True:
        return "TIMING_REJECTED","Routed hardware violates the requested clock constraint"
    if row.get("comparison_status")!="QUALIFIED":
        return "OTHER_REJECTED","Routed measurement is not an accepted controlled ablation"
    delta=row.get("delta_hls_latency_cycles")
    if type(delta) not in (int,float):
        raise ValueError("Qualified case has no attested latency delta")
    if delta<0:
        return "QUALIFIED","HLS estimated latency decreases; no board-runtime claim"
    if delta>0:
        return "QUALIFIED","HLS estimated latency increases; replication is not a guaranteed speedup"
    return "QUALIFIED","HLS estimated latency is unchanged; a changed IR is not a performance gain"


def analyze_ir_suite(directory)->dict:
    """Read/verify all E0/E1 artifacts; append native observations by case.

    Complete Functional IR contains model constants. Deliberately export
    ONLY hashes, counts, requested mapping fields and lowered mechanisms.
    """
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("A regular prepared research suite is required")
    audit=audit_suite(root,require_host=True,require_all_axes=False)
    if not audit["all_models_source_ready"]:
        raise ValueError("Scientific IR report requires fully source-prepared studies")
    manifest=root/"suite_results.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("A regular pinned suite inventory is required")
    studies=json.loads(manifest.read_text())["studies"]
    model_data=[]
    native=[]
    rows=[]
    for member in studies:
        if member.get("status")!="E1-PREPARED":
            raise ValueError("Cannot explain blocked study as source-prepared")
        name=member["id"]
        family=member["family"]
        folder=root/name
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("A regular model study directory is required")
        campaign_file=folder/"campaign.json"
        if campaign_file.is_symlink() or not campaign_file.is_file():
            raise ValueError("Missing or linked prepared campaign")
        campaign=json.loads(campaign_file.read_text())
        available=[entry for entry in campaign["cases"] if entry["status"]=="READY"]
        if len(available)!=len(campaign["cases"]):
            raise ValueError("Scientific IR study requires all declared variants to be READY; blocked variants must not disappear")
        if not available or available[0]["case"]!="case_0000" or available[0]["dimension"]!="baseline":
            raise ValueError("A source-attested first-case baseline is required")
        baseline_path=folder/"case_0000"/"progressive_ir.json"
        baseline=read_ir_artifact(baseline_path)
        bp,bl=_op_maps(baseline)
        functional=baseline["stages"]["functional"]
        functional_sha=baseline["functional_sha256"]
        entries=[]
        attested=[]
        for spec in available:
            label=spec["case"]
            case_dir=folder/label
            if case_dir.is_symlink() or not case_dir.is_dir():
                raise ValueError("A regular prepared case directory is required")
            artifact=read_ir_artifact(case_dir/"progressive_ir.json")
            checked=compare_ir_artifacts(baseline,artifact)
            if not checked["functional_preserved"]:
                raise ValueError("Architectural search changed the functional graph")
            ap,al=_op_maps(artifact)
            # Reconstruct verified typed IR, then independently enumerate the
            # actual HLS loop ordering against the portable Phi mapping.
            typed=verify_ir_artifact(artifact)
            phi_checks={point.operator:verify_hls_phi(point)
                        for point in typed.architectural.points}
            if set(phi_checks)!=set(ap):
                raise ValueError("Phi refinement lost an architectural operator")
            if set(bp)!=set(ap) or set(bl)!=set(al):
                raise ValueError("Architecture/Lowered Model IR operators changed")
            cpp=artifact["stages"]["hls"]["manifest"]["cpp_sha256"]
            if cpp!=spec["cpp_sha256"]:
                raise ValueError("Saved IR HLS source differs from pinned campaign")
            native_case=summarize_case(str(case_dir))
            if native_case["source_sha256"]!=cpp:
                raise ValueError("Native evidence disagrees with the compiled IR source")
            attested.append(native_case)
            operators=[]
            for op in sorted(ap):
                arch_change=_changes(bp[op]["effective"],ap[op]["effective"],ARCH_FIELDS)
                lowered_change=_changes(bl[op],al[op],LOWERED_FIELDS)
                point=ap[op]
                operators.append({
                    "operator":op,"kind":point["kind"],
                    "iteration_shape":point["iteration_shape"],
                    "requested":point["requested"],
                    "effective":point["effective"],
                    "source_scope":point["source_scope"],
                    "decision_sources":point["decision_sources"],
                    "changed_effective_fields":arch_change,
                    "lowered_mechanisms":{k:al[op].get(k) for k in LOWERED_FIELDS},
                    "changed_lowering_fields":lowered_change,
                    "phi_refinement":phi_checks[op],
                })
            entries.append({
                "model":name,"family":family,"case":label,
                "axis":spec["dimension"],
                "functional_sha256":functional_sha,
                "functional_preserved":checked["functional_preserved"],
                "stage_sha256":artifact["stage_sha256"],
                "hls_source_changed":checked["hls_source_changed"],
                "source_sha256":cpp,
                "changed_architectural_operators":checked["changed_effective_mapping_operators"],
                "changed_lowered_operators":checked["changed_lowered_operators"],
                "changed_scope_provenance_operators":checked["changed_scope_provenance_operators"],
                "phi_verified_operators":sorted(op for op,receipt in phi_checks.items()
                    if receipt["status"]=="EXHAUSTIVE_PHI_REFINEMENT"),
                "phi_unverified_operators":sorted(
                    op+":"+receipt["status"] for op,receipt in phi_checks.items()
                    if receipt["status"]!="EXHAUSTIVE_PHI_REFINEMENT"),
                "operators":operators,
            })
        rows.extend(entries)
        evidence=make_rows(campaign,{"schema":"fpgai-research/paper-evidence/v2",
                                     "results":attested})
        for item in evidence:
            item.update(model=name,family=family,functional_sha256=functional_sha)
        native.extend(evidence)
        model_data.append({
            "model":name,"family":family,"functional_sha256":functional_sha,
            "functional_operator_count":len(functional["operators"]),
            "functional_tensor_count":len(functional["tensors"]),
            "operator_kinds":[op["kind"] for op in functional["operators"]],
            "campaign_sha256":_digest(campaign_file),
            "cases":len(available)
        })
    comparison=compare_suite_tables({
        "schema":"fpgai-research/full-paper-suite-tables/v1",
        "case_count":len(native),"suite_sha256":_digest(manifest),"rows":native})
    qualified_by_key={(r["model"],r["case"]):r for r in comparison["rows"]}
    for item in rows:
        native_row=qualified_by_key[(item["model"],item["case"])]
        status,interp=_classification(native_row)
        item.update({
            "evidence":native_row["evidence"],
            "numerical_pass":native_row["numerical_pass"],
            "outcome":status,"interpretation":interp,
            "hls_latency_cycles":native_row["observed_hls_latency_cycles"],
            "hls_top_interval_cycles":native_row["observed_hls_ii_cycles"],
            "hls_latency_delta_cycles":native_row["delta_hls_latency_cycles"],
            "routed_lut":native_row["observed_lut"],
            "routed_wns_ns":native_row["observed_wns_ns"],
            "qualified_lut_delta":native_row["delta_lut"],
            "routed_resources": ({
                "lut":native_row["observed_lut"],
                "ff":native_row["observed_ff"],
                "dsp":native_row["observed_dsp"],
                "bram_tiles":native_row["observed_bram_tiles"],
                "uram":native_row["observed_uram"],
            } if native_row["evidence"]=="E4-Vivado-implementation" else None),
            "qualification_blocker":native_row["blocker"],
        })
    phi_receipts=[
        op["phi_refinement"]
        for record in rows for op in record["operators"]
    ]
    phi_statuses=[proof["status"] for proof in phi_receipts]
    samples=[
        proof["sampled_refinement"]
        for proof in phi_receipts if proof.get("sampled_refinement")
    ]
    return {
        "schema":SCHEMA,"source_suite_sha256":_digest(manifest),
        "models":model_data,"case_count":len(rows),
        "phi_refinement_summary":{
            "exhaustive":phi_statuses.count("EXHAUSTIVE_PHI_REFINEMENT"),
            "bound_exceeded":phi_statuses.count("BOUND_EXCEEDED"),
            "unsupported_kind":phi_statuses.count("UNSUPPORTED_KIND"),
            "sampled_large_operators":sum(
                r["status"]=="SAMPLED_COUNTERS_MATCH_PHI" for r in samples),
            "sampled_iteration_witnesses":sum(
                r["sampled_iterations"] for r in samples),
            "algebraically_characterized_large_operators":sum(
                proof.get("algebraic_partition",{}).get("status") ==
                "ALGEBRAIC_PARTITION_DERIVED" for proof in phi_receipts),
            "algebraic_certificates_are_not_native_or_exhaustive":True,
            "sampled_not_exhaustively_verified":True,
            "checked_source_loops_only":True,
        },
        "e4_routed_cases":comparison["e4_routed_cases"],
        "qualified_cases":comparison["qualified_cases"],
        "cases":rows,
        "weight_values_exported":False,
        "native_tools_executed":False,
        "evidence_boundaries":{
            "functional":"Verified E0 model operator/tensor identity, not full accuracy",
            "architectural":"E1 requested/effective mappings and scope provenance",
            "lowered":"E1 explicit execution mechanisms and refinement checks",
            "phi_loop_consistency":"E0 exhaustive source-loop checks for small domains; large domains retain bounded status with deterministic samples and separately labeled Euclidean partition derivations (not actual emitted-C++ proofs); other kinds unverified",
            "hls":"E1 regenerated source SHA; E3 latency is a synthesis estimate",
            "physical":"E4 OOC routed QoR, gated by numerical and timing acceptance",
        },
        "limitations":(
            "Identical Functional Model IR hashes prove identical saved mathematical "
            "graphs, not universal numerical equivalence under quantization. "
            "Lowered replicas/PEs/SIMD specify intended hardware structures; "
            "they do not guarantee Vivado scheduling or throughput gains. "
            "No model constants, on-board measurement or full-dataset accuracy "
            "are exported."
        ),
    }


def export_ir_suite(directory, output)->dict:
    root=Path(directory)
    study=analyze_ir_suite(root)
    dest=Path(output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Refusing to overwrite an existing scientific IR study")
    with TemporaryDirectory(prefix=".fpgai-ir-science-",dir=dest.parent) as temp:
        stage=Path(temp)
        (stage/"ir_mechanisms.json").write_text(
            json.dumps(study,sort_keys=True,indent=2,allow_nan=False)+"\n")
        with (stage/"ir_mechanisms.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=COLUMNS)
            writer.writeheader()
            for record in study["cases"]:
                changes=[(op["operator"],list(op["changed_effective_fields"]),
                          list(op["changed_lowering_fields"]))
                         for op in record["operators"] if op["changed_effective_fields"]
                         or op["changed_lowering_fields"]]
                writer.writerow({
                    field:("TBD" if value is None else value)
                    for field,value in {
                        "model":record["model"],"family":record["family"],
                        "case":record["case"],"axis":record["axis"],
                        "functional_preserved":record["functional_preserved"],
                        "changed_architectural_operators":";".join(record["changed_architectural_operators"]),
                        "changed_lowered_operators":";".join(record["changed_lowered_operators"]),
                        "changed_architecture_fields":json.dumps([
                            {"operator":op,"fields":arch} for op,arch,_ in changes
                        ],sort_keys=True),
                        "changed_lowering_fields":json.dumps([
                            {"operator":op,"fields":lowered} for op,_,lowered in changes
                        ],sort_keys=True),
                        "phi_verified_operators":";".join(record["phi_verified_operators"]),
                        "phi_unverified_operators":";".join(record["phi_unverified_operators"]),
                        "hls_source_changed":record["hls_source_changed"],
                        "source_sha256":record["source_sha256"],
                        "evidence":record["evidence"],
                        "numerical_pass":record["numerical_pass"],
                        "outcome":record["outcome"],
                        "hls_latency_cycles":record["hls_latency_cycles"],
                        "hls_latency_delta_cycles":record["hls_latency_delta_cycles"],
                        "routed_lut":record["routed_lut"],
                        "routed_wns_ns":record["routed_wns_ns"],
                        "qualified_lut_delta":record["qualified_lut_delta"],
                        "interpretation":record["interpretation"],
                    }.items()
                })
        lines=["# Scientific Model IR mechanism study","",
               "**Research question:** how do source-preserved architecture choices "
               "change explicitly lowered mechanisms and observed HLS/OOC QoR?","",
               "The original Functional Model IR is hash-verified across cases. "
               "No trained weights or other tensor constant values are exported.",""]
        coverage=study["phi_refinement_summary"]
        lines.extend([
            "## E0 Phi refinement coverage", "",
            f"Exhaustively checked operator-cases: {coverage['exhaustive']}. "
            f"Large unenumerated operator-cases: {coverage['bound_exceeded']}. "
            f"Other unsupported kinds: {coverage['unsupported_kind']}.", "",
            f"Large-domain Euclidean partition derivations: "
            f"{coverage['algebraically_characterized_large_operators']} "
            "(mapping contract only; not a proof of emitted C++ or RTL).", "",
            f"Deterministic nonexhaustive sample witnesses: "
            f"{coverage['sampled_iteration_witnesses']}. "
            "Samples and algebraic derivations are independent evidence "
            "categories and do not upgrade BOUND_EXCEEDED to an exhaustive pass.", "",
        ])
        for model in study["models"]:
            lines.extend([f"## {model['model']} ({model['family']})",
                f"Functional Model IR: {model['functional_operator_count']} operators, "
                f"{model['functional_tensor_count']} tensors; "
                f"SHA-256 `{model['functional_sha256']}`.","",
                "| Case | Axis | Architecture → Lowering changes | Verified Phi ops | Unverified Phi ops | Native outcome | HLS Δ cycles | Routed LUT | WNS (ns) |",
                "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |"])
            for record in study["cases"]:
                if record["model"]!=model["model"]:continue
                changes=[]
                for op in record["operators"]:
                    for field,diff in op["changed_effective_fields"].items():
                        corresponding=[name for name in op["changed_lowering_fields"]]
                        changes.append(f"{field}: {diff['baseline']}→{diff['variant']}; "
                                       f"lowered: {','.join(corresponding) or 'none'}")
                change="; ".join(changes) or "baseline/no effective mapping changes"
                change=change.replace("|","/")
                values=[record["case"],record["axis"],change,
                        len(record["phi_verified_operators"]),
                        len(record["phi_unverified_operators"]),
                        record["outcome"],record["hls_latency_delta_cycles"],
                        record["routed_lut"],record["routed_wns_ns"]]
                lines.append("| "+" | ".join("TBD" if x is None else str(x) for x in values)+" |")
            lines.extend(["","The architecture and lowered mechanisms are source-level"
                          " contracts. The native deltas are accepted only after"
                          " source-attested E4/numerical/physical qualification.",""])
        lines.extend(["## Scientific limitations","",study["limitations"],""])
        (stage/"ir_mechanisms.md").write_text("\n".join(lines))
        from fpgai_research.experiments.ir_pair_study import export_ir_pairs
        export_ir_pairs(study, stage)
        from fpgai_research.experiments.ir_schedule_evidence import export_ir_hls_schedule
        schedule=export_ir_hls_schedule(study, stage)
        from fpgai_research.experiments.native_schedule_reports import export_native_schedule_inventory
        inventory=export_native_schedule_inventory(root, study, stage)
        from fpgai_research.experiments.ir_research_questions import export_ir_research_questions
        export_ir_research_questions(schedule,inventory,stage)
        stage.rename(dest)
    return study
