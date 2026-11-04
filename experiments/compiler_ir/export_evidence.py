"""Read-only consolidated CST experiment evidence (not a paper or vendor proof).

Refuses to merge different compiler revisions or changed original CSV/JSON
assets. Does not infer Vitis, FPGA, trained-model or prior-art results.
"""
from __future__ import annotations
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

SERIES_SCHEMA = "fpgai-research/compiler-ir-experiment-series/v1"
FIELDS = {
    "array": ("array/summary.json", ("cases","host_checked_cases","host_numeric_passes")),
    "row": ("row/summary.json", ("paired_cases","host_checked_pairs","host_both_qualified")),
    "chain": ("chain/summary.json", ("cases","host_checked","host_numeric_passes",
                                   "phi_and_row_epoch_traces")),
    "schedule_mutations": ("schedule_mutations/summary.json",
        ("positive_controls","positive_control_accepts",
         "negative_interventions","negative_interventions_correctly_rejected")),
    "independent_oracle": ("independent_oracle/summary.json",
        ("complete_permutations","reference_legal","reference_illegal",
         "false_accepts","false_rejects")),
    "scaling": ("scaling/summary.json",
        ("distinct_graphs","schedule_samples","max_checked_element_events")),
    "nonlinear": ("nonlinear/summary.json",
        ("generated_cases","selected_host_checks","host_numerical_passes",
         "executed_row_event_passes")),
    "decision_interface": ("decision_interface/summary.json",
        ("graphs","candidate_actions","valid_actions","rejected_actions",
         "false_legal","false_illegal","host_numerical_passes",
         "actual_mapped_value_lifetime_passes",
         "policy_trajectories","trained_rl_policies")),
    "rl_compatible_ir_r3_r4": ("rl_compatible_ir_r3_r4/summary.json",
        ("graphs","renaming_invariance_cases",
         "identical_proposer_certificate_cases",
         "certified_action_proposals","certified_legal_actions",
         "source_admissible_but_unknown_actions",
         "native_vitis_calls","rl_policies_trained",
         "rl_sample_efficiency_measured")),
    "two_phi_counterfactual_e1": ("two_phi_counterfactual_e1/summary.json",
        ("original_cases","controlled_phi_contrasts",
         "effective_phi_interventions","ineffective_phi_interventions",
         "changed_last_use_contrasts","changed_scratch_capacity_contrasts",
         "lifetime_discrimination_established_by_this_study",
         "novelty_relative_to_prior_art_established","original_vendor_calls")),
    "value_lifetime": ("value_lifetime/summary.json",
        ("cases_attempted","source_value_trace_passes",
         "total_mapped_reduction_iterations",
         "total_logical_element_commits",
         "total_verified_consumer_releases")),
    "cpu_paper_table": ("cpu_paper_table/report.json",
        ("original_host_checked_total","original_host_numerical_passes")),
    "tensor_source_report": ("tensor_source_report/report.json",
        ("paired_variant_rows","paired_rows_with_source_attested_numerical_and_cycles")),
    "row_block_source_report": ("row_block_source_report/summary.json",
        ("paired_comparisons","pairs_with_original_csim_csynth_measurements")),
    "tensor_native_report": ("tensor_native_report/report.json",
        ("paired_rows_with_source_attested_numerical_and_cycles",)),
    "row_block_native_report": ("row_block_native_report/summary.json",
        ("pairs_with_original_csim_csynth_measurements",)),
    "mapped_event_causality": ("mapped_event_causality/summary.json",
        ("candidate_mappings","canonical_verified",
         "stale_mapping_rejections","missing_reduction_rejections",
         "early_read_rejections")),
    "mapped_source_e1": ("mapped_source_e1/summary.json",
        ("matched_source_cases","qualified_source_cases",
         "all_source_last_reads_released")),
    "fanout_e0": ("fanout_e0/summary.json",
        ("cases","valid_controls","early_alias_rejections","executed_source_cases")),
    "trained_mlp_host": ("trained_mlp_host/evidence.json",
        ("trained_samples_checked","onnx_vs_model_ir_qualified",
         "gcc_host_vs_model_ir_qualified")),
}


def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing/symlinked original experiment evidence: " + str(path))
    h=sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def collect(run: str | Path) -> dict:
    base=Path(run)
    if base.is_symlink() or not base.is_dir():
        raise ValueError("Expected real experiment series directory")
    manifest_path=base/"series_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Missing original source-pinned series manifest")
    meta=json.loads(manifest_path.read_text())
    if meta.get("schema") != SERIES_SCHEMA:
        raise ValueError("Unexpected series schema")
    revision=meta.get("source_revision")
    if not isinstance(revision,str) or len(revision)!=40:
        raise ValueError("Malformed full Git source SHA")
    if digest(base/"source_plan.json") != meta.get("configuration_sha256"):
        raise ValueError("Source plan does not match original config checksum")
    observations=[]
    for name, step in meta["steps"].items():
        status=step.get("status")
        if status not in ("PASSED","FAILED","BLOCKED"):
            raise ValueError("Unknown series stage status")
        for entry in step.get("artifacts",[]):
            orig=base/name/entry["path"]
            if digest(orig)!=entry["sha256"]:
                raise ValueError("Original experiment case changed: "+str(orig))
        measured={}
        if status=="PASSED" and name in FIELDS:
            rel, keys=FIELDS[name]
            file=base/rel
            if file.is_symlink() or not file.is_file():
                raise ValueError("Original stage result missing: "+str(file))
            record=json.loads(file.read_text())
            rev=record.get("source_revision",record.get("revision"))
            if rev is not None and rev != revision:
                raise ValueError("Cannot combine scientific measurements across revisions")
            measured={key:record[key] for key in keys if key in record}
        observations.append({
            "stage":name,"status":status,"evidence":step.get("evidence","E0/E1"),
            "original_counters":measured,
            "reason":step.get("reason"),
        })
    # The post-hoc controlled Phi analysis must refer to exactly the
    # original source-executed GCC CSV in the SAME series. A stand-alone
    # CSV with plausible numbers does not qualify as causal provenance.
    stages=meta["steps"]
    if stages.get("two_phi_counterfactual_e1",{}).get("status")=="PASSED":
        if stages.get("two_phi_readers_e1",{}).get("status")!="PASSED":
            raise ValueError("Phi causal analysis lacks original GCC parent stage")
        original=base/"two_phi_readers_e1"/"cases.csv"
        secondary=base/"two_phi_counterfactual_e1"/"summary.json"
        if secondary.is_symlink() or not secondary.is_file():
            raise ValueError("Phi causal source manifest missing")
        if json.loads(secondary.read_text()).get("input_csv_sha256")!=digest(original):
            raise ValueError("Phi causal CSV does not match same-run original GCC case bytes")
    # These are *protocol readiness* counters, not priority/physical claims.
    groups={
        "C1_compositional_CST_IR":(
            "two_phi_readers_e1","nchw_real_source_rw_e1"),
        "C2_bounded_certificate_and_independent_oracle":(
            "mapped_event_causality","graph_semantic_oracle_e0"),
        "C3_policy_independent_RL_compatible_interface":(
            "decision_interface","rl_compatible_ir_r3_r4"),
    }
    contribution_protocols={
        name:{
            "required_stages":list(needed),
            "passed_stages":sum(stages.get(stage,{}).get("status")=="PASSED"
                                for stage in needed),
            "source_evidence_complete":all(
                stages.get(stage,{}).get("status")=="PASSED" for stage in needed),
            "novelty_established":False,
            "physical_or_RL_advantage_established":False,
        }
        for name,needed in groups.items()
    }
    return {
        "schema":"fpgai-research/compiler-ir-scientific-index/v1",
        "source_revision":revision,
        "stages":observations,
        "stage_counts":{state:sum(s["status"]==state for s in observations)
                        for state in ("PASSED","FAILED","BLOCKED")},
        "three_research_contribution_source_protocols":contribution_protocols,
        "all_three_contribution_source_protocols_complete":all(
            item["source_evidence_complete"]
            for item in contribution_protocols.values()),
        "actual_RL_training_benefit_established":False,
        "original_vitis_hardware_quality_claim_established":False,
        "published_prior_art_novelty_established":False,
        "physical_fpga_timing_established":False,
        "unmeasured_hardware_fields_not_imputed":True,
        "limits":[
            "Bounded CPU/GCC observations are not universal compiler correctness",
            "Only original fixed-point CSim and CSynth may support HLS QoR",
            "Source scratch arrays do not establish physical BRAM reduction",
            "This index does not reproduce ST-Flow/Allo/StreamTensor baselines",
        ],
    }


def export(run: str | Path, output: str | Path) -> dict:
    dst=Path(output)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError("Create-only paper evidence index already exists")
    result=collect(run)
    dst.mkdir(parents=True)
    (dst/"paper_evidence.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    with (dst/"stage_status.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=[
            "stage","status","evidence","original_counters_json"])
        writer.writeheader()
        for x in result["stages"]:
            writer.writerow({
                "stage":x["stage"],"status":x["status"],"evidence":x["evidence"],
                "original_counters_json":json.dumps(x["original_counters"],sort_keys=True),
            })
    lines=[
        "# FPGAI CST Model IR — audited source-specific experiment index",
        "",
        "**Source revision:** "+chr(96)+result["source_revision"]+chr(96),
        "",
        "This is a source-verified results INDEX, not a scientific novelty "
        "claim, native synthesis result, or manuscript.",
        "CST C1/C2/C3 protocol completeness is only a bounded SOURCE evidence"
        " status: it is not external novelty, native QoR or improved RL learning.",
        "",
        "| Experiment | Status | Scope | Original metrics |",
        "| --- | --- | --- | --- |",
    ]
    for x in result["stages"]:
        metrics=", ".join(f"{k}={v}" for k,v in x["original_counters"].items())
        lines.append("| "+x["stage"]+" | "+x["status"]+" | "+
                     x["evidence"]+" | "+(metrics or "—")+" |")
    lines+=["", "## Mandatory unresolved research gates", "",
            "- Real trained MNIST must use original model and input hashes.",
            "- Native HLS figures require original numerical CSim and "
            "bounded synthesized HLS reports; physical claims need routing.",
            "- ST-Flow/Allo/StreamTensor feature parity and comparable "
            "artifact experiments remain open.",
            "- Broader Conv2D and branching composition require implemented "
            "source lowering and independent oracles.", ""]
    (dst/"paper_evidence.md").write_text("\n".join(lines))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series-root",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    result=export(args.series_root,args.output)
    print(json.dumps({"source_revision":result["source_revision"],
                      "stages":len(result["stages"]),
                      "stage_counts":result["stage_counts"]},indent=2))


if __name__=="__main__":
    main()
