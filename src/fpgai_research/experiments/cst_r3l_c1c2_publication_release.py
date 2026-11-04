"""R3l: read-only, revision-pinned scientific publication evidence release.

Aggregates the author's *existing* create-only original receipts from R3c
and R3d..R3h, each independently re-audited. R3j/R3k are optional until
original experiments have actually run. The release is derived in a NEW
folder and never modifies any original source/native artifact.

This audits internal evidence, NOT external scientific novelty. It
intentionally cannot set submission_ready=True without an independently
verified competing artifact and a final reviewer decision. No GCC/Vitis/
Vivado/agent invocations. Frozen RL code remains untouched.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

from fpgai_research.experiments import (
    cst_r3c_cnn_publication_tables as r3c,
    cst_r3d_c1c2_finite_contract_ablation as r3d,
    cst_r3e_c1_fanout_counterexamples as r3e,
    cst_r3f_c2_variable_graph_coverage as r3f,
    cst_r3g_c1_independent_event_semantics as r3g,
    cst_r3h_c2_compiled_once_selection as r3h,
    cst_r3j_c1_heldout_semantics as r3j,
    cst_r3k_c2_target_limit_intervention as r3k,
)

SCHEMA="fpgai-research/R3l-C1C2-readonly-publication-evidence/v1"
ORIGINAL_SOURCE_REVISIONS={
    "R3c":"f665b62ca81ee181a1447e3eddd23d7eba578186",
    "R3d":"fc40369f9493d93cc78c4886ae2e32c5d1f64093",
    "R3e":"fc40369f9493d93cc78c4886ae2e32c5d1f64093",
    "R3f":"fc40369f9493d93cc78c4886ae2e32c5d1f64093",
    "R3g":"3309076814714e61906363e21e3d50b932519f9d",
    "R3h":"3309076814714e61906363e21e3d50b932519f9d",
}
DEFAULT_FOLDER={
    "R3c":"r3c_cnn_original_publication_tables_v1",
    "R3d":"r3d_c1c2_finite_contract_gate_v1",
    "R3e":"r3e_c1_bounded_counterexamples_v1",
    "R3f":"r3f_c2_variable_graph_coverage_v1",
    "R3g":"r3g_c1_independent_event_oracle_v1",
    "R3h":"r3h_c2_compiled_once_selection_v1",
    "R3j":"r3j_c1_heldout_semantics_v1",
    "R3k":"r3k_c2_target_limit_intervention_v1",
}
REQUIRED=("R3c","R3d","R3e","R3f","R3g","R3h")
OPTIONAL=("R3j","R3k")
SOURCE_KEYS={
    "R3d":"source_revision","R3e":"source_revision",
    "R3f":"original_source_revision","R3g":"source_revision",
    "R3h":"source_revision","R3j":"original_source_revision",
    "R3k":"original_source_revision",
}
READERS={
    "R3d":r3d.audit,"R3e":r3e.audit,"R3f":r3f.audit,
    "R3g":r3g.audit,"R3h":r3h.audit,"R3j":r3j.audit,"R3k":r3k.audit,
}


def _sha(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()


def _dir(path:Path)->None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("R3l original evidence directory absent or linked: "+str(path))


def audit_r3c(original:Path)->dict:
    """Rehash all derived R3c file bytes and inspect strict E2/E3 denominators."""
    _dir(original)
    names=(
        "summary.json","paper_results.md","case_population.csv",
        "matched_pair_population.csv","derived_export_sha256.json",
    )
    for name in names:
        file=original/name
        if file.is_symlink() or not file.is_file():
            raise ValueError("R3l original R3c derived cohort missing: "+name)
    digests=json.loads((original/"derived_export_sha256.json").read_text())
    if (not isinstance(digests,dict)
            or set(digests)!=set(names)-{"derived_export_sha256.json"}
            or any(digests[name]!=_sha(original/name) for name in digests)):
        raise ValueError("R3l original R3c exported file digest mismatch")
    data=json.loads((original/"summary.json").read_text())
    if (data.get("schema")!=r3c.SCHEMA
            or data.get("source_revision")!=ORIGINAL_SOURCE_REVISIONS["R3c"]
            or data.get("original_native_attempts")!=6
            or data.get("original_CSim_E2_qualified")!=6
            or data.get("original_CSynth_E3_qualified")!=4
            or data.get("original_CSynth_E3_rejected")!=2
            or data.get("matched_pairs_prespecified")!=3
            or data.get("matched_pairs_both_E3_qualified")!=2
            or data.get("physical_E4") is not None
            or data.get("learning_improvement") is not None
            or data.get("scientific_novelty_established") is not False
            or data.get("original_vendor_calls_during_analysis")!=0):
        raise ValueError("R3l R3c original source, evidence tier or denominator changed")
    if (not isinstance(data.get("cases"),list) or len(data["cases"])!=6
            or not isinstance(data.get("pairs"),list) or len(data["pairs"])!=3
            or sum(bool(x.get("both_E3")) for x in data["pairs"])!=2):
        raise ValueError("R3l R3c case-level matched population corrupted")
    with (original/"case_population.csv").open(newline="",encoding="utf-8") as handle:
        cases=list(csv.DictReader(handle))
    with (original/"matched_pair_population.csv").open(newline="",encoding="utf-8") as handle:
        pairs=list(csv.DictReader(handle))
    if len(cases)!=6 or len(pairs)!=3:
        raise ValueError("R3l R3c case/pair CSV denominator changed")
    # A failed HLS E3 arm is a scientific failure/unknown, never a
    # resource/cycle comparison with a fabricated numeric delta.
    for pair in data["pairs"]:
        if not pair["both_E3"] and any(pair.get(k) is not None for k in (
                "cycles_control","cycles_independent","cycles_delta",
                "estimated_latency_delta_ns","estimated_latency_change_percent",
                "bram_18k_delta","dsp_delta","lut_delta","ff_delta")):
            raise ValueError("R3l R3c rejected E3 pair contains imputed QoR data")
    if (data.get("qualified_pairs_independent_phi_more_cycles")!=2
            or data.get("qualified_pairs_estimated_BRAM_ties")!=2):
        raise ValueError("R3l R3c negative HLS pair outcomes modified")
    return {
        "original_source_revision":data["source_revision"],
        "original_cases":6,"e2_qualified":6,"e3_qualified":4,
        "e3_rejected":2,"matched_pairs":3,"matched_e3_pairs":2,
        "original_vitis_calls_during_analysis":0,
        "original_source_receipt_sha256":_sha(original/"summary.json"),
        "export_digest_manifest_sha256":_sha(original/"derived_export_sha256.json"),
    }


def _read_source_study(key:str,folder:Path)->dict:
    if key not in READERS:
        raise ValueError("R3l cannot audit unsupported original source study")
    _dir(folder)
    try:
        audit=READERS[key](folder)
    except Exception as exc:
        raise ValueError("R3l failed read-only original "+key+" audit") from exc
    if not isinstance(audit,dict):
        raise ValueError("R3l source study auditor did not return a record")
    rev=audit.get(SOURCE_KEYS[key])
    if (not isinstance(rev,str)
            or re.fullmatch(r"[a-f0-9]{40}",rev) is None):
        raise ValueError("R3l original source revision missing for "+key)
    expected=ORIGINAL_SOURCE_REVISIONS.get(key)
    if expected is not None and rev!=expected:
        raise ValueError("R3l original "+key+" revision does not match frozen cohort")
    if any(audit.get(flag) not in (None,0) for flag in (
            "original_compiler_or_vendor_calls_during_audit",
            "new_compiler_gcc_vitis_vivado_calls",
            "new_compiler_or_vendor_calls_during_audit",
            "new_compiler_vendor_calls",
            "original_compiler_or_vendor_calls_during_audit")):
        raise ValueError("R3l auditor invoked compilation unexpectedly for "+key)
    return audit


def gather(build:Path,*,include_optional:bool=False)->dict:
    entries={}
    for key in REQUIRED+OPTIONAL:
        original=build/DEFAULT_FOLDER[key]
        if key in OPTIONAL and not include_optional:
            continue
        if key in OPTIONAL and not original.exists():
            entries[key]={
                "status":"UNEXECUTED_OPTIONAL_SOURCE_EXPERIMENT",
                "original_folder":str(original),
                "audit":None,
            }
            continue
        _dir(original)
        receipt=(audit_r3c(original) if key=="R3c"
                 else _read_source_study(key,original))
        entries[key]={
            "status":"VERIFIED_ORIGINAL_RECEIPT",
            "original_folder":str(original),
            "audit":receipt,
        }
    if any(entries[k]["status"]!="VERIFIED_ORIGINAL_RECEIPT" for k in REQUIRED):
        raise ValueError("R3l required source/native receipt missing")
    # Every cited original has its own provenance; do NOT pretend these
    # are iid trials or merge source rows into a new native campaign.
    r3c_a=entries["R3c"]["audit"]
    r3d_a=entries["R3d"]["audit"]
    r3e_a=entries["R3e"]["audit"]
    r3f_a=entries["R3f"]["audit"]
    r3g_a=entries["R3g"]["audit"]
    r3h_a=entries["R3h"]["audit"]
    if (r3d_a.get("finite_candidate_denominator")!=6
            or (r3d_a.get("e0_certified"),r3d_a.get("rejected"),
                r3d_a.get("unknown_e0"))!=(3,2,1)
            or (r3e_a.get("original_case_count"),r3e_a.get("rejected_mutants"),
                r3e_a.get("independent_minimum_slot_oracle_agreements"))!=(5,25,5)
            or (r3f_a.get("candidate_population"),r3f_a.get("certified"),
                r3f_a.get("rejected"),r3f_a.get("unknown_e0"))!=(48,21,24,3)
            or (r3g_a.get("tested_event_trace_variants"),
                r3g_a.get("independent_oracle_disagreements"))!=(133,0)
            or (r3h_a.get("candidate_population"),r3h_a.get("catalog_compiler_calls"),
                r3h_a.get("compiled_once_selection_additional_compiler_calls"))!=(48,48,0)):
        raise ValueError("R3l required C1/C2 source population deviates from original observed evidence")
    return {
        "schema":SCHEMA,
        "required_original_cohorts":list(REQUIRED),
        "optional_cohorts":list(OPTIONAL) if include_optional else [],
        "original_cohorts":entries,
        "c1_checked_source_trace_variants_r3g":133,
        "c1_r3g_oracle_disagreements":0,
        "c1_r3e_targeted_mutants":25,
        "c1_r3e_oracle_cases":5,
        "c2_r3f_actions":48,
        "c2_r3f_e0_certified":21,
        "c2_r3f_e0_unknown":3,
        "c2_r3f_rejected":24,
        "c2_r3h_catalog_transitions":48,
        "c2_r3h_cached_selection_extra_transitions":0,
        "native_r3c_source_revision":r3c_a["original_source_revision"],
        "native_r3c_e2":6,"native_r3c_e3":4,
        "native_r3c_matched_e3":2,
        "independent_external_compiler_parity":"NOT_ESTABLISHED",
        "arbitrary_graph_formal_soundness":"NOT_ESTABLISHED",
        "trained_agent_performance":"SEPARATE_PAPER_NOT_MEASURED",
        "hardware_physical_E4_improvement":"NOT_ESTABLISHED",
        "submission_ready":False,
        "decision":"MAJOR_REVISION_REQUIRES_EXTERNAL_PARITY_AND_REVIEW",
        "native_or_source_compiler_calls_during_aggregation":0,
    }


def markdown(report:dict)->str:
    lines=[
        "# C1/C2 original evidence synthesis — read-only publication snapshot",
        "",
        "**Submission decision: MAJOR REVISION; this report is NOT an acceptance or novelty claim.**",
        "",
        "All rows come from separately executed original study folders. No source or native experiments were rerun.",
        "",
        "| Original cohort | Grounded result | Revision / provenance |",
        "|---|---|---|",
    ]
    facts={
        "R3c":"6/6 E2, 4/6 strict E3, 2/3 both-qualified matched E3 pairs; HLS only",
        "R3d":"6 finite source actions: 3 E0 certified, 2 rejected, 1 UNKNOWN",
        "R3e":"5/5 independent fixed-order source slot oracle; 25/25 selected mutants rejected",
        "R3f":"48 actions: 21 E0 certified, 24 rejected, 3 UNKNOWN",
        "R3g":"133/133 finite source trace verdicts agree with separate interpreter",
        "R3h":"48 source compiler transitions; cached certified selection requires 0 additional transitions",
        "R3j":"Optional held-out 24-source-profile/216-trace source study",
        "R3k":"Optional 12-pair/192-action source target-limit study",
    }
    for key,entry in report["original_cohorts"].items():
        audit=entry["audit"]
        if audit is None:
            lines.append(f"| {key} | NOT EXECUTED / NO RESULT | --- |")
        else:
            source=audit.get("original_source_revision",audit.get("source_revision"))
            lines.append(f"| {key} | {facts[key]} | `{source}` |")
    lines.extend([
        "",
        "## What is not demonstrated",
        "",
        "- Independent equivalence/capability parity with Allo, CODO or other source IR",
        "- Formal correctness for all possible graphs, memory ports or FPGA cycles",
        "- Trained-agent or policy learning improvements (future separate paper)",
        "- Strict physical E4 FPGA resource, achieved latency or power improvements",
        "",
        "Individual source campaigns and original CNN/native pairs share architecture patterns and **are not independent random samples**.",
        "The full E3-rejected pair remains unqualified; no E3 resources/cycles were imputed.",
        "",
    ])
    return "\n".join(lines)


def export_create_only(report:dict,*,output:str|Path)->None:
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("R3l publication aggregate is strictly create-only")
    if report.get("schema")!=SCHEMA or report.get("submission_ready") is not False:
        raise ValueError("R3l publication report could not be independently re-audited")
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"evidence_manifest.json").write_text(
        json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n",
        encoding="utf-8")
    (dest/"paper_evidence_table.md").write_text(markdown(report),encoding="utf-8")
    with (dest/"original_cohort_provenance.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=(
            "cohort","original_folder","receipt_status","source_revision",
            "audit_sha256"))
        writer.writeheader()
        for key,entry in report["original_cohorts"].items():
            audit=entry["audit"] or {}
            writer.writerow({
                "cohort":key,
                "original_folder":entry["original_folder"],
                "receipt_status":entry["status"],
                "source_revision":audit.get(
                    "original_source_revision",audit.get("source_revision","")),
                "audit_sha256":next((audit[f] for f in (
                    "saved_receipt_sha256","read_only_receipt_sha256",
                    "source_receipt_sha256","receipt_sha256",
                    "original_source_receipt_sha256") if f in audit),""),
            })
    names=("evidence_manifest.json","paper_evidence_table.md",
           "original_cohort_provenance.csv")
    (dest/"derived_sha256.json").write_text(json.dumps({
        k:_sha(dest/k) for k in names},indent=2,sort_keys=True)+"\n",
        encoding="utf-8")


def main(argv:list[str]|None=None)->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build",default="build")
    parser.add_argument("--output",required=True)
    parser.add_argument("--include-optional",action="store_true")
    args=parser.parse_args(argv)
    out=Path(args.output)
    if out.exists() or out.is_symlink():
        raise FileExistsError("R3l original release output already exists")
    result=gather(Path(args.build),include_optional=args.include_optional)
    export_create_only(result,output=out)
    print(json.dumps({
        "schema":SCHEMA,
        "read_only_verified_required_cohorts":len(REQUIRED),
        "independent_external_compiler_parity":
            result["independent_external_compiler_parity"],
        "submission_ready":False,
        "new_compiler_vendor_calls":0,
        "output":str(out),
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
