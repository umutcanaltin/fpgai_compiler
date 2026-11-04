"""Export a nine-case CPU Model IR replay appendix from a VERIFIED private audit.

Never read raw trained weights or convert CPU source execution into Vitis E2.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

EXPECTED={
    "semantic_property":"Semantic property campaign",
    "model_ir":"Model IR representation",
    "refinement":"Progressive refinement",
    "materialization":"Source-loop materialization",
    "scoped":"Scoped expressiveness",
    "semantic_stress":"Multi-graph stress",
    "branched_conv":"Branched convolution",
    "intermediate_tensors":"Intermediate tensor semantics",
    "expanded_semantics":"Expanded operator semantics",
}


def export_cpu_appendix(audit_path,output):
    original=Path(audit_path)
    dest=Path(output)
    if (original.is_symlink() or not original.is_file()
            or original.stat().st_size>1024*1024):
        raise ValueError("Missing safe private CPU audit")
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("CPU appendix destination already exists")
    audited=json.loads(original.read_text(encoding="utf-8"))
    cases=audited.get("cpu_studies")
    if (audited.get("schema")!="fpgai-research/cpu-semantic-campaign-independent-audit/v1"
            or audited.get("nine_source_studies_reattested") is not True
            or audited.get("verified_log_count")!=9
            or audited.get("native_vitis_or_vivado_executed") is not False
            or audited.get("numerical_fixed_point_e2_qualified") is not False
            or audited.get("physical_e3_e4_qualified") is not False
            or not isinstance(cases,list) or len(cases)!=9
            or {x.get("experiment") for x in cases}!=set(EXPECTED)
            or not all(x.get("log_hash_verified") is True
                       and x.get("output_files_verified_present") is True
                       and isinstance(x.get("output_file_count"),int)
                       and x["output_file_count"]>0
                       for x in cases)):
        raise ValueError("CPU appendix requires independently verified nine-study audit")
    lines=[
        r"\begin{table}[t]",r"\centering",r"\small",
        r"\caption{Fresh CPU/source-only Model IR campaign, independently "
        r"audited at the receipt and output-file level. This is E0/E1 "
        r"evidence, not fixed-point native CSim or hardware quality of result.}",
        r"\label{tab:cpu-campaign-reattested}",
        r"\begin{tabular}{lr}",r"\toprule",
        r"CPU Model IR study & Verified outputs \\",
        r"\midrule",
    ]
    for case in cases:
        lines.append(EXPECTED[case["experiment"]]+" & "+
                     str(case["output_file_count"])+r" \\")
    lines.extend([r"\midrule",
                  r"Completed case receipts & 9 \\",
                  r"\bottomrule",r"\end{tabular}",
                  r"\end{table}",""])
    receipt={
        "schema":"fpgai-research/cpu-campaign-paper-appendix/v1",
        "original_cpu_audit_sha256":sha256(original.read_bytes()).hexdigest(),
        "completed_and_audited_cpu_studies":9,
        "native_numerical_e2_executed":False,
        "native_e3_e4_evidence_claimed":False,
        "original_run_source_git_attestation":
            audited.get("source_git_revision_attested_by_original_run",False),
        "table_is_standalone_optional_supplement":True,
    }
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-cpu-appendix-",dir=dest.parent) as tmp:
        stage=Path(tmp)/"appendix"
        stage.mkdir()
        (stage/"cpu_replay_table.tex").write_text("\n".join(lines),encoding="utf-8")
        (stage/"appendix_receipt.json").write_text(
            json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Private CPU appendix appeared concurrently")
        stage.rename(dest)
    return receipt
