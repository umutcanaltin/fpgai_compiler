"""Independent file/log audit of nine executed CPU Model IR paper studies.

Can validate an EXISTING prior campaign even when its original runner
did not record the git source SHA. Evidence is CPU E0/E1 only.
This audit reads original outputs and writes to a separate NEW directory.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fpgai_research.experiments.cpu_semantic_campaign import SCHEMA, commands

AUDIT_SCHEMA="fpgai-research/cpu-semantic-campaign-independent-audit/v1"


def audit_cpu_campaign(directory):
    root=Path(directory)
    manifest=root/"campaign.json"
    if (root.is_symlink() or not root.is_dir() or manifest.is_symlink()
            or not manifest.is_file() or manifest.stat().st_size>8*1024*1024):
        raise ValueError("No safe completed CPU source campaign")
    record=json.loads(manifest.read_text(encoding="utf-8"))
    if (record.get("schema")!=SCHEMA or record.get("complete") is not True
            or record.get("executed") is not True
            or record.get("vendor_tools_executed") is not False
            or record.get("physical_results_claimed") is not False
            or record.get("full_fixed_point_equivalence_claimed") is not False):
        raise ValueError("CPU-only completed campaign contract not satisfied")
    receipts=record.get("receipts")
    expected=commands(root)
    if (not isinstance(receipts,list) or len(receipts)!=len(expected)
            or record.get("studies_completed")!=len(expected)):
        raise ValueError("CPU study completion count does not match expected cases")
    audit=[]
    for receipt,(name,args,artifact) in zip(receipts,expected):
        log=root/(name+".log")
        if (not isinstance(receipt,dict) or receipt.get("experiment")!=name
                or receipt.get("evidence")!="E0/E1-CPU-SOURCE-ONLY"
                or receipt.get("returncode")!=0
                or receipt.get("artifact_present") is not True
                or receipt.get("vendor_tools_executed") is not False):
            raise ValueError("CPU per-case receipt invalid: "+name)
        if (log.is_symlink() or not log.is_file()
                or log.stat().st_size>250000
                or sha256(log.read_bytes()).hexdigest()!=receipt.get("log_sha256")):
            raise ValueError("CPU per-case log missing or changed: "+name)
        if artifact.is_symlink() or not artifact.exists():
            raise ValueError("CPU per-case artifact missing: "+name)
        files=([artifact] if artifact.is_file()
               else sorted(p for p in artifact.rglob("*") if p.is_file()))
        if not files or any(p.is_symlink() or p.stat().st_size>16*1024*1024
                            for p in files):
            raise ValueError("CPU result artifact missing, linked or oversized: "+name)
        digest=sha256()
        for f in files:
            rel=f.relative_to(root).as_posix().encode()
            digest.update(len(rel).to_bytes(4,"big")+rel)
            filebytes=f.read_bytes()
            digest.update(sha256(filebytes).digest())
        audit.append({
            "experiment":name,
            "output_file_count":len(files),
            "output_tree_sha256":digest.hexdigest(),
            "original_log_sha256":receipt["log_sha256"],
            "log_hash_verified":True,
            "output_files_verified_present":True,
        })
    return {
        "schema":AUDIT_SCHEMA,
        "original_campaign_manifest_sha256":sha256(manifest.read_bytes()).hexdigest(),
        "nine_source_studies_reattested":len(audit)==9,
        "cpu_studies":audit,"verified_log_count":len(audit),
        "source_git_revision_attested_by_original_run":False,
        "native_vitis_or_vivado_executed":False,
        "numerical_fixed_point_e2_qualified":False,
        "physical_e3_e4_qualified":False,
        "paper_results_may_be_reported_as_cpu_e0_e1_only":True,
    }


def export_cpu_campaign_audit(directory,output):
    destination=Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Independent CPU audit must use a fresh output")
    report=audit_cpu_campaign(directory)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-cpu-audit-",dir=destination.parent) as tmp:
        stage=Path(tmp)/"audit"
        stage.mkdir()
        (stage/"cpu_campaign_audit.json").write_text(
            json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n",
            encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("CPU audit target appeared during export")
        stage.rename(destination)
    return report
