"""Read-only scientific publication gate for original CST NCHW experiments.

This checks primary original HLS/physical counts and original identity
against their independently audited source trees. Neither an original
Vivado route nor a compiler unit test can establish a novel research
contribution; trained-network validity and peer-artifact parity are
always explicitly listed as human/independent research obligations.
No native or external tools are executed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from fpgai_research.experiments.cst_nchw_native_execute import audit_native
from fpgai_research.experiments.cst_nchw_physical import audit_routed_study

SCHEMA="fpgai-research/cst-publication-readiness-gate/v1"
_PEERS=("ST-Flow","Allo","HIDA","StreamTensor","Wavelet")


def _hash_file(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing original source-pinned study manifest")
    return sha256(path.read_bytes()).hexdigest()


def evaluate(*,native_study:str|Path,physical_study:str|Path)->dict:
    """Attest what is present, never infer absent external evidence."""
    native_root=Path(native_study)
    physical_root=Path(physical_study)
    native=audit_native(native_root,require_current_source=False)
    physical=audit_routed_study(physical_root)
    ncases=native.get("cases_audited")
    pcases=physical.get("cases")
    if (native.get("cases")!=12 or native.get("matched_pairs")!=6
            or not isinstance(ncases,list) or len(ncases)!=12
            or physical.get("prespecified_cases")!=12
            or not isinstance(pcases,list) or len(pcases)!=12):
        raise ValueError("Original scientific NCHW denominators must be exactly 12/6")
    if native.get("source_revision")!=physical.get("original_hls_source_revision"):
        raise ValueError("Native and physical studies have different source provenance")
    for i,(n,p) in enumerate(zip(ncases,pcases)):
        if (n.get("case")!=f"case_{i:03d}"
                or p.get("case")!=n["case"]
                or p.get("source_storage")!=n.get("source_storage")
                or p.get("graph")!=n.get("graph")
                or p.get("mapping")!=n.get("mapping")
                or p.get("source_cpp_sha256")!=n.get("source_cpp_sha256")):
            raise ValueError("Original HLS and Vivado case identities do not match")
    pairs=native.get("pairs_audited")
    if not isinstance(pairs,list) or len(pairs)!=6:
        raise ValueError("Original HLS six-pair population missing")
    strict_cases=sum(c.get("routed_qualified") is True for c in pcases)
    observed_cases=sum(c.get("routed_observation_valid") is True for c in pcases)
    strict_pairs=sum(all(c.get("routed_qualified") is True
                         for c in pcases[2*i:2*i+2]) for i in range(6))
    observed_pairs=sum(all(c.get("routed_observation_valid") is True
                           for c in pcases[2*i:2*i+2]) for i in range(6))
    hls_cases=sum(c.get("observed_numerical_and_hls_report") is True
                  for c in ncases)
    hls_pairs=sum(p.get("both_original_CSim_CSynth_attested") is True
                  for p in pairs)
    return {
        "schema":SCHEMA,
        "source_revision":native["source_revision"],
        "original_source_manifest_sha256":_hash_file(
            native_root/"native_source_manifest.json"),
        "original_physical_manifest_sha256":_hash_file(
            physical_root/"physical_study_manifest.json"),
        "prespecified_cases":12,"prespecified_pairs":6,
        "qualified_original_hls_cases":hls_cases,
        "qualified_original_hls_pairs":hls_pairs,
        "strict_clean_drc_e4_cases":strict_cases,
        "strict_clean_drc_e4_pairs":strict_pairs,
        "descriptive_routed_observation_cases":observed_cases,
        "descriptive_routed_observation_pairs":observed_pairs,
        "strict_physical_primary_gate":(
            hls_cases==12 and hls_pairs==6
            and strict_cases==12 and strict_pairs==6),
        "exploratory_observation_complete":(
            hls_cases==12 and hls_pairs==6
            and observed_cases==12 and observed_pairs==6),
        "trained_real_model_out_of_sample_independently_reproduced":False,
        "independently_executed_exact_peer_artifact_parity":{
            name:"NOT_ESTABLISHED" for name in _PEERS},
        "mapping_conditioned_certificate_firstness_established":False,
        "release_ip_rights_verified":False,
        "paper_submission_ready":False,
        "publication_claims_allowed":[
            "Original E3 results only for individually qualified HLS cases",
            "Strict E4 pairs only where both original cases have zero DRC violations",
            "Warning-only routed pairs only as explicitly labelled exploratory observations",
        ],
        "unmet_scientific_gates":[
            "Independently executed matching published-compiler artifact parity",
            "Actual per-Phi counterfactual demonstrating distinctive certificate capability beyond prior art",
            "Trained-model independent external-validity verification",
            "Artifact reproducer, institutional IP/release approval and reviewed manuscript",
        ],
        "original_vendor_calls_this_command":0,
    }


def create_report(*,native_study:str|Path,physical_study:str|Path,
                  output:str|Path)->dict:
    """Create-only separate paper gate. Never modify the studied evidence."""
    dest=Path(output)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Source-pinned paper gate output is create-only")
    result=evaluate(native_study=native_study,physical_study=physical_study)
    dest.mkdir(parents=True)
    (dest/"readiness.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-study",required=True)
    p.add_argument("--physical-study",required=True)
    p.add_argument("--output")
    a=p.parse_args(argv)
    result=(create_report(native_study=a.native_study,
                          physical_study=a.physical_study,output=a.output)
            if a.output else evaluate(
                native_study=a.native_study,physical_study=a.physical_study))
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
