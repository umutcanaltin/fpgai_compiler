"""CPU-only protocol controls: these are not original Vivado results."""
from hashlib import sha256
from pathlib import Path

import pytest

from fpgai_research.toolchains.vivado.rom_read_attestation import (
    verify_post_run_readmem_absence,
)


def fixture(tmp_path):
    name="fpgai_weights_ROM_AUTO_1R.dat"
    digest=sha256(b"1234abcd\n").hexdigest()
    original={name:digest}
    stdout=tmp_path/"original_vivado.stdout.log"
    stdout.write_text(
        "INFO: [Synth 8-3876] $readmem data file './"+
        name+"' is read successfully [original.v:33]\n")
    before={"matches_original_roms":True,"observed_sha256":original}
    after={"matches_original_roms":False,"missing_original_roms":[name],
           "modified_original_roms":[],"unattributed_rom_files":[],
           "invalid_rom_files":[],"observed_sha256":{}}
    return name,original,stdout,before,after


def verify(name,original,stdout,before,after,hash=None):
    return verify_post_run_readmem_absence(
        expected=original,literal_refs=[name],before=before,after=after,
        stdout_file=stdout,expected_stdout_sha256=hash,
    )


def test_vivado_original_readmem_post_absence_witness_is_narrow(tmp_path):
    name,original,stdout,before,after=fixture(tmp_path)
    digest=sha256(stdout.read_bytes()).hexdigest()
    result=verify(name,original,stdout,before,after,hash=digest)
    assert result["qualified_postrun_absence"] is True
    assert result["missing_roms"]==[name]
    assert result["reported_successful_roms"]==[name]
    assert result["stdout_sha256"]==digest
    assert result["reason"]=="SOURCE_ATTESTED_VENDOR_READ_POST_RUN_ABSENCE"


@pytest.mark.parametrize("mutation,reason",[
    ("not_referenced","NOT_ONLY_LITERAL_ROM_POST_RUN_ABSENCE"),
    ("extra","NOT_ONLY_LITERAL_ROM_POST_RUN_ABSENCE"),
    ("changed","NOT_ONLY_LITERAL_ROM_POST_RUN_ABSENCE"),
    ("invalid","NOT_ONLY_LITERAL_ROM_POST_RUN_ABSENCE"),
    ("unverified_before","PRE_RUN_STAGED_ROM_NOT_ATTESTED"),
    ("not_missing","NOT_ONLY_LITERAL_ROM_POST_RUN_ABSENCE"),
])
def test_post_absence_rejects_other_side_effects(tmp_path,mutation,reason):
    name,original,stdout,before,after=fixture(tmp_path)
    refs=[name]
    if mutation=="not_referenced":refs=[]
    if mutation=="extra":after["unattributed_rom_files"]=["untrusted.dat"]
    if mutation=="changed":after["modified_original_roms"]=[name]
    if mutation=="invalid":after["invalid_rom_files"]=[name]
    if mutation=="unverified_before":before["observed_sha256"]={}
    if mutation=="not_missing":after["missing_original_roms"]=[]
    receipt=verify_post_run_readmem_absence(
        expected=original,literal_refs=refs,before=before,after=after,
        stdout_file=stdout)
    assert receipt["qualified_postrun_absence"] is False
    assert receipt["reason"]==reason


def test_original_vendor_read_and_hash_are_both_required(tmp_path):
    name,original,stdout,before,after=fixture(tmp_path)
    expected_digest=sha256(stdout.read_bytes()).hexdigest()
    stdout.write_text("INFO: [Synth 8-3876] $readmem data file './"+
                      name+"' was NOT read successfully\n")
    changed=verify(name,original,stdout,before,after,hash=expected_digest)
    assert changed["qualified_postrun_absence"] is False
    assert changed["reason"]=="LOG_HASH_DIFFERS_FROM_VENDOR_RECEIPT"
    assert verify(name,original,stdout,before,after)[
        "reason"]=="ORIGINAL_VENDOR_READMEM_SUCCESS_NOT_FOUND"


def test_unsafe_relative_path_cannot_count_as_literal_source_read(tmp_path):
    name,original,stdout,before,after=fixture(tmp_path)
    stdout.write_text(
        "INFO: [Synth 8-3876] $readmem data file './../"+
        name+"' is read successfully\n")
    assert verify(name,original,stdout,before,after)[
        "qualified_postrun_absence"] is False


def test_log_symlink_never_serves_as_vendor_attestation(tmp_path):
    name,original,stdout,before,after=fixture(tmp_path)
    link=tmp_path/"bad_stdout.log"
    link.symlink_to(stdout)
    result=verify(name,original,link,before,after)
    assert result["qualified_postrun_absence"] is False
    assert result["reason"]=="MISSING_UNSAFE_OR_UNBOUNDED_VENDOR_LOG"


def test_postrun_witness_requires_hash_at_preexecution(tmp_path):
    name,original,stdout,before,after=fixture(tmp_path)
    before["matches_original_roms"]=False
    assert not verify(name,original,stdout,before,after)[
        "qualified_postrun_absence"]
