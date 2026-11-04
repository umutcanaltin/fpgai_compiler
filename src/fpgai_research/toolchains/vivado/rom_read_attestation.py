"""Prospective and independently repeatable witness for readmem data post-run absence.

A completed Vivado process may have read a hash-attested staged ROM then
left that working-directory file absent. Only a NEW source-pinned study
that explicitly opts in may classify this precise scenario as a
vendor-reported successful read, not as post-run byte-for-byte stability.
Old studies and unexpected changes are not retroactively qualified.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re

_SCHEMA="fpgai-research/vivado-readmem-post-absence-witness/v1"
_SUCCESS=re.compile(
    r"""\bINFO:\s*\[Synth 8-3876\]\s*\$readmem data file ['"]([^'"]+)['"] is read successfully\b"""
)
_NAME=re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+\-]*\Z",re.ASCII)
_MAX_LOG=64*1024*1024


def verify_post_run_readmem_absence(*,expected:dict[str,str],
                                    literal_refs:list[str],
                                    before:dict,after:dict,
                                    stdout_file:str|Path,
                                    expected_stdout_sha256:str|None=None)->dict:
    """Return conservative witness; never run tools or modify original files.

    IMPORTANT: a Vivado log line establishes reported read success, not
    independently measured internal memory values. The pre-execution
    staged ROM sha256 and unchanged RTL source ROM remain separately
    mandatory. No witness can be made for an unsupported parameterized
    or indirect $readmem file name.
    """
    result={"schema":_SCHEMA,"qualified_postrun_absence":False,
            "missing_roms":[],"reported_successful_roms":[],
            "stdout_sha256":None,"reason":"NOT_PROVEN"}
    if (not isinstance(expected,dict) or not expected or
            not isinstance(literal_refs,list) or
            any(not isinstance(x,str) or not _NAME.fullmatch(x)
                for x in (*expected,*literal_refs)) or
            not all(isinstance(x,str) and re.fullmatch(r"[a-f0-9]{64}",x)
                    for x in expected.values()) or
            not isinstance(before,dict) or not isinstance(after,dict)):
        result["reason"]="INVALID_EXPECTED_ROM_CONTRACT"
        return result
    if (before.get("matches_original_roms") is not True or
            before.get("observed_sha256")!=expected):
        result["reason"]="PRE_RUN_STAGED_ROM_NOT_ATTESTED"
        return result
    missing=after.get("missing_original_roms")
    result["missing_roms"]=missing if isinstance(missing,list) else []
    if (not isinstance(missing,list) or not missing
            or len(set(missing))!=len(missing)
            or set(missing)-set(expected)
            or set(missing)-set(literal_refs)
            or after.get("modified_original_roms")!=[]
            or after.get("unattributed_rom_files")!=[]
            or after.get("invalid_rom_files")!=[]
            or after.get("observed_sha256")!={
                k:v for k,v in expected.items() if k not in missing
            }):
        result["reason"]="NOT_ONLY_LITERAL_ROM_POST_RUN_ABSENCE"
        return result
    log=Path(stdout_file)
    if log.is_symlink() or not log.is_file() or log.stat().st_size>_MAX_LOG:
        result["reason"]="MISSING_UNSAFE_OR_UNBOUNDED_VENDOR_LOG"
        return result
    contents=log.read_bytes()
    digest=sha256(contents).hexdigest()
    result["stdout_sha256"]=digest
    if (expected_stdout_sha256 is not None and
            digest!=expected_stdout_sha256):
        result["reason"]="LOG_HASH_DIFFERS_FROM_VENDOR_RECEIPT"
        return result
    found=set()
    for match in _SUCCESS.finditer(contents.decode("utf-8",errors="replace")):
        raw=match.group(1)
        name=raw[2:] if raw.startswith("./") else raw
        if _NAME.fullmatch(name) and name in expected:
            found.add(name)
    result["reported_successful_roms"]=sorted(found)
    if not set(missing).issubset(found):
        result["reason"]="ORIGINAL_VENDOR_READMEM_SUCCESS_NOT_FOUND"
        return result
    result["reason"]="SOURCE_ATTESTED_VENDOR_READ_POST_RUN_ABSENCE"
    result["qualified_postrun_absence"]=True
    return result
