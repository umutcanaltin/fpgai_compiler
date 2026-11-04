"""Conservative original Vitis HLS RTL + ROM-data inventory for Vivado OOC.

Only Verilog/SystemVerilog and plain, named local .dat/.mem ROM payloads
are supported. Never invent or regenerate ROM contents. Reject symlinks,
unknown adjacent design dependencies, missing referenced ROMs and path
escapes. Original HLS HDL hashes remain distinct from ROM-data hashes
for compatibility with historical Vitis invocation receipts.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re

RTL_SUFFIXES=(".v",".sv")
ROM_SUFFIXES=(".dat",".mem")
BAD_SUFFIXES=(".vhd",".vhdl",".xci",".xcix",".dcp",".edif",".edf",
              ".tcl",".coe",".hex",".mif",".bin")
MAX_FILES=4096
MAX_ROM_BYTES=64*1024*1024
_LOCAL_NAME=re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+\-]*\Z",re.ASCII)
_READMEM=re.compile(
    r"\$readmem[hb]\s*\(\s*\"([^\"]+)\"",re.I,
)


def _named_regular(path:Path):
    if (not _LOCAL_NAME.fullmatch(path.name)
            or path.is_symlink() or not path.is_file()):
        raise ValueError("Unsafe/symlinked original HLS RTL input: "+path.name)
    return path


def inventory_vitis_rtl(case_directory:str|Path)->dict:
    root=Path(case_directory)/"fpgai_hls_project"/"solution1"/"syn"/"verilog"
    if root.is_symlink() or not root.is_dir():
        return {"rtl_sha256":{},"rom_sha256":{},"literal_readmem_references":[]}
    files=sorted(p for p in root.iterdir() if p.is_file() or p.is_symlink())
    if len(files)>MAX_FILES:
        raise ValueError("Excessive Vitis RTL/ROM source count")
    if any(p.is_symlink() or p.name.startswith(".") or p.is_dir()
           for p in root.iterdir()):
        raise ValueError("Source RTL/ROM directory contains symlink/hidden/unsupported directory")
    unexpected=[p.name for p in files
                if p.suffix.lower() not in RTL_SUFFIXES+ROM_SUFFIXES]
    if unexpected:
        raise ValueError("UNSUPPORTED_HLS_RTL_IP_OR_MEMORY_ASSETS: "+", ".join(unexpected[:8]))
    rtl={p.name:sha256(_named_regular(p).read_bytes()).hexdigest()
         for p in files if p.suffix.lower() in RTL_SUFFIXES}
    rom={}
    for p in files:
        if p.suffix.lower() not in ROM_SUFFIXES:
            continue
        _named_regular(p)
        if p.stat().st_size>MAX_ROM_BYTES or p.stat().st_size==0:
            raise ValueError("Unbounded/empty HLS ROM initialization data: "+p.name)
        rom[p.name]=sha256(p.read_bytes()).hexdigest()
    if not rtl:
        raise ValueError("Missing original Vitis-generated Verilog HDL")
    references=set()
    for p in files:
        if p.suffix.lower() not in RTL_SUFFIXES:
            continue
        contents=p.read_text(errors="replace")
        for value in _READMEM.findall(contents):
            # Vitis HLS 2023.2 emits both "weights.dat" and
            # "./weights.dat". Only a SINGLE explicit current-directory
            # prefix is allowed; never resolve paths or admit subdirectories,
            # absolute paths, Windows separators or traversal.
            name=value[2:] if value.startswith("./") else value
            if (not _LOCAL_NAME.fullmatch(name) or
                    Path(name).suffix.lower() not in ROM_SUFFIXES):
                raise ValueError("UNSAFE_OR_UNSUPPORTED_RTL_READMEM_PATH: "+value)
            # Record the canonical basename for exact manifest auditing and
            # comparison with the original hashed ROM inventory.
            references.add(name)
    missing=references-set(rom)
    if missing:
        raise ValueError("MISSING_ORIGINAL_HLS_ROM_INIT: "+", ".join(sorted(missing)))
    # For ROM filenames passed via Verilog string parameters, literal
    # readmem parsing cannot establish the consumer. We still preserve
    # every source ROM asset and make no inferred semantic claim.
    return {"rtl_sha256":rtl,"rom_sha256":rom,
            "literal_readmem_references":sorted(references)}


def compare_staged_roms(directory:str|Path,expected:dict[str,str])->dict:
    """Read-only, bounded, fail-closed comparison with original source ROMs.

    Extra .dat/.mem produced by Vivado are NOT assumed to be original
    inputs. Distinguish unknown extra files from missing or altered original
    ROMs; neither condition may qualify as original E4 evidence.
    """
    folder=Path(directory)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Unsafe or missing staged ROM directory")
    if not isinstance(expected,dict) or any(
        not isinstance(name,str) or not _LOCAL_NAME.fullmatch(name)
        or Path(name).suffix.lower() not in ROM_SUFFIXES
        or not isinstance(digest,str) or not re.fullmatch(r"[a-f0-9]{64}",digest)
        for name,digest in expected.items()
    ):
        raise ValueError("Invalid original ROM inventory")
    candidates=[p for p in folder.iterdir()
                if p.suffix.lower() in ROM_SUFFIXES]
    if len(candidates)>MAX_FILES:
        raise ValueError("Excessive staged ROM data count")
    observed={}
    invalid=[]
    for p in sorted(candidates):
        if (p.is_symlink() or not p.is_file() or
                not _LOCAL_NAME.fullmatch(p.name) or
                p.stat().st_size==0 or p.stat().st_size>MAX_ROM_BYTES):
            invalid.append(p.name)
            continue
        observed[p.name]=sha256(p.read_bytes()).hexdigest()
    missing=sorted(set(expected)-set(observed))
    extra=sorted(set(observed)-set(expected))
    changed=sorted(name for name in set(expected)&set(observed)
                   if expected[name]!=observed[name])
    return {
        "matches_original_roms":not (missing or changed or invalid or extra),
        "original_inputs_unchanged":not (missing or changed or
            any(name in expected for name in invalid)),
        "missing_original_roms":missing,
        "modified_original_roms":changed,
        "unattributed_rom_files":extra,
        "invalid_rom_files":sorted(invalid),
        "observed_sha256":observed,
    }
