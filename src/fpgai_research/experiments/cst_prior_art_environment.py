"""Read-only pinned peer-compiler checkout and reproducibility preflight.

This is NOT peer-semantic parity: verifying a Git revision and a README
cannot demonstrate that its compiler does or does not express a CST
tensor-lifetime certificate. Original peer commands, source/output
hashes, numerical parity and unsupported semantics require independent
execution in the appropriate separate environment.

The documented revisions below are EXACT pointers; several are source
snapshots, not confirmed publication evaluation revisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

SCHEMA="fpgai-research/cst-pinned-peer-environment-check/v1"
PINS={
    "ST-Flow":{
        "repository":"UCLA-VAST/ST-Flow",
        "revision":"b249f322fbfb6e746e7c8a3a09800a2298c79757",
        "pin_type":"PACT26_EVALUATION_TAG",
        "requirements":"Python 3.12; Vitis HLS 2024.2 for native CSim",
    },
    "Allo":{
        "repository":"cornell-zhang/allo",
        "revision":"abc270c7a1919c780df2b7d7e99de45e81c175b9",
        "pin_type":"MAIN_SNAPSHOT_NOT_PLDI_EVALUATION_REVISION",
        "requirements":"Use upstream documented Allo build and examples",
    },
    "HIDA":{
        "repository":"UIUC-ChenLab/ScaleHLS-HIDA",
        "revision":"fd79f6c946593c290d6d285c2a2efcab72fbe9c6",
        "pin_type":"MAIN_SNAPSHOT_NOT_ASPLOS_EVALUATION_REVISION",
        "requirements":"MLIR/Clang/CMake/Ninja; source build",
    },
    "Wavelet":{
        "repository":"plum-umd/wavelet",
        "revision":"9ef3a89aa6b28254820a81e23367c24562eea903",
        "pin_type":"MAIN_SNAPSHOT_NOT_PLDI_ARCHIVE_VERSION",
        "requirements":"Rust/Cargo, Lean/Elan, cvc5; async-dataflow backend",
    },
    "StreamTensor":{
        "repository":None,"revision":None,
        "pin_type":"PAPER_DOCUMENTATION_ONLY_NO_VERIFIED_RUNNABLE_COMMIT",
        "requirements":"No independently source-pinned runnable artifact",
    },
}


def _git(cwd:Path,*args)->str:
    try:
        command=subprocess.run(
            ["git","-C",str(cwd),*args],capture_output=True,text=True,
            check=False,timeout=10)
    except (FileNotFoundError,subprocess.TimeoutExpired) as err:
        raise ValueError("Git checkout inspection unavailable") from err
    if command.returncode!=0:
        raise ValueError("Git checkout inspection failed: "+" ".join(args))
    return command.stdout.strip()


def inspect_checkout(*,peer:str,checkout:str|Path)->dict:
    """No network, no peer compiler command, no file writes."""
    if peer not in PINS:
        raise ValueError("Unsupported peer-compiler comparison label")
    pin=PINS[peer]
    if pin["revision"] is None:
        return {
            "schema":SCHEMA,"peer":peer,"pin_type":pin["pin_type"],
            "source_checkout_status":"NO_VERIFIED_RUNNABLE_REVISION",
            "exact_checkout":False,"peer_compiler_executed":False,
            "numerical_parity_established":False,
            "certificate_parity_established":False,
        }
    path=Path(checkout)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Peer checkout must be a real directory")
    root=_git(path,"rev-parse","--show-toplevel")
    if Path(root).resolve()!=path.resolve():
        raise ValueError("Peer checkout must be its exact repository root")
    head=_git(path,"rev-parse","HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}",head):
        raise ValueError("Unexpected peer revision identifier")
    dirty=_git(path,"status","--porcelain")
    remote=_git(path,"remote","get-url","origin")
    expected=pin["repository"]
    recognized=bool(re.fullmatch(
        r"(?:https://github[.]com/|git@github[.]com:|"
        r"ssh://git@github[.]com/)"+re.escape(expected)+r"(?:[.]git)?/?",
        remote))
    return {
        "schema":SCHEMA,"peer":peer,
        "expected_repository":expected,
        "expected_revision":pin["revision"],
        "observed_revision":head,
        "origin_matches_expected_repository":recognized,
        "clean_checkout":dirty=="",
        "exact_checkout":(
            head==pin["revision"] and recognized and dirty==""),
        "source_checkout_status":(
            "PINNED_CLEAN_SOURCE_CHECKOUT" if
            head==pin["revision"] and recognized and dirty==""
            else "MISMATCHED_OR_DIRTY_SOURCE_CHECKOUT"),
        "pin_type":pin["pin_type"],
        "required_toolchain_context":pin["requirements"],
        "peer_compiler_executed":False,
        "numerical_parity_established":False,
        "certificate_parity_established":False,
        "matched_hardware_qor_established":False,
        "original_vendor_calls_this_command":0,
        "limitation":"Checkout inspection is not peer capability or novelty proof",
    }


def create_report(*,peer:str,checkout:str|Path,output:str|Path)->dict:
    dst=Path(output)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError("Peer preflight evidence is create-only")
    record=inspect_checkout(peer=peer,checkout=checkout)
    dst.mkdir(parents=True)
    (dst/"peer_preflight.json").write_text(
        json.dumps(record,indent=2,sort_keys=True)+"\n")
    return record


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--peer",choices=sorted(PINS),required=True)
    p.add_argument("--checkout",required=True)
    p.add_argument("--output")
    a=p.parse_args(argv)
    result=(create_report(peer=a.peer,checkout=a.checkout,output=a.output)
            if a.output else inspect_checkout(peer=a.peer,checkout=a.checkout))
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
