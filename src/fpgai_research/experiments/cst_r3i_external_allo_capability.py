"""R3i external-artifact test: pinned Allo two-reader MatMul source probe.

This is an OPTIONAL separate toolchain capability experiment, not a substitute
for a proven same-IR semantics comparison. Requires a separately prepared
editable Allo checkout exactly at the pinned public source revision, with
PAST and Allo dependencies installed in a *separate* environment. No Python
code in this module imports or installs Allo unless explicitly executing.

The candidate tests an observable result: a shared MatMul producer with
two MatMul consumers, where the invalid treatment overwrites the producer
after consumer 1 but before consumer 2. Allo may detect different outputs
through its own schedule-equivalence verifier. Such detection does NOT
prove an explicit last-reader source-event checker is present or absent.

Do not run from the production compiler environment unless its external
dependencies are explicitly isolated. Never rerun/overwrite original
C1/C2 source evidence. No Vitis/Vivado or trained RL.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

SCHEMA="fpgai-research/R3i-pinned-Allo-external-capability/v1"
PINNED_ALLO="abc270c7a1919c780df2b7d7e99de45e81c175b9"


def protocol()->dict:
    return {
        "schema":SCHEMA,
        "external_project":"Allo PLDI 2024",
        "external_repository":"https://github.com/cornell-zhang/allo",
        "external_source_revision":PINNED_ALLO,
        "capability_questions":[
            "same functional 2-reader MatMul fanout across normal/reordered source schedules",
            "detect changed result if shared producer H is overwritten before last reader",
            "do not equate output-equivalence detection with source event last-reader proof",
        ],
        "external_artifact_executed":False,
        "result_status":"NOT_RUN",
        "hardware_source_comparison":"NOT_SAME_COMPILER_IR",
        "original_vitis_calls":0,"original_vivado_calls":0,
        "trained_rl_agents":0,
    }


def _git_revision(checkout:Path)->str:
    if checkout.is_symlink() or not checkout.is_dir():
        raise ValueError("R3i external Allo checkout missing or linked")
    rev=subprocess.run(
        ["git","-C",str(checkout),"rev-parse","HEAD"],
        check=True,text=True,capture_output=True,timeout=10).stdout.strip()
    if rev!=PINNED_ALLO:
        raise ValueError("R3i Allo revision mismatch; pinned source required")
    unclean=subprocess.run(
        ["git","-C",str(checkout),"status","--porcelain","--untracked-files=no"],
        check=True,text=True,capture_output=True,timeout=10).stdout
    if unclean.strip():
        raise ValueError("R3i Allo tracked source modified; strict exact checkout required")
    for name in ("allo/verify.py","tests/test_verify.py","tests/test_schedule_memory.py"):
        if not (checkout/name).is_file():
            raise ValueError("R3i pinned Allo source verification files missing")
    return rev


def reference_inputs()->tuple[np.ndarray,...]:
    x=np.array([[1.0,2.0],[-1.0,3.0]],dtype=np.float32)
    wp=np.array([[2.0,1.0],[0.0,1.0]],dtype=np.float32)
    wa=np.array([[1.0,0.0],[2.0,1.0]],dtype=np.float32)
    wb=np.array([[1.0,-1.0],[0.0,2.0]],dtype=np.float32)
    return x,wp,wa,wb


def reference_outputs()->dict:
    x,wp,wa,wb=reference_inputs()
    h=x@wp
    good=(h@wa)+(h@wb)
    overwritten=h@wa
    if np.allclose(good,overwritten,rtol=0,atol=1e-6):
        raise AssertionError("R3i negative source value mutation has no impact")
    return {
        "reference_good":good.tolist(),
        "reference_early_overwrite":overwritten.tolist(),
        "same_output":False,
        "input_sha256":sha256(b"".join(x.tobytes() for x in reference_inputs())).hexdigest(),
        "dtype":"float32",
        "shape":[2,2],
    }


def execute_allo(checkout:Path)->dict:
    revision=_git_revision(checkout)
    allo=importlib.import_module("allo")
    root=Path(allo.__file__).resolve()
    if not root.is_relative_to(checkout.resolve()):
        raise ValueError("R3i imported Allo is not from the exact checked-out artifact")
    from allo.ir.types import float32
    # Allo source introspection may resolve DSL annotations against globals.
    # These are only installed after the explicit isolated opt-in check.
    globals()["allo"] = allo
    globals()["float32"] = float32

    # Define all three tiny independent source programs. This is Allo
    # source, not a translation of our typed C1 event language.
    def safe(X:float32[2,2], WP:float32[2,2],
             WA:float32[2,2], WB:float32[2,2])->float32[2,2]:
        H:float32[2,2]=0
        for i,j,k in allo.grid(2,2,2):
            H[i,j]+=X[i,k]*WP[k,j]
        A:float32[2,2]=0
        for i,j,k in allo.grid(2,2,2):
            A[i,j]+=H[i,k]*WA[k,j]
        B:float32[2,2]=0
        for i,j,k in allo.grid(2,2,2):
            B[i,j]+=H[i,k]*WB[k,j]
        O:float32[2,2]=0
        for i,j in allo.grid(2,2):
            O[i,j]=A[i,j]+B[i,j]
        return O

    def reordered(X:float32[2,2], WP:float32[2,2],
                  WA:float32[2,2], WB:float32[2,2])->float32[2,2]:
        H:float32[2,2]=0
        for i,j,k in allo.grid(2,2,2):
            H[i,j]+=X[i,k]*WP[k,j]
        B:float32[2,2]=0
        for i,j,k in allo.grid(2,2,2):
            B[i,j]+=H[i,k]*WB[k,j]
        A:float32[2,2]=0
        for i,j,k in allo.grid(2,2,2):
            A[i,j]+=H[i,k]*WA[k,j]
        O:float32[2,2]=0
        for i,j in allo.grid(2,2):
            O[i,j]=A[i,j]+B[i,j]
        return O

    def overwritten(X:float32[2,2], WP:float32[2,2],
                    WA:float32[2,2], WB:float32[2,2])->float32[2,2]:
        H:float32[2,2]=0
        for i,j,k in allo.grid(2,2,2):
            H[i,j]+=X[i,k]*WP[k,j]
        A:float32[2,2]=0
        for i,j,k in allo.grid(2,2,2):
            A[i,j]+=H[i,k]*WA[k,j]
        # Invalid all-reader lifetime example: B reads changed H.
        for i,j in allo.grid(2,2):
            H[i,j]=0.0
        B:float32[2,2]=0
        for i,j,k in allo.grid(2,2,2):
            B[i,j]+=H[i,k]*WB[k,j]
        O:float32[2,2]=0
        for i,j in allo.grid(2,2):
            O[i,j]=A[i,j]+B[i,j]
        return O

    schedules=[allo.customize(fn) for fn in (safe,reordered,overwritten)]
    comparisons={}
    # The pinned Allo verify.py writes relative tmp/a.c and tmp/b.c.
    # Run in an isolated scratch directory, NEVER in research source/artifacts.
    old_cwd=Path.cwd()
    with tempfile.TemporaryDirectory(prefix="fpgai_allo_c1_probe_") as scratch:
        try:
            os.chdir(scratch)
            for name,i in (("self_equivalence",0),("reordered_equivalence",1),
                           ("early_overwrite_equivalence",2)):
                try:
                    verified=allo.verify(schedules[0],schedules[i])
                    if type(verified) is not bool and not isinstance(verified,np.bool_):
                        raise ValueError("Allo verifier did not return a boolean")
                    verified=bool(verified)
                    comparisons[name]={
                        "status":"EXECUTED_TRUE" if verified else "EXECUTED_FALSE",
                        "meaning":"Allo/PAST whole-output schedule equivalence result",
                    }
                except Exception as exc:
                    comparisons[name]={
                        "status":"ERROR_NOT_A_NEGATIVE_CAPABILITY_RESULT",
                        "error_type":type(exc).__name__,
                        "reason":str(exc)[:500],
                    }
        finally:
            os.chdir(old_cwd)
    return {
        "schema":SCHEMA,
        "external_source_revision":revision,
        "external_allo_module_file":str(root),
        "external_artifact_executed":True,
        "reference":reference_outputs(),
        "original_comparisons":comparisons,
        "source_semantics_e0_lifetime_certificate_available":None,
        "source_semantics_phi_issue_certificate_available":None,
        "trivial_external_feature_absence_claim":False,
        "new_vitis_calls":0, "new_vivado_calls":0,"trained_agents":0,
        "limits":[
            "The Allo programs are hand-specified small float32 MatMul fanout probes",
            "Allo verification compares observable source output semantics, not automatically the C1 event certificate",
            "An ERROR is neither acceptance nor proof the capability is absent",
            "A failed verification may reflect translation or PAST limitations and needs manual inspection",
            "The compiler versions and source languages are different, so timing/area comparisons are excluded",
        ],
    }


def create_only(report:dict,*,output:str|Path)->None:
    root=Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("R3i: external artifact report create-only")
    if report.get("schema")!=SCHEMA:
        raise ValueError("R3i: external report schema invalid")
    root.mkdir(parents=True,exist_ok=False)
    (root/"r3i_pinned_allo_capability.json").write_text(
        json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n",
        encoding="utf-8")


def main(argv:list[str]|None=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",action="store_true")
    parser.add_argument("--allo-checkout")
    parser.add_argument("--output")
    parser.add_argument("--confirm-isolated-external",action="store_true")
    args=parser.parse_args(argv)
    if args.protocol and not any((
            args.allo_checkout,args.output,args.confirm_isolated_external)):
        print(json.dumps(protocol(),indent=2,sort_keys=True))
        return
    if (not args.allo_checkout or not args.output
            or args.confirm_isolated_external is not True):
        parser.error("External execution requires Allo checkout, new output and explicit isolated confirmation")
    target=Path(args.output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("R3i: refuse to overwrite any original external evidence")
    report=execute_allo(Path(args.allo_checkout))
    create_only(report,output=target)
    print(json.dumps({
        "schema":SCHEMA,
        "external_source_revision":report["external_source_revision"],
        "original_comparisons":report["original_comparisons"],
        "new_native_vendor_calls":0,
        "output":args.output,
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
