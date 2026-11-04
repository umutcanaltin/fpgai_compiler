"""Execute a fresh bounded E0/E1 Model IR experiment suite, never Vitis.

Dry-run is default. --execute runs only Python CPU/source studies, each
with a fixed seed and a private, fresh artifact path. No original trained
weights, vendor tools, deployment or manuscript receipts are overwritten.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

SCHEMA="fpgai-research/cpu-semantic-campaign/v1"
ROOT=Path(__file__).resolve().parents[3]


def commands(dest):
    root=Path(dest)
    return [
        ("semantic_property",[
            "experiments/semantic_property_campaign.py",
            "--seed","20261022","--legal","100","--illegal","25",
            "--out",str(root/"semantic_property.json")],
         root/"semantic_property.json"),
        ("model_ir",["experiments/run_model_ir_study.py",
                     "--output",str(root/"model_ir")],root/"model_ir"),
        ("refinement",["experiments/run_refinement_ablation.py",
                       "--seed","2026","--output",str(root/"refinement")],
         root/"refinement"),
        ("materialization",["experiments/run_materialization_study.py",
                            "--seed","2026","--cases","96",
                            "--output",str(root/"materialization")],
         root/"materialization"),
        ("scoped",["experiments/run_scoped_expressiveness.py",
                   "--seed","2026","--max-operators","4",
                   "--output",str(root/"scoped")],root/"scoped"),
        ("semantic_stress",["experiments/run_semantic_stress.py",
                            "--seed","73","--cases","32",
                            "--output",str(root/"stress")],root/"stress"),
        ("branched_conv",["experiments/run_branch_ir_study.py",
                          "--output",str(root/"branched_conv")],
         root/"branched_conv"),
        ("intermediate_tensors",["experiments/run_intermediate_tensor_study.py",
                                 "--seed","2026","--samples","3",
                                 "--output",str(root/"intermediate")],
         root/"intermediate"),
        ("expanded_semantics",["experiments/run_expanded_semantics.py",
                               "--seed","2026","--fixtures","8",
                               "--output",str(root/"expanded_semantics")],
         root/"expanded_semantics"),
    ]


def validate_steps(steps,*,repository=ROOT):
    ids=[]
    for label,args,output in steps:
        if (label in ids or not args or not args[0].startswith("experiments/")
                or not str(args[0]).endswith(".py")
                or any(a in {"--execute","--vendor","--vivado"} for a in args)):
            raise ValueError("Unsupported CPU experiment step")
        source=Path(repository)/args[0]
        if source.is_symlink() or not source.is_file():
            raise ValueError("Missing CPU experiment entrypoint")
        if not Path(output).name:
            raise ValueError("Invalid CPU output")
        ids.append(label)
    if len(steps)<9:
        raise ValueError("Semantic paper study coverage incomplete")
    return steps


def run_campaign(output,*,execute=False,python=sys.executable,repository=ROOT):
    path=Path(output)
    if path.exists() or path.is_symlink() or str(path) in ("",".","/"):
        raise FileExistsError("CPU experiment destination must be new")
    steps=validate_steps(commands(path),repository=repository)
    planned=[{"experiment":name,"argv":[python,*args],
              "artifact":str(out)} for name,args,out in steps]
    if not execute:
        return {"schema":SCHEMA,"executed":False,"studies_planned":len(steps),
                "studies_completed":0,"steps":planned,
                "vendor_tools_executed":False,"physical_results_claimed":False}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.mkdir()
    receipts=[]
    for label,args,artifact in steps:
        print("CPU source-only experiment:",label,flush=True)
        try:
            proc=subprocess.run([python,*args],cwd=repository,
                                capture_output=True,text=True,
                                timeout=300,check=False)
            output_text=(proc.stdout+"\n"+proc.stderr)[-200000:]
            code=proc.returncode
        except (OSError,subprocess.TimeoutExpired) as exc:
            output_text="Execution interrupted: "+type(exc).__name__
            code=-1
        log=path/(label+".log")
        log.write_text(output_text,encoding="utf-8")
        receipt={"experiment":label,"returncode":code,
                 "artifact_present":artifact.exists(),
                 "log_sha256":sha256(output_text.encode()).hexdigest(),
                 "evidence":"E0/E1-CPU-SOURCE-ONLY",
                 "vendor_tools_executed":False}
        receipts.append(receipt)
        if code!=0 or not artifact.exists():
            (path/"incomplete_campaign.json").write_text(
                json.dumps({"schema":SCHEMA,"complete":False,
                            "receipts":receipts},indent=2)+"\n")
            raise RuntimeError(
                "CPU study failed; private logs preserved, no completion claim: "+label)
    report={"schema":SCHEMA,"executed":True,"complete":True,
            "studies_completed":len(receipts),"receipts":receipts,
            "vendor_tools_executed":False,
            "physical_results_claimed":False,
            "full_fixed_point_equivalence_claimed":False,
            "timestamp_utc":datetime.now(timezone.utc).isoformat()}
    (path/"campaign.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--execute",action="store_true",
                   help="Opt-in CPU Python experiments only; no native tools")
    opts=p.parse_args(argv)
    result=run_campaign(opts.output,execute=opts.execute)
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
