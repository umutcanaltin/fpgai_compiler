"""Publication experiment registry: frozen evidence and planned work, never execution."""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

SCHEMA="fpgai-research/publication-experiment-register/v1"
STATES={"verified","recorded","negative_observed","manual_pending","planned","blocked"}
STAGES={"E0","E1","E2","E3","E4","E0/E1","E2/E3","E3/E4","E1/E2/E3/E4","publication"}
TRACKS=("bounded","expanded")
REGISTRY=Path(__file__).resolve().parents[3]/"paper"/"EXPERIMENT_REGISTER_2026-10-27.json"


def validate_registry(obj, *, root=None):
    if not isinstance(obj,dict) or obj.get("schema")!=SCHEMA:
        raise ValueError("Invalid paper experiment registry schema")
    entries=obj.get("experiments")
    if not isinstance(entries,list) or len(entries)<20:
        raise ValueError("Incomplete experiment inventory")
    lookup={}
    for e in entries:
        if not isinstance(e,dict):
            raise ValueError("Non-object experiment")
        ident=e.get("id")
        if not isinstance(ident,str) or not ident or ident in lookup:
            raise ValueError("Empty or duplicate experiment identifier")
        if (e.get("track") not in TRACKS or e.get("stage") not in STAGES
                or e.get("state") not in STATES
                or not isinstance(e.get("title"),str) or not e["title"]
                or not isinstance(e.get("note"),str) or not e["note"]):
            raise ValueError("Unrecognized experiment fields")
        requires=e.get("requires")
        if (not isinstance(requires,list)
                or any(not isinstance(x,str) or x==ident for x in requires)
                or len(set(requires))!=len(requires)):
            raise ValueError("Invalid experiment dependencies")
        command=e.get("command")
        if command is not None:
            if (e["state"] not in {"verified","recorded","planned"}
                    or not isinstance(command,str) or "\n" in command
                    or any(x in command for x in ("&&","||",";","$(","rm -rf"))
                    or not (command.startswith("python experiments/")
                            or command.startswith("bash experiments/")
                            or command=="bash paper/run_final_verification.sh")):
                raise ValueError("Disallowed experiment command")
            if root is not None:
                script=Path(root)/command.split()[1]
                if script.is_symlink() or not script.is_file():
                    raise ValueError("Missing experiment CLI")
        artifacts=e.get("artifacts")
        if (not isinstance(artifacts,list)
                or any(not isinstance(a,str) or not a.startswith("build/")
                       or ".." in Path(a).parts for a in artifacts)):
            raise ValueError("Invalid private artifact reference")
        lookup[ident]=e
    for ident,e in lookup.items():
        if any(dep not in lookup for dep in e["requires"]):
            raise ValueError("Unknown experiment prerequisite")
        if e["track"]=="bounded" and any(
                lookup[d]["track"]=="expanded" for d in e["requires"]):
            raise ValueError("Expanded experiment blocks bounded paper")
    visited=set()
    active=set()
    def dfs(ident):
        if ident in active:
            raise ValueError("Cyclic experiment dependencies")
        if ident in visited:
            return
        active.add(ident)
        for dep in lookup[ident]["requires"]:
            dfs(dep)
        active.remove(ident)
        visited.add(ident)
    for ident in lookup:
        dfs(ident)
    if (not any(e["track"]=="bounded" and e["stage"]=="E4"
                and e["state"]=="verified" for e in entries)
            or not any(e["track"]=="expanded" and e["stage"]=="E3/E4"
                       for e in entries)):
        raise ValueError("Missing physical experiment domains")
    return obj


def load_registry(path=REGISTRY, *, root=None):
    source=Path(path)
    if source.is_symlink() or not source.is_file() or source.stat().st_size>1024*1024:
        raise ValueError("Missing safe experiment registry")
    return validate_registry(json.loads(source.read_text(encoding="utf-8")),
                             root=root)


def summarize_registry(registry,track="all"):
    if track not in ("all",*TRACKS):
        raise ValueError("Unknown campaign track")
    items=[e for e in registry["experiments"]
           if track=="all" or e["track"]==track]
    return {
        "schema":"fpgai-research/publication-experiment-plan/v1",
        "track":track,"total_experiments":len(items),
        "evidence_states":dict(sorted(Counter(e["state"] for e in items).items())),
        "stages":dict(sorted(Counter(e["stage"] for e in items).items())),
        "expanded_native_e2_e3_e4_qualified":False,
        "native_vendor_tools_executed":False,
    }


def render_plan(registry,track="all"):
    selected=[e for e in registry["experiments"]
              if track=="all" or e["track"]==track]
    lines=["# FPGAI paper: full experiment register","",
           "This register separates independently attested evidence from future "
           "runs. Nothing is executed by this planner.","",
           "| ID | Track | Stage | State | Depends on |",
           "|---|---|---|---|---|"]
    for e in selected:
        lines.append(f'| {e["id"]} | {e["track"]} | {e["stage"]} | '
                     f'{e["state"]} | {", ".join(e["requires"]) or "—"} |')
    lines.extend(["","## Available source-only or verification commands",""])
    for e in selected:
        if e["command"]:
            lines.extend([f'### {e["id"]}: {e["title"]}',"~~~bash",
                          e["command"],"~~~",""])
    lines.extend(["## Missing / blocked evidence",""])
    for e in selected:
        if e["state"] in ("planned","blocked","manual_pending"):
            lines.append(f'- **{e["id"]}** {e["title"]}: {e["note"]}')
    lines.extend(["","CSim output alone does not certify numerical E2; "
                  "synthesis/route require real native reports; "
                  "all model/weight files remain private.",""])
    return "\n".join(lines)


def export_plan(registry,output,track="all",*,registry_bytes=None):
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Experiment plan destination already exists")
    result=summarize_registry(registry,track)
    if registry_bytes is not None:
        result["registry_sha256"]=sha256(registry_bytes).hexdigest()
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".fpgai-campaign-",dir=target.parent) as tmp:
        folder=Path(tmp)/"plan"
        folder.mkdir()
        (folder/"experiment_plan.md").write_text(render_plan(registry,track),
                                                 encoding="utf-8")
        (folder/"experiment_plan.json").write_text(
            json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Experiment destination appeared")
        folder.rename(target)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track",choices=("all",*TRACKS),default="all")
    p.add_argument("--output")
    p.add_argument("--registry",type=Path,default=REGISTRY)
    args=p.parse_args(argv)
    registry=load_registry(args.registry,root=Path(__file__).resolve().parents[3])
    if args.output:
        result=export_plan(registry,args.output,args.track,
                           registry_bytes=args.registry.read_bytes())
    else:
        print(render_plan(registry,args.track))
        result=summarize_registry(registry,args.track)
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
