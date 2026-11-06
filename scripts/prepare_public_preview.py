"""Create a non-publishable, explicit-allowlist release preview.

This is not an open-source release mechanism or a security clearance.
It NEVER changes GitHub visibility, modifies source evidence, copies Git
history, chooses a license or publishes to PyPI/GitHub.

Default profile is a small compiler-only preview. A full research preview
includes RL code/tests but deliberately excludes internal research
journals, native private artifacts and vendor workflows.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

SCHEMA="fpgai-research/clean-room-preview/v1"

PUBLIC_FILES=frozenset({
    ".gitignore", "pyproject.toml", "MANIFEST.in", "Makefile",
    "README.md","CONTRIBUTING.md","CITATION.md","NOTICE.md","SECURITY.md",
    "CHANGELOG.md","SUPPORT.md","CODE_OF_CONDUCT.md",
    "examples/README.md","examples/NETWORK_GALLERY.md",
    "examples/make_toy_onnx.py","examples/make_neural_network_onnx.py",
    "examples/compile_neural_network.py",
    "examples/make_mlp_onnx.py",
    "examples/make_sample_input.py",
    "examples/architecture.json","examples/architecture_scopes.json",
    "docs/README.md","docs/USING_FPGAI.md","docs/ARCHITECTURE.md",
    "docs/NEUROSCIENTIST_QUICKSTART.md",
    "docs/RESEARCH_PUBLICATION_STATUS.md",
    "docs/SUPPORTED_FEATURES.md","docs/SCIENTIFIC_REPRODUCIBILITY.md",
    "docs/COMPILER_OPERATOR_CONTRIBUTION.md",
    "docs/FAQ.md","docs/RELEASE_PREVIEW.md","docs/ROADMAP.md",
    "docs/LICENSE_DECISION.md","docs/OPEN_SOURCE_RELEASE_CHECKLIST.md",
    "docs/ACADEMIC_RESEARCH_LICENSE_PROPOSAL.md",
    "docs/WHO_CAN_USE_FPGAI.md",
    "docs/AMD_TOOLCHAIN_SETUP.md","docs/FPGA_DEPLOYMENT.md",
    "scripts/check_open_source_readiness.py",
    "scripts/prepare_public_preview.py","scripts/check_public_docs.py",
    "scripts/setup_amd_environment.sh",
    "scripts/check_amd_environment.py",
    "scripts/check_fpga_deployment.py",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
})
PUBLIC_TESTS=frozenset({
    "tests/test_architecture_interfaces.py",
    "tests/test_cst_pipeline.py",
    "tests/test_compositional_model_ir.py",
    "tests/test_public_release_preview.py","tests/test_public_docs_links.py",
    "tests/test_public_neural_network_examples.py",
    "tests/test_public_distribution_surface.py",
    "tests/test_amd_environment_and_deployment_docs.py",
    "tests/test_fpga_deployment_inventory.py",
})
APPROVED_PREFIXES={
    "compiler":("src/fpgai_research/",),
    "full-research":(
        "src/fpgai_research/","rl_environment/","tests/",
        "experiments/compiler_ir/",
    ),
}
REJECTED_COMPONENTS={
    ".git","__pycache__",".venv",".venv-compiler-ir","build","dist",
    "weights","datasets","data","results","maintainers",
}
REJECTED_SUFFIXES=(
    ".pem",".key",".p12",".pfx",".pt",".pth",".onnx",".npy",".npz",
    ".h5",".hdf5",".bit",".hwh",".xpr",".log",".jou",".wdb",
)
MAX_BYTES=2_000_000


def allowed(path:str,*,profile:str)->bool:
    """Pure strict allowlist predicate, independently unit-testable."""
    if profile not in APPROVED_PREFIXES:
        raise ValueError("Unknown release preview profile")
    p=PurePosixPath(path)
    if (p.is_absolute() or not path or ".." in p.parts
            or p.as_posix()!=path or set(p.parts)&REJECTED_COMPONENTS
            or any(part.startswith(".env") for part in p.parts)
            or path.lower().endswith(REJECTED_SUFFIXES)):
        return False
    if path in PUBLIC_FILES or path in PUBLIC_TESTS:
        return True
    if profile=="compiler":
        return path.startswith("src/fpgai_research/") and path.endswith(".py")
    if path.startswith("src/fpgai_research/"):
        return path.endswith(".py")
    if path.startswith("rl_environment/"):
        # Include frozen Python implementation but not historical sprint,
        # raw result and manuscript files inside the RL research tree.
        return path.endswith(".py")
    if path.startswith("tests/"):
        return path.endswith(".py")
    if path.startswith("experiments/compiler_ir/"):
        return path.endswith(".py") or path in {
            "experiments/compiler_ir/series.json",
            "experiments/compiler_ir/run_all.sh",
        }
    return False


def tracked(root:Path)->tuple[str,...]:
    result=subprocess.run(
        ["git","ls-files","-z"],cwd=root,check=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,
    )
    return tuple(sorted(
        x.decode("utf-8") for x in result.stdout.split(b"\0") if x
    ))


def plan(root:Path,profile:str)->dict:
    root=root.resolve()
    candidates=tracked(root)
    selected=[p for p in candidates if allowed(p,profile=profile)]
    if not selected or "README.md" not in selected or "pyproject.toml" not in selected:
        raise ValueError("Missing tracked public README/project metadata")
    for rel in selected:
        path=root/rel
        if (not path.is_file() or path.is_symlink()
                or any(x.is_symlink() for x in path.parents if x!=root)
                or path.stat().st_size>MAX_BYTES):
            raise ValueError("Unsafe or missing tracked public file: "+rel)
    missing=sorted(PUBLIC_FILES-set(selected))
    if missing:
        raise ValueError("Missing required public release-preview files: "+repr(missing))
    counts=dict(Counter(p.split("/",1)[0] for p in selected))
    return {
        "schema":SCHEMA,"profile":profile,
        "source_checkout":"tracked-files-only",
        "tracked_file_count":len(candidates),
        "selected_file_count":len(selected),
        "excluded_file_count":len(candidates)-len(selected),
        "selected_by_top_level":counts,
        "internal_paper_docs_included":any(p.startswith("paper/") for p in selected),
        "internal_maintainer_docs_included":any("maintainers/" in p for p in selected),
        "native_tool_workflow_included":any(p.startswith(".github/workflows/") for p in selected),
        "vendor_binaries_or_weights_included":any(
            p.lower().endswith(REJECTED_SUFFIXES) for p in selected),
        "redistribution_license_approved":False,
        "history_security_reviewed":False,
        "clean_room_tested":False,
        "ready_to_publish":False,
        "selected_paths":selected,
        "disclaimer":[
            "A preview is NOT permission to make the repository public",
            "Full Git history, licensing, IP, private assets and dependencies require independent owner review",
            "No original study folders, source/native evidence or Git objects were copied",
            "Full clean-room install, tests and security scanning have NOT been executed by this preview",
        ],
    }


def stage(root:Path,dest:Path,report:dict)->dict:
    """Create a new preview; never overwrite anything, including a symlink."""
    root=root.resolve()
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("Release preview is create-only")
    if report.get("schema")!=SCHEMA or report.get("ready_to_publish") is not False:
        raise ValueError("Invalid preview plan or mistaken publication approval")
    # Inside this checkout, write only to ignored build/; never create
    # a source/docs/tests folder or mutate existing paths during preview.
    destination=dest.resolve()
    if (destination==root or
            (root in destination.parents
             and destination!=root/"build"
             and root/"build" not in destination.parents)):
        raise ValueError("Preview output inside checkout must be under build/")
    if any(part.is_symlink() for part in dest.parents if part!=root):
        raise ValueError("Preview output has a linked parent directory")
    dest.mkdir(parents=True,exist_ok=False)
    for rel in report["selected_paths"]:
        source=root/rel
        target=dest/rel
        if source.is_symlink() or not source.is_file() or source.stat().st_size>MAX_BYTES:
            raise ValueError("Source file changed after preview planning: "+rel)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target,follow_symlinks=False)
    manifest={k:v for k,v in report.items() if k!="selected_paths"}
    manifest["selected_paths"]=report["selected_paths"]
    (dest/"PREVIEW_NOT_FOR_PUBLICATION.json").write_text(
        json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return {"preview":str(dest),"selected_file_count":len(report["selected_paths"]),
            "ready_to_publish":False}


def main(argv:list[str]|None=None)->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=sorted(APPROVED_PREFIXES),default="compiler")
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan",action="store_true",help="Read-only selected-file inventory")
    action.add_argument("--output",help="NEW create-only preview directory")
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parents[1]
    report=plan(root,args.profile)
    if args.plan:
        out={k:v for k,v in report.items() if k!="selected_paths"}
        print(json.dumps(out,sort_keys=True,indent=2))
        return
    print(json.dumps(stage(root,Path(args.output),report),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
