from pathlib import Path

SAFE_PATTERNS = [
    ".DS_Store",
    ".pytest_cache",
    "__pycache__",
]

SAFE_SUFFIXES = [
    ".pyc",
    ".pyo",
    ".bak",
]

SAFE_EXACT = [
    "repo_audit",
]

SAFE_DIR_PREFIXES = [
    "build",
    "experiments/p5_sweep_smoke",
    "experiments/p5_sweep_real_smoke",
    "experiments/p6_precision_dryrun",
    "experiments/p6_precision_real",
]

DELETE_TRACKED = [
    "scripts/patch_sprint13b_compiler.py",
    "scripts/patch_sprint13b_compiler_runtime_preload.py",
    "scripts/patch_sprint13b_compiler_runtime_preload_v6.py",
    "scripts/patch_sprint13b_compiler_v2.py",
    "scripts/patch_sprint13b_compiler_v3.py",
    "scripts/patch_sprint13b_compiler_v4.py",
    "scripts/patch_sprint13e_top_native_accum.py",
    "scripts/patch_sprint13f_top_loss_eval.py",
    "scripts/patch_sprint15c_pipeline_policy.py",
    "scripts/create_sprint9_precision_effect_sweep.py",
    "scripts/validate_sprint4_yaml_pipeline.py",
]

KEEP_UNTRACKED_HINTS = [
    "configs/examples",
    "configs/paper",
    "docs/",
    "scripts/MANIFEST.md",
    "scripts/README.md",
    "tests/test_cli_",
    "fpgai/backends/vivado",
]

def is_safe_generated(path: Path) -> bool:
    s = str(path)
    name = path.name

    if ".git" in path.parts or ".venv" in path.parts:
        return False

    if name in SAFE_PATTERNS:
        return True

    if path.suffix in SAFE_SUFFIXES:
        return True

    if s in SAFE_EXACT:
        return True

    for prefix in SAFE_DIR_PREFIXES:
        if s == prefix or s.startswith(prefix + "/"):
            return True

    return False

def main():
    root = Path(".")
    candidates = []

    for p in root.rglob("*"):
        if is_safe_generated(p):
            candidates.append(p)

    print("## SAFE GENERATED DELETE CANDIDATES")
    for p in sorted(candidates, key=str):
        print(p)

    print("\n## TRACKED SPRINT/OBSOLETE FILES THAT SHOULD BE DELETED IF PRESENT")
    for item in DELETE_TRACKED:
        p = Path(item)
        if p.exists():
            print(p)

    print("\n## IMPORTANT UNTRACKED FILES TO KEEP/COMMIT, NOT DELETE")
    for p in sorted(root.rglob("*"), key=str):
        s = str(p)
        if any(s.startswith(prefix) for prefix in KEEP_UNTRACKED_HINTS):
            if p.is_file():
                print(p)

if __name__ == "__main__":
    main()
