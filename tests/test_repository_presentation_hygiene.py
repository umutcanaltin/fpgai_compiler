from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_tree_uses_capability_oriented_names() -> None:
    forbidden = []
    for path in [ROOT / "docs", ROOT / "fpgai", ROOT / "scripts", ROOT / "configs"]:
        if not path.exists():
            continue
        for item in path.rglob("*"):
            if ("spr" + "int") in item.name.lower():
                forbidden.append(str(item.relative_to(ROOT)))
    assert forbidden == []
    assert not (ROOT / "README_PATCH.md").exists()
    assert not (ROOT / "contributing.md").exists()
    assert (ROOT / "CONTRIBUTING.md").is_file()
    assert (ROOT / "docs" / "IMPLEMENTATION_STATUS.md").is_file()
    assert (ROOT / "docs" / "DEVELOPMENT_ROADMAP.md").is_file()


def test_gitignore_excludes_local_generated_artifacts() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for required in ("__pycache__/", "*.py[cod]", "build/", "vitis_hls*.log"):
        assert required in text


def test_public_docs_do_not_expose_internal_iteration_labels() -> None:
    offenders = []
    for path in (ROOT / "docs").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".json"}:
            content = path.read_text(encoding="utf-8", errors="replace").lower()
            if ("spr" + "int") in content:
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_canonical_package_ownership_has_no_parallel_legacy_packages() -> None:
    forbidden = [
        ROOT / "fpgai" / "reports",
        ROOT / "fpgai" / "benchmarking",
        ROOT / "fpgai" / "compiler",
    ]
    assert [str(path.relative_to(ROOT)) for path in forbidden if path.exists()] == []
    for required in ("reporting", "benchmark", "capabilities", "engine"):
        assert (ROOT / "fpgai" / required).is_dir()


def test_public_repository_avoids_internal_or_publication_terminology() -> None:
    forbidden_terms = ("pa" + "per", "ar" + "xiv", "spr" + "int")
    offenders = []
    roots = [ROOT / "fpgai", ROOT / "tests", ROOT / "docs", ROOT / "configs", ROOT / "examples", ROOT / "scripts"]
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = str(path.relative_to(ROOT)).lower()
            if any(term in rel for term in forbidden_terms):
                offenders.append(rel)
                continue
            if path.suffix.lower() in {".py", ".md", ".yml", ".yaml", ".json", ".txt"}:
                content = path.read_text(encoding="utf-8", errors="replace").lower()
                if any(term in content for term in forbidden_terms):
                    offenders.append(rel)
    assert offenders == []
