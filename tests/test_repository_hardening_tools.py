from pathlib import Path

from fpgai.devtools.clean_generated_artifacts import (
    find_generated_paths,
    remove_generated_paths,
)
from fpgai.devtools.repository_audit import audit_repository, write_reports


def test_repository_audit_reports_generated_roots_and_oversized_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "fpgai"
    source_dir.mkdir()
    (source_dir / "large_module.py").write_text("value = 1\n" * 1201, encoding="utf-8")

    generated_dir = tmp_path / "paper_results"
    generated_dir.mkdir()
    (generated_dir / "result.json").write_text("{}", encoding="utf-8")

    report = audit_repository(tmp_path)

    assert report.status == "failed"
    assert report.generated_roots_present == ["paper_results"]
    assert any(
        finding.category == "oversized_source_module"
        and finding.path == "fpgai/large_module.py"
        and finding.severity == "blocking"
        for finding in report.findings
    )


def test_repository_audit_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    (tmp_path / "fpgai").mkdir()
    (tmp_path / "fpgai" / "small_module.py").write_text("value = 1\n", encoding="utf-8")

    report = audit_repository(tmp_path)
    json_path, markdown_path = write_reports(report, tmp_path / "audit")

    assert json_path.exists()
    assert markdown_path.exists()
    assert '"status": "passed"' in json_path.read_text(encoding="utf-8")
    assert "Status: `passed`" in markdown_path.read_text(encoding="utf-8")


def test_generated_artifact_cleanup_is_explicit_and_limited(tmp_path: Path) -> None:
    source_file = tmp_path / "fpgai" / "compiler.py"
    source_file.parent.mkdir()
    source_file.write_text("value = 1\n", encoding="utf-8")

    generated_file = tmp_path / "build" / "result.json"
    generated_file.parent.mkdir()
    generated_file.write_text("{}", encoding="utf-8")

    cache_file = tmp_path / "tests" / "__pycache__" / "cached.pyc"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"cache")

    paths = find_generated_paths(tmp_path)
    relative_paths = {path.relative_to(tmp_path).as_posix() for path in paths}

    assert "build" in relative_paths
    assert "tests/__pycache__" in relative_paths
    assert "fpgai" not in relative_paths

    remove_generated_paths(paths)

    assert source_file.exists()
    assert not generated_file.exists()
    assert not cache_file.exists()


def test_cleanup_does_not_enter_virtual_environment(tmp_path: Path) -> None:
    repository_cache = tmp_path / "fpgai" / "__pycache__"
    repository_cache.mkdir(parents=True)

    virtualenv_cache = tmp_path / ".venv" / "lib" / "package" / "__pycache__"
    virtualenv_cache.mkdir(parents=True)

    paths = find_generated_paths(tmp_path)
    relative_paths = {path.relative_to(tmp_path).as_posix() for path in paths}

    assert "fpgai/__pycache__" in relative_paths
    assert not any(path.startswith(".venv/") for path in relative_paths)


def test_cleanup_preserves_maintained_build_examples(tmp_path: Path) -> None:
    example_file = tmp_path / "examples" / "build" / "cpp_only.yml"
    example_file.parent.mkdir(parents=True)
    example_file.write_text("project:\n  name: cpp_only\n", encoding="utf-8")

    paths = find_generated_paths(tmp_path)
    relative_paths = {path.relative_to(tmp_path).as_posix() for path in paths}

    assert "examples/build" not in relative_paths
    assert example_file.exists()
