"""Audit repository structure, naming, generated artifacts, and file sizes.

Run with:

    python -m fpgai.devtools.repository_audit

The command prints a short summary and writes detailed JSON and Markdown reports.
It does not modify the repository.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from fpgai.devtools.path_policy import iter_directories

SOURCE_DIRECTORIES = ("fpgai", "scripts")
TEST_DIRECTORY = "tests"
GENERATED_ROOTS = (
    "build",
    "dev_audits",
    "paper_results",
    "paper_outputs",
    "paper_tables",
    "repo_audit",
    "reports",
)
CACHE_DIRECTORY_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
AMBIGUOUS_FILENAMES = {
    "common.py",
    "helper.py",
    "helpers.py",
    "misc.py",
    "new.py",
    "run_all.py",
    "test_script.py",
    "utils.py",
    "utils2.py",
}
SOURCE_WARNING_LINES = 800
SOURCE_BLOCKING_LINES = 1200
TEST_WARNING_LINES = 1200
TEST_BLOCKING_LINES = 2000


@dataclass(frozen=True)
class FileFinding:
    """One file-level repository audit finding."""

    path: str
    category: str
    message: str
    severity: str
    line_count: int | None = None


@dataclass(frozen=True)
class RepositoryAuditReport:
    """Serializable result produced by the repository audit."""

    repository_root: str
    python_source_files: int
    test_files: int
    markdown_files: int
    generated_roots_present: list[str]
    cache_directories_present: list[str]
    findings: list[FileFinding]

    @property
    def blocking_failure_count(self) -> int:
        return sum(finding.severity == "blocking" for finding in self.findings)

    @property
    def warning_count(self) -> int:
        return sum(finding.severity == "warning" for finding in self.findings)

    @property
    def status(self) -> str:
        if self.blocking_failure_count:
            return "failed"
        if self.warning_count:
            return "warnings"
        return "passed"


def _iter_files(root: Path, suffix: str, directories: Iterable[str]) -> list[Path]:
    files: list[Path] = []

    for directory_name in directories:
        directory = root / directory_name
        if not directory.exists():
            continue
        files.extend(path for path in directory.rglob(f"*{suffix}") if path.is_file())

    return sorted(files)


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as source_file:
        return sum(1 for _ in source_file)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _size_finding(path: Path, root: Path, *, is_test: bool) -> FileFinding | None:
    line_count = _line_count(path)

    if is_test:
        warning_limit = TEST_WARNING_LINES
        blocking_limit = TEST_BLOCKING_LINES
        category = "oversized_test"
    else:
        warning_limit = SOURCE_WARNING_LINES
        blocking_limit = SOURCE_BLOCKING_LINES
        category = "oversized_source_module"

    if line_count > blocking_limit:
        severity = "blocking"
    elif line_count > warning_limit:
        severity = "warning"
    else:
        return None

    return FileFinding(
        path=_relative(path, root),
        category=category,
        severity=severity,
        line_count=line_count,
        message=(
            f"File has {line_count} lines. Split it by clear responsibility before "
            "adding more behavior."
        ),
    )


def _ambiguous_name_finding(path: Path, root: Path) -> FileFinding | None:
    if path.name not in AMBIGUOUS_FILENAMES:
        return None

    return FileFinding(
        path=_relative(path, root),
        category="ambiguous_filename",
        severity="warning",
        message="Filename does not describe the file's responsibility clearly.",
    )


def audit_repository(repository_root: str | Path) -> RepositoryAuditReport:
    """Inspect repository structure without changing files."""

    root = Path(repository_root).resolve()
    source_files = _iter_files(root, ".py", SOURCE_DIRECTORIES)
    test_files = _iter_files(root, ".py", (TEST_DIRECTORY,))
    markdown_files = sorted(path for path in root.rglob("*.md") if path.is_file())

    generated_roots_present = [
        path_name for path_name in GENERATED_ROOTS if (root / path_name).exists()
    ]

    cache_directories_present = sorted(
        _relative(path, root)
        for path in iter_directories(root)
        if path.name in CACHE_DIRECTORY_NAMES
    )

    findings: list[FileFinding] = []

    for path_name in generated_roots_present:
        findings.append(
            FileFinding(
                path=path_name,
                category="generated_root_in_repository",
                severity="warning",
                message="Generated output should not be committed to the source repository.",
            )
        )

    for path_name in cache_directories_present:
        findings.append(
            FileFinding(
                path=path_name,
                category="cache_directory_in_repository",
                severity="warning",
                message="Local cache directory should be removed and ignored.",
            )
        )

    for path in source_files:
        size_finding = _size_finding(path, root, is_test=False)
        if size_finding is not None:
            findings.append(size_finding)

        name_finding = _ambiguous_name_finding(path, root)
        if name_finding is not None:
            findings.append(name_finding)

    for path in test_files:
        size_finding = _size_finding(path, root, is_test=True)
        if size_finding is not None:
            findings.append(size_finding)

        name_finding = _ambiguous_name_finding(path, root)
        if name_finding is not None:
            findings.append(name_finding)

    findings.sort(key=lambda finding: (finding.severity != "blocking", finding.category, finding.path))

    return RepositoryAuditReport(
        repository_root=root.as_posix(),
        python_source_files=len(source_files),
        test_files=len(test_files),
        markdown_files=len(markdown_files),
        generated_roots_present=generated_roots_present,
        cache_directories_present=cache_directories_present,
        findings=findings,
    )


def _report_as_dict(report: RepositoryAuditReport) -> dict[str, object]:
    data = asdict(report)
    data["blocking_failure_count"] = report.blocking_failure_count
    data["warning_count"] = report.warning_count
    data["status"] = report.status
    return data


def write_reports(report: RepositoryAuditReport, output_directory: str | Path) -> tuple[Path, Path]:
    """Write detailed JSON and Markdown audit reports."""

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "repository_audit.json"
    markdown_path = output_dir / "repository_audit.md"

    json_path.write_text(
        json.dumps(_report_as_dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# FPGAI repository audit",
        "",
        f"- Status: `{report.status}`",
        f"- Python source files: {report.python_source_files}",
        f"- Test files: {report.test_files}",
        f"- Markdown files: {report.markdown_files}",
        f"- Blocking findings: {report.blocking_failure_count}",
        f"- Warnings: {report.warning_count}",
        "",
        "## Findings",
        "",
    ]

    if not report.findings:
        lines.append("No findings.")
    else:
        for finding in report.findings:
            line_count = ""
            if finding.line_count is not None:
                line_count = f" ({finding.line_count} lines)"
            lines.append(
                f"- **{finding.severity}** `{finding.category}`: "
                f"`{finding.path}`{line_count} — {finding.message}"
            )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root to inspect")
    parser.add_argument(
        "--output-dir",
        default="build/repository_audit",
        help="directory for detailed JSON and Markdown reports",
    )
    parser.add_argument(
        "--fail-on-blocking",
        action="store_true",
        help="return a non-zero exit code when blocking findings exist",
    )
    return parser


def main() -> int:
    args = _build_argument_parser().parse_args()
    report = audit_repository(args.root)
    json_path, markdown_path = write_reports(report, args.output_dir)

    print("Repository audit")
    print("----------------")
    print(f"python_source_files: {report.python_source_files}")
    print(f"test_files: {report.test_files}")
    print(f"markdown_files: {report.markdown_files}")
    print(f"generated_roots_present: {len(report.generated_roots_present)}")
    print(f"cache_directories_present: {len(report.cache_directories_present)}")
    print(f"blocking_failures: {report.blocking_failure_count}")
    print(f"warnings: {report.warning_count}")
    print(f"status: {report.status}")
    print(f"json_report: {json_path}")
    print(f"markdown_report: {markdown_path}")

    if args.fail_on_blocking and report.blocking_failure_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
