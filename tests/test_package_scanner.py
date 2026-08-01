from pathlib import Path

from fpgai.discovery import scan_package_manifests


def test_scanner_finds_bounded_packages_and_skips_generated(tmp_path: Path) -> None:
    (tmp_path / "provider" / "package").mkdir(parents=True)
    (tmp_path / "provider" / "package" / "fpgai.yaml").write_text("schema: x\n", encoding="utf-8")
    (tmp_path / "generated" / "bad").mkdir(parents=True)
    (tmp_path / "generated" / "bad" / "fpgai.yaml").write_text("schema: x\n", encoding="utf-8")

    result = scan_package_manifests(tmp_path, max_depth=2)

    assert result.manifests == (tmp_path / "provider" / "package" / "fpgai.yaml",)


def test_scanner_does_not_follow_symlink_directories(tmp_path: Path) -> None:
    target = tmp_path / "outside"
    target.mkdir()
    (target / "fpgai.yaml").write_text("schema: x\n", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "linked").symlink_to(target, target_is_directory=True)

    result = scan_package_manifests(root, max_depth=2)

    assert not result.manifests
    assert root / "linked" in result.skipped_symlinks
