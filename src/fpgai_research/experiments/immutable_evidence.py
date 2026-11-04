"""Immutable, idempotent publication of a small experimental evidence directory.

Never silently replace a previous research receipt. Distinct researchers or
processes must use distinct output paths; rename is not a portable no-replace
lock against a simultaneous writer.
"""
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory


def _fingerprints(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Evidence directory must be regular and non-linked")
    files = {}
    for item in root.rglob("*"):
        if item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise ValueError("Evidence contains linked or non-regular entries")
        if item.is_file():
            digest = sha256()
            with item.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[item.relative_to(root).as_posix()] = digest.hexdigest()
    return files


@contextmanager
def immutable_evidence_directory(destination: str | Path):
    """Prepare privately and publish once; identical reruns preserve bytes.

    If the destination is present, the new result must have exactly the same
    file set and file bytes. On mismatch, leave the prior evidence untouched.
    """
    target = Path(destination)
    if target.is_symlink():
        raise FileExistsError("Cannot replace symlinked evidence destination")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{target.name}.stage-", dir=target.parent) as tmp:
        stage = Path(tmp)
        yield stage
        candidate = _fingerprints(stage)
        if not candidate:
            raise ValueError("Cannot publish an empty evidence directory")
        if target.is_symlink():
            raise FileExistsError("Cannot replace symlinked evidence destination")
        if target.exists():
            if candidate != _fingerprints(target):
                raise FileExistsError("Evidence already exists with different content")
            return
        stage.rename(target)
