"""Source checkout path resolution without importing compiler or vendor code."""
from pathlib import Path


def source_checkout_root(anchor: str | Path) -> Path:
    """Find the current research checkout from an existing package file."""
    return Path(anchor).resolve().parents[2]


def compiler_src_root(anchor: str | Path) -> Path:
    root = source_checkout_root(anchor)
    return root / "src"
