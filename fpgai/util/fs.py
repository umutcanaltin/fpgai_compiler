from __future__ import annotations

from pathlib import Path
import shutil


def ensure_clean_dir(path: Path, *, clean: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if clean:
        for p in path.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
