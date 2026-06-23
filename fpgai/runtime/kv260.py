from __future__ import annotations

import argparse
from pathlib import Path


def generate_kv260_runtime(output_dir: str | Path, app_name: str = "fpgai_kv260") -> Path:
    """Generate a small KV260 runtime README scaffold for deployment artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    readme = out / "README.md"
    readme.write_text(
        f"# {app_name}\n\n"
        "This directory is a KV260 runtime scaffold for FPGAI-generated artifacts.\n\n"
        "Place the generated bitstream, hardware handoff, and host/runtime files here.\n",
        encoding="utf-8",
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a KV260 runtime scaffold.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--app-name", default="fpgai_kv260")
    ns = parser.parse_args(argv)

    out = generate_kv260_runtime(ns.out, app_name=ns.app_name)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
