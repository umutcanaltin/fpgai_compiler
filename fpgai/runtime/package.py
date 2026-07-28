"""Public runtime-package API and command-line entry point.

Implementation details live in focused modules under :mod:`fpgai.runtime`.
Existing imports of ``emit_runtime_package`` remain supported.
"""
from __future__ import annotations

import argparse
import json

from fpgai.runtime.package_builder import emit_runtime_package

__all__ = ["emit_runtime_package", "main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create an FPGAI runtime package from a compile output directory."
    )
    parser.add_argument("out_dir")
    parser.add_argument("--board")
    parser.add_argument("--pipeline-mode")
    parser.add_argument("--top-name")
    args = parser.parse_args(argv)

    result = emit_runtime_package(
        args.out_dir,
        board=args.board,
        pipeline_mode=args.pipeline_mode,
        top_name=args.top_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
