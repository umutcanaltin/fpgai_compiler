#!/usr/bin/env python3
"""Sprint 15C patch: make pipeline_style lower to distinct pipeline II values.

This patch updates fpgai/engine/planner.py so pipeline_style is not just metadata:
  conservative -> II 3/4
  balanced     -> II 2
  aggressive   -> II 1

The codegen already consumes LayerPlan.pipeline_ii, so this makes conservative,
balanced, and aggressive generate different HLS PIPELINE II directives while
keeping PE/SIMD/unroll/partition fixed in the Sprint 15C sweep.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()
PLANNER = ROOT / "fpgai" / "engine" / "planner.py"

NEW_FUNC = '''def _pipeline_ii_for(policy: Policy, compute_hint: str = "") -> int:
    """Lower pipeline_style into distinct HLS initiation intervals.

    Sprint 15C goal: conservative, balanced, and aggressive must produce
    different generated HLS directives even when PE/SIMD/unroll/partition
    are held constant by the sweep.

    These values are intentionally simple and evidence-driven:
    - conservative prioritizes timing/resource safety, so it relaxes II.
    - balanced uses moderate pipelining.
    - aggressive targets II=1.
    """
    style = str(getattr(policy, "pipeline_style", "balanced") or "balanced").lower()
    hint = str(compute_hint or "").lower()

    if style == "aggressive":
        return 1
    if style == "balanced":
        return 2
    if style == "conservative":
        return 4 if hint == "memory_bound" else 3
    return 2

'''


def main() -> int:
    if not PLANNER.exists():
        raise SystemExit(f"Missing {PLANNER}")

    src = PLANNER.read_text()
    pattern = re.compile(r"def _pipeline_ii_for\(policy: Policy, compute_hint: str = \"\"\) -> int:\n.*?\ndef _layer_notes\(", re.S)
    match = pattern.search(src)
    if not match:
        raise SystemExit("Could not locate _pipeline_ii_for(...) followed by _layer_notes(...). No files changed.")

    replacement = NEW_FUNC + "def _layer_notes("
    updated = src[: match.start()] + replacement + src[match.end():]

    if updated == src:
        print("[OK] planner.py already up to date")
        return 0

    backup = PLANNER.with_suffix(".py.sprint15c_before_pipeline_policy")
    backup.write_text(src)
    PLANNER.write_text(updated)
    print(f"[OK] Patched {PLANNER}")
    print(f"[OK] Backup written to {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
