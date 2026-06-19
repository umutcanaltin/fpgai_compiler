#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "fpgai" / "engine" / "compiler.py"


def main() -> None:
    text = COMPILER.read_text(encoding="utf-8")
    original = text

    old = (
        'emit_csim_train_tcl( top_name=top_name, part=part, input_bin_path=input_bin, '
        'target_bin_path=target_bin, weights_mode=weights_mode, intermediate_dump=intermediate_dump, )'
    )
    new = (
        'emit_csim_train_tcl( top_name=top_name, part=part, input_bin_path=input_bin, '
        'target_bin_path=target_bin, weights_mode=weights_mode, '
        'preload_bin_path=str((out_dir / "training_reference" / "weights_before_ref.bin").resolve()), '
        'intermediate_dump=intermediate_dump, )'
    )

    if old in text:
        text = text.replace(old, new, 1)
    elif 'preload_bin_path=str((out_dir / "training_reference" / "weights_before_ref.bin").resolve())' in text:
        print("[OK] compiler.py already passes training preload path")
    else:
        raise SystemExit(
            "Could not patch compiler.py automatically. Search for emit_csim_train_tcl(...) "
            "inside _emit_hls training_on_device branch and add:\n"
            "  preload_bin_path=str((out_dir / 'training_reference' / 'weights_before_ref.bin').resolve()),"
        )

    # Optional robustness: compare artifacts should be found anywhere below build/, not only hls/.
    old_find_block = (
        'hls_grads = self._find_file_recursive(hls_dir, "grads.bin") '
        'hls_w_before = self._find_file_recursive(hls_dir, "weights_before.bin") '
        'hls_w_after = self._find_file_recursive(hls_dir, "weights_after.bin")'
    )
    new_find_block = (
        'search_root = out_dir if out_dir is not None else hls_dir '
        'hls_grads = self._find_file_recursive(search_root, "grads.bin") '
        'hls_w_before = self._find_file_recursive(search_root, "weights_before.bin") '
        'hls_w_after = self._find_file_recursive(search_root, "weights_after.bin")'
    )
    if old_find_block in text:
        text = text.replace(old_find_block, new_find_block, 1)

    if text != original:
        COMPILER.write_text(text, encoding="utf-8")
        print("[OK] patched", COMPILER)
    else:
        print("[OK] no compiler.py changes needed")


if __name__ == "__main__":
    main()
