#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"[ERROR] Expected exactly one {label} block, found {count}.")
    return text.replace(old, new, 1)


def main() -> int:
    path = Path("fpgai/engine/compiler.py")
    if not path.exists():
        raise SystemExit("[ERROR] Run this from the fpgai_compiler repository root.")

    text = path.read_text(encoding="utf-8")
    original = text

    old_call = '''emit_tb_train_cpp(
                src_dir,
                graph=g,
                top_name=top_name,
                in_words=int(np.fromfile(input_bin, dtype=np.float32).size),
                out_words=0,
                weights_mode=weights_mode,
                weight_words=total_param_words,
                preload_weights=[],
                training_cfg=training_cfg,
            )'''
    new_call = '''training_hls_output_dir = out_dir / "training_hls_outputs"
            training_hls_output_dir.mkdir(parents=True, exist_ok=True)
            training_preload_weights = []
            if runtime_weight_mode:
                ref_weights_before = out_dir / "training_reference" / "weights_before_ref.bin"
                if ref_weights_before.exists():
                    training_preload_weights = np.fromfile(ref_weights_before, dtype=np.float32).astype(np.float32).tolist()

            emit_tb_train_cpp(
                src_dir,
                graph=g,
                top_name=top_name,
                in_words=int(np.fromfile(input_bin, dtype=np.float32).size),
                out_words=0,
                weights_mode=weights_mode,
                weight_words=total_param_words,
                preload_weights=training_preload_weights,
                training_cfg=training_cfg,
                output_dir=str(training_hls_output_dir.resolve()),
            )'''

    if old_call in text:
        text = replace_once(text, old_call, new_call, "emit_tb_train_cpp")
    elif "training_hls_output_dir = out_dir / \"training_hls_outputs\"" in text:
        print("[OK] compiler.py already contains Sprint 13B testbench output/preload patch")
    else:
        raise SystemExit(
            "[ERROR] Could not find the expected emit_tb_train_cpp block. "
            "Please paste the block around emit_tb_train_cpp from compiler.py."
        )

    old_search = '''if hls_run is not None and hls_dir is not None:
            hls_grads = self._find_file_recursive(hls_dir, "grads.bin")
            hls_w_before = self._find_file_recursive(hls_dir, "weights_before.bin")
            hls_w_after = self._find_file_recursive(hls_dir, "weights_after.bin")'''
    new_search = '''if hls_run is not None and hls_dir is not None:
            # Sprint 13B: training CSim artifacts may be written to an explicit
            # build/training_hls_outputs directory instead of the Vitis CSim
            # working directory. Search the full build root first so embedded,
            # stream, and DDR training modes are handled consistently.
            hls_grads = self._find_file_recursive(out_dir, "grads.bin")
            hls_w_before = self._find_file_recursive(out_dir, "weights_before.bin")
            hls_w_after = self._find_file_recursive(out_dir, "weights_after.bin")'''

    if old_search in text:
        text = replace_once(text, old_search, new_search, "training artifact search")
    elif "training_hls_outputs" in text and 'self._find_file_recursive(out_dir, "grads.bin")' in text:
        print("[OK] compiler.py already contains Sprint 13B artifact-search patch")
    else:
        raise SystemExit(
            "[ERROR] Could not find the expected hls artifact search block. "
            "Please paste the block around hls_grads from compiler.py."
        )

    if text != original:
        backup = path.with_suffix(path.suffix + ".sprint13b.bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        print(f"[OK] Patched {path}")
        print(f"[OK] Backup: {backup}")
    else:
        print("[OK] No compiler.py changes needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
