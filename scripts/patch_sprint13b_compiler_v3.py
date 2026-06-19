#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


def main() -> int:
    path = Path("fpgai/engine/compiler.py")
    if not path.exists():
        raise SystemExit("[ERROR] Run this from the fpgai_compiler repository root.")

    text = path.read_text(encoding="utf-8")
    original = text

    if "training_hls_output_dir = out_dir / \"training_hls_outputs\"" not in text:
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
        if old_call not in text:
            raise SystemExit("[ERROR] Could not find original emit_tb_train_cpp call. Paste compiler.py around emit_tb_train_cpp.")
        text = text.replace(old_call, new_call, 1)
    else:
        print("[OK] compiler.py already has training_hls_output_dir setup")

    if "emit_tb_train_cpp(" in text and "output_dir=str(training_hls_output_dir.resolve())," not in text:
        text, n = re.subn(
            r"(\n\s*training_cfg=training_cfg,\n\s*\))",
            "\n                training_cfg=training_cfg,\n                output_dir=str(training_hls_output_dir.resolve()),\n            )",
            text,
            count=1,
        )
        if n != 1:
            raise SystemExit("[ERROR] Could not add output_dir to emit_tb_train_cpp call.")
    else:
        print("[OK] compiler.py already passes output_dir to emit_tb_train_cpp")

    if "output_dir_path=str(training_hls_output_dir.resolve())," not in text:
        text, n = re.subn(
            r"(\n\s*intermediate_dump=intermediate_dump,\n\s*\))",
            "\n                    intermediate_dump=intermediate_dump,\n                    output_dir_path=str(training_hls_output_dir.resolve()),\n                )",
            text,
            count=1,
        )
        if n != 1:
            raise SystemExit("[ERROR] Could not add output_dir_path to emit_csim_train_tcl call.")
    else:
        print("[OK] compiler.py already passes output_dir_path to emit_csim_train_tcl")

    old = '''hls_grads = self._find_file_recursive(hls_dir, "grads.bin")
            hls_w_before = self._find_file_recursive(hls_dir, "weights_before.bin")
            hls_w_after = self._find_file_recursive(hls_dir, "weights_after.bin")'''
    new = '''hls_grads = self._find_file_recursive(out_dir, "grads.bin")
            hls_w_before = self._find_file_recursive(out_dir, "weights_before.bin")
            hls_w_after = self._find_file_recursive(out_dir, "weights_after.bin")'''
    if old in text:
        text = text.replace(old, new, 1)
    elif 'self._find_file_recursive(out_dir, "grads.bin")' in text:
        print("[OK] compiler.py already searches out_dir for training bins")
    else:
        raise SystemExit("[ERROR] Could not patch HLS training artifact search.")

    if text != original:
        backup = path.with_suffix(path.suffix + ".sprint13b_v3.bak")
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
