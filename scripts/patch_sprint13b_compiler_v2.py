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
        text = replace_once(text, old_call, new_call, "training testbench call")
    elif "training_hls_output_dir = out_dir / \"training_hls_outputs\"" in text and "output_dir=str(training_hls_output_dir.resolve())" in text:
        print("[OK] compiler.py already has explicit training_hls_outputs testbench patch")
    else:
        raise SystemExit("[ERROR] Could not patch emit_tb_train_cpp call. Paste the compiler.py block around emit_tb_train_cpp.")

    old_tcl = '''emit_csim_train_tcl(
                    top_name=top_name,
                    part=part,
                    input_bin_path=input_bin,
                    target_bin_path=target_bin,
                    weights_mode=weights_mode,
                    intermediate_dump=intermediate_dump,
                )'''
    new_tcl = '''emit_csim_train_tcl(
                    top_name=top_name,
                    part=part,
                    input_bin_path=input_bin,
                    target_bin_path=target_bin,
                    weights_mode=weights_mode,
                    intermediate_dump=intermediate_dump,
                    output_dir_path=str(training_hls_output_dir.resolve()),
                )'''
    if old_tcl in text:
        text = replace_once(text, old_tcl, new_tcl, "training TCL call")
    elif "output_dir_path=str(training_hls_output_dir.resolve())" in text:
        print("[OK] compiler.py already passes output_dir_path to emit_csim_train_tcl")
    else:
        raise SystemExit("[ERROR] Could not patch emit_csim_train_tcl call. Paste the compiler.py block around emit_csim_train_tcl.")

    old_search = '''if hls_run is not None and hls_dir is not None:
            hls_grads = self._find_file_recursive(hls_dir, "grads.bin")
            hls_w_before = self._find_file_recursive(hls_dir, "weights_before.bin")
            hls_w_after = self._find_file_recursive(hls_dir, "weights_after.bin")'''
    new_search = '''if hls_run is not None and hls_dir is not None:
            # Sprint 13B: CSim training artifacts may be written either under
            # build/training_hls_outputs or inside Vitis CSim directories.
            # Search the complete build root so embedded, stream, and DDR modes
            # are handled consistently.
            hls_grads = self._find_file_recursive(out_dir, "grads.bin")
            hls_w_before = self._find_file_recursive(out_dir, "weights_before.bin")
            hls_w_after = self._find_file_recursive(out_dir, "weights_after.bin")'''
    if old_search in text:
        text = replace_once(text, old_search, new_search, "training artifact search")
    elif 'self._find_file_recursive(out_dir, "grads.bin")' in text:
        print("[OK] compiler.py already searches out_dir for training artifacts")
    else:
        raise SystemExit("[ERROR] Could not patch training artifact search. Paste the compiler.py block around hls_grads.")

    if text != original:
        backup = path.with_suffix(path.suffix + ".sprint13b_v2.bak")
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
