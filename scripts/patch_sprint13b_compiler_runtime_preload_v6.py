from __future__ import annotations

from pathlib import Path

p = Path("fpgai/engine/compiler.py")
text = p.read_text(encoding="utf-8")
orig = text

old_tb = "emit_tb_train_cpp( src_dir, graph=g, top_name=top_name, in_words=int(np.fromfile(input_bin, dtype=np.float32).size), out_words=0, weights_mode=weights_mode, weight_words=total_param_words, preload_weights=[], training_cfg=training_cfg, )"
new_tb = "emit_tb_train_cpp( src_dir, graph=g, top_name=top_name, in_words=int(np.fromfile(input_bin, dtype=np.float32).size), out_words=0, weights_mode=weights_mode, weight_words=total_param_words, preload_weights=[], training_cfg=training_cfg, output_dir=str((out_dir / \"training_hls_outputs\").resolve()), )"
if old_tb in text:
    text = text.replace(old_tb, new_tb, 1)
elif "output_dir=str((out_dir / \"training_hls_outputs\").resolve())" not in text:
    raise SystemExit("Could not patch emit_tb_train_cpp call. Inspect fpgai/engine/compiler.py manually.")

old_tcl = "emit_csim_train_tcl( top_name=top_name, part=part, input_bin_path=input_bin, target_bin_path=target_bin, weights_mode=weights_mode, intermediate_dump=intermediate_dump, ),"
new_tcl = "emit_csim_train_tcl( top_name=top_name, part=part, input_bin_path=input_bin, target_bin_path=target_bin, weights_mode=weights_mode, intermediate_dump=intermediate_dump, preload_bin_path=str((out_dir / \"training_reference\" / \"weights_before_ref.bin\").resolve()), output_dir_path=str((out_dir / \"training_hls_outputs\").resolve()), ),"
if old_tcl in text:
    text = text.replace(old_tcl, new_tcl, 1)
elif "preload_bin_path=str((out_dir / \"training_reference\" / \"weights_before_ref.bin\").resolve())" not in text:
    raise SystemExit("Could not patch emit_csim_train_tcl call. Inspect fpgai/engine/compiler.py manually.")

# Make training compare artifact discovery robust across both hls_dir and the full build dir.
old_find = "hls_grads = self._find_file_recursive(hls_dir, \"grads.bin\") hls_w_before = self._find_file_recursive(hls_dir, \"weights_before.bin\") hls_w_after = self._find_file_recursive(hls_dir, \"weights_after.bin\")"
new_find = "hls_grads = self._find_file_recursive(out_dir, \"grads.bin\") or self._find_file_recursive(hls_dir, \"grads.bin\") hls_w_before = self._find_file_recursive(out_dir, \"weights_before.bin\") or self._find_file_recursive(hls_dir, \"weights_before.bin\") hls_w_after = self._find_file_recursive(out_dir, \"weights_after.bin\") or self._find_file_recursive(hls_dir, \"weights_after.bin\")"
if old_find in text:
    text = text.replace(old_find, new_find, 1)

if text == orig:
    print("[WARN] No changes were made; file may already be patched.")
else:
    p.write_text(text, encoding="utf-8")
    print("[OK] Patched fpgai/engine/compiler.py for Sprint 13B runtime preload v6")
