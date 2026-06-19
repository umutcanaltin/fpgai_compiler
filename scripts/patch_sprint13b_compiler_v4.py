from __future__ import annotations

from pathlib import Path

COMPILER = Path('fpgai/engine/compiler.py')

MARKER_START = '        # FPGAI Sprint 13B v4: preload runtime training weights for stream/ddr CSim\n'
MARKER_END = '        # FPGAI Sprint 13B v4 end\n'

BLOCK = '''        # FPGAI Sprint 13B v4: preload runtime training weights for stream/ddr CSim
        # The training reference step is executed before HLS emission and writes
        # build/training_reference/weights_before_ref.bin.  Runtime-weight training
        # modes need these words in the generated testbench; otherwise CSim starts
        # correctly but exits with: "Runtime preload size mismatch. got=0 expected=...".
        training_preload_weights = []
        if runtime_weight_mode:
            ref_weights_path = out_dir / "training_reference" / "weights_before_ref.bin"
            if ref_weights_path.exists():
                try:
                    training_preload_weights = [
                        float(x) for x in np.fromfile(ref_weights_path, dtype=np.float32)
                    ]
                except Exception:
                    training_preload_weights = []
        # FPGAI Sprint 13B v4 end
'''


def remove_old_block(text: str) -> str:
    while MARKER_START in text:
        a = text.index(MARKER_START)
        b = text.index(MARKER_END, a) + len(MARKER_END)
        text = text[:a] + text[b:]
    return text


def main() -> None:
    if not COMPILER.exists():
        raise SystemExit(f"Missing {COMPILER}; run from repository root")

    text = COMPILER.read_text(encoding='utf-8')
    text = remove_old_block(text)

    # Replace any earlier Sprint 13B patch that still passes an empty preload list.
    text = text.replace('preload_weights=training_preload_weights,', 'preload_weights=[],')

    needle = '        emit_tb_train_cpp(\n'
    idx = text.find(needle)
    if idx < 0:
        raise SystemExit('Could not find emit_tb_train_cpp call in compiler.py')

    # Insert the preload block directly before the training testbench emission.
    text = text[:idx] + BLOCK + text[idx:]

    # Replace only the first training testbench argument occurrence after the insertion.
    idx = text.find(needle)
    call_end = text.find('        write_text(\n            hls_dir / "run_hls.tcl"', idx)
    if call_end < 0:
        call_end = text.find('        else:', idx)
    if call_end < 0:
        raise SystemExit('Could not locate end of emit_tb_train_cpp call block')

    call = text[idx:call_end]
    if 'preload_weights=[],' not in call:
        raise SystemExit('emit_tb_train_cpp call does not contain preload_weights=[]')
    call = call.replace('preload_weights=[],', 'preload_weights=training_preload_weights,', 1)
    text = text[:idx] + call + text[call_end:]

    COMPILER.write_text(text, encoding='utf-8')
    print('[OK] Patched fpgai/engine/compiler.py for Sprint 13B v4 runtime training preload weights')


if __name__ == '__main__':
    main()
