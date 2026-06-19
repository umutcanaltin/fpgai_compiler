from __future__ import annotations

import re
from pathlib import Path

TARGET = Path('fpgai/backends/hls/emit/top_train_cpp.py')


def _find_function_span(text: str, name: str) -> tuple[int, int]:
    m = re.search(rf"^def\s+{re.escape(name)}\s*\(", text, flags=re.M)
    if not m:
        raise SystemExit(f"Could not find {name} definition")
    start = m.start()
    m2 = re.search(r"^def\s+\w+\s*\(", text[m.end():], flags=re.M)
    end = (m.end() + m2.start()) if m2 else len(text)
    return start, end


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Missing {TARGET}")
    text = TARGET.read_text(encoding='utf-8')

    if '_inject_native_accumulated_modes_from_cpp' not in text:
        raise SystemExit(
            'Missing _inject_native_accumulated_modes_from_cpp. Apply/commit Sprint 13E native accumulation first.'
        )

    start, end = _find_function_span(text, '_inject_native_accumulated_modes_from_cpp')
    body = text[start:end]

    if 'if (mode == 6)' in body and 'loss_eval mode' in body:
        print('[OK] Native loss-evaluation mode already present in generator helper')
        return

    snippet = r'''
    # Sprint 13F: add evaluation-only loss mode to the generated HLS top.
    # mode 6 reads one input/target record, computes loss_value, emits one loss float,
    # and returns before gradient computation or optimizer update.
    if "if (mode == 6)" not in cpp:
        loss_pos = cpp.find("loss_value +=")
        if loss_pos < 0:
            raise RuntimeError("Could not find loss_value accumulation for mode 6 injection")
        insert_pos = cpp.find("\n\n  for (int i = 0; i <", loss_pos)
        if insert_pos < 0:
            raise RuntimeError("Could not find gradient loop marker after loss_value for mode 6 injection")
        loss_eval_block = "\n\n  // FPGAI loss_eval mode.\n  if (mode == 6) {\n    write_f32(out, (float)loss_value, true);\n    return;\n  }"
        cpp = cpp[:insert_pos] + loss_eval_block + cpp[insert_pos:]
'''

    # Insert immediately before the final return cpp in the helper.
    idx = body.rfind('    return cpp')
    if idx < 0:
        raise SystemExit('Could not find return cpp in _inject_native_accumulated_modes_from_cpp')
    body = body[:idx] + snippet + body[idx:]
    text = text[:start] + body + text[end:]
    TARGET.write_text(text, encoding='utf-8')
    print('[OK] Patched HLS top generator with loss-evaluation mode 6')
    print('[OK] Verify with generated deeplearn.cpp grep: mode == 6 | loss_eval mode')


if __name__ == '__main__':
    main()
