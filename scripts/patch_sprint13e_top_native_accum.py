from __future__ import annotations

import re
from pathlib import Path

TARGET = Path('fpgai/backends/hls/emit/top_train_cpp.py')

HELPER = r'''

def _inject_native_accumulated_modes_from_cpp(
    cpp: str,
    input_size: int,
    learning_rate: float,
) -> str:
    """Post-process generated training C++ with native mini-batch modes.

    Added top modes:
      mode 3: run forward/backward, accumulate parameter gradients, no update
      mode 4: average accumulated gradients and apply one SGD update in the HLS top
      mode 5: reset accumulated gradients and counter

    This helper discovers trainable parameters from the generated C++ itself, so it is
    robust to changes in the Python generator's local variable names.
    """
    if "FPGAI_NATIVE_ACC_BATCH_COUNT" in cpp and "if (mode == 3)" in cpp:
        return cpp

    params = []
    # Weight/bias gradients already emitted by the training generator.
    grad_w = {m.group(1): int(m.group(2)) for m in re.finditer(r"static\s+grad_wgt_t\s+dW_([A-Za-z0-9_]+)\[(\d+)\];", cpp)}
    grad_b = {m.group(1): int(m.group(2)) for m in re.finditer(r"static\s+grad_bias_t\s+dB_([A-Za-z0-9_]+)\[(\d+)\];", cpp)}

    for tag in sorted(grad_w.keys()):
        if tag not in grad_b:
            continue
        w_arr = f"W_{tag}"
        b_arr = f"B_{tag}"
        if f"static wgt_t {w_arr}[" not in cpp or f"static bias_t {b_arr}[" not in cpp:
            # BatchNorm or other trainable parameter forms can be added later.
            continue
        if f"OUT_grad_{tag}" not in cpp:
            continue
        params.append({
            "tag": tag,
            "w": grad_w[tag],
            "b": grad_b[tag],
            "w_arr": w_arr,
            "b_arr": b_arr,
            "dw": f"dW_{tag}",
            "db": f"dB_{tag}",
            "out": f"OUT_grad_{tag}",
        })

    if not params:
        raise RuntimeError("Native accumulation injection found no dense/conv trainable parameters in generated C++")

    decl = ["", "// FPGAI native accumulated mini-batch optimizer state."]
    for p in params:
        decl.append(f"static acc_t ACC_{p['dw']}[{p['w']}];")
        decl.append(f"static acc_t ACC_{p['db']}[{p['b']}];")
    decl.append("static int FPGAI_NATIVE_ACC_BATCH_COUNT = 0;")
    decl_text = "\n".join(decl) + "\n"

    extern_marker = '\nextern "C" void '
    if extern_marker not in cpp:
        raise RuntimeError("Could not find extern C top marker for native accumulation injection")
    cpp = cpp.replace(extern_marker, decl_text + extern_marker, 1)

    def reset_lines(indent: str) -> list[str]:
        out = []
        for p in params:
            out.append(f"{indent}for (int i = 0; i < {p['w']}; ++i) ACC_{p['dw']}[i] = (acc_t)0;")
            out.append(f"{indent}for (int i = 0; i < {p['b']}; ++i) ACC_{p['db']}[i] = (acc_t)0;")
        out.append(f"{indent}FPGAI_NATIVE_ACC_BATCH_COUNT = 0;")
        return out

    def emit_grad_lines(indent: str) -> list[str]:
        out = []
        for idx, p in enumerate(params):
            last = "true" if idx == len(params) - 1 else "false"
            total = p['w'] + p['b']
            out.append(f"{indent}for (int i = 0; i < {p['w']}; ++i) {p['out']}[i] = (float){p['dw']}[i];")
            out.append(f"{indent}for (int i = 0; i < {p['b']}; ++i) {p['out']}[{p['w']} + i] = (float){p['db']}[i];")
            out.append(f"{indent}emit_stream_block<{total}>(out, {p['out']}, {last});")
        return out

    pre_input = [
        "",
        "  if (mode == 5) {",
        *reset_lines("    "),
        "    return;",
        "  }",
        "",
        "  if (mode == 4) {",
        "    const int denom = (FPGAI_NATIVE_ACC_BATCH_COUNT > 0) ? FPGAI_NATIVE_ACC_BATCH_COUNT : 1;",
    ]
    for p in params:
        pre_input.append(f"    for (int i = 0; i < {p['w']}; ++i) {p['dw']}[i] = (grad_wgt_t)(((float)ACC_{p['dw']}[i]) / (float)denom);")
        pre_input.append(f"    for (int i = 0; i < {p['b']}; ++i) {p['db']}[i] = (grad_bias_t)(((float)ACC_{p['db']}[i]) / (float)denom);")
        pre_input.append(f"    fpgai::sgd_update_wgt_typed<{p['w']}, wgt_t, grad_wgt_t, upd_t, acc_t, 1, 4>({p['w_arr']}, {p['dw']}, (upd_t){learning_rate:.8f}f);")
        pre_input.append(f"    fpgai::sgd_update_bias_typed<{p['b']}, bias_t, grad_bias_t, upd_t, acc_t, 1, 2>({p['b_arr']}, {p['db']}, (upd_t){learning_rate:.8f}f);")
    pre_input.extend(reset_lines("    "))
    pre_input.extend(emit_grad_lines("    "))
    pre_input.extend(["    return;", "  }", ""])

    input_marker = f"\n  for (int i = 0; i < {int(input_size)}; ++i)"
    pos = cpp.find(input_marker)
    if pos < 0:
        raise RuntimeError(f"Could not find training input read marker: {input_marker!r}")
    cpp = cpp[:pos] + "\n".join(pre_input) + cpp[pos:]

    accum = ["", "  if (mode == 3) {"]
    for p in params:
        accum.append(f"    for (int i = 0; i < {p['w']}; ++i) ACC_{p['dw']}[i] += (acc_t){p['dw']}[i];")
        accum.append(f"    for (int i = 0; i < {p['b']}; ++i) ACC_{p['db']}[i] += (acc_t){p['db']}[i];")
    accum.append("    FPGAI_NATIVE_ACC_BATCH_COUNT += 1;")
    accum.extend(emit_grad_lines("    "))
    accum.extend(["    return;", "  }", ""])

    update_marker = "\n  fpgai::sgd_update_"
    pos = cpp.find(update_marker)
    if pos < 0:
        raise RuntimeError("Could not find SGD update marker for mode 3 insertion")
    cpp = cpp[:pos] + "\n".join(accum) + cpp[pos:]
    return cpp
'''


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
        marker = 'def emit_top_train_cpp('
        if marker not in text:
            raise SystemExit('Could not find emit_top_train_cpp definition')
        text = text.replace(marker, HELPER + '\n' + marker, 1)

    start, end = _find_function_span(text, 'emit_top_train_cpp')
    body = text[start:end]

    # Remove older v1 wrapping if present; v2 wraps with generated-C++ parser.
    body = body.replace('_inject_native_accumulated_modes(', '_inject_native_accumulated_modes_from_cpp(')
    body = re.sub(
        r"_inject_native_accumulated_modes_from_cpp\((['\"]\\n['\"]\.join\(lines\)),\s*parameter_specs,\s*input_size,\s*learning_rate\)",
        r"_inject_native_accumulated_modes_from_cpp(\1, input_size, learning_rate)",
        body,
    )

    if '_inject_native_accumulated_modes_from_cpp(' not in body:
        pattern = re.compile(r"return\s+((?:['\"]\\n['\"]\.join\(lines\))(?:\s*\+\s*['\"]\\n['\"])?)(?P<tail>\s*(?:#.*)?$)", re.M)
        matches = list(pattern.finditer(body))
        if not matches:
            raise SystemExit('Could not patch return statement in emit_top_train_cpp; inspect file ending manually')
        # Patch the last join(lines) return in the function.
        m = matches[-1]
        expr = m.group(1)
        replacement = f"return _inject_native_accumulated_modes_from_cpp({expr}, input_size, learning_rate)"
        body = body[:m.start()] + replacement + body[m.end():]

    text = text[:start] + body + text[end:]
    TARGET.write_text(text, encoding='utf-8')
    print('[OK] Patched native accumulated mini-batch modes into', TARGET)
    print('[OK] Verify with: grep -n "native_accumulated_modes_from_cpp\\|mode == 3\\|mode == 4\\|mode == 5" fpgai/backends/hls/emit/top_train_cpp.py | head -80')


if __name__ == '__main__':
    main()
