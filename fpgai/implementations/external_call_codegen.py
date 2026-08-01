from __future__ import annotations

import math
from typing import Any

from fpgai.implementations.hls_integration import parse_flat_array_abi


def _literal(value: Any, cpp_type: str) -> str:
    if cpp_type in {"int", "unsigned"}:
        return str(int(value)) + ("u" if cpp_type == "unsigned" else "")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("HLSCOMP010: non-finite external attribute")
    text = f"{number:.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + ("f" if cpp_type == "float" else "")


def package_declarations(plan) -> list[str]:
    declarations: list[str] = []
    seen: set[tuple[str, str]] = set()
    for binding in plan.bindings:
        contract = binding.contract
        key = (contract.package_id, contract.top)
        if key in seen:
            continue
        seen.add(key)
        abi = parse_flat_array_abi(contract)
        params = [f"const {abi.scalar_type}* input", f"{abi.scalar_type}* output", "int count"]
        params.extend(f"{item.cpp_type} {item.name}" for item in abi.attributes)
        declarations.append(f"void {contract.top}({', '.join(params)});")
    return declarations


def emit_external_call(binding, *, current_buffer: str, current_type: str, output_buffer: str, output_type: str) -> list[str]:
    abi = parse_flat_array_abi(binding.contract)
    values = []
    for item in abi.attributes:
        value = binding.attributes.get(item.name, item.default)
        values.append(_literal(value, item.cpp_type))
    lines: list[str] = [f"    // External implementation: {binding.contract.package_id}"]
    in_arg = current_buffer
    out_arg = output_buffer
    if current_type != abi.scalar_type:
        in_arg = f"{binding.wrapper_symbol}_input"
        lines.extend([
            f"    {abi.scalar_type} {in_arg}[{binding.input_words}];",
            f"    for (int i = 0; i < {binding.input_words}; ++i) {{",
            "#pragma HLS PIPELINE II=1",
            f"        {in_arg}[i] = ({abi.scalar_type}){current_buffer}[i];",
            "    }",
        ])
    if output_type != abi.scalar_type:
        out_arg = f"{binding.wrapper_symbol}_output"
        lines.append(f"    {abi.scalar_type} {out_arg}[{binding.output_words}];")
    args = [in_arg, out_arg, str(binding.output_words), *values]
    lines.append(f"    {binding.contract.top}({', '.join(args)});")
    if output_type != abi.scalar_type:
        lines.extend([
            f"    for (int i = 0; i < {binding.output_words}; ++i) {{",
            "#pragma HLS PIPELINE II=1",
            f"        {output_buffer}[i] = ({output_type}){out_arg}[i];",
            "    }",
        ])
    return lines
