from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from fpgai.implementations.implementation_contract import ImplementationContract

from .abi import HLSFlatArrayABI, parse_flat_array_abi, validate_hls_integration_contract
from .errors import HLSIntegrationIssue
from .types import ExternalHLSProjectRequest, ExternalHLSProjectResult


def _slug(package_id: str) -> str:
    return package_id.replace(".", "_").replace("-", "_")


def _safe_package_file(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"HLSINT007: symlinked package file is not allowed: {relative}")
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"HLSINT008: package file escapes package root: {relative}") from exc
    if not resolved.is_file():
        raise ValueError(f"HLSINT009: package file does not exist: {relative}")
    return resolved


def _cpp_literal(value: Any, cpp_type: str) -> str:
    if cpp_type in {"int", "unsigned"}:
        return str(int(value)) + ("u" if cpp_type == "unsigned" else "")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("HLSINT010: non-finite operator attribute")
    text = f"{numeric:.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + ("f" if cpp_type == "float" else "")


def _copy_files(contract: ImplementationContract, hls_dir: Path) -> tuple[tuple[Path, ...], tuple[Path, ...], Path, Path]:
    if contract.package_root is None:
        raise ValueError("HLSINT011: implementation contract has no package_root")
    package_root = Path(contract.package_root)
    slug = _slug(contract.package_id)
    external_root = hls_dir / "external" / slug
    src_root = external_root / "src"
    include_root = external_root / "include"
    src_root.mkdir(parents=True, exist_ok=True)
    include_root.mkdir(parents=True, exist_ok=True)

    copied_sources: list[Path] = []
    copied_headers: list[Path] = []
    ordered_sources = contract.source_order or contract.sources
    for index, relative in enumerate(ordered_sources):
        source = _safe_package_file(package_root, relative)
        destination = src_root / f"{index:03d}_{source.name}"
        shutil.copy2(source, destination)
        copied_sources.append(destination)
    for index, relative in enumerate(contract.headers):
        source = _safe_package_file(package_root, relative)
        destination = include_root / source.name
        shutil.copy2(source, destination)
        copied_headers.append(destination)
    return tuple(copied_sources), tuple(copied_headers), src_root, include_root


def _wrapper_header(request: ExternalHLSProjectRequest, abi: HLSFlatArrayABI) -> str:
    return (
        "#pragma once\n\n"
        f"extern \"C\" void {request.top_name}(\n"
        f"    const {abi.scalar_type} input[{request.input_words}],\n"
        f"    {abi.scalar_type} output[{request.output_words}]\n"
        ");\n"
    )


def _package_declaration(contract: ImplementationContract, abi: HLSFlatArrayABI) -> str:
    params = [f"const {abi.scalar_type}* input", f"{abi.scalar_type}* output", "int count"]
    params.extend(f"{item.cpp_type} {item.name}" for item in abi.attributes)
    return f"void {contract.top}({', '.join(params)});"


def _wrapper_source(request: ExternalHLSProjectRequest, abi: HLSFlatArrayABI) -> str:
    contract = request.contract
    values = []
    for item in abi.attributes:
        value = request.operator_attributes.get(item.name, item.default)
        values.append(_cpp_literal(value, item.cpp_type))
    call_args = ["input", "output", str(request.output_words), *values]
    return "\n".join(
        [
            '#include "external_wrapper.h"',
            "",
            _package_declaration(contract, abi),
            "",
            f'extern "C" void {request.top_name}(',
            f"    const {abi.scalar_type} input[{request.input_words}],",
            f"    {abi.scalar_type} output[{request.output_words}]",
            ") {",
            "#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0",
            "#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1",
            "#pragma HLS INTERFACE s_axilite port=input bundle=control",
            "#pragma HLS INTERFACE s_axilite port=output bundle=control",
            "#pragma HLS INTERFACE s_axilite port=return bundle=control",
            f"    {contract.top}({', '.join(call_args)});",
            "}",
            "",
        ]
    )


def _testbench_source(request: ExternalHLSProjectRequest, abi: HLSFlatArrayABI) -> str:
    attrs = {item.name: request.operator_attributes.get(item.name, item.default) for item in abi.attributes}
    scale = float(attrs.get("scale", 1.0))
    bias = float(attrs.get("bias", 0.0))
    return f'''#include "external_wrapper.h"
#include <cmath>
#include <cstdio>

int main() {{
    {abi.scalar_type} input[{request.input_words}];
    {abi.scalar_type} output[{request.output_words}] = {{0}};
    for (int i = 0; i < {request.input_words}; ++i) input[i] = ({abi.scalar_type})(i - 2);
    {request.top_name}(input, output);
    int failures = 0;
    for (int i = 0; i < {request.output_words}; ++i) {{
        {abi.scalar_type} expected = input[i] * ({abi.scalar_type}){_cpp_literal(scale, abi.scalar_type)} + ({abi.scalar_type}){_cpp_literal(bias, abi.scalar_type)};
        if (std::fabs((double)(output[i] - expected)) > 1.0e-6) ++failures;
    }}
    std::printf("[FPGAI-EXTERNAL-HLS] failures=%d\\n", failures);
    return failures == 0 ? 0 : 1;
}}
'''


def _run_tcl(request: ExternalHLSProjectRequest, hls_dir: Path, copied_sources: tuple[Path, ...], external_include: Path) -> str:
    source_lines = []
    include_rel = external_include.relative_to(hls_dir).as_posix()
    for path in copied_sources:
        source_rel = path.relative_to(hls_dir).as_posix()
        source_lines.append(f'add_files "./{source_rel}" -cflags "-I./include -I./{include_rel}"')
    source_text = "\n".join(source_lines)
    return f'''# Auto-generated by FPGAI external HLS integration
open_project -reset fpgai_external_hls
set_top {request.top_name}
add_files ./src/{request.top_name}.cpp -cflags "-I./include -I./{include_rel}"
{source_text}
add_files -tb ./src/tb.cpp -cflags "-I./include -I./{include_rel}"
open_solution -reset sol1
set_part {request.part}
create_clock -period {request.clock_period_ns} -name default
csim_design
csynth_design
export_design -format ip_catalog -vendor fpgai -version 1.0
exit
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def emit_external_hls_operator_project(request: ExternalHLSProjectRequest) -> ExternalHLSProjectResult:
    issues = list(validate_hls_integration_contract(request.contract))
    if request.input_words <= 0 or request.output_words <= 0:
        issues.append(HLSIntegrationIssue("HLSINT012", "tensor_words", "Input and output word counts must be positive"))
    if request.input_words != request.output_words:
        issues.append(HLSIntegrationIssue("HLSINT013", "tensor_words", "flat_array_v1 currently requires equal input and output sizes"))
    if issues:
        return ExternalHLSProjectResult(False, None, None, None, None, None, issues=tuple(issues))

    try:
        abi = parse_flat_array_abi(request.contract)
        out_dir = Path(request.out_dir)
        hls_dir = out_dir / "hls"
        if hls_dir.exists():
            shutil.rmtree(hls_dir)
        (hls_dir / "src").mkdir(parents=True)
        (hls_dir / "include").mkdir(parents=True)
        (out_dir / "reports").mkdir(parents=True, exist_ok=True)
        copied_sources, copied_headers, _src_root, include_root = _copy_files(request.contract, hls_dir)

        header = hls_dir / "include" / "external_wrapper.h"
        top_cpp = hls_dir / "src" / f"{request.top_name}.cpp"
        tb_cpp = hls_dir / "src" / "tb.cpp"
        run_tcl = hls_dir / "run_hls.tcl"
        header.write_text(_wrapper_header(request, abi), encoding="utf-8")
        top_cpp.write_text(_wrapper_source(request, abi), encoding="utf-8")
        tb_cpp.write_text(_testbench_source(request, abi), encoding="utf-8")
        run_tcl.write_text(_run_tcl(request, hls_dir, copied_sources, include_root), encoding="utf-8")

        report_path = out_dir / "reports" / "external_hls_integration.json"
        report = {
            "schema": "fpgai.external-hls-integration/v1",
            "status": "generated",
            "usage": {"platform_scope": "research", "production_path": "morfics"},
            "operator": {"name": request.operator_name, "attributes": dict(request.operator_attributes)},
            "implementation": request.contract.to_dict(),
            "abi": {"name": abi.abi, "scalar_type": abi.scalar_type, "input_words": request.input_words, "output_words": request.output_words},
            "ownership": {
                "package_sources": "user_owned_source_copied_read_only",
                "generated_wrapper": "compiler_owned",
                "package_root_modified": False,
            },
            "artifacts": [
                {"path": str(path), "sha256": _sha256(path), "kind": "copied_source"} for path in copied_sources
            ] + [
                {"path": str(path), "sha256": _sha256(path), "kind": "copied_header"} for path in copied_headers
            ] + [
                {"path": str(path), "sha256": _sha256(path), "kind": "generated"} for path in (header, top_cpp, tb_cpp, run_tcl)
            ],
            "validation": {
                "reference_testbench_generated": True,
                "c_simulation_requested": False,
                "hls_synthesis_requested": False,
                "tool_result": "not_run",
            },
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        generated = (header, top_cpp, tb_cpp, run_tcl, report_path)
        return ExternalHLSProjectResult(True, hls_dir, top_cpp, tb_cpp, run_tcl, report_path, copied_sources, copied_headers, generated)
    except ValueError as exc:
        code, _, message = str(exc).partition(": ")
        issue = HLSIntegrationIssue(code or "HLSINT014", "integration", message or str(exc))
        return ExternalHLSProjectResult(False, None, None, None, None, None, issues=(issue,))
