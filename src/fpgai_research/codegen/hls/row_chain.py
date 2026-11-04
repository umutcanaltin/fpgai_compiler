"""Actual HLS source emitter for typed linear neural-network row stages.

Unlike a metadata-only lifetime planner, emits one C++ row loop with N
independently mapped MatMul nests and exactly N-1 intermediate row buffers.
This implementation explicitly does not generate concurrent DATAFLOW/FIFOs.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from math import prod

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.codegen.hls.row_interleaved import _emit_row_matmul
from fpgai_research.model_ir.row_chain import RowChainIR, row_chain_receipt
from fpgai_research.model_ir.semantics import functional_fingerprint, verify_refinement


def emit_row_chain_hls(ir: RowChainIR, *, row_block: int = 1) -> Emission:
    if not isinstance(ir, RowChainIR):
        raise TypeError("Expected typed RowChainIR")
    if type(row_block) is not int or not 1 <= row_block <= ir.row_count:
        raise ValueError("Invalid row-block lifetime extent")
    lowered = ir.composed.lowered
    verify_refinement(lowered)
    f = lowered.architectural.functional
    ops = f.operators
    if len(ops) != len(ir.chain) or tuple(x.name for x in ops) != ir.chain:
        raise ValueError("Row-chain compilation operator identities differ")
    widths = {(p.effective.precision_bits, p.effective.integer_bits)
              for p in lowered.architectural.points}
    if len(widths) != 1:
        raise ValueError("Mixed numerical formats are unsupported in row-chain lowering")
    precision, integer = next(iter(widths))
    input_shape = f.tensor(f.inputs[0]).shape
    m, original_k = input_shape[0], prod(input_shape[1:])
    output_n = f.tensor(f.outputs[0]).shape[1]
    src = [
        "// FPGAI experimental compositional row-chain Model IR; E1 source.",
        "#ifdef FPGAI_HOST_FLOAT",
        "#include <cmath>", "#include <cstdio>",
        "using data_t = float; // Not ap_fixed bit-exact",
        "#else", "#include <ap_fixed.h>", "#include <hls_math.h>",
        f"using data_t = ap_fixed<{precision}, {integer}, AP_RND, AP_SAT>;",
        "#endif", "",
    ]
    for i, op in enumerate(ops):
        if op.kind not in ("MatMul", "Gemm"):
            continue
        w = f.tensor(op.inputs[1])
        if w.data is None:
            raise ValueError("Row-chain HLS needs constant weights")
        if op.kind == "Gemm" and int(op.attr("transB", 0)) == 1:
            output_width, inner_width = w.shape
            canonical = [
                w.data[ni*inner_width+ki]
                for ki in range(inner_width) for ni in range(output_width)
            ]
        else:
            canonical = w.data
        values = ", ".join(f"data_t({v:.9g})" for v in canonical)
        src.append(f"static const data_t w{i}[{len(canonical)}] = {{{values}}};")
        if op.kind == "Gemm" and len(op.inputs) == 3:
            bias = f.tensor(op.inputs[2])
            if bias.data is None:
                raise ValueError("Gemm row-chain bias must be static")
            values = ", ".join(f"data_t({v:.9g})" for v in bias.data)
            src.append(f"static const data_t bias{i}[{len(bias.data)}] = {{{values}}};")
    src.append(
        f'extern "C" void fpgai_model(const data_t input[{m*original_k}], '
        f'data_t output[{m*output_n}]) {{'
    )
    for i, (_name, width) in enumerate(ir.scratch_by_tensor):
        src.append(f"  data_t scratch_{i}[{row_block*width}];")
    if row_block == 1:
        src.append(f"  for (int mi=0; mi<{m}; ++mi) {{")
    else:
        src.append(f"  for (int rb=0; rb<{m}; rb+={row_block}) {{")
    manifest_operators = []
    for ordinal, (op, lo, point) in enumerate(
            zip(ops, lowered.operators, lowered.architectural.points)):
        if lo.iteration_shape[0] != m:
            raise ValueError("Row-chain pipeline changed source row dimension")
        input_array = "input" if ordinal == 0 else f"scratch_{ordinal-1}"
        output_array = "output" if ordinal == len(ops)-1 else f"scratch_{ordinal}"
        src.append(f"    // {op.name}: {op.kind}; complete row before next stage")
        if row_block > 1:
            src.extend([
                f"    for (int ri=0; ri<{row_block}; ++ri) {{",
                "      const int mi = rb + ri;",
                f"      if (mi >= {m}) continue;",
            ])
        if op.kind in ("MatMul", "Gemm"):
            rows, n, k = lo.iteration_shape
            _emit_row_matmul(
                src, ordinal=ordinal, lo=lo,
                input_array=input_array, weight_array=f"w{ordinal}",
                output_array=output_array, input_k=k, output_n=n,
                lhs_is_row_buffer=(ordinal > 0),
                output_is_row_buffer=(ordinal < len(ops)-1),
                row_slot="ri" if row_block > 1 else None,
                bias_array=(f"bias{ordinal}" if op.kind == "Gemm"
                            and len(op.inputs) == 3 else None),
            )
        elif op.kind in ("Relu", "Flatten"):
            if (lo.replica_count, lo.pe_count, lo.simd_lanes,
                    lo.fold_factor) != (1, 1, 1, 1):
                raise ValueError("Unary row stages need an implemented spatial mapping")
            n = f.tensor(op.outputs[0]).shape[1]
            if f.tensor(op.inputs[0]).shape[0] != m:
                raise ValueError("Unary stage input changed row dimension")
            input_width = prod(f.tensor(op.inputs[0]).shape[1:])
            if input_width != n:
                raise ValueError("Unary stage must preserve per-row element count")
            read = (f"{input_array}[mi*{input_width}+ni]" if ordinal == 0
                    else (f"{input_array}[ri*{n}+ni]" if row_block > 1
                          else f"{input_array}[ni]"))
            write = (f"{output_array}[mi*{n}+ni]" if ordinal == len(ops)-1
                     else (f"{output_array}[ri*{n}+ni]" if row_block > 1
                           else f"{output_array}[ni]"))
            src.append(f"    for (int ni=0; ni<{n}; ++ni) {{")
            if lo.pipeline_ii is not None:
                src.append(f"#pragma HLS PIPELINE II={lo.pipeline_ii}")
            src.append(f"      data_t value = {read};")
            src.extend([
                "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_TENSOR_VALUES)",
                f'      std::printf("FPGAI_VALUE_READ {ordinal} %d %d\\n", mi,ni);',
                "#endif",
            ])
            if op.kind == "Relu":
                src.append("      if (value < data_t(0)) value = data_t(0);")
            src.append(f"      {write} = value;")
            src.extend([
                "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_TENSOR_VALUES)",
                f'      std::printf("FPGAI_VALUE_READY {ordinal} %d %d\\n", mi,ni);',
                "#endif",
            ])
            src.append("    }")
        else:
            raise ValueError("Unsupported nonlinear row-chain stage")
        if ordinal < len(ops)-1:
            src.extend([
                "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_ROWS)",
                f'    std::printf("FPGAI_CHAIN_ROW_READY {ordinal} %d\\n", mi);',
                "#endif",
            ])
        if ordinal > 0:
            src.extend([
                "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_ROWS)",
                f'    std::printf("FPGAI_CHAIN_ROW_CONSUMED {ordinal-1} %d\\n", mi);',
                "#endif",
            ])
        if row_block > 1:
            src.append("    }")
        manifest_operators.append({
            "operator": op.name, "kind": op.kind,
            "requested": asdict(point.requested),
            "effective": asdict(point.effective),
            "lowered_mechanisms": {
                "replicas": lo.replica_count,
                "pe_count": lo.pe_count,
                "simd_lanes": lo.simd_lanes,
                "fold_factor": lo.fold_factor,
                "tile_shape": list(lo.tile_shape),
                "pipeline_ii": lo.pipeline_ii,
            },
        })
    src.extend(["  }", "}"])
    cpp = "\n".join(src) + "\n"
    return Emission(cpp, {
        "schema": "fpgai-research/hls-row-chain-e1/v1",
        "functional_sha256": functional_fingerprint(f),
        "cpp_sha256": sha256(cpp.encode()).hexdigest(),
        "precision_bits": precision,
        "integer_bits": integer,
        "input_shape": list(f.tensor(f.inputs[0]).shape),
        "output_shape": list(f.tensor(f.outputs[0]).shape),
        "operators": manifest_operators,
        "row_chain_contract": row_chain_receipt(ir),
        "row_block": row_block,
        "source_intermediate_scratch_elements": row_block*ir.physical_scratch_elements,
        "source_full_intermediate_elements": ir.full_intermediate_elements,
        "evidence": "E1-generated-row-chain-HLS-source-only",
        "native_vitis_executed": False,
        "physical_hardware_buffer_proven": False,
    })
