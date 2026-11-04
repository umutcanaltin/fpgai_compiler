"""Source-level row-interleaved HLS materialization for two MatMul operators.

A deliberately bounded compiler research backend: unlike the ordinary
whole-graph array emitter, the consumer uses each completed intermediate
row before the producer starts the next row. This changes the loop
schedule and bounds the intermediate scratch to one logical row.

This is not streaming/FIFO hardware, bit-exact HLS validation, an achieved
initiation interval, or placed-and-routed evidence.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.model_ir.row_lifetime import (
    RowInterleavedModelIR, row_lifetime_receipt,
)
from fpgai_research.model_ir.semantics import (
    functional_fingerprint, verify_refinement,
)


def _emit_row_matmul(src: list[str], *, ordinal: int, lo, input_array: str,
                     weight_array: str, output_array: str, input_k: int,
                     output_n: int, lhs_is_row_buffer: bool,
                     output_is_row_buffer: bool | None = None,
                     row_slot: str | None = None,
                     bias_array: str | None = None,
                     fanout_commit_tensor: str | None = None,
                     fanout_read_tensor: str | None = None,
                     fanout_read_consumer: str | None = None,
                     lhs_is_full_matrix: bool = False,
                     tie_fanout_trace_to_load: bool = False) -> None:
    """Generate one complete per-row MatMul reduction and unique output write."""
    _tm, tn, tk = lo.tile_shape
    width = lo.replica_count * lo.pe_count
    pipeline = (f"#pragma HLS PIPELINE II={lo.pipeline_ii}"
                if lo.pipeline_ii is not None else "")
    if row_slot is not None and (not isinstance(row_slot, str) or row_slot not in ("ri",)):
        raise ValueError("Unrecognized row-block scratch slot expression")
    if fanout_commit_tensor is not None or fanout_read_tensor is not None:
        import re
        for token in (fanout_commit_tensor,fanout_read_tensor,fanout_read_consumer):
            if token is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*",token):
                raise ValueError("Invalid fanout trace identifier")
    if (fanout_read_tensor is None) != (fanout_read_consumer is None):
        raise ValueError("Fanout source read requires both tensor and consumer")
    if lhs_is_full_matrix and (not lhs_is_row_buffer or row_slot is not None):
        raise ValueError("Full-matrix fanout reader requires flat scratch input")
    if tie_fanout_trace_to_load and (fanout_read_tensor is None or
                                         not lhs_is_full_matrix):
        raise ValueError("Tied graph source read index requires a traced flat matrix")
    if lhs_is_full_matrix:
        read_lhs = (f"{input_array}[fpgai_source_element]"
                    if tie_fanout_trace_to_load
                    else f"{input_array}[mi*{input_k}+kk]")
    elif lhs_is_row_buffer:
        read_lhs = (f"{input_array}[{row_slot}*{input_k}+kk]"
                    if row_slot is not None else f"{input_array}[kk]")
    else:
        read_lhs = f"{input_array}[mi*{input_k}+kk]"
    if output_is_row_buffer is None:
        output_is_row_buffer = ordinal == 0
    if output_is_row_buffer:
        write_dest = (f"{output_array}[{row_slot}*{output_n}+ni]"
                      if row_slot is not None else f"{output_array}[ni]")
    else:
        write_dest = f"{output_array}[mi*{output_n}+ni]"
    src.extend([
        f"    for (int nb=0; nb<{output_n}; nb+={tn}) {{",
        f"      for (int ob=0; ob<{tn}; ob+={width*lo.fold_factor}) {{",
        f"        for (int fold=0; fold<{lo.fold_factor}; ++fold) {{",
        f"          for (int replica=0; replica<{lo.replica_count}; ++replica) {{",
        "#pragma HLS UNROLL",
        f"            for (int pe=0; pe<{lo.pe_count}; ++pe) {{",
        "#pragma HLS UNROLL",
        f"              const int ni=nb+ob+fold*{width}+replica*{lo.pe_count}+pe;",
        f"              if (ni<{output_n} && ni<nb+{tn}) {{",
        (f"                data_t acc={bias_array}[ni];" if bias_array is not None
         else "                data_t acc=data_t(0);"),
        f"                for (int kb=0; kb<{input_k}; kb+={tk}) {{",
        f"                  for (int ko=0; ko<{tk}; ko+={lo.simd_lanes}) {{",
    ])
    if pipeline:
        src.append(pipeline)
    src.extend([
        f"                    data_t partial[{lo.simd_lanes}];",
        "#pragma HLS ARRAY_PARTITION variable=partial complete",
        f"                    for (int lane=0; lane<{lo.simd_lanes}; ++lane) {{",
        "#pragma HLS UNROLL",
        "                      const int kk=kb+ko+lane;",
        "                      partial[lane] = data_t(0);",
        f"                      if (kk<{input_k} && ko+lane<{tk}) {{",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_ITERATIONS)",
        f'                        std::printf("FPGAI_ITER {ordinal} %d %d %d %d %d %d %d\\n", mi,ni,kk,replica,pe,lane,fold);',
        "#endif",
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_TENSOR_VALUES)",
        f'                        std::printf("FPGAI_VALUE_READ {ordinal} %d %d\\n", mi,kk);',
        "#endif",
    ])
    if tie_fanout_trace_to_load:
        src.append(f"                        const int fpgai_source_element=mi*{input_k}+kk;")
    src.append(f"                        partial[lane]={read_lhs}*{weight_array}[kk*{output_n}+ni];")
    if fanout_read_tensor is not None:
        src.extend([
            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_BRANCH)",
            f'                        std::printf("FPGAI_BRANCH READ {fanout_read_tensor} %d {fanout_read_consumer} %d %d %d %d %d\\n",'+
            (f'fpgai_source_element,mi*{output_n}+ni,replica,pe,lane,fold);'
             if tie_fanout_trace_to_load else
             f'mi*{input_k}+kk,mi*{output_n}+ni,replica,pe,lane,fold);'),
            "#endif",
        ])
    src.extend([
        "                      }",
        "                    }",
        f"                    for (int lane=0; lane<{lo.simd_lanes}; ++lane) acc+=partial[lane];",
        "                  }",
        "                }",
        f"                {write_dest}=acc;",
    ])
    if fanout_commit_tensor is not None:
        src.extend([
            "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_FANOUT)",
            f'                std::printf("FPGAI_FANOUT COMMIT {fanout_commit_tensor} %d\\n",mi*{output_n}+ni);',
            "#endif",
        ])
    src.extend([
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_TENSOR_VALUES)",
        f'                std::printf("FPGAI_VALUE_READY {ordinal} %d %d\\n", mi,ni);',
        "#endif",
        "              }",
        "            }",
        "          }",
        "        }",
        "      }",
        "    }",
    ])


def emit_row_interleaved_hls(ir: RowInterleavedModelIR) -> Emission:
    if not isinstance(ir, RowInterleavedModelIR):
        raise TypeError("Expected typed row-interleaved Model IR")
    lowered = ir.base.lowered
    verify_refinement(lowered)
    f = lowered.architectural.functional
    first, second = f.operators
    lo0, lo1 = lowered.operators
    m, n, k = lo0.iteration_shape
    m1, out_n, k1 = lo1.iteration_shape
    if m1 != m or k1 != n or ir.row_contract.shape != (m, n):
        raise ValueError("Row-interleaving tensor geometry does not match Lowered Model IR")
    if len(f.inputs) != 1 or len(f.outputs) != 1:
        raise ValueError("Row-interleaved emitted top must have one input/output")
    global_format = {
        (p.effective.precision_bits, p.effective.integer_bits)
        for p in lowered.architectural.points
    }
    if len(global_format) != 1:
        raise ValueError("Mixed numerical formats require implemented cast lowering")
    width, integer = next(iter(global_format))
    w0 = f.tensor(first.inputs[1])
    w1 = f.tensor(second.inputs[1])
    if w0.shape != (k, n) or w1.shape != (n, out_n):
        raise ValueError("Unsupported MatMul weight geometry")
    src = [
        "// FPGAI experimental row-lifetime HLS, E1 intent not hardware QoR.",
        "#ifdef FPGAI_HOST_FLOAT",
        "#include <cmath>", "#include <cstdio>",
        "using data_t = float; // not bit-exact ap_fixed",
        "#else", "#include <ap_fixed.h>", "#include <hls_math.h>",
        f"using data_t = ap_fixed<{width}, {integer}, AP_RND, AP_SAT>;",
        "#endif", "",
    ]
    for label, tensor in (("w0", w0), ("w1", w1)):
        values = ", ".join(f"data_t({v:.9g})" for v in tensor.data)
        src.append(f"static const data_t {label}[{len(tensor.data)}] = {{{values}}};")
    src.extend([
        f'extern "C" void fpgai_model(const data_t input[{m*k}], data_t output[{m*out_n}]) {{',
        f"  data_t row_buffer[{n}];",
        f"  for (int mi=0; mi<{m}; ++mi) {{",
        "    // Producer completes all reductions for this row.",
    ])
    _emit_row_matmul(
        src, ordinal=0, lo=lo0, input_array="input", weight_array="w0",
        output_array="row_buffer", input_k=k, output_n=n,
        lhs_is_row_buffer=False,
    )
    src.extend([
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_ROWS)",
        '    std::printf("FPGAI_ROW_READY %d\\n", mi);',
        "#endif",
        "    // Only after the full producer row is complete, consumer may read.",
    ])
    _emit_row_matmul(
        src, ordinal=1, lo=lo1, input_array="row_buffer", weight_array="w1",
        output_array="output", input_k=n, output_n=out_n,
        lhs_is_row_buffer=True,
    )
    src.extend([
        "#if defined(FPGAI_HOST_FLOAT) && defined(FPGAI_TRACE_ROWS)",
        '    std::printf("FPGAI_ROW_CONSUMED %d\\n", mi);',
        "#endif",
        "  }",
        "}",
    ])
    cpp = "\n".join(src) + "\n"
    manifest = {
        "schema": "fpgai-research/hls-row-lifetime-e1/v1",
        "functional_sha256": functional_fingerprint(f),
        "cpp_sha256": sha256(cpp.encode("utf-8")).hexdigest(),
        "precision_bits": width, "integer_bits": integer,
        "input_shape": list(f.tensor(f.inputs[0]).shape),
        "output_shape": list(f.tensor(f.outputs[0]).shape),
        "operators": [{
            "operator": op.name, "kind": op.kind,
            "requested": asdict(p.requested),
            "effective": asdict(p.effective),
            "lowered_mechanisms": {
                "replicas": lo.replica_count, "pe_count": lo.pe_count,
                "simd_lanes": lo.simd_lanes, "fold_factor": lo.fold_factor,
                "tile_shape": list(lo.tile_shape),
                "pipeline_ii": lo.pipeline_ii,
            },
        } for op, p, lo in zip(f.operators, lowered.architectural.points,
                               lowered.operators)],
        "row_lifetime_contract": row_lifetime_receipt(ir),
        "intermediate_scratch_elements": n,
        "full_materialized_intermediate_elements": m*n,
        "exchange_policy": "producer_row_complete_then_consumer_row",
        "evidence": "E1-generated-HLS-row-schedule-only",
        "physical_schedule_verified": False,
        "vendor_tools_executed": False,
    }
    return Emission(cpp=cpp, manifest=manifest)
