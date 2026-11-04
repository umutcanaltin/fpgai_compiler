"""Generate an AXI4-Stream + AXI4-Lite wrapper around canonical FPGAI HLS.

The original model.cpp, its constant weights, and native E1-E4 receipts are
never modified. A board wrapper uses *raw fixed-point bits* (NOT host float32):
one 32-bit AXI stream beat per model tensor element, little-endian on the PS.
Only a whole, statically shaped tensor is processed per invocation.
"""
from __future__ import annotations

from math import prod


def validate_tensor_contract(manifest: dict) -> tuple[int, int, int, int]:
    ins = manifest.get("input_shape")
    outs = manifest.get("output_shape")
    width = manifest.get("precision_bits")
    integer = manifest.get("integer_bits")
    if (not isinstance(ins, list) or not isinstance(outs, list)
            or not ins or not outs
            or any(type(x) is not int or x < 1 for x in ins + outs)
            or type(width) is not int or type(integer) is not int
            or width < 2 or width > 32 or integer < 1 or integer > width):
        raise ValueError("Board DMA requires positive static shapes and 2..32-bit fixed-point format")
    nin, nout = prod(ins), prod(outs)
    if not 1 <= nin <= 262144 or not 1 <= nout <= 262144:
        raise ValueError("Board DMA supports 1..262144 fixed-point values per tensor")
    return nin, nout, width, integer


def bridge_cpp(manifest: dict) -> str:
    nin, nout, width, integer = validate_tensor_contract(manifest)
    return f"""// EXPERIMENTAL FPGAI AXI DMA adapter. Emitted from source-attested model.cpp.
// Exactly one {nin}-element input and one {nout}-element output per invocation.
// 32-bit TDATA; low {width} bits represent ap_fixed<{width},{integer}>.
// Input TLAST is expected at element {nin - 1}. Output TLAST is asserted at {nout - 1}.
// Fixed-length input framing is the host's responsibility; not an error-resync protocol.
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include "model.cpp"
using dma_word_t = ap_axiu<32,0,0,0>;
extern "C" void fpgai_dma(hls::stream<dma_word_t> &axis_in,
                           hls::stream<dma_word_t> &axis_out) {{
#pragma HLS INTERFACE axis port=axis_in
#pragma HLS INTERFACE axis port=axis_out
#pragma HLS INTERFACE s_axilite port=return bundle=control
  data_t x[{nin}];
  data_t y[{nout}];
  for (int i=0; i<{nin}; ++i) {{
#pragma HLS PIPELINE II=1
    dma_word_t word=axis_in.read();
    x[i].range({width - 1},0)=word.data.range({width - 1},0);
  }}
  fpgai_model(x,y);
  for (int i=0; i<{nout}; ++i) {{
#pragma HLS PIPELINE II=1
    dma_word_t word;
    word.data=0;
    word.data.range({width - 1},0)=y[i].range({width - 1},0);
    word.keep=-1;
    word.strb=-1;
    word.last=(i=={nout - 1});
    axis_out.write(word);
  }}
}}
"""


def bridge_testbench_cpp(manifest: dict) -> str:
    nin, nout, width, integer = validate_tensor_contract(manifest)
    return f"""// Native Vitis CSim: verify AXIS element bits, framing and source model call.
#include <cstdio>
#include <ap_fixed.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
using data_t = ap_fixed<{width},{integer},AP_RND,AP_SAT>;
using dma_word_t = ap_axiu<32,0,0,0>;
extern "C" void fpgai_model(const data_t*, data_t*);
extern "C" void fpgai_dma(hls::stream<dma_word_t>&, hls::stream<dma_word_t>&);
int main() {{
  hls::stream<dma_word_t> src("src"), dest("dest");
  data_t x[{nin}], golden[{nout}];
  for (int i=0; i<{nin}; ++i) {{
    x[i]=data_t((i%7-3)/8.0);
    dma_word_t beat;
    beat.data=0;
    beat.data.range({width - 1},0)=x[i].range({width - 1},0);
    beat.keep=-1;
    beat.strb=-1;
    beat.last=(i=={nin - 1});
    src.write(beat);
  }}
  fpgai_model(x,golden);
  fpgai_dma(src,dest);
  for (int i=0; i<{nout}; ++i) {{
    if (dest.empty()) return 10;
    dma_word_t beat=dest.read();
    // Materialize each proxy into the same explicit unsigned bit-vector
    // before comparing. Vivado/Vitis HLS 2023.2 cannot choose operator!=
    // between ap_range_ref (TDATA) and af_range_ref (ap_fixed).
    ap_uint<{width}> observed_bits = beat.data.range({width - 1},0);
    ap_uint<{width}> expected_bits = golden[i].range({width - 1},0);
    if (observed_bits != expected_bits) return 11;
    if (beat.last!=(i=={nout - 1})) return 12;
    if (beat.keep!=15 || beat.strb!=15) return 13;
  }}
  if (!dest.empty()) return 14;
  std::puts("FPGAI_BOARD_DMA_CSIM_PASS");
  return 0;
}}
"""


def bridge_hls_tcl(*, part: str, clock_ns: float) -> str:
    from math import isfinite
    if not isfinite(clock_ns) or not 2.0 <= clock_ns <= 40.0:
        raise ValueError("Explicit HLS clock must be finite and 2..40 ns")
    if not isinstance(part, str) or not part.isascii() or not all(
        c.isalnum() or c in "-_" for c in part
    ):
        raise ValueError("Unsafe hardware part")
    return f"""# Generated experimental board IP export; separate from original OOC receipt.
open_project -reset fpgai_dma_hls_project
set_top fpgai_dma
add_files fpgai_dma.cpp
add_files -tb fpgai_dma_tb.cpp
open_solution solution1 -flow_target vivado
set_part {{{part}}}
create_clock -period {clock_ns:.6f} -name default
csim_design
csynth_design
export_design -format ip_catalog -rtl verilog
exit
"""
