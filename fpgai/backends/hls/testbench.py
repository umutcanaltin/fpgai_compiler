from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from fpgai.numerics.precision_policy import default_precision_policy


def _precision_bits(raw_cfg: Mapping[str, Any] | None) -> tuple[int, int]:
    policy = default_precision_policy(raw_cfg or {})
    activation = policy.get("activation", {})
    total_bits = int(activation.get("total_bits", 16))
    int_bits = int(activation.get("int_bits", 6))
    return total_bits, int_bits


def _common_helpers(*, act_bits: int, act_int_bits: int, out_words: int) -> str:
    act_per_axis = max(1, 32 // int(act_bits))
    frac_bits = int(act_bits) - int(act_int_bits)

    return f"""
static const int FPGAI_ACT_BITS = {int(act_bits)};
static const int FPGAI_ACT_INT_BITS = {int(act_int_bits)};
static const int FPGAI_ACT_FRAC_BITS = {int(frac_bits)};
static const int FPGAI_ACT_PER_AXIS = {int(act_per_axis)};
static const int FPGAI_OUTPUT_VALUES = {int(out_words)};

static unsigned int fpgai_float_to_raw(float value) {{
    const double scale = (double)(1ULL << FPGAI_ACT_FRAC_BITS);
    const long long qmin = -(1LL << (FPGAI_ACT_BITS - 1));
    const long long qmax =  (1LL << (FPGAI_ACT_BITS - 1)) - 1LL;

    double scaled_d = (double)value * scale;
    long long scaled = (long long)(scaled_d >= 0.0 ? scaled_d + 0.5 : scaled_d - 0.5);

    if (scaled < qmin) scaled = qmin;
    if (scaled > qmax) scaled = qmax;

    unsigned long long mask =
        (FPGAI_ACT_BITS >= 32)
            ? 0xffffffffULL
            : ((1ULL << FPGAI_ACT_BITS) - 1ULL);

    return (unsigned int)(((unsigned long long)scaled) & mask);
}}

static float fpgai_raw_to_float(unsigned int raw) {{
    unsigned int mask =
        (FPGAI_ACT_BITS >= 32)
            ? 0xffffffffU
            : ((1U << FPGAI_ACT_BITS) - 1U);

    raw &= mask;

    int signed_value = (int)raw;
    if (FPGAI_ACT_BITS < 32) {{
        unsigned int sign_bit = 1U << (FPGAI_ACT_BITS - 1);
        if (raw & sign_bit) {{
            signed_value = (int)(raw | (~mask));
        }}
    }}

    const float scale = (float)(1U << FPGAI_ACT_FRAC_BITS);
    return ((float)signed_value) / scale;
}}

static void push_packed_activations(
    hls::stream<axis_t>& stream,
    const std::vector<float>& values
) {{
    const int n = (int)values.size();

    for (int base = 0; base < n; base += FPGAI_ACT_PER_AXIS) {{
        axis_t pkt;
        pkt.data = 0;
        pkt.keep = -1;
        pkt.strb = -1;
        pkt.last = (base + FPGAI_ACT_PER_AXIS >= n) ? 1 : 0;

        for (int lane = 0; lane < FPGAI_ACT_PER_AXIS; ++lane) {{
            int index = base + lane;
            if (index < n) {{
                unsigned int raw = fpgai_float_to_raw(values[index]);
                unsigned int lo = lane * FPGAI_ACT_BITS;
                unsigned int hi = ((lane + 1) * FPGAI_ACT_BITS) - 1;
                pkt.data.range(hi, lo) = raw;
            }}
        }}

        stream.write(pkt);
    }}
}}

static std::vector<float> read_packed_activations(
    hls::stream<axis_t>& stream,
    int expected_values
) {{
    std::vector<float> output_data;
    output_data.reserve(expected_values);

    while (!stream.empty() && (int)output_data.size() < expected_values) {{
        axis_t pkt = stream.read();

        for (int lane = 0; lane < FPGAI_ACT_PER_AXIS; ++lane) {{
            int index = (int)output_data.size();
            if (index >= expected_values) break;

            unsigned int lo = lane * FPGAI_ACT_BITS;
            unsigned int hi = ((lane + 1) * FPGAI_ACT_BITS) - 1;
            unsigned int raw = pkt.data.range(hi, lo).to_uint();

            output_data.push_back(fpgai_raw_to_float(raw));
        }}
    }}

    return output_data;
}}
"""



def _cfg_get(raw: Mapping[str, Any] | None, path: str, default: Any = None) -> Any:
    if not isinstance(raw, Mapping):
        return default
    current: Any = raw
    for part in path.split('.'):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _normalise_token(value: Any) -> str:
    return str(value or '').strip().lower().replace('-', '_')


def _io_m_axi_requested(raw_cfg: Mapping[str, Any] | None, kind: str) -> bool:
    kind = str(kind).strip().lower()
    if kind == 'input':
        prefixes = ('data_movement.inputs.import', 'data_movement.inputs', 'data_movement.input.load')
    else:
        prefixes = ('data_movement.outputs.export', 'data_movement.outputs', 'data_movement.output.store')

    interface = ''
    policy = ''
    for prefix in prefixes:
        interface = interface or _normalise_token(_cfg_get(raw_cfg, f'{prefix}.interface', ''))
        policy = policy or _normalise_token(_cfg_get(raw_cfg, f'{prefix}.policy', ''))
        tiled = _cfg_get(raw_cfg, f'{prefix}.tiled', None)
        if not policy and isinstance(tiled, Mapping) and bool(tiled.get('enabled', False)):
            policy = 'tiled'
    return interface in {'m_axi', 'maxi', 'ddr'} and policy in {'', 'full', 'tiled'}

def emit_tb_cpp(
    tb_dir: Path,
    *,
    top_name: str,
    in_words: int,
    out_words: int,
    weights_mode: str,
    weight_words: int = 0,
    raw_cfg: Mapping[str, Any] | None = None,
) -> None:
    tb_path = tb_dir / "tb.cpp"
    mode = str(weights_mode).strip().lower()
    act_bits, act_int_bits = _precision_bits(raw_cfg)
    helpers = _common_helpers(
        act_bits=act_bits,
        act_int_bits=act_int_bits,
        out_words=out_words,
    )

    input_m_axi = _io_m_axi_requested(raw_cfg, "input")
    output_m_axi = _io_m_axi_requested(raw_cfg, "output")

    if mode in {"embedded", "compile_time", "static", ""} and (input_m_axi or output_m_axi):
        input_arg = "const ap_uint<32>* input_mem" if input_m_axi else "hls::stream<axis_t>& in_stream"
        output_arg = "ap_uint<32>* output_mem" if output_m_axi else "hls::stream<axis_t>& out_stream"
        input_decl = f"    std::vector<ap_uint<32> > input_mem({max(1, int(in_words))});\n" if input_m_axi else "    hls::stream<axis_t> in_stream;\n"
        output_decl = f"    std::vector<ap_uint<32> > output_mem({max(1, int(out_words))});\n" if output_m_axi else "    hls::stream<axis_t> out_stream;\n"
        input_load = f"""    for (int index = 0; index < {max(1, int(in_words))}; ++index) {{
        float value = (index < n_floats) ? input_data[index] : 0.0f;
        input_mem[index] = fpgai_float_to_bits(value);
    }}
""" if input_m_axi else "    push_packed_activations(in_stream, input_data);\n"
        output_capture = f"""    std::vector<float> output_data;
    output_data.reserve({max(1, int(out_words))});
    for (int index = 0; index < {max(1, int(out_words))}; ++index) {{
        output_data.push_back(fpgai_bits_to_float(output_mem[index].to_uint()));
    }}
""" if output_m_axi else "    std::vector<float> output_data = read_packed_activations(out_stream, FPGAI_OUTPUT_VALUES);\n"
        call_input = "input_mem.data()" if input_m_axi else "in_stream"
        call_output = "output_mem.data()" if output_m_axi else "out_stream"
        tb_text = f"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include "fpgai_params.h"

using std::size_t;
typedef ap_axis<32,0,0,0> axis_t;

extern "C" void {top_name}(
    {input_arg},
    {output_arg}
);

{helpers}

static unsigned int fpgai_float_to_bits(float value) {{
    union {{ float f; unsigned int i; }} u;
    u.f = value;
    return u.i;
}}

static float fpgai_bits_to_float(unsigned int value) {{
    union {{ unsigned int i; float f; }} u;
    u.i = value;
    return u.f;
}}

int main(int argc, char** argv) {{
    const char* in_path = "input.bin";
    const char* out_path = "output.bin";
    if (argc >= 2) in_path = argv[1];
    if (argc >= 3) out_path = argv[2];

    std::ifstream f(in_path, std::ios::binary);
    if (!f) {{
        fprintf(stderr, "[TB] Error: Could not open input file: %s\\n", in_path);
        return 1;
    }}

    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    int n_floats = size / sizeof(float);

    std::vector<float> input_data(n_floats);
    f.read(reinterpret_cast<char*>(input_data.data()), size);
    f.close();

    printf("[TB] Loaded %d inputs from %s\\n", n_floats, in_path);
    printf("[TB] Inference I/O ABI: input_m_axi=%d output_m_axi=%d weights_mode=embedded\\n", {1 if input_m_axi else 0}, {1 if output_m_axi else 0});

{input_decl}{output_decl}
{input_load}
    printf("[TB] Running inference...\\n");
    {top_name}({call_input}, {call_output});

{output_capture}
    printf("[TB] Received %zu outputs.\\n", output_data.size());

    std::ofstream of(out_path, std::ios::binary);
    if (!of) {{
        fprintf(stderr, "[TB] Error: Could not open output file for writing: %s\\n", out_path);
        return 3;
    }}
    of.write(reinterpret_cast<const char*>(output_data.data()), output_data.size() * sizeof(float));
    of.close();

    printf("[TB] Wrote %s\\n", out_path);
    return 0;
}}
"""
    elif mode in {"stream", "streamed"}:
        tb_text = f"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include "fpgai_params.h"

using std::size_t;
typedef ap_axis<32,0,0,0> axis_t;

extern "C" void {top_name}(
    hls::stream<axis_t>& in,
    hls::stream<axis_t>& out,
    hls::stream<axis_t>& weight_stream,
    int mode
);

{helpers}

int main(int argc, char** argv) {{
    const char* in_path = "input.bin";
    const char* out_path = "output.bin";
    if (argc >= 2) in_path = argv[1];
    if (argc >= 3) out_path = argv[2];

    std::ifstream f(in_path, std::ios::binary);
    if (!f) {{
        fprintf(stderr, "[TB] Error: Could not open input file: %s\\n", in_path);
        return 1;
    }}

    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    int n_floats = size / sizeof(float);

    std::vector<float> input_data(n_floats);
    f.read(reinterpret_cast<char*>(input_data.data()), size);
    f.close();

    printf("[TB] Loaded %d inputs from %s\\n", n_floats, in_path);
    printf("[TB] Precision AXIS packing: act_bits=%d act_int_bits=%d values_per_axis=%d\\n",
           FPGAI_ACT_BITS, FPGAI_ACT_INT_BITS, FPGAI_ACT_PER_AXIS);

    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;
    hls::stream<axis_t> weight_stream;

    const int expected_weight_words = {int(weight_words)};
    const int actual_weight_words = fpgai::fpgai_runtime_weight_word_count();
    printf("[TB] Preloading %d runtime weight words through AXI stream...\\n", actual_weight_words);
    if (expected_weight_words != 0 && expected_weight_words != actual_weight_words) {{
        fprintf(stderr, "[TB] Error: expected %d runtime weight words, generated %d\\n", expected_weight_words, actual_weight_words);
        return 2;
    }}

    fpgai::fpgai_preload_runtime_weights(weight_stream);
    {top_name}(in_stream, out_stream, weight_stream, 0);

    push_packed_activations(in_stream, input_data);

    printf("[TB] Running inference...\\n");
    {top_name}(in_stream, out_stream, weight_stream, 1);

    std::vector<float> output_data = read_packed_activations(out_stream, FPGAI_OUTPUT_VALUES);

    printf("[TB] Received %zu outputs.\\n", output_data.size());

    std::ofstream of(out_path, std::ios::binary);
    if (!of) {{
        fprintf(stderr, "[TB] Error: Could not open output file for writing: %s\\n", out_path);
        return 3;
    }}
    of.write(reinterpret_cast<const char*>(output_data.data()), output_data.size() * sizeof(float));
    of.close();

    printf("[TB] Wrote %s\\n", out_path);
    return 0;
}}
"""
    elif mode in {"ddr", "dma_ddr", "ddr_tiled", "runtime_ddr", "m_axi", "external_ddr", "uram"}:
        tb_text = f"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include "fpgai_params.h"

using std::size_t;
typedef ap_axis<32,0,0,0> axis_t;

extern "C" void {top_name}(
    hls::stream<axis_t>& in,
    hls::stream<axis_t>& out,
    const ap_uint<32>* weights_mem
);

{helpers}

int main(int argc, char** argv) {{
    const char* in_path = "input.bin";
    const char* out_path = "output.bin";
    if (argc >= 2) in_path = argv[1];
    if (argc >= 3) out_path = argv[2];

    std::ifstream f(in_path, std::ios::binary);
    if (!f) {{
        fprintf(stderr, "[TB] Error: Could not open input file: %s\\n", in_path);
        return 1;
    }}

    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    int n_floats = size / sizeof(float);

    std::vector<float> input_data(n_floats);
    f.read(reinterpret_cast<char*>(input_data.data()), size);
    f.close();

    printf("[TB] Loaded %d inputs from %s\\n", n_floats, in_path);
    printf("[TB] Precision AXIS packing: act_bits=%d act_int_bits=%d values_per_axis=%d\\n",
           FPGAI_ACT_BITS, FPGAI_ACT_INT_BITS, FPGAI_ACT_PER_AXIS);

    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;

    const int expected_weight_words = {int(weight_words)};
    const int actual_weight_words = fpgai::fpgai_runtime_weight_word_count();
    printf("[TB] Preparing %d runtime weight words in DDR buffer...\\n", actual_weight_words);
    if (expected_weight_words != 0 && expected_weight_words != actual_weight_words) {{
        fprintf(stderr, "[TB] Error: expected %d runtime weight words, generated %d\\n", expected_weight_words, actual_weight_words);
        return 2;
    }}

    std::vector<ap_uint<32> > weights_mem(actual_weight_words);
    fpgai::fpgai_fill_runtime_weight_words(weights_mem.data(), actual_weight_words);

    push_packed_activations(in_stream, input_data);

    printf("[TB] Running inference...\\n");
    {top_name}(in_stream, out_stream, weights_mem.data());

    std::vector<float> output_data = read_packed_activations(out_stream, FPGAI_OUTPUT_VALUES);

    printf("[TB] Received %zu outputs.\\n", output_data.size());

    std::ofstream of(out_path, std::ios::binary);
    if (!of) {{
        fprintf(stderr, "[TB] Error: Could not open output file for writing: %s\\n", out_path);
        return 3;
    }}
    of.write(reinterpret_cast<const char*>(output_data.data()), output_data.size() * sizeof(float));
    of.close();

    printf("[TB] Wrote %s\\n", out_path);
    return 0;
}}
"""
    else:
        tb_text = f"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

using std::size_t;
typedef ap_axis<32,0,0,0> axis_t;

extern "C" void {top_name}(hls::stream<axis_t>& in, hls::stream<axis_t>& out);

{helpers}

int main(int argc, char** argv) {{
    const char* in_path = "input.bin";
    const char* out_path = "output.bin";
    if (argc >= 2) in_path = argv[1];
    if (argc >= 3) out_path = argv[2];

    std::ifstream f(in_path, std::ios::binary);
    if (!f) {{
        fprintf(stderr, "[TB] Error: Could not open input file: %s\\n", in_path);
        return 1;
    }}

    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    int n_floats = size / sizeof(float);

    std::vector<float> input_data(n_floats);
    f.read(reinterpret_cast<char*>(input_data.data()), size);
    f.close();

    printf("[TB] Loaded %d inputs from %s\\n", n_floats, in_path);
    printf("[TB] Precision AXIS packing: act_bits=%d act_int_bits=%d values_per_axis=%d\\n",
           FPGAI_ACT_BITS, FPGAI_ACT_INT_BITS, FPGAI_ACT_PER_AXIS);

    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;

    push_packed_activations(in_stream, input_data);

    printf("[TB] Running {top_name}...\\n");
    {top_name}(in_stream, out_stream);

    std::vector<float> output_data = read_packed_activations(out_stream, FPGAI_OUTPUT_VALUES);

    printf("[TB] Received %zu outputs.\\n", output_data.size());

    std::ofstream of(out_path, std::ios::binary);
    if (!of) {{
        fprintf(stderr, "[TB] Error: Could not open output file for writing: %s\\n", out_path);
        return 3;
    }}
    of.write(reinterpret_cast<const char*>(output_data.data()), output_data.size() * sizeof(float));
    of.close();

    printf("[TB] Wrote %s\\n", out_path);
    return 0;
}}
"""

    tb_path.write_text(tb_text, encoding="utf-8")
