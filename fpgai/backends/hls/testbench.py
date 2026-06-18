from __future__ import annotations
from pathlib import Path


def emit_tb_cpp(
    tb_dir: Path,
    *,
    top_name: str,
    in_words: int,
    out_words: int,
    weights_mode: str,
    weight_words: int = 0,
) -> None:
    tb_path = tb_dir / "tb.cpp"
    mode = str(weights_mode).strip().lower()

    if mode in {"stream", "streamed"}:
        tb_text = f"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

typedef ap_axis<32,0,0,0> axis_t;

extern "C" void {top_name}(
    hls::stream<axis_t>& in,
    hls::stream<axis_t>& out,
    hls::stream<axis_t>& weight_stream,
    int mode
);

static void push_f32(hls::stream<axis_t>& s, float v, bool last=false) {{
    union {{ float f; unsigned int i; }} u;
    u.f = v;

    axis_t pkt;
    pkt.data = u.i;
    pkt.keep = -1;
    pkt.strb = -1;
    pkt.last = last ? 1 : 0;
    s.write(pkt);
}}

int main(int argc, char** argv) {{
    const char* in_path = "input.bin";
    if (argc >= 2) in_path = argv[1];

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

    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;
    hls::stream<axis_t> weight_stream;

    const int weight_words = {int(weight_words)};
    printf("[TB] Preloading %d runtime weight words through AXI stream...\\n", weight_words);
    for (int i = 0; i < weight_words; i++) {{
        float v = 0.001f * (float)(i + 1);
        push_f32(weight_stream, v, i == weight_words - 1);
    }}

    {top_name}(in_stream, out_stream, weight_stream, 0);

    for (int i = 0; i < n_floats; i++) {{
        push_f32(in_stream, input_data[i], i == n_floats - 1);
    }}

    printf("[TB] Running inference...\\n");
    {top_name}(in_stream, out_stream, weight_stream, 1);

    std::vector<float> output_data;
    while (!out_stream.empty()) {{
        axis_t pkt = out_stream.read();
        union {{ unsigned int i; float f; }} u;
        u.i = pkt.data.to_uint();
        output_data.push_back(u.f);
    }}

    printf("[TB] Received %zu outputs.\\n", output_data.size());

    std::ofstream of("output.bin", std::ios::binary);
    of.write(reinterpret_cast<const char*>(output_data.data()), output_data.size() * sizeof(float));
    of.close();

    printf("[TB] Wrote output.bin\\n");
    return 0;
}}
"""
    elif mode in {"ddr", "dma_ddr"}:
        tb_text = f"""
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

typedef ap_axis<32,0,0,0> axis_t;

extern "C" void {top_name}(
    hls::stream<axis_t>& in,
    hls::stream<axis_t>& out,
    const ap_uint<32>* weights_mem
);

static axis_t make_axis_f32(float v, bool last=false) {{
    union {{ float f; unsigned int i; }} u;
    u.f = v;

    axis_t pkt;
    pkt.data = u.i;
    pkt.keep = -1;
    pkt.strb = -1;
    pkt.last = last ? 1 : 0;
    return pkt;
}}

static ap_uint<32> make_word_f32(float v) {{
    union {{ float f; unsigned int i; }} u;
    u.f = v;
    return ap_uint<32>(u.i);
}}

int main(int argc, char** argv) {{
    const char* in_path = "input.bin";
    if (argc >= 2) in_path = argv[1];

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

    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;

    const int weight_words = {int(weight_words)};
    printf("[TB] Preparing %d runtime weight words in DDR buffer...\\n", weight_words);
    std::vector<ap_uint<32> > weights_mem(weight_words);
    for (int i = 0; i < weight_words; i++) {{
        float v = 0.001f * (float)(i + 1);
        weights_mem[i] = make_word_f32(v);
    }}

    for (int i = 0; i < n_floats; i++) {{
        in_stream.write(make_axis_f32(input_data[i], i == n_floats - 1));
    }}

    printf("[TB] Running inference...\\n");
    {top_name}(in_stream, out_stream, weights_mem.data());

    std::vector<float> output_data;
    while (!out_stream.empty()) {{
        axis_t pkt = out_stream.read();
        union {{ unsigned int i; float f; }} u;
        u.i = pkt.data.to_uint();
        output_data.push_back(u.f);
    }}

    printf("[TB] Received %zu outputs.\\n", output_data.size());

    std::ofstream of("output.bin", std::ios::binary);
    of.write(reinterpret_cast<const char*>(output_data.data()), output_data.size() * sizeof(float));
    of.close();

    printf("[TB] Wrote output.bin\\n");
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

typedef ap_axis<32,0,0,0> axis_t;

extern "C" void {top_name}(hls::stream<axis_t>& in, hls::stream<axis_t>& out);

int main(int argc, char** argv) {{
    const char* in_path = "input.bin";
    if (argc >= 2) in_path = argv[1];

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

    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;

    for(int i = 0; i < n_floats; i++) {{
        union {{ float f; unsigned int i; }} u;
        u.f = input_data[i];

        axis_t pkt;
        pkt.data = u.i;
        pkt.keep = -1;
        pkt.strb = -1;
        pkt.last = (i == n_floats - 1);
        in_stream.write(pkt);
    }}

    printf("[TB] Running {top_name}...\\n");
    {top_name}(in_stream, out_stream);

    std::vector<float> output_data;
    while(!out_stream.empty()) {{
        axis_t pkt = out_stream.read();
        union {{ unsigned int i; float f; }} u;
        u.i = pkt.data.to_uint();
        output_data.push_back(u.f);
    }}

    printf("[TB] Received %zu outputs.\\n", output_data.size());

    std::ofstream of("output.bin", std::ios::binary);
    of.write(reinterpret_cast<const char*>(output_data.data()), output_data.size() * sizeof(float));
    of.close();

    printf("[TB] Wrote output.bin\\n");
    return 0;
}}
"""

    tb_path.write_text(tb_text, encoding="utf-8")
