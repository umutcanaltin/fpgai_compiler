from __future__ import annotations

from pathlib import Path
from typing import List
from fpgai.ir.graph import Graph


def emit_tb_train_cpp(
    tb_dir: Path,
    *,
    graph: Graph,
    top_name: str,
    in_words: int,
    out_words: int,
    weights_mode: str,
    weight_words: int,
    preload_weights: List[float],
    training_cfg: dict,
) -> None:
    tb_path = tb_dir / "tb.cpp"

    preload_vals = ", ".join(f"{float(v):.8f}f" for v in preload_weights)

    tb_text = f"""\
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

typedef ap_axis<32,0,0,0> axis_t;

extern "C" void {top_name}(
    hls::stream<axis_t>& in,
    hls::stream<axis_t>& out,
    hls::stream<axis_t>& aux,
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

static std::vector<float> read_bin(const char* path) {{
    std::ifstream f(path, std::ios::binary);
    if (!f) {{
        fprintf(stderr, "[TB-TRAIN] Error: cannot open %s\\n", path);
        std::exit(1);
    }}
    f.seekg(0, std::ios::end);
    size_t size = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}}

static void write_bin(const char* path, const std::vector<float>& data) {{
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}}

int main(int argc, char** argv) {{
    const char* in_path = "input.bin";
    const char* target_path = "target.bin";

    if (argc >= 2) in_path = argv[1];
    if (argc >= 3) target_path = argv[2];

    std::vector<float> input_data = read_bin(in_path);
    std::vector<float> target_data = read_bin(target_path);

    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;
    hls::stream<axis_t> aux_stream;

    if ("{weights_mode}" == std::string("stream") || "{weights_mode}" == std::string("ddr")) {{
        std::vector<float> preload = {{ {preload_vals} }};
        for (size_t i = 0; i < preload.size(); ++i) {{
            push_f32(aux_stream, preload[i], i + 1 == preload.size());
        }}
        {top_name}(in_stream, out_stream, aux_stream, 0);
    }}

    for (size_t i = 0; i < input_data.size(); ++i) {{
        push_f32(in_stream, input_data[i], i + 1 == input_data.size());
    }}

    for (size_t i = 0; i < target_data.size(); ++i) {{
        push_f32(aux_stream, target_data[i], i + 1 == target_data.size());
    }}

    {top_name}(in_stream, out_stream, aux_stream, 2);

    std::vector<float> all_out;
    while (!out_stream.empty()) {{
        axis_t pkt = out_stream.read();
        union {{ unsigned int i; float f; }} u;
        u.i = pkt.data.to_uint();
        all_out.push_back(u.f);
    }}

    const int grad_words = {int(weight_words)};
    const int w_before_words = {int(weight_words)};
    const int w_after_words = {int(weight_words)};
    const int expected_total = grad_words + w_before_words + w_after_words;

    if ((int)all_out.size() != expected_total) {{
        fprintf(stderr, "[TB-TRAIN] Unexpected output words. got=%zu expected=%d\\n", all_out.size(), expected_total);
        return 2;
    }}

    std::vector<float> grads(all_out.begin(), all_out.begin() + grad_words);
    std::vector<float> w_before(all_out.begin() + grad_words, all_out.begin() + grad_words + w_before_words);
    std::vector<float> w_after(all_out.begin() + grad_words + w_before_words, all_out.end());

    write_bin("grads.bin", grads);
    write_bin("weights_before.bin", w_before);
    write_bin("weights_after.bin", w_after);

    printf("[TB-TRAIN] Wrote grads.bin, weights_before.bin, weights_after.bin\\n");
    return 0;
}}
"""
    tb_path.write_text(tb_text, encoding="utf-8")