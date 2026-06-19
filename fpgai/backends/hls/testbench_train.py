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

static float pop_f32(hls::stream<axis_t>& s) {{
    axis_t pkt = s.read();
    union {{ unsigned int i; float f; }} u;
    u.i = pkt.data.to_uint();
    return u.f;
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

static std::vector<float> drain_exact(
    hls::stream<axis_t>& out_stream,
    int expected_words,
    const char* label
) {{
    std::vector<float> values;
    while (!out_stream.empty()) {{
        values.push_back(pop_f32(out_stream));
    }}
    if ((int)values.size() != expected_words) {{
        fprintf(
            stderr,
            "[TB-TRAIN] Unexpected %s words. got=%zu expected=%d delta=%lld\\n",
            label,
            values.size(),
            expected_words,
            (long long)values.size() - (long long)expected_words
        );
        std::exit(2);
    }}
    return values;
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

    {top_name}(in_stream, out_stream, aux_stream, 1);
    std::vector<float> weights_before = drain_exact(out_stream, {int(weight_words)}, "weights_before");
    write_bin("weights_before.bin", weights_before);

    for (size_t i = 0; i < input_data.size(); ++i) {{
        push_f32(in_stream, input_data[i], i + 1 == input_data.size());
    }}

    for (size_t i = 0; i < target_data.size(); ++i) {{
        push_f32(aux_stream, target_data[i], i + 1 == target_data.size());
    }}

    {top_name}(in_stream, out_stream, aux_stream, 2);
    std::vector<float> grads = drain_exact(out_stream, {int(weight_words)}, "grads");
    write_bin("grads.bin", grads);

    {top_name}(in_stream, out_stream, aux_stream, 1);
    std::vector<float> weights_after = drain_exact(out_stream, {int(weight_words)}, "weights_after");
    write_bin("weights_after.bin", weights_after);

    printf("[TB-TRAIN] Wrote weights_before.bin (%zu floats)\\n", weights_before.size());
    printf("[TB-TRAIN] Wrote grads.bin (%zu floats)\\n", grads.size());
    printf("[TB-TRAIN] Wrote weights_after.bin (%zu floats)\\n", weights_after.size());
    return 0;
}}
"""
    tb_path.write_text(tb_text, encoding="utf-8")
