from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fpgai.ir.graph import Graph


def _cpp_string_literal(value: str) -> str:
    return str(value).replace('\\', '\\\\').replace('"', '\\"')


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
    output_dir: Optional[str] = None,
) -> None:
    """Emit the Vitis HLS C-simulation testbench for one training step.

    Sprint 13B v2:
    - Supports embedded and runtime-weight training modes.
    - Accepts an explicit output directory through argv[3].
    - Also writes compatibility copies in the CSim working directory.
    - Fails CSim loudly if runtime preload size or file writes are wrong.
    """

    del graph, in_words, out_words, training_cfg

    tb_path = tb_dir / "tb.cpp"
    normalized_mode = str(weights_mode).strip().lower()
    preload_vals = ", ".join(f"{float(v):.8f}f" for v in preload_weights)
    default_out_dir = _cpp_string_literal(str(output_dir or "."))

    runtime_mode_expr = (
        f'(std::string("{_cpp_string_literal(normalized_mode)}") == "stream" || '
        f'std::string("{_cpp_string_literal(normalized_mode)}") == "streamed" || '
        f'std::string("{_cpp_string_literal(normalized_mode)}") == "ddr" || '
        f'std::string("{_cpp_string_literal(normalized_mode)}") == "dma_ddr")'
    )

    tb_text = f"""\
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

typedef ap_axis<32,0,0,0> axis_t;

extern "C" void {top_name}(
    hls::stream<axis_t>& in,
    hls::stream<axis_t>& out,
    hls::stream<axis_t>& aux,
    int mode
);

static std::string join_path(const std::string& dir, const char* name) {{
    if (dir.empty() || dir == ".") return std::string(name);
    const char last = dir[dir.size() - 1];
    // Avoid a generated C++ backslash character literal. ASCII 92 is backslash.
    if (last == '/' || ((int)last) == 92) return dir + name;
    return dir + "/" + name;
}}

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

static void write_one_bin(const std::string& path, const std::vector<float>& data) {{
    std::ofstream f(path.c_str(), std::ios::binary);
    if (!f) {{
        fprintf(stderr, "[TB-TRAIN] Error: cannot write %s\\n", path.c_str());
        std::exit(3);
    }}
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    if (!f) {{
        fprintf(stderr, "[TB-TRAIN] Error: failed while writing %s\\n", path.c_str());
        std::exit(3);
    }}
}}

static void write_bin_both(const std::string& out_dir, const char* name, const std::vector<float>& data) {{
    write_one_bin(join_path(out_dir, name), data);
    if (!(out_dir.empty() || out_dir == ".")) {{
        write_one_bin(std::string(name), data);
    }}
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

    std::string out_dir = "{default_out_dir}";
    if (argc >= 4) out_dir = argv[3];

    printf("[TB-TRAIN] input=%s\\n", in_path);
    printf("[TB-TRAIN] target=%s\\n", target_path);
    printf("[TB-TRAIN] output_dir=%s\\n", out_dir.c_str());
    printf("[TB-TRAIN] mode={_cpp_string_literal(normalized_mode)} expected_weight_words={int(weight_words)}\\n");

    std::vector<float> input_data = read_bin(in_path);
    std::vector<float> target_data = read_bin(target_path);

    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;
    hls::stream<axis_t> aux_stream;

    if ({runtime_mode_expr}) {{
        std::vector<float> preload = {{ {preload_vals} }};
        if ((int)preload.size() != {int(weight_words)}) {{
            fprintf(
                stderr,
                "[TB-TRAIN] Runtime preload size mismatch. got=%zu expected=%d mode={_cpp_string_literal(normalized_mode)}\\n",
                preload.size(),
                {int(weight_words)}
            );
            std::exit(4);
        }}
        for (size_t i = 0; i < preload.size(); ++i) {{
            push_f32(aux_stream, preload[i], i + 1 == preload.size());
        }}
        {top_name}(in_stream, out_stream, aux_stream, 0);
        printf("[TB-TRAIN] Preloaded runtime weights (%zu floats)\\n", preload.size());
    }}

    {top_name}(in_stream, out_stream, aux_stream, 1);
    std::vector<float> weights_before = drain_exact(out_stream, {int(weight_words)}, "weights_before");
    write_bin_both(out_dir, "weights_before.bin", weights_before);

    for (size_t i = 0; i < input_data.size(); ++i) {{
        push_f32(in_stream, input_data[i], i + 1 == input_data.size());
    }}
    for (size_t i = 0; i < target_data.size(); ++i) {{
        push_f32(aux_stream, target_data[i], i + 1 == target_data.size());
    }}

    {top_name}(in_stream, out_stream, aux_stream, 2);
    std::vector<float> grads = drain_exact(out_stream, {int(weight_words)}, "grads");
    write_bin_both(out_dir, "grads.bin", grads);

    {top_name}(in_stream, out_stream, aux_stream, 1);
    std::vector<float> weights_after = drain_exact(out_stream, {int(weight_words)}, "weights_after");
    write_bin_both(out_dir, "weights_after.bin", weights_after);

    printf("[TB-TRAIN] Wrote %s (%zu floats)\\n", join_path(out_dir, "weights_before.bin").c_str(), weights_before.size());
    printf("[TB-TRAIN] Wrote %s (%zu floats)\\n", join_path(out_dir, "grads.bin").c_str(), grads.size());
    printf("[TB-TRAIN] Wrote %s (%zu floats)\\n", join_path(out_dir, "weights_after.bin").c_str(), weights_after.size());
    return 0;
}}
"""
    tb_path.write_text(tb_text, encoding="utf-8")
