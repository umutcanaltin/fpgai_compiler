from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from fpgai.ir.graph import Graph


def _cpp_string_literal(value: str) -> str:
    return str(value).replace('\\', '\\\\').replace('"', '\\"')


def _cfg_lookup(cfg: Any, *paths: str, default: Any = None) -> Any:
    """Best-effort nested config lookup for dict/object compiler configs."""
    for path in paths:
        cur: Any = cfg
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            elif hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return default


def _as_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return max(int(minimum), out)


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
    """Emit the Vitis HLS C-simulation testbench for training.

    Sprint 13C:
    - Keeps the Sprint 13B one-step path unchanged when epochs/train_steps=1 and batch_size=1.
    - Supports runtime preload for stream/ddr modes from argv[3].
    - Supports small multi-step smoke tests by repeating mode=2 training calls.
    - Supports batch/microbatch replay by cycling through records in input.bin/target.bin.

    This is a smoke-test path, not full accumulated mini-batch SGD. Each replayed sample
    calls the training kernel once and therefore validates repeated stateful update execution.
    """
    del graph

    tb_path = tb_dir / "tb.cpp"
    normalized_mode = str(weights_mode).strip().lower()
    preload_vals = ", ".join(f"{float(v):.8f}f" for v in preload_weights)
    default_out_dir = _cpp_string_literal(str(output_dir or "."))

    # Existing sweeps use training.execution.epochs and training.execution.batch_size.
    # For 13C, train_steps can be explicit, but epochs also works as the step count.
    train_steps = _as_int(
        _cfg_lookup(
            training_cfg,
            "execution.train_steps",
            "train_steps",
            "execution.steps",
            "steps",
            "execution.epochs",
            "epochs",
            default=1,
        ),
        default=1,
    )
    batch_size = _as_int(
        _cfg_lookup(
            training_cfg,
            "execution.batch_size",
            "batch_size",
            "execution.micro_batch_size",
            "micro_batch_size",
            default=1,
        ),
        default=1,
    )

    runtime_mode_expr = (
        f'(std::string("{_cpp_string_literal(normalized_mode)}") == "stream" || '
        f'std::string("{_cpp_string_literal(normalized_mode)}") == "streamed" || '
        f'std::string("{_cpp_string_literal(normalized_mode)}") == "ddr" || '
        f'std::string("{_cpp_string_literal(normalized_mode)}") == "dma_ddr" || '
        f'std::string("{_cpp_string_literal(normalized_mode)}") == "external_ddr")'
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
    if (last == '/' || ((int)last) == 92) return dir + name;
    return dir + "/" + name;
}}

static void mkdir_best_effort(const std::string& path) {{
    if (path.empty() || path == ".") return;
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0777);
#endif
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

static void write_text_file(const std::string& path, const std::string& text) {{
    std::ofstream f(path.c_str());
    if (!f) {{
        fprintf(stderr, "[TB-TRAIN] Error: cannot write %s\\n", path.c_str());
        std::exit(3);
    }}
    f << text;
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

static void push_record(
    hls::stream<axis_t>& s,
    const std::vector<float>& data,
    int words_per_record,
    int record_index
) {{
    if (words_per_record <= 0) return;
    int n_records = (int)data.size() / words_per_record;
    if (n_records <= 0) {{
        fprintf(stderr, "[TB-TRAIN] Input/target record count is zero. words_per_record=%d size=%zu\\n", words_per_record, data.size());
        std::exit(5);
    }}
    int rec = record_index % n_records;
    int base = rec * words_per_record;
    for (int i = 0; i < words_per_record; ++i) {{
        push_f32(s, data[(size_t)(base + i)], i + 1 == words_per_record);
    }}
}}

int main(int argc, char** argv) {{
    const char* in_path = "input.bin";
    const char* target_path = "target.bin";
    const char* preload_path = "__FPGAI_NO_PRELOAD__";
    if (argc >= 2) in_path = argv[1];
    if (argc >= 3) target_path = argv[2];
    if (argc >= 4) preload_path = argv[3];
    std::string out_dir = "{default_out_dir}";
    if (argc >= 5) out_dir = argv[4];
    mkdir_best_effort(out_dir);

    printf("[TB-TRAIN] input=%s\\n", in_path);
    printf("[TB-TRAIN] target=%s\\n", target_path);
    printf("[TB-TRAIN] preload_path=%s\\n", preload_path);
    printf("[TB-TRAIN] output_dir=%s\\n", out_dir.c_str());
    printf("[TB-TRAIN] mode={_cpp_string_literal(normalized_mode)} expected_weight_words={int(weight_words)}\\n");
    printf("[TB-TRAIN] train_steps={int(train_steps)} batch_size={int(batch_size)} in_words={int(in_words)} out_words={int(out_words)}\\n");

    std::vector<float> input_data = read_bin(in_path);
    std::vector<float> target_data = read_bin(target_path);
    int input_words_per_record = {int(in_words)};
    if (input_words_per_record <= 0) input_words_per_record = (int)input_data.size();
    int target_words_per_record = {int(out_words)};
    if (target_words_per_record <= 0) target_words_per_record = (int)target_data.size();

    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;
    hls::stream<axis_t> aux_stream;

    if ({runtime_mode_expr}) {{
        std::vector<float> preload;
        std::string preload_s(preload_path ? preload_path : "");
        if (!preload_s.empty() && preload_s != "__FPGAI_NO_PRELOAD__") {{
            preload = read_bin(preload_path);
            printf("[TB-TRAIN] Loaded runtime preload from %s (%zu floats)\\n", preload_path, preload.size());
        }} else {{
            preload = std::vector<float>{{ {preload_vals} }};
            printf("[TB-TRAIN] Loaded compile-time preload (%zu floats)\\n", preload.size());
        }}
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

    std::vector<float> last_grads;
    int total_train_calls = 0;
    for (int step = 0; step < {int(train_steps)}; ++step) {{
        for (int b = 0; b < {int(batch_size)}; ++b) {{
            int rec = step * {int(batch_size)} + b;
            push_record(in_stream, input_data, input_words_per_record, rec);
            push_record(aux_stream, target_data, target_words_per_record, rec);
            {top_name}(in_stream, out_stream, aux_stream, 2);
            last_grads = drain_exact(out_stream, {int(weight_words)}, "grads");
            total_train_calls += 1;
        }}
    }}
    if (last_grads.empty()) {{
        fprintf(stderr, "[TB-TRAIN] No train calls were executed. train_steps={int(train_steps)} batch_size={int(batch_size)}\\n");
        std::exit(6);
    }}
    write_bin_both(out_dir, "grads.bin", last_grads);

    {top_name}(in_stream, out_stream, aux_stream, 1);
    std::vector<float> weights_after = drain_exact(out_stream, {int(weight_words)}, "weights_after");
    write_bin_both(out_dir, "weights_after.bin", weights_after);

    std::string summary = "{{\\n";
    summary += "  \\\"mode\\\": \\\"{_cpp_string_literal(normalized_mode)}\\\",\\n";
    summary += "  \\\"train_steps\\\": " + std::to_string({int(train_steps)}) + ",\\n";
    summary += "  \\\"batch_size\\\": " + std::to_string({int(batch_size)}) + ",\\n";
    summary += "  \\\"total_train_calls\\\": " + std::to_string(total_train_calls) + ",\\n";
    summary += "  \\\"weight_words\\\": " + std::to_string({int(weight_words)}) + ",\\n";
    summary += "  \\\"in_words\\\": " + std::to_string({int(in_words)}) + ",\\n";
    summary += "  \\\"out_words\\\": " + std::to_string({int(out_words)}) + "\\n";
    summary += "}}\\n";
    write_text_file(join_path(out_dir, "training_multistep_summary.json"), summary);
    if (!(out_dir.empty() || out_dir == ".")) {{
        write_text_file("training_multistep_summary.json", summary);
    }}

    printf("[TB-TRAIN] Wrote %s (%zu floats)\\n", join_path(out_dir, "weights_before.bin").c_str(), weights_before.size());
    printf("[TB-TRAIN] Wrote %s (%zu floats)\\n", join_path(out_dir, "grads.bin").c_str(), last_grads.size());
    printf("[TB-TRAIN] Wrote %s (%zu floats)\\n", join_path(out_dir, "weights_after.bin").c_str(), weights_after.size());
    printf("[TB-TRAIN] Multi-step summary: train_steps=%d batch_size=%d total_train_calls=%d\\n", {int(train_steps)}, {int(batch_size)}, total_train_calls);
    return 0;
}}
"""
    tb_path.write_text(tb_text, encoding="utf-8")
