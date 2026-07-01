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

    Training accumulation mode:
    - Keeps batch replay mode by default.
    - Adds an accumulated-gradient CSim smoke mode for runtime-weight designs.
    - In accumulated mode, gradients for each microbatch sample are computed from the same
      base weights, averaged in the HLS testbench, and a reference SGD update is emitted.

    Native accumulation mode:
    - In accumulated mode, the testbench drives native HLS top modes:
      mode 3 = accumulate gradients, mode 4 = average/apply update, mode 5 = reset.
    - This validates that the generated HLS top owns the mini-batch optimizer update.

    Training convergence mode:
    - Adds multi-epoch convergence-smoke instrumentation.
    - The testbench calls generated HLS top mode 6 to emit evaluation loss without updating weights.
    - It records initial/final/epoch losses in training_multistep_summary.json.
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

    batch_mode = str(
        _cfg_lookup(
            training_cfg,
            "execution.batch_mode",
            "batch_mode",
            "optimizer.batch_mode",
            default="replay",
        )
    ).strip().lower()
    accumulated_batch = batch_mode in {"accumulate", "accumulated", "true_minibatch", "mini_batch", "minibatch"}
    learning_rate = float(
        _cfg_lookup(
            training_cfg,
            "optimizer.learning_rate",
            "learning_rate",
            default=0.01,
        )
    )

    convergence_smoke = bool(
        _cfg_lookup(
            training_cfg,
            "execution.convergence_smoke",
            "convergence_smoke",
            "validation.convergence_smoke",
            default=False,
        )
    )
    loss_eval_records = _as_int(
        _cfg_lookup(
            training_cfg,
            "execution.loss_eval_records",
            "loss_eval_records",
            "validation.loss_eval_records",
            default=batch_size,
        ),
        default=batch_size,
    )

    runtime_mode_expr = (
        f'(std::string("{_cpp_string_literal(normalized_mode)}") == "stream" || '
        f'std::string("{_cpp_string_literal(normalized_mode)}") == "streamed" || '
        f'std::string("{_cpp_string_literal(normalized_mode)}") == "ddr" || '
        f'std::string("{_cpp_string_literal(normalized_mode)}") == "dma_ddr" || '
        f'std::string("{_cpp_string_literal(normalized_mode)}") == "external_ddr" || std::string("{_cpp_string_literal(normalized_mode)}") == "ddr_tiled" || std::string("{_cpp_string_literal(normalized_mode)}") == "ddr_tiled_mutable")'
    )
    m_axi_weight_runtime = normalized_mode in {"ddr", "dma_ddr", "external_ddr", "ddr_tiled", "ddr_tiled_mutable"}
    extern_weights_arg = "    ap_uint<32>* weights_mem,\n" if m_axi_weight_runtime else ""
    call_weights_arg = "weights_mem.data(), " if m_axi_weight_runtime else ""
    weight_mem_decl = f"    std::vector<ap_uint<32> > weights_mem({int(weight_words)});\n" if m_axi_weight_runtime else ""
    weight_mem_pack = """        for (size_t i = 0; i < preload.size(); ++i) {
            union { float f; unsigned int i; } u;
            u.f = preload[i];
            weights_mem[i] = u.i;
        }
""" if m_axi_weight_runtime else ""
    aux_preload_push = "" if m_axi_weight_runtime else """        for (size_t i = 0; i < preload.size(); ++i) {
            push_f32(aux_stream, preload[i], i + 1 == preload.size());
        }
"""


    tb_text = f"""\
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
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
{extern_weights_arg}    int mode
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
{weight_mem_decl}
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
{weight_mem_pack}{aux_preload_push}        {top_name}(in_stream, out_stream, aux_stream, {call_weights_arg}0);
        printf("[TB-TRAIN] Preloaded runtime weights (%zu floats)\\n", preload.size());
    }}

    {top_name}(in_stream, out_stream, aux_stream, {call_weights_arg}1);
    std::vector<float> weights_before = drain_exact(out_stream, {int(weight_words)}, "weights_before");
    write_bin_both(out_dir, "weights_before.bin", weights_before);

    const bool convergence_smoke = {str(convergence_smoke).lower()};
    const int loss_eval_records = {int(loss_eval_records)};
    std::vector<float> epoch_losses;
    float initial_loss = -1.0f;
    float final_loss = -1.0f;

    auto evaluate_loss = [&]() -> float {{
        int input_records = input_words_per_record > 0 ? ((int)input_data.size() / input_words_per_record) : 0;
        int target_records = target_words_per_record > 0 ? ((int)target_data.size() / target_words_per_record) : 0;
        int n_records = loss_eval_records;
        if (input_records > 0 && n_records > input_records) n_records = input_records;
        if (target_records > 0 && n_records > target_records) n_records = target_records;
        if (n_records <= 0) {{
            fprintf(stderr, "[TB-TRAIN] Loss evaluation record count is zero.\\n");
            std::exit(7);
        }}
        float total_loss = 0.0f;
        for (int r = 0; r < n_records; ++r) {{
            push_record(in_stream, input_data, input_words_per_record, r);
            push_record(aux_stream, target_data, target_words_per_record, r);
            {top_name}(in_stream, out_stream, aux_stream, {call_weights_arg}6);
            std::vector<float> loss_words = drain_exact(out_stream, 1, "loss_eval");
            total_loss += loss_words[0];
        }}
        return total_loss / (float)n_records;
    }};

    if (convergence_smoke) {{
        initial_loss = evaluate_loss();
        epoch_losses.push_back(initial_loss);
        printf("[TB-TRAIN] convergence_initial_loss=%f loss_eval_records=%d\\n", initial_loss, loss_eval_records);
    }}

    std::vector<float> last_grads;
    int total_train_calls = 0;
    int optimizer_update_calls = 0;

    if ({str(accumulated_batch).lower()}) {{
        printf("[TB-TRAIN] native_accumulated_batch=true optimizer_location=hls_top_accumulated_optimizer\\n");
        for (int step = 0; step < {int(train_steps)}; ++step) {{
            {top_name}(in_stream, out_stream, aux_stream, {call_weights_arg}5);
            for (int b = 0; b < {int(batch_size)}; ++b) {{
                int rec = step * {int(batch_size)} + b;
                push_record(in_stream, input_data, input_words_per_record, rec);
                push_record(aux_stream, target_data, target_words_per_record, rec);
                {top_name}(in_stream, out_stream, aux_stream, {call_weights_arg}3);
                last_grads = drain_exact(out_stream, {int(weight_words)}, "accum_grads");
                total_train_calls += 1;
            }}
            {top_name}(in_stream, out_stream, aux_stream, {call_weights_arg}4);
            last_grads = drain_exact(out_stream, {int(weight_words)}, "avg_grads");
            optimizer_update_calls += 1;
            if (convergence_smoke) {{
                float epoch_loss = evaluate_loss();
                epoch_losses.push_back(epoch_loss);
                printf("[TB-TRAIN] convergence_epoch=%d loss=%f\\n", step + 1, epoch_loss);
            }}
        }}
    }} else {{
        for (int step = 0; step < {int(train_steps)}; ++step) {{
            for (int b = 0; b < {int(batch_size)}; ++b) {{
                int rec = step * {int(batch_size)} + b;
                push_record(in_stream, input_data, input_words_per_record, rec);
                push_record(aux_stream, target_data, target_words_per_record, rec);
                {top_name}(in_stream, out_stream, aux_stream, {call_weights_arg}2);
                last_grads = drain_exact(out_stream, {int(weight_words)}, "grads");
                total_train_calls += 1;
                optimizer_update_calls += 1;
            }}
        }}
    }}
    if (last_grads.empty()) {{
        fprintf(stderr, "[TB-TRAIN] No train calls were executed. train_steps={int(train_steps)} batch_size={int(batch_size)}\\n");
        std::exit(6);
    }}
    write_bin_both(out_dir, "grads.bin", last_grads);

    {top_name}(in_stream, out_stream, aux_stream, {call_weights_arg}1);
    std::vector<float> weights_after = drain_exact(out_stream, {int(weight_words)}, "weights_after");
    write_bin_both(out_dir, "weights_after.bin", weights_after);

    std::string summary = "{{\\n";
    summary += "  \\\"mode\\\": \\\"{_cpp_string_literal(normalized_mode)}\\\",\\n";
    summary += "  \\\"train_steps\\\": " + std::to_string({int(train_steps)}) + ",\\n";
    summary += "  \\\"batch_size\\\": " + std::to_string({int(batch_size)}) + ",\\n";
    summary += "  \\\"total_train_calls\\\": " + std::to_string(total_train_calls) + ",\\n";
    summary += "  \\\"total_forward_backward_calls\\\": " + std::to_string(total_train_calls) + ",\\n";
    summary += "  \\\"optimizer_update_calls\\\": " + std::to_string(optimizer_update_calls) + ",\\n";
    summary += "  \\\"accumulated_batch\\\": " + std::string({str(accumulated_batch).lower()} ? "true" : "false") + ",\\n";
    summary += "  \\\"averaged_gradients\\\": " + std::string({str(accumulated_batch).lower()} ? "true" : "false") + ",\\n";
    summary += "  \\\"gradient_accumulation_mode\\\": " + std::string({str(accumulated_batch).lower()} ? "true" : "false") + ",\\n";
    summary += "  \\\"optimizer_apply_mode\\\": " + std::string({str(accumulated_batch).lower()} ? "true" : "false") + ",\\n";
    summary += "  \\\"reset_accumulator_mode\\\": " + std::string({str(accumulated_batch).lower()} ? "true" : "false") + ",\\n";
    summary += "  \\\"optimizer_location\\\": \\\"" + std::string({str(accumulated_batch).lower()} ? "hls_top_accumulated_optimizer" : "hls_top_step_optimizer") + "\\\",\\n";
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
