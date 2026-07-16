from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from fpgai.ir.graph import Graph
from fpgai.engine.training import resolve_training_execution_schedule


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
    raw_cfg: Optional[dict] = None,
    dataset_sample_count: Optional[int] = None,
    held_out_sample_count: Optional[int] = None,
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

    schedule_cfg = raw_cfg if isinstance(raw_cfg, dict) and raw_cfg else {"training": training_cfg}
    schedule = resolve_training_execution_schedule(
        schedule_cfg,
        sample_count=(int(dataset_sample_count) if dataset_sample_count is not None else None),
    )
    train_steps = int(schedule.explicit_train_steps or 0)
    batch_size = int(schedule.batch_size)
    epochs = int(schedule.epochs)
    shuffle = bool(schedule.shuffle)
    shuffle_seed = int(schedule.seed)
    drop_last = bool(schedule.drop_last)

    batch_mode = str(schedule.batch_mode).strip().lower()
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
            default=(epochs > 1),
        )
    )
    loss_eval_records = _as_int(
        _cfg_lookup(
            training_cfg,
            "execution.loss_eval_records",
            "loss_eval_records",
            "validation.loss_eval_records",
            default=(int(dataset_sample_count) if dataset_sample_count else batch_size),
        ),
        default=(int(dataset_sample_count) if dataset_sample_count else batch_size),
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

    raw_for_movement = raw_cfg or {}

    def _movement_requested(kind: str, direction: str, *, interface: str = "m_axi") -> bool:
        movement = _cfg_lookup(raw_for_movement, f"data_movement.{kind}.{direction}", default={})
        if not isinstance(movement, dict) or not any(k in movement for k in {"interface", "transport", "policy", "tiled"}):
            direct = _cfg_lookup(raw_for_movement, f"data_movement.{kind}", default={})
            movement = direct if isinstance(direct, dict) else {}
        if not isinstance(movement, dict):
            return False
        iface = str(movement.get("interface", "")).strip().lower().replace("-", "_")
        raw_policy = movement.get("policy", "")
        tiled = movement.get("tiled", False)
        tiled_enabled = bool(tiled.get("enabled", False)) if isinstance(tiled, dict) else bool(tiled)
        policy = str(raw_policy or ("tiled" if tiled_enabled else "")).strip().lower().replace("-", "_")
        return iface == interface and policy in {"full", "tiled"}

    input_m_axi_tiled_requested = _movement_requested("inputs", "import")
    label_m_axi_tiled_requested = _movement_requested("labels", "import")
    output_m_axi_tiled_requested = _movement_requested("outputs", "export")
    gradient_export_requested = _movement_requested("gradients", "export")
    optimizer_state_export_requested = _movement_requested("optimizer_state", "export")
    optimizer_type = str(_cfg_lookup(raw_for_movement, "training.optimizer.type", default=_cfg_lookup(training_cfg, "optimizer.type", default="sgd"))).strip().lower().replace("-", "_")
    optimizer_state_words = int(weight_words) if optimizer_type == "momentum" else (2 * int(weight_words) if optimizer_type == "adam" else 0)
    if not optimizer_state_export_requested or optimizer_state_words <= 0:
        optimizer_state_export_requested = False
        optimizer_state_words = 0

    input_mem_extern_arg = "    ap_uint<32>* input_mem,\n" if input_m_axi_tiled_requested else ""
    label_mem_extern_arg = "    ap_uint<32>* label_mem,\n" if label_m_axi_tiled_requested else ""
    output_mem_extern_arg = "    ap_uint<32>* output_mem,\n" if output_m_axi_tiled_requested else ""
    gradient_mem_extern_arg = "    ap_uint<32>* gradients_mem,\n" if gradient_export_requested else ""
    optimizer_state_mem_extern_arg = "    ap_uint<32>* optimizer_state_mem,\n" if optimizer_state_export_requested else ""
    # m_axi training input/label/output buffers must be sized from the runtime record
    # width, not only from compile-time graph metadata. Some training paths pass
    # out_words=0 while target.bin still contains a non-empty label vector; allocating
    # label_mem from out_words would make CSim fail before the hardware behavior is tested.
    input_mem_decl = ""
    label_mem_decl = ""
    output_mem_decl = ""
    dynamic_movement_mem_decl = (
        ("    std::vector<ap_uint<32> > input_mem((size_t)(input_words_per_record > 0 ? input_words_per_record : 1));\n" if input_m_axi_tiled_requested else "")
        + ("    std::vector<ap_uint<32> > label_mem((size_t)(target_words_per_record > 0 ? target_words_per_record : 1));\n" if label_m_axi_tiled_requested else "")
        + ("    std::vector<ap_uint<32> > output_mem((size_t)(target_words_per_record > 0 ? target_words_per_record : 1));\n" if output_m_axi_tiled_requested else "")
    )
    gradient_mem_decl = f"    std::vector<ap_uint<32> > gradients_mem({int(weight_words)});\n" if gradient_export_requested else ""
    optimizer_state_mem_decl = f"    std::vector<ap_uint<32> > optimizer_state_mem({optimizer_state_words});\n" if optimizer_state_export_requested else ""
    # Keep this argument order aligned with the final generated training top wrappers in
    # fpgai/backends/hls/emit/top_train_cpp.py:
    # weights_mem, input_mem, label_mem, output_mem, gradients_mem, optimizer_state_mem, mode.
    input_mem_call_arg = "input_mem.data(), " if input_m_axi_tiled_requested else ""
    label_mem_call_arg = "label_mem.data(), " if label_m_axi_tiled_requested else ""
    output_mem_call_arg = "output_mem.data(), " if output_m_axi_tiled_requested else ""
    gradient_mem_call_arg = "gradients_mem.data(), " if gradient_export_requested else ""
    optimizer_state_mem_call_arg = "optimizer_state_mem.data(), " if optimizer_state_export_requested else ""
    movement_call_args = f"{call_weights_arg}{input_mem_call_arg}{label_mem_call_arg}{output_mem_call_arg}{gradient_mem_call_arg}{optimizer_state_mem_call_arg}"
    eval_record_mem_pack = (
        "            pack_record_mem(input_mem, input_data, input_words_per_record, r, \"input\");\n"
        if input_m_axi_tiled_requested else ""
    ) + (
        "            pack_record_mem(label_mem, target_data, target_words_per_record, r, \"label\");\n"
        if label_m_axi_tiled_requested else ""
    )
    held_out_eval_mem_pack = (
        "            pack_record_mem(input_mem, held_out_input_data, input_words_per_record, r, \"held_out_input\");\n"
        if input_m_axi_tiled_requested else ""
    )

    train_record_mem_pack = (
        "                pack_record_mem(input_mem, input_data, input_words_per_record, rec, \"input\");\n"
        if input_m_axi_tiled_requested else ""
    ) + (
        "                pack_record_mem(label_mem, target_data, target_words_per_record, rec, \"label\");\n"
        if label_m_axi_tiled_requested else ""
    )

    runtime_preload_call = "" if m_axi_weight_runtime else (
        f"        {top_name}(in_stream, out_stream, aux_stream, "
        f"{movement_call_args}0);\n"
    )


    gradient_capture_block = ""
    if gradient_export_requested:
        gradient_capture_block = (
            "\n"
            "    // FPGAI CSim automatic gradient-export capture.\n"
            "    // Calls generated mode 8 after training and writes dedicated capture files.\n"
            "    {\n"
            f"        {top_name}(in_stream, out_stream, aux_stream, {call_weights_arg}{input_mem_call_arg}{label_mem_call_arg}{output_mem_call_arg}gradients_mem.data(), {optimizer_state_mem_call_arg}8);\n"
            f"        std::vector<float> gradients_after = unpack_words_to_f32(gradients_mem, {int(weight_words)});\n"
            "        write_bin_both(out_dir, \"gradients_after.bin\", gradients_after);\n"
            "        write_bin_both(out_dir, \"gradients_export.bin\", gradients_after);\n"
            "        printf(\"[TB-TRAIN] Wrote %s (%zu floats) from FPGAI_MODE_EXPORT_GRADIENTS\\n\", join_path(out_dir, \"gradients_after.bin\").c_str(), gradients_after.size());\n"
            "    }\n"
        )

    optimizer_state_capture_block = ""
    if optimizer_state_export_requested:
        optimizer_state_capture_block = (
            "\n"
            "    // FPGAI CSim automatic optimizer-state export capture.\n"
            "    // Calls generated mode 9 after training and writes optimizer_state_after.bin.\n"
            "    {\n"
            f"        {top_name}(in_stream, out_stream, aux_stream, {call_weights_arg}{input_mem_call_arg}{label_mem_call_arg}{output_mem_call_arg}{gradient_mem_call_arg}optimizer_state_mem.data(), 9);\n"
            f"        std::vector<float> optimizer_state_after = unpack_words_to_f32(optimizer_state_mem, {optimizer_state_words});\n"
            "        write_bin_both(out_dir, \"optimizer_state_after.bin\", optimizer_state_after);\n"
            "        printf(\"[TB-TRAIN] Wrote %s (%zu floats) from FPGAI_MODE_EXPORT_OPTIMIZER_STATE\\n\", join_path(out_dir, \"optimizer_state_after.bin\").c_str(), optimizer_state_after.size());\n"
            "    }\n"
        )


    accuracy_eval_block = f"""
    auto evaluate_accuracy = [&]() -> float {{
        int input_records = input_words_per_record > 0 ? ((int)input_data.size() / input_words_per_record) : 0;
        int target_records = target_words_per_record > 0 ? ((int)target_data.size() / target_words_per_record) : 0;
        int n_records = loss_eval_records;
        if (input_records > 0 && n_records > input_records) n_records = input_records;
        if (target_records > 0 && n_records > target_records) n_records = target_records;
        if (n_records <= 0) return -1.0f;
        int correct = 0;
        for (int r = 0; r < n_records; ++r) {{
            push_record(in_stream, input_data, input_words_per_record, r);
{held_out_eval_mem_pack}            {top_name}(in_stream, out_stream, aux_stream, {movement_call_args}0);
            std::vector<float> prediction = drain_exact(out_stream, {int(out_words)}, "accuracy_eval");
            int predicted_class = 0;
            for (int i = 1; i < (int)prediction.size(); ++i) {{
                if (prediction[(size_t)i] > prediction[(size_t)predicted_class]) predicted_class = i;
            }}
            int expected_class = 0;
            const float *target_record = target_data.data() + (size_t)r * (size_t)target_words_per_record;
            for (int i = 1; i < target_words_per_record; ++i) {{
                if (target_record[i] > target_record[expected_class]) expected_class = i;
            }}
            if (predicted_class == expected_class) correct += 1;
        }}
        return (float)correct / (float)n_records;
    }};
""" if convergence_smoke else """
    auto evaluate_accuracy = [&]() -> float {{ return -1.0f; }};
"""


    held_out_enabled = bool(held_out_sample_count and int(held_out_sample_count) > 0)
    held_out_arg_block = (
        '    const char* held_out_input_path = argc >= 6 ? argv[5] : "__FPGAI_NO_HELD_OUT__";\n'
        '    const char* held_out_target_path = argc >= 7 ? argv[6] : "__FPGAI_NO_HELD_OUT__";\n'
        if held_out_enabled
        else
        '    const char* held_out_input_path = "__FPGAI_NO_HELD_OUT__";\n'
        '    const char* held_out_target_path = "__FPGAI_NO_HELD_OUT__";\n'
    )
    held_out_load_block = ""
    held_out_eval_block = ""
    held_out_after_block = ""
    if held_out_enabled:
        held_out_load_block = (
            '    std::vector<float> held_out_input_data = read_bin(held_out_input_path);\n'
            '    std::vector<float> held_out_target_data = read_bin(held_out_target_path);\n'
        )
        held_out_eval_block = f"""
    struct HeldOutMetrics {{
        int sample_count;
        int correct_count;
        float average_loss;
        float accuracy;
        std::string predictions_csv;
    }};

    auto evaluate_held_out = [&](const char* phase) -> HeldOutMetrics {{
        HeldOutMetrics metrics = {{0, 0, 0.0f, 0.0f, "sample,target,prediction,loss\\n"}};
        int held_input_records = input_words_per_record > 0 ? ((int)held_out_input_data.size() / input_words_per_record) : 0;
        int held_target_records = target_words_per_record > 0 ? ((int)held_out_target_data.size() / target_words_per_record) : 0;
        int held_records = held_input_records < held_target_records ? held_input_records : held_target_records;
        if (held_records != {int(held_out_sample_count)}) {{
            fprintf(stderr, "[TB-TRAIN] Held-out record-count mismatch. configured=%d runtime=%d\\n", {int(held_out_sample_count)}, held_records);
            std::exit(7);
        }}
        for (int r = 0; r < held_records; ++r) {{
            push_record(in_stream, held_out_input_data, input_words_per_record, r);
{eval_record_mem_pack}            {top_name}(in_stream, out_stream, aux_stream, {movement_call_args}0);
            std::vector<float> prediction = drain_exact(out_stream, {int(out_words)}, "held_out_eval");
            int base = r * target_words_per_record;
            int target_index = 0;
            int prediction_index = 0;
            float best_target = held_out_target_data[(size_t)base];
            float best_prediction = prediction[0];
            float record_loss = 0.0f;
            for (int i = 0; i < target_words_per_record; ++i) {{
                float target_value = held_out_target_data[(size_t)(base + i)];
                float prediction_value = prediction[(size_t)i];
                if (target_value > best_target) {{ best_target = target_value; target_index = i; }}
                if (prediction_value > best_prediction) {{ best_prediction = prediction_value; prediction_index = i; }}
                float clipped = prediction_value < 1.0e-7f ? 1.0e-7f : prediction_value;
                record_loss -= target_value * std::log(clipped);
            }}
            if (prediction_index == target_index) metrics.correct_count += 1;
            metrics.average_loss += record_loss;
            metrics.predictions_csv += std::to_string(r) + "," + std::to_string(target_index) + "," + std::to_string(prediction_index) + "," + std::to_string(record_loss) + "\\n";
        }}
        metrics.sample_count = held_records;
        metrics.average_loss /= (float)held_records;
        metrics.accuracy = (float)metrics.correct_count / (float)held_records;
        printf("[TB-TRAIN] held_out_%s samples=%d loss=%f accuracy=%f\\n", phase, metrics.sample_count, metrics.average_loss, metrics.accuracy);
        return metrics;
    }};

    HeldOutMetrics held_out_before = evaluate_held_out("before");
"""
        held_out_after_block = """
    HeldOutMetrics held_out_after = evaluate_held_out("after");
    write_text_file(join_path(out_dir, "held_out_predictions_before.csv"), held_out_before.predictions_csv);
    write_text_file(join_path(out_dir, "held_out_predictions_after.csv"), held_out_after.predictions_csv);
    std::string held_curve = "phase,sample_count,correct_count,average_loss,accuracy,checkpoint\\n";
    held_curve += "before," + std::to_string(held_out_before.sample_count) + "," + std::to_string(held_out_before.correct_count) + "," + std::to_string(held_out_before.average_loss) + "," + std::to_string(held_out_before.accuracy) + ",weights_before.bin\\n";
    held_curve += "after," + std::to_string(held_out_after.sample_count) + "," + std::to_string(held_out_after.correct_count) + "," + std::to_string(held_out_after.average_loss) + "," + std::to_string(held_out_after.accuracy) + ",weights_after.bin\\n";
    write_text_file(join_path(out_dir, "held_out_curve.csv"), held_curve);
    std::string held_summary = "{\\n";
    held_summary += "  \\\"artifact_kind\\\": \\\"fpgai_hls_held_out_validation_execution\\\",\\n";
    held_summary += "  \\\"schema_version\\\": 1,\\n";
    held_summary += "  \\\"sample_count\\\": " + std::to_string(held_out_after.sample_count) + ",\\n";
    held_summary += "  \\\"before\\\": {\\\"average_loss\\\": " + std::to_string(held_out_before.average_loss) + ", \\\"accuracy\\\": " + std::to_string(held_out_before.accuracy) + ", \\\"correct_count\\\": " + std::to_string(held_out_before.correct_count) + "},\\n";
    held_summary += "  \\\"after\\\": {\\\"average_loss\\\": " + std::to_string(held_out_after.average_loss) + ", \\\"accuracy\\\": " + std::to_string(held_out_after.accuracy) + ", \\\"correct_count\\\": " + std::to_string(held_out_after.correct_count) + "},\\n";
    held_summary += "  \\\"checkpoint_before\\\": \\\"weights_before.bin\\\",\\n";
    held_summary += "  \\\"checkpoint_after\\\": \\\"weights_after.bin\\\",\\n";
    held_summary += "  \\\"claim_scope\\\": \\\"held_out_validation_mechanism_demonstrated\\\"\\n";
    held_summary += "}\\n";
    write_text_file(join_path(out_dir, "held_out_validation_summary.json"), held_summary);
"""

    tb_text = f"""\
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
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
{extern_weights_arg}{input_mem_extern_arg}{label_mem_extern_arg}{output_mem_extern_arg}{gradient_mem_extern_arg}{optimizer_state_mem_extern_arg}    int mode
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

static std::vector<int> make_epoch_order(int sample_count, int epoch_index, bool shuffle, uint32_t seed) {{
    std::vector<int> order((size_t)sample_count);
    for (int i = 0; i < sample_count; ++i) order[(size_t)i] = i;
    if (!shuffle || sample_count <= 1) return order;
    uint32_t state = seed ^ ((uint32_t)(epoch_index + 1) * 0x9E3779B9u);
    for (int i = sample_count - 1; i > 0; --i) {{
        state = state * 1664525u + 1013904223u;
        int j = (int)(state % (uint32_t)(i + 1));
        int tmp = order[(size_t)i];
        order[(size_t)i] = order[(size_t)j];
        order[(size_t)j] = tmp;
    }}
    return order;
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

static std::vector<float> unpack_words_to_f32(const std::vector<ap_uint<32> >& words, int expected_words) {{
    std::vector<float> out;
    out.reserve((size_t)expected_words);
    for (int i = 0; i < expected_words; ++i) {{
        union {{ unsigned int i; float f; }} u;
        u.i = words[(size_t)i].to_uint();
        out.push_back(u.f);
    }}
    return out;
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

static ap_uint<32> pack_f32_word(float value) {{
    union {{ float f; unsigned int i; }} u;
    u.f = value;
    return ap_uint<32>(u.i);
}}

static void pack_record_mem(
    std::vector<ap_uint<32> >& mem,
    const std::vector<float>& data,
    int words_per_record,
    int record_index,
    const char* label
) {{
    if (words_per_record <= 0) return;
    int n_records = (int)data.size() / words_per_record;
    if (n_records <= 0) {{
        fprintf(stderr, "[TB-TRAIN] %s m_axi record count is zero. words_per_record=%d size=%zu\\n", label, words_per_record, data.size());
        std::exit(5);
    }}
    int rec = record_index % n_records;
    int base = rec * words_per_record;
    if ((int)mem.size() < words_per_record) {{
        fprintf(stderr, "[TB-TRAIN] %s m_axi buffer too small. got=%zu expected=%d\\n", label, mem.size(), words_per_record);
        std::exit(5);
    }}
    for (int i = 0; i < words_per_record; ++i) {{
        mem[(size_t)i] = pack_f32_word(data[(size_t)(base + i)]);
    }}
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
{held_out_arg_block}    mkdir_best_effort(out_dir);

    printf("[TB-TRAIN] input=%s\\n", in_path);
    printf("[TB-TRAIN] target=%s\\n", target_path);
    printf("[TB-TRAIN] preload_path=%s\\n", preload_path);
    printf("[TB-TRAIN] output_dir=%s\\n", out_dir.c_str());
    printf("[TB-TRAIN] held_out_input=%s\\n", held_out_input_path);
    printf("[TB-TRAIN] held_out_target=%s\\n", held_out_target_path);
    printf("[TB-TRAIN] mode={_cpp_string_literal(normalized_mode)} expected_weight_words={int(weight_words)}\\n");
    printf("[TB-TRAIN] train_steps={int(train_steps)} batch_size={int(batch_size)} in_words={int(in_words)} out_words={int(out_words)}\\n");

    std::vector<float> input_data = read_bin(in_path);
    std::vector<float> target_data = read_bin(target_path);
{held_out_load_block}    int input_words_per_record = {int(in_words)};
    if (input_words_per_record <= 0) input_words_per_record = (int)input_data.size();
    int target_words_per_record = {int(out_words)};
    if (target_words_per_record <= 0) target_words_per_record = (int)target_data.size();
    int input_records = input_words_per_record > 0 ? ((int)input_data.size() / input_words_per_record) : 0;
    int target_records = target_words_per_record > 0 ? ((int)target_data.size() / target_words_per_record) : 0;
    int dataset_records = input_records < target_records ? input_records : target_records;
    if (dataset_records <= 0) {{
        fprintf(stderr, "[TB-TRAIN] Dataset contains no complete input/target records.\\n");
        std::exit(7);
    }}
    const int configured_dataset_records = {int(dataset_sample_count or 0)};
    if (configured_dataset_records > 0 && configured_dataset_records != dataset_records) {{
        fprintf(
            stderr,
            "[TB-TRAIN] Dataset record-count mismatch. configured=%d runtime=%d\\n",
            configured_dataset_records,
            dataset_records
        );
        std::exit(7);
    }}
    const int batches_per_epoch = {str(drop_last).lower()}
        ? (dataset_records / {int(batch_size)})
        : ((dataset_records + {int(batch_size)} - 1) / {int(batch_size)});
    if (batches_per_epoch <= 0) {{
        fprintf(stderr, "[TB-TRAIN] Resolved zero batches per epoch.\\n");
        std::exit(7);
    }}
    const int configured_epochs = {int(epochs)};
    const int explicit_update_limit = {int(train_steps)};
    const int total_update_target = explicit_update_limit > 0
        ? explicit_update_limit
        : configured_epochs * batches_per_epoch;

{dynamic_movement_mem_decl}    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;
    hls::stream<axis_t> aux_stream;
{weight_mem_decl}{input_mem_decl}{label_mem_decl}{output_mem_decl}{gradient_mem_decl}{optimizer_state_mem_decl}
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
{weight_mem_pack}{aux_preload_push}{runtime_preload_call}
        printf("[TB-TRAIN] Preloaded runtime weights (%zu floats)\\n", preload.size());
    }}

    {top_name}(in_stream, out_stream, aux_stream, {movement_call_args}1);
    std::vector<float> weights_before = drain_exact(out_stream, {int(weight_words)}, "weights_before");
    write_bin_both(out_dir, "weights_before.bin", weights_before);

    const bool convergence_smoke = {str(convergence_smoke).lower()};
    const int loss_eval_records = {int(loss_eval_records)};
    std::vector<float> epoch_losses;
    float initial_loss = -1.0f;
    float final_loss = -1.0f;
    float initial_accuracy = -1.0f;
    float final_accuracy = -1.0f;

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
{eval_record_mem_pack}            {top_name}(in_stream, out_stream, aux_stream, {movement_call_args}6);
            std::vector<float> loss_words = drain_exact(out_stream, 1, "loss_eval");
            total_loss += loss_words[0];
        }}
        return total_loss / (float)n_records;
    }};

{accuracy_eval_block}
{held_out_eval_block}
    if (convergence_smoke) {{
        initial_loss = evaluate_loss();
        initial_accuracy = evaluate_accuracy();
        epoch_losses.push_back(initial_loss);
        printf("[TB-TRAIN] convergence_initial_loss=%f loss_eval_records=%d\\n", initial_loss, loss_eval_records);
    }}

    std::vector<float> last_grads;
    std::vector<float> accumulated_grads_before_reduce;
    std::vector<float> previous_accumulator((size_t){int(weight_words)}, 0.0f);
    int total_train_calls = 0;
    int optimizer_update_calls = 0;
    int epochs_completed = 0;
    int records_consumed = 0;
    std::string epoch_curve = "epoch,optimizer_updates,records_consumed,dataset_loss,accuracy,checkpoint\\n";
    std::string batch_curve = "epoch,batch,optimizer_update,actual_batch_size,records_consumed,gradient_l2_norm,gradient_max_abs\\n";
    if (convergence_smoke) {{
        epoch_curve += "0,0,0," + std::to_string(initial_loss) + "," + std::to_string(initial_accuracy) + ",weights_before.bin\\n";
    }}
    std::string checkpoint_dir = join_path(out_dir, "checkpoints");
    mkdir_best_effort(checkpoint_dir);

    if ({str(accumulated_batch).lower()}) {{
        printf("[TB-TRAIN] native_accumulated_batch=true optimizer_location=hls_top_accumulated_optimizer\\n");
        for (int update_index = 0; update_index < total_update_target; ++update_index) {{
            int epoch_index = update_index / batches_per_epoch;
            int batch_index = update_index % batches_per_epoch;
            std::vector<int> order = make_epoch_order(
                dataset_records,
                epoch_index,
                {str(shuffle).lower()},
                (uint32_t){int(shuffle_seed)}u
            );
            int batch_start = batch_index * {int(batch_size)};
            int actual_batch_size = dataset_records - batch_start;
            if (actual_batch_size > {int(batch_size)}) actual_batch_size = {int(batch_size)};
            if ({str(drop_last).lower()} && actual_batch_size < {int(batch_size)}) actual_batch_size = 0;
            if (actual_batch_size <= 0) continue;

            {top_name}(in_stream, out_stream, aux_stream, {movement_call_args}5);
            std::fill(previous_accumulator.begin(), previous_accumulator.end(), 0.0f);
            for (int b = 0; b < actual_batch_size; ++b) {{
                int rec = order[(size_t)(batch_start + b)];
                push_record(in_stream, input_data, input_words_per_record, rec);
                push_record(aux_stream, target_data, target_words_per_record, rec);
{train_record_mem_pack}                {top_name}(in_stream, out_stream, aux_stream, {movement_call_args}3);
                last_grads = drain_exact(out_stream, {int(weight_words)}, "accum_grads");
                std::vector<float> sample_gradient(last_grads.size(), 0.0f);
                for (size_t gi = 0; gi < last_grads.size(); ++gi) {{
                    sample_gradient[gi] = last_grads[gi] - previous_accumulator[gi];
                }}
                char canonical_sample_name[96];
                char canonical_accum_name[96];
                char detailed_sample_name[160];
                char detailed_accum_name[160];
                std::snprintf(canonical_sample_name, sizeof(canonical_sample_name), "per_sample_gradient_%04d.bin", b);
                std::snprintf(canonical_accum_name, sizeof(canonical_accum_name), "accumulator_after_%04d.bin", b);
                std::snprintf(
                    detailed_sample_name,
                    sizeof(detailed_sample_name),
                    "trace_epoch_%04d_batch_%04d_slot_%04d_record_%04d_gradient.bin",
                    epoch_index + 1,
                    batch_index + 1,
                    b,
                    rec
                );
                std::snprintf(
                    detailed_accum_name,
                    sizeof(detailed_accum_name),
                    "trace_epoch_%04d_batch_%04d_slot_%04d_record_%04d_accumulator.bin",
                    epoch_index + 1,
                    batch_index + 1,
                    b,
                    rec
                );
                write_bin_both(out_dir, canonical_sample_name, sample_gradient);
                write_bin_both(out_dir, canonical_accum_name, last_grads);
                write_one_bin(join_path(out_dir, detailed_sample_name), sample_gradient);
                write_one_bin(join_path(out_dir, detailed_accum_name), last_grads);
                previous_accumulator = last_grads;
                total_train_calls += 1;
                records_consumed += 1;
            }}

            accumulated_grads_before_reduce = last_grads;
            write_bin_both(out_dir, "gradient_accumulated_pre_reduce.bin", accumulated_grads_before_reduce);
            {top_name}(in_stream, out_stream, aux_stream, {movement_call_args}4);
            last_grads = drain_exact(out_stream, {int(weight_words)}, "avg_grads");
            write_bin_both(out_dir, "gradient_reduced_export.bin", last_grads);
            optimizer_update_calls += 1;

            float batch_grad_l2_sq = 0.0f;
            float batch_grad_max_abs = 0.0f;
            for (float value : last_grads) {{
                float av = value < 0.0f ? -value : value;
                batch_grad_l2_sq += value * value;
                if (av > batch_grad_max_abs) batch_grad_max_abs = av;
            }}
            batch_curve += std::to_string(epoch_index + 1) + ",";
            batch_curve += std::to_string(batch_index + 1) + ",";
            batch_curve += std::to_string(optimizer_update_calls) + ",";
            batch_curve += std::to_string(actual_batch_size) + ",";
            batch_curve += std::to_string(records_consumed) + ",";
            batch_curve += std::to_string(std::sqrt(batch_grad_l2_sq)) + ",";
            batch_curve += std::to_string(batch_grad_max_abs) + "\\n";

            bool end_of_epoch = batch_index == batches_per_epoch - 1;
            bool end_of_run = update_index == total_update_target - 1;
            if (end_of_epoch || end_of_run) {{
                float epoch_loss = -1.0f;
                if (convergence_smoke) {{
                    epoch_loss = evaluate_loss();
                    final_accuracy = evaluate_accuracy();
                    epoch_losses.push_back(epoch_loss);
                    printf("[TB-TRAIN] convergence_epoch=%d loss=%f\\n", epoch_index + 1, epoch_loss);
                }}
                {top_name}(in_stream, out_stream, aux_stream, {movement_call_args}1);
                std::vector<float> checkpoint_weights = drain_exact(out_stream, {int(weight_words)}, "checkpoint_weights");
                char checkpoint_name[96];
                std::snprintf(checkpoint_name, sizeof(checkpoint_name), "epoch_%04d_weights.bin", epoch_index + 1);
                write_one_bin(join_path(checkpoint_dir, checkpoint_name), checkpoint_weights);
                epoch_curve += std::to_string(epoch_index + 1) + ",";
                epoch_curve += std::to_string(optimizer_update_calls) + ",";
                epoch_curve += std::to_string(records_consumed) + ",";
                epoch_curve += (convergence_smoke ? std::to_string(epoch_loss) : std::string("")) + ",";
                epoch_curve += (convergence_smoke ? std::to_string(final_accuracy) : std::string("")) + ",";
                epoch_curve += std::string("checkpoints/") + checkpoint_name + "\\n";
                epochs_completed = epoch_index + 1;
            }}
        }}
    }} else {{
        for (int update_index = 0; update_index < total_update_target; ++update_index) {{
            int epoch_index = update_index / batches_per_epoch;
            int batch_index = update_index % batches_per_epoch;
            std::vector<int> order = make_epoch_order(
                dataset_records,
                epoch_index,
                {str(shuffle).lower()},
                (uint32_t){int(shuffle_seed)}u
            );
            int batch_start = batch_index * {int(batch_size)};
            int actual_batch_size = dataset_records - batch_start;
            if (actual_batch_size > {int(batch_size)}) actual_batch_size = {int(batch_size)};
            if ({str(drop_last).lower()} && actual_batch_size < {int(batch_size)}) actual_batch_size = 0;
            for (int b = 0; b < actual_batch_size; ++b) {{
                int rec = order[(size_t)(batch_start + b)];
                push_record(in_stream, input_data, input_words_per_record, rec);
                push_record(aux_stream, target_data, target_words_per_record, rec);
{train_record_mem_pack}                {top_name}(in_stream, out_stream, aux_stream, {movement_call_args}2);
                last_grads = drain_exact(out_stream, {int(weight_words)}, "grads");
                total_train_calls += 1;
                optimizer_update_calls += 1;
                records_consumed += 1;
            }}
            if (batch_index == batches_per_epoch - 1 || update_index == total_update_target - 1) {{
                epochs_completed = epoch_index + 1;
            }}
        }}
    }}
    if (last_grads.empty()) {{
        fprintf(stderr, "[TB-TRAIN] No train calls were executed. updates=%d batch_size={int(batch_size)}\\n", total_update_target);
        std::exit(6);
    }}
    write_bin_both(out_dir, "grads.bin", last_grads);
    if (accumulated_grads_before_reduce.empty()) {{
        accumulated_grads_before_reduce = last_grads;
        write_bin_both(out_dir, "gradient_accumulated_pre_reduce.bin", accumulated_grads_before_reduce);
        write_bin_both(out_dir, "gradient_reduced_export.bin", last_grads);
    }}
    write_text_file(join_path(out_dir, "training_epoch_curve.csv"), epoch_curve);
    write_text_file(join_path(out_dir, "training_batch_curve.csv"), batch_curve);
    if (!(out_dir.empty() || out_dir == ".")) {{
        write_text_file("training_epoch_curve.csv", epoch_curve);
        write_text_file("training_batch_curve.csv", batch_curve);
    }}

    {top_name}(in_stream, out_stream, aux_stream, {movement_call_args}1);
    std::vector<float> weights_after = drain_exact(out_stream, {int(weight_words)}, "weights_after");
    write_bin_both(out_dir, "weights_after.bin", weights_after);
{held_out_after_block}{gradient_capture_block}{optimizer_state_capture_block}

    std::string summary = "{{\\n";
    summary += "  \\\"artifact_kind\\\": \\\"fpgai_hls_training_multiepoch_execution\\\",\\n";
    summary += "  \\\"schema_version\\\": 2,\\n";
    summary += "  \\\"mode\\\": \\\"{_cpp_string_literal(normalized_mode)}\\\",\\n";
    summary += "  \\\"train_steps\\\": " + std::to_string({int(train_steps)}) + ",\\n";
    summary += "  \\\"epochs_requested\\\": " + std::to_string(configured_epochs) + ",\\n";
    summary += "  \\\"epochs_completed\\\": " + std::to_string(epochs_completed) + ",\\n";
    summary += "  \\\"batch_size\\\": " + std::to_string({int(batch_size)}) + ",\\n";
    summary += "  \\\"batches_per_epoch\\\": " + std::to_string(batches_per_epoch) + ",\\n";
    summary += "  \\\"shuffle\\\": " + std::string({str(shuffle).lower()} ? "true" : "false") + ",\\n";
    summary += "  \\\"shuffle_seed\\\": " + std::to_string({int(shuffle_seed)}) + ",\\n";
    summary += "  \\\"drop_last\\\": " + std::string({str(drop_last).lower()} ? "true" : "false") + ",\\n";
    summary += "  \\\"total_update_target\\\": " + std::to_string(total_update_target) + ",\\n";
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
    int dataset_input_records = input_words_per_record > 0 ? ((int)input_data.size() / input_words_per_record) : 0;
    int dataset_target_records = target_words_per_record > 0 ? ((int)target_data.size() / target_words_per_record) : 0;
    summary += "  \\\"out_words\\\": " + std::to_string({int(out_words)}) + ",\\n";
    summary += "  \\\"dataset_input_records\\\": " + std::to_string(dataset_input_records) + ",\\n";
    summary += "  \\\"dataset_target_records\\\": " + std::to_string(dataset_target_records) + ",\\n";
    summary += "  \\\"dataset_records_consumed\\\": " + std::to_string(records_consumed) + ",\\n";
    summary += "  \\\"training_epoch_curve_csv\\\": \\\"training_epoch_curve.csv\\\",\\n";
    summary += "  \\\"training_batch_curve_csv\\\": \\\"training_batch_curve.csv\\\",\\n";
    summary += "  \\\"checkpoint_directory\\\": \\\"checkpoints\\\",\\n";
    summary += "  \\\"initial_loss\\\": " + std::string(convergence_smoke ? std::to_string(initial_loss) : "null") + ",\\n";
    summary += "  \\\"final_loss\\\": " + std::string(convergence_smoke && !epoch_losses.empty() ? std::to_string(epoch_losses.back()) : "null") + ",\\n";
    float grad_l1 = 0.0f; float grad_l2_sq = 0.0f; float grad_max_abs = 0.0f;
    for (float value : last_grads) {{ float av = value < 0.0f ? -value : value; grad_l1 += av; grad_l2_sq += value * value; if (av > grad_max_abs) grad_max_abs = av; }}
    summary += "  \\\"gradient_l1_norm\\\": " + std::to_string(grad_l1) + ",\\n";
    summary += "  \\\"gradient_l2_norm\\\": " + std::to_string(std::sqrt(grad_l2_sq)) + ",\\n";
    summary += "  \\\"gradient_max_abs\\\": " + std::to_string(grad_max_abs) + "\\n";
    summary += "}}\\n";
    write_text_file(join_path(out_dir, "training_multistep_summary.json"), summary);
    if (!(out_dir.empty() || out_dir == ".")) {{
        write_text_file("training_multistep_summary.json", summary);
    }}

    printf("[TB-TRAIN] Wrote %s (%zu floats)\\n", join_path(out_dir, "weights_before.bin").c_str(), weights_before.size());
    printf("[TB-TRAIN] Wrote %s (%zu floats)\\n", join_path(out_dir, "grads.bin").c_str(), last_grads.size());
    printf("[TB-TRAIN] Wrote %s (%zu floats)\\n", join_path(out_dir, "weights_after.bin").c_str(), weights_after.size());
    printf("[TB-TRAIN] Multi-epoch summary: epochs=%d batches_per_epoch=%d updates=%d records=%d\\n", epochs_completed, batches_per_epoch, optimizer_update_calls, records_consumed);
    return 0;
}}
"""
    tb_path.write_text(tb_text, encoding="utf-8")
