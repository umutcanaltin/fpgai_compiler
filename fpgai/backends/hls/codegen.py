from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json

from fpgai.util.fs import ensure_clean_dir, write_text


@dataclass(frozen=True)
class HLSProject:
    hls_dir: Path
    top_name: str
    project_name: str
    run_tcl: Path
    top_cpp: Path
    tb_cpp: Path


def _cfg_get(raw: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = raw
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _tcl_bool(v: bool) -> str:
    return "1" if bool(v) else "0"


def _emit_training_tb_cpp(
    *,
    graph,
    weights_mode: str,
) -> str:
    import numpy as np

    def _try_to_numpy(value):
        if value is None:
            return None
        if isinstance(value, str):
            return None
        try:
            arr = np.asarray(value, dtype=np.float32)
            if arr.dtype.kind in ("U", "S", "O"):
                return None
            return arr
        except Exception:
            return None

    def _resolve_named_array(name: str):
        if not name:
            return None
        if hasattr(graph, "constants") and name in getattr(graph, "constants", {}):
            arr = _try_to_numpy(graph.constants[name])
            if arr is not None:
                return arr
        if hasattr(graph, "params") and name in getattr(graph, "params", {}):
            arr = _try_to_numpy(graph.params[name])
            if arr is not None:
                return arr
        try:
            t = graph.get_tensor(name)
        except Exception:
            t = None
        if t is not None:
            for a in ("data", "initializer", "value", "values"):
                if hasattr(t, a):
                    arr = _try_to_numpy(getattr(t, a))
                    if arr is not None:
                        return arr
        return None

    def _resolve_attr_array_or_ref(op, *keys):
        attrs = getattr(op, "attrs", {}) or {}
        for k in keys:
            if k not in attrs:
                continue
            v = attrs[k]
            arr = _try_to_numpy(v)
            if arr is not None:
                return arr
            if isinstance(v, str):
                arr = _resolve_named_array(v)
                if arr is not None:
                    return arr
        return None

    def _resolve_dense_arrays(op):
        w_arr = None
        b_arr = None
        if len(op.inputs) > 1:
            w_arr = _resolve_named_array(op.inputs[1])
        if len(op.inputs) > 2:
            b_arr = _resolve_named_array(op.inputs[2])
        if w_arr is None:
            w_arr = _resolve_attr_array_or_ref(
                op, "weights", "weight", "W", "kernel", "weights_name", "weight_name"
            )
        if b_arr is None:
            b_arr = _resolve_attr_array_or_ref(op, "bias", "biases", "B", "bias_name")
        return w_arr, b_arr

    def _resolve_conv_arrays(op):
        w_arr = None
        b_arr = None
        if len(op.inputs) > 1:
            w_arr = _resolve_named_array(op.inputs[1])
        if len(op.inputs) > 2:
            b_arr = _resolve_named_array(op.inputs[2])
        if w_arr is None:
            w_arr = _resolve_attr_array_or_ref(
                op, "weights", "weight", "W", "kernel", "weights_name", "weight_name"
            )
        if b_arr is None:
            b_arr = _resolve_attr_array_or_ref(op, "bias", "biases", "B", "bias_name")
        return w_arr, b_arr

    def _resolve_bn_arrays(op):
        arrs = []
        for idx in range(1, min(len(op.inputs), 5)):
            arr = _resolve_named_array(op.inputs[idx])
            if arr is not None:
                arrs.append(arr)
        if not arrs:
            arrs.extend(
                x
                for x in [
                    _resolve_attr_array_or_ref(op, "scale", "gamma", "scale_name", "gamma_name"),
                    _resolve_attr_array_or_ref(op, "bias", "beta", "bias_name", "beta_name"),
                    _resolve_attr_array_or_ref(op, "mean", "running_mean", "mean_name", "running_mean_name"),
                    _resolve_attr_array_or_ref(op, "var", "running_var", "var_name", "running_var_name"),
                ]
                if x is not None
            )
        return arrs

    preload: list[float] = []
    total_param_words = 0

    for op in getattr(graph, "ops", []):
        if op.op_type == "Dense":
            w_arr, b_arr = _resolve_dense_arrays(op)
            if w_arr is not None:
                preload.extend(np.asarray(w_arr, dtype=np.float32).reshape(-1).tolist())
                total_param_words += int(np.asarray(w_arr).size)
            if b_arr is not None:
                preload.extend(np.asarray(b_arr, dtype=np.float32).reshape(-1).tolist())
                total_param_words += int(np.asarray(b_arr).size)

        elif op.op_type == "Conv":
            w_arr, b_arr = _resolve_conv_arrays(op)
            if w_arr is not None:
                preload.extend(np.asarray(w_arr, dtype=np.float32).reshape(-1).tolist())
                total_param_words += int(np.asarray(w_arr).size)
            if b_arr is not None:
                preload.extend(np.asarray(b_arr, dtype=np.float32).reshape(-1).tolist())
                total_param_words += int(np.asarray(b_arr).size)

        elif op.op_type == "BatchNormalization":
            arrs = _resolve_bn_arrays(op)
            for a in arrs:
                preload.extend(np.asarray(a, dtype=np.float32).reshape(-1).tolist())
                total_param_words += int(np.asarray(a).size)

    preload_txt = ", ".join(f"{float(x):.8f}f" for x in preload)

    return f"""#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <ap_axi_sdata.h>
#include <hls_stream.h>

typedef ap_axis<32,0,0,0> axis_t;

extern "C" void deeplearn(
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

    if (std::string("{weights_mode}") == "stream" || std::string("{weights_mode}") == "ddr") {{
        std::vector<float> preload = {{ {preload_txt} }};
        for (size_t i = 0; i < preload.size(); ++i) {{
            push_f32(aux_stream, preload[i], i + 1 == preload.size());
        }}
        deeplearn(in_stream, out_stream, aux_stream, 0);
    }}

    for (size_t i = 0; i < input_data.size(); ++i) {{
        push_f32(in_stream, input_data[i], false);
    }}
    for (size_t i = 0; i < target_data.size(); ++i) {{
        push_f32(in_stream, target_data[i], i + 1 == target_data.size());
    }}

    deeplearn(in_stream, out_stream, aux_stream, 2);

    std::vector<float> all_out;
    while (!out_stream.empty()) {{
        axis_t pkt = out_stream.read();
        union {{ unsigned int i; float f; }} u;
        u.i = pkt.data.to_uint();
        all_out.push_back(u.f);
    }}

    const int grad_words = {total_param_words};
    const int w_before_words = {total_param_words};
    const int w_after_words = {total_param_words};
    const int expected_total = grad_words + w_before_words + w_after_words;

    if ((int)all_out.size() != expected_total) {{
        fprintf(
            stderr,
            "[TB-TRAIN] Unexpected output words: got=%zu expected=%d\\n",
            all_out.size(),
            expected_total
        );
        return 2;
    }}

    std::vector<float> grads(
        all_out.begin(),
        all_out.begin() + grad_words
    );
    std::vector<float> w_before(
        all_out.begin() + grad_words,
        all_out.begin() + grad_words + w_before_words
    );
    std::vector<float> w_after(
        all_out.begin() + grad_words + w_before_words,
        all_out.end()
    );

    write_bin("grads.bin", grads);
    write_bin("weights_before.bin", w_before);
    write_bin("weights_after.bin", w_after);

    printf("[TB-TRAIN] Wrote grads.bin, weights_before.bin, weights_after.bin\\n");
    return 0;
}}
"""


def _emit_training_run_tcl(
    *,
    part: str,
    clk_mhz: int,
    top_name: str,
    run_csim: bool,
    run_csynth: bool,
    export_ip: bool,
) -> str:
    period_ns = 1000.0 / float(clk_mhz)

    lines = [
        "# Auto-generated by fpgai (training)",
        "open_project -reset fpgai_hls_proj",
        f"set_top {top_name}",
        "set SRC_DIR ./src",
        "set INC_DIR ./include",
        "set LAYERS_INC_DIR ./include/layers",
        'add_files ${SRC_DIR}/' + f'{top_name}.cpp -cflags "-I${{INC_DIR}} -I${{LAYERS_INC_DIR}} -DFPGAI_DEBUG_DUMP"',
        'add_files -tb ${SRC_DIR}/tb.cpp -cflags "-I${INC_DIR} -I${LAYERS_INC_DIR} -DFPGAI_DEBUG_DUMP"',
        "open_solution -reset sol1",
        f"set_part {part}",
        f"create_clock -period {period_ns:.1f} -name default",
    ]

    if run_csim:
        lines.append('csim_design -argv "{build_input} {build_target}"')
    if run_csynth:
        lines.append("csynth_design")
    if export_ip and run_csynth:
        lines.append(
            'export_design -format ip_catalog -description "FPGAI Neural Network Training" '
            '-vendor "fpgai" -version "1.0"'
        )

    lines.append("exit")
    return "\n".join(lines) + "\n"


def _emit_inference_run_tcl(
    *,
    part: str,
    clk_mhz: int,
    top_name: str,
    run_csim: bool,
    run_csynth: bool,
    export_ip: bool,
) -> str:
    period_ns = 1000.0 / float(clk_mhz)

    lines = [
        "# Auto-generated by fpgai (inference)",
        "open_project -reset fpgai_hls_proj",
        f"set_top {top_name}",
        "set SRC_DIR ./src",
        "set INC_DIR ./include",
        "set LAYERS_INC_DIR ./include/layers",
        'if {[file exists "${SRC_DIR}/' + f'{top_name}.cpp"]' + '} {',
        '  add_files ${SRC_DIR}/' + f'{top_name}.cpp -cflags "-I${{INC_DIR}} -I${{LAYERS_INC_DIR}} -DFPGAI_DEBUG_DUMP"',
        "} else {",
        '  add_files ${SRC_DIR}/deeplearn.cpp -cflags "-I${INC_DIR} -I${LAYERS_INC_DIR} -DFPGAI_DEBUG_DUMP"',
        "}",
        'add_files ${SRC_DIR}/fpgai_params.cpp -cflags "-I${INC_DIR} -I${LAYERS_INC_DIR} -DFPGAI_DEBUG_DUMP"',
        'if {[file exists "${SRC_DIR}/layers/dense.cpp"]} {',
        '  add_files ${SRC_DIR}/layers/dense.cpp -cflags "-I${INC_DIR} -I${LAYERS_INC_DIR} -DFPGAI_DEBUG_DUMP"',
        "}",
        'if {[file exists "${SRC_DIR}/layers/conv.cpp"]} {',
        '  add_files ${SRC_DIR}/layers/conv.cpp -cflags "-I${INC_DIR} -I${LAYERS_INC_DIR} -DFPGAI_DEBUG_DUMP"',
        "}",
        'if {[file exists "${SRC_DIR}/layers/pool.cpp"]} {',
        '  add_files ${SRC_DIR}/layers/pool.cpp -cflags "-I${INC_DIR} -I${LAYERS_INC_DIR} -DFPGAI_DEBUG_DUMP"',
        "}",
        'if {[file exists "${SRC_DIR}/layers/activations.cpp"]} {',
        '  add_files ${SRC_DIR}/layers/activations.cpp -cflags "-I${INC_DIR} -I${LAYERS_INC_DIR} -DFPGAI_DEBUG_DUMP"',
        "}",
        'if {[file exists "${SRC_DIR}/layers/model_inst.cpp"]} {',
        '  add_files ${SRC_DIR}/layers/model_inst.cpp -cflags "-I${INC_DIR} -I${LAYERS_INC_DIR} -DFPGAI_DEBUG_DUMP"',
        "}",
        'add_files -tb ${SRC_DIR}/tb.cpp -cflags "-I${INC_DIR} -I${LAYERS_INC_DIR} -DFPGAI_DEBUG_DUMP"',
        "open_solution -reset sol1",
        f"set_part {part}",
        f"create_clock -period {period_ns:.1f} -name default",
    ]

    if run_csim:
        lines.append("csim_design")
    if run_csynth:
        lines.append("csynth_design")
    if export_ip and run_csynth:
        lines.append(
            'export_design -format ip_catalog -description "FPGAI Neural Network Inference" '
            '-vendor "fpgai" -version "1.0"'
        )

    lines.append("exit")
    return "\n".join(lines) + "\n"


def emit_hls_stub(
    *,
    graph,
    out_dir: Path,
    top_name: str,
    hls_options: Dict[str, Any],
    compile_plan=None,
    memory_plan=None,
    communication_plan=None,
) -> HLSProject:
    from fpgai.backends.hls.emit.top_train_cpp import emit_top_train_cpp

    hls_dir = out_dir / "hls"
    src_dir = hls_dir / "src"
    inc_dir = hls_dir / "include"
    layers_inc_dir = inc_dir / "layers"

    ensure_clean_dir(hls_dir, clean=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    inc_dir.mkdir(parents=True, exist_ok=True)
    layers_inc_dir.mkdir(parents=True, exist_ok=True)

    pipeline_mode = str(hls_options.get("pipeline_mode", "inference")).lower()
    weights_mode = str(hls_options.get("weights_mode", "embedded")).lower()
    part = str(hls_options.get("part", "xck26-sfvc784-2LV-c"))
    clk_mhz = int(hls_options.get("clk_mhz", 200))
    training_cfg = hls_options.get("training_cfg", {}) or {}

    # Debug-friendly defaults:
    # keep csim/csynth on, but do NOT export IP unless explicitly requested.
    run_csim = bool(hls_options.get("run_csim", True))
    run_csynth = bool(hls_options.get("run_csynth", True))
    export_ip = bool(hls_options.get("export_ip", False))

    top_cpp = src_dir / f"{top_name}.cpp"
    tb_cpp = src_dir / "tb.cpp"
    run_tcl = hls_dir / "run_hls.tcl"

    write_text(inc_dir / "fpgai_types.h", "// auto-generated placeholder\n")
    write_text(inc_dir / "fpgai_params.h", "// auto-generated placeholder\n")
    write_text(layers_inc_dir / "dense.h", "// auto-generated placeholder\n")
    write_text(layers_inc_dir / "conv.h", "// auto-generated placeholder\n")
    write_text(layers_inc_dir / "pool.h", "// auto-generated placeholder\n")
    write_text(layers_inc_dir / "activations.h", "// auto-generated placeholder\n")

    if pipeline_mode == "training_on_device":
        top_src = emit_top_train_cpp(
            graph=graph,
            top_name=top_name,
            weights_mode=weights_mode,
            training_cfg=training_cfg,
        )
        tb_src = _emit_training_tb_cpp(
            graph=graph,
            weights_mode=weights_mode,
        )
        tcl_src = _emit_training_run_tcl(
            part=part,
            clk_mhz=clk_mhz,
            top_name=top_name,
            run_csim=run_csim,
            run_csynth=run_csynth,
            export_ip=export_ip,
        )
    else:
        if (src_dir / "deeplearn.cpp").exists():
            top_src = (src_dir / "deeplearn.cpp").read_text(encoding="utf-8")
        else:
            top_src = f'extern "C" void {top_name}() {{}}\n'

        tb_src = "int main(){return 0;}\n"
        tcl_src = _emit_inference_run_tcl(
            part=part,
            clk_mhz=clk_mhz,
            top_name=top_name,
            run_csim=run_csim,
            run_csynth=run_csynth,
            export_ip=export_ip,
        )

    write_text(top_cpp, top_src)
    write_text(tb_cpp, tb_src)

    build_input = str((out_dir / "input.bin").resolve())
    build_target = str((out_dir / "target.bin").resolve())
    tcl_src = tcl_src.replace("{build_input}", build_input).replace("{build_target}", build_target)
    write_text(run_tcl, tcl_src)

    meta = {
        "pipeline_mode": pipeline_mode,
        "top_name": top_name,
        "weights_mode": weights_mode,
        "part": part,
        "clk_mhz": clk_mhz,
        "run_csim": run_csim,
        "run_csynth": run_csynth,
        "export_ip": export_ip,
        "compile_plan_present": compile_plan is not None,
        "memory_plan_present": memory_plan is not None,
        "communication_plan_present": communication_plan is not None,
    }
    write_text(hls_dir / "codegen_meta.json", json.dumps(meta, indent=2))

    return HLSProject(
        hls_dir=hls_dir,
        top_name=top_name,
        project_name="fpgai_hls_proj",
        run_tcl=run_tcl,
        top_cpp=top_cpp,
        tb_cpp=tb_cpp,
    )