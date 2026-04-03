from __future__ import annotations


def emit_csim_train_tcl(
    *,
    top_name: str = "deeplearn",
    part: str,
    input_bin_path: str,
    target_bin_path: str,
    weights_mode: str = "embedded",
    intermediate_dump: bool = False,
) -> str:
    if weights_mode in ("stream", "ddr"):
        weight_src_line = 'add_files ${SRC_DIR}/weights_runtime.cpp -cflags "-I${INC_DIR} -I${LAYERS_INC_DIR}'
    else:
        weight_src_line = 'add_files ${SRC_DIR}/fpgai_params.cpp -cflags "-I${INC_DIR} -I${LAYERS_INC_DIR}'

    extra_cflags = ""
    if intermediate_dump:
        extra_cflags += " -DFPGAI_DEBUG_DUMP"

    weight_src_line += extra_cflags + '"'
    top_cflags = f'-I${{INC_DIR}} -I${{LAYERS_INC_DIR}}{extra_cflags}'
    tb_cflags = f'-I${{INC_DIR}} -I${{LAYERS_INC_DIR}}{extra_cflags}'

    return f"""\
open_project -reset fpgai_hls_proj
set_top {top_name}

set SRC_DIR ./src
set INC_DIR ./include
set LAYERS_INC_DIR ./include/layers

if {{[file exists "${{SRC_DIR}}/{top_name}.cpp"]}} {{
  add_files ${{SRC_DIR}}/{top_name}.cpp -cflags "{top_cflags}"
}} else {{
  add_files ${{SRC_DIR}}/deeplearn.cpp -cflags "{top_cflags}"
}}

{weight_src_line}

if {{[file exists "${{SRC_DIR}}/layers/dense.cpp"]}} {{
  add_files ${{SRC_DIR}}/layers/dense.cpp -cflags "{top_cflags}"
}}
if {{[file exists "${{SRC_DIR}}/layers/conv.cpp"]}} {{
  add_files ${{SRC_DIR}}/layers/conv.cpp -cflags "{top_cflags}"
}}
if {{[file exists "${{SRC_DIR}}/layers/pool.cpp"]}} {{
  add_files ${{SRC_DIR}}/layers/pool.cpp -cflags "{top_cflags}"
}}
if {{[file exists "${{SRC_DIR}}/layers/activations.cpp"]}} {{
  add_files ${{SRC_DIR}}/layers/activations.cpp -cflags "{top_cflags}"
}}
if {{[file exists "${{SRC_DIR}}/layers/batchnorm.cpp"]}} {{
  add_files ${{SRC_DIR}}/layers/batchnorm.cpp -cflags "{top_cflags}"
}}

add_files -tb ${{SRC_DIR}}/tb.cpp -cflags "{tb_cflags}"

open_solution -reset sol1
set_part {part}
create_clock -period 5.0 -name default

csim_design -argv "{input_bin_path} {target_bin_path}"
csynth_design
export_design -format ip_catalog -description "FPGAI Neural Network Training" -vendor "fpgai" -version "1.0"
exit
"""