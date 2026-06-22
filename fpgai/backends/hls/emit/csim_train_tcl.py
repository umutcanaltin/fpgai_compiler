from __future__ import annotations

from pathlib import Path


def _tcl_quote(value: str) -> str:
    """Return a TCL-safe value for paths used inside a quoted -argv string."""
    return str(value).replace('\\', '/').replace('"', '\\"')


def emit_csim_train_tcl(
    *,
    top_name: str = "deeplearn",
    part: str,
    input_bin_path: str,
    target_bin_path: str,
    weights_mode: str = "embedded",
    intermediate_dump: bool = False,
) -> str:
    """Emit the Vitis HLS TCL script for training CSim/Csynth.

    Training stream compatibility:
    Runtime-weight training modes (stream/DDR) need initial weights for the
    preload phase before mode=2 training. The generated tb.cpp already accepts
    argv[3] as a runtime preload file and argv[4] as an output directory. Older
    scripts passed __FPGAI_NO_PRELOAD__, so stream CSim loaded 0 weights and
    failed with:

        Runtime preload size mismatch. got=0 expected=<N> mode=stream

    We infer the reference preload file from the build directory of input.bin:

        <build>/training_reference/weights_before_ref.bin

    This path is valid by the time Vitis runs CSim because the Python training
    reference stage runs before HLS execution in the compiler pipeline.
    Embedded mode is unaffected; the testbench ignores runtime preload for
    embedded weights.
    """

    del weights_mode  # mode-specific behavior is handled inside tb.cpp.

    extra_cflags = ""
    if intermediate_dump:
        extra_cflags += " -DFPGAI_DEBUG_DUMP"

    top_cflags = f'-I${{INC_DIR}} -I${{LAYERS_INC_DIR}}{extra_cflags}'
    tb_cflags = f'-I${{INC_DIR}} -I${{LAYERS_INC_DIR}}{extra_cflags}'

    build_dir = Path(input_bin_path).resolve().parent
    preload_bin_path = build_dir / "training_reference" / "weights_before_ref.bin"
    output_dir = "."

    input_arg = _tcl_quote(str(Path(input_bin_path).resolve()))
    target_arg = _tcl_quote(str(Path(target_bin_path).resolve()))
    preload_arg = _tcl_quote(str(preload_bin_path))
    output_arg = _tcl_quote(output_dir)

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

# argv[1]=input.bin argv[2]=target.bin argv[3]=runtime preload weights argv[4]=output dir
csim_design -clean -argv "{input_arg} {target_arg} {preload_arg} {output_arg}"
csynth_design
export_design -format ip_catalog -description "FPGAI Neural Network Training" -vendor "fpgai" -version "1.0"
exit
"""
