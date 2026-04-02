from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from fpgai.util.fs import ensure_clean_dir, write_text


def _cfg_get(raw: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = raw
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _infer_in_out_words(graph) -> Tuple[int, int]:
    in_words = 1
    out_words = 1

    if getattr(graph, "inputs", None):
        x = graph.get_tensor(graph.inputs[0])
        if x is not None and getattr(x, "shape", None):
            shape = tuple(int(d) for d in x.shape)
            if len(shape) > 1 and shape[0] == 1:
                shape = shape[1:]
            n = 1
            for d in shape:
                n *= d
            in_words = n if shape else 1

    if getattr(graph, "outputs", None):
        y = graph.get_tensor(graph.outputs[0])
        if y is not None and getattr(y, "shape", None):
            shape = tuple(int(d) for d in y.shape)
            if len(shape) > 1 and shape[0] == 1:
                shape = shape[1:]
            n = 1
            for d in shape:
                n *= d
            out_words = n if shape else 1

    return int(in_words), int(out_words)


def _emit_runtime_h(
    top_name: str,
    in_words: int,
    out_words: int,
    *,
    optimizer_type: str,
    learning_rate: float,
    loss_type: str,
    batch_size: int,
    epochs: int,
) -> str:
    return f"""#pragma once

namespace fpgai_train_runtime {{
static constexpr const char* TOP_NAME = "{top_name}";
static constexpr int INPUT_WORDS = {in_words};
static constexpr int TARGET_WORDS = {out_words};
static constexpr const char* OPTIMIZER = "{optimizer_type}";
static constexpr float LEARNING_RATE = {learning_rate}f;
static constexpr const char* LOSS_TYPE = "{loss_type}";
static constexpr int BATCH_SIZE = {batch_size};
static constexpr int EPOCHS = {epochs};
}} // namespace fpgai_train_runtime
"""


def _emit_readme(
    top_name: str,
    *,
    optimizer_type: str,
    learning_rate: float,
    loss_type: str,
    batch_size: int,
    epochs: int,
) -> str:
    return f"""FPGAI training host artifacts
============================

Top name      : {top_name}
Optimizer     : {optimizer_type}
Learning rate : {learning_rate}
Loss          : {loss_type}
Batch size    : {batch_size}
Epochs        : {epochs}

Files expected by the training flow:
- input.bin
- target.bin

This hostcpp training directory is a training-mode artifact scaffold.
It is separate from the inference-only host emitter and keeps training
build outputs isolated while on-board training interfaces evolve.
"""


def _emit_run_cpp() -> str:
    return r'''#include "train_runtime.h"

#include <cstdio>
#include <fstream>
#include <vector>

static bool read_f32(const char* path, std::vector<float>& v) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    size_t bytes = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    v.resize(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return true;
}

int main(int argc, char** argv) {
    const char* input_path = (argc > 1) ? argv[1] : "input.bin";
    const char* target_path = (argc > 2) ? argv[2] : "target.bin";

    std::vector<float> x;
    std::vector<float> t;

    if (!read_f32(input_path, x)) {
        std::fprintf(stderr, "failed to read input: %s\n", input_path);
        return 2;
    }
    if (!read_f32(target_path, t)) {
        std::fprintf(stderr, "failed to read target: %s\n", target_path);
        return 2;
    }

    std::printf("[HostCpp/Train] top=%s\n", fpgai_train_runtime::TOP_NAME);
    std::printf("[HostCpp/Train] optimizer=%s lr=%.6f loss=%s batch=%d epochs=%d\n",
                fpgai_train_runtime::OPTIMIZER,
                fpgai_train_runtime::LEARNING_RATE,
                fpgai_train_runtime::LOSS_TYPE,
                fpgai_train_runtime::BATCH_SIZE,
                fpgai_train_runtime::EPOCHS);
    std::printf("[HostCpp/Train] input floats=%zu target floats=%zu\n", x.size(), t.size());
    std::printf("[HostCpp/Train] expected input=%d expected target=%d\n",
                fpgai_train_runtime::INPUT_WORDS,
                fpgai_train_runtime::TARGET_WORDS);
    std::printf("[HostCpp/Train] training host scaffold generated successfully.\n");
    return 0;
}
'''


def emit_hostcpp_project_train(graph, out_dir: Path, *, top_name: str, raw_cfg: Dict[str, Any]) -> Path:
    host_dir = out_dir / "hostcpp"
    include_dir = host_dir / "include"
    src_dir = host_dir / "src"

    ensure_clean_dir(host_dir, clean=True)
    include_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    in_words, out_words = _infer_in_out_words(graph)

    optimizer_type = str(_cfg_get(raw_cfg, "training.optimizer.type", "sgd")).lower()
    learning_rate = float(_cfg_get(raw_cfg, "training.optimizer.learning_rate", 0.01))
    loss_type = str(_cfg_get(raw_cfg, "training.loss.type", "mse")).lower()
    batch_size = int(_cfg_get(raw_cfg, "training.execution.batch_size", 1))
    epochs = int(_cfg_get(raw_cfg, "training.execution.epochs", 1))

    write_text(
        include_dir / "train_runtime.h",
        _emit_runtime_h(
            top_name,
            in_words,
            out_words,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            loss_type=loss_type,
            batch_size=batch_size,
            epochs=epochs,
        ),
    )
    write_text(src_dir / "run_train.cpp", _emit_run_cpp())
    write_text(
        host_dir / "README.txt",
        _emit_readme(
            top_name,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            loss_type=loss_type,
            batch_size=batch_size,
            epochs=epochs,
        ),
    )
    return host_dir