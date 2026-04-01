from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json

from fpgai.util.fs import write_text


def _cfg_get(raw: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = raw
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@dataclass(frozen=True)
class TrainingPlan:
    optimizer_type: str
    learning_rate: float
    loss_type: str
    batch_size: int
    epochs: int
    supported_train_ops: List[str]
    skipped_non_trainable_ops: List[str]
    numerics: Dict[str, Any]
    weights_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimizer_type": self.optimizer_type,
            "learning_rate": self.learning_rate,
            "loss_type": self.loss_type,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "supported_train_ops": self.supported_train_ops,
            "skipped_non_trainable_ops": self.skipped_non_trainable_ops,
            "numerics": self.numerics,
            "weights_mode": self.weights_mode,
        }


TRAINABLE_OPS_STAGE1 = {"Dense", "Relu", "Sigmoid", "Flatten", "Reshape"}


def build_training_plan(graph, raw_cfg: Dict[str, Any]) -> TrainingPlan:
    optimizer_type = str(_cfg_get(raw_cfg, "training.optimizer.type", "sgd")).lower()
    learning_rate = float(_cfg_get(raw_cfg, "training.optimizer.learning_rate", 0.01))
    loss_type = str(_cfg_get(raw_cfg, "training.loss.type", "mse")).lower()
    batch_size = int(_cfg_get(raw_cfg, "training.execution.batch_size", 1))
    epochs = int(_cfg_get(raw_cfg, "training.execution.epochs", 1))
    weights_mode = str(_cfg_get(raw_cfg, "data_movement.ps_pl.weights.mode", "embedded")).lower()

    supported_train_ops: List[str] = []
    skipped_non_trainable_ops: List[str] = []

    for op in graph.ops:
        op_type = getattr(op, "op_type", "")
        if op_type in TRAINABLE_OPS_STAGE1:
            supported_train_ops.append(op_type)
        else:
            skipped_non_trainable_ops.append(op_type)

    numerics = {
        "forward": {
            "activation": _cfg_get(raw_cfg, "numerics.defaults.activation"),
            "weight": _cfg_get(raw_cfg, "numerics.defaults.weight"),
            "bias": _cfg_get(raw_cfg, "numerics.defaults.bias"),
            "accum": _cfg_get(raw_cfg, "numerics.defaults.accum"),
        },
        "training": _cfg_get(raw_cfg, "numerics.training", {}) or {},
    }

    return TrainingPlan(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        loss_type=loss_type,
        batch_size=batch_size,
        epochs=epochs,
        supported_train_ops=supported_train_ops,
        skipped_non_trainable_ops=skipped_non_trainable_ops,
        numerics=numerics,
        weights_mode=weights_mode,
    )


def emit_training_artifacts(out_dir: Path, plan: TrainingPlan) -> Path:
    training_dir = out_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    json_path = training_dir / "training_plan.json"
    txt_path = training_dir / "summary.txt"

    write_text(json_path, json.dumps(plan.to_dict(), indent=2))

    lines = []
    lines.append("=============== FPGAI Training Plan ===============")
    lines.append(f"optimizer_type        : {plan.optimizer_type}")
    lines.append(f"learning_rate         : {plan.learning_rate}")
    lines.append(f"loss_type             : {plan.loss_type}")
    lines.append(f"batch_size            : {plan.batch_size}")
    lines.append(f"epochs                : {plan.epochs}")
    lines.append(f"weights_mode          : {plan.weights_mode}")
    lines.append(f"supported_train_ops   : {sorted(set(plan.supported_train_ops))}")
    lines.append(f"skipped_non_trainable : {sorted(set(plan.skipped_non_trainable_ops))}")
    lines.append("===================================================")
    write_text(txt_path, "\n".join(lines))

    return json_path