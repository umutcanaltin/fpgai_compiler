from __future__ import annotations

import json
from pathlib import Path

from fpgai.quantization.model_ptq import ModelPTQResult
from fpgai.quantization.model_qat import ModelQATResult


def write_model_ptq_report(result: ModelPTQResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_model_qat_report(result: ModelQATResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
