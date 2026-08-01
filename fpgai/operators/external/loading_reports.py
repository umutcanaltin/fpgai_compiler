from __future__ import annotations
import json
from pathlib import Path
from .operator_loader import OperatorLoadResult

def write_external_operator_loading_report(result: OperatorLoadResult, output: str|Path) -> Path:
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result.to_dict(),indent=2,sort_keys=True),encoding="utf-8"); return path
