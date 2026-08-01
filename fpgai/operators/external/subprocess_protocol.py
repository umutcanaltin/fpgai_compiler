from __future__ import annotations
import json
from dataclasses import dataclass

@dataclass(frozen=True)
class SubprocessLoadResponse:
    status: str
    package_id: str
    message: str = ""
    def to_json(self): return json.dumps(self.__dict__,sort_keys=True)
