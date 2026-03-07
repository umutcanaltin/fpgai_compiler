from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from fpgai.ir.graph import Graph


@dataclass
class PartitionPlan:
    device_id: str

    def to_dict(self) -> Dict:
        return {"device_id": self.device_id, "mode": "single_device"}


def single_device_plan(g: Graph, device_id: str = "fpga0") -> PartitionPlan:
    return PartitionPlan(device_id=device_id)
