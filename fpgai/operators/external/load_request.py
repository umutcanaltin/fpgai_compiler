from __future__ import annotations
from dataclasses import dataclass
from fpgai.registries import RegistryCatalogue

@dataclass(frozen=True)
class OperatorLoadRequest:
    catalogue: RegistryCatalogue
    package_ids: tuple[str,...]
    trust_level: str = "metadata_only"
    load_mode: str = "in_process"
    def __post_init__(self): object.__setattr__(self,"package_ids",tuple(self.package_ids))
