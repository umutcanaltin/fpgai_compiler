from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HLSProject:
    """
    Layout:
      <out_dir>/hls/
        run_hls.tcl
        include/
          fpgai_types.h
          fpgai_params.h
          layers/
            dense.h
            activations.h
        src/
          <top>.cpp
          fpgai_params.cpp  (optional)
          tb.cpp
          layers/
            dense.cpp
            activations.cpp
            model_inst.cpp
    """
    out_dir: Path
    top_name: str

    @property
    def hls_dir(self) -> Path:
        return self.out_dir / "hls"

    @property
    def include_dir(self) -> Path:
        return self.hls_dir / "include"

    @property
    def include_layers_dir(self) -> Path:
        return self.include_dir / "layers"

    @property
    def src_dir(self) -> Path:
        return self.hls_dir / "src"

    @property
    def src_layers_dir(self) -> Path:
        return self.src_dir / "layers"
