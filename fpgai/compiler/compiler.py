from dataclasses import dataclass
from typing import Any, Optional

from fpgai.engine.legacy_engine import fpgai_engine


@dataclass
class CompileArtifact:
    # In v1 we can store paths or strings; start with paths since your engine writes files.
    out_dir: str
    success: bool = True
    meta: Optional[dict] = None


class Compiler:
    def __init__(self, cfg: Any):
        self.cfg = cfg

    @classmethod
    def from_cfg(cls, cfg: Any) -> "Compiler":
        return cls(cfg)

    def compile(self, *, input_data, first_layer_shape, output_shape, onnx_path: str, verbose: bool = False) -> CompileArtifact:
        # Map YAML → legacy engine args
        precision = "float"
        if self.cfg.numerics.kind == "fixed":
            precision = self.cfg.numerics.activation

        _ = fpgai_engine(
            input_data=input_data,
            first_layer_shape=first_layer_shape,
            output_shape=output_shape,
            learning_rate=getattr(self.cfg.raw.get("training", {}).get("optimizer", {}), "learning_rate", 0.1)
            if self.cfg.raw.get("training", {}).get("enabled", False) else 0.1,
            mode=self.cfg.pipeline.mode,
            onnx_file_name=onnx_path,
            precision=precision,
            verbose=verbose,
        )

        out_dir = self.cfg.raw.get("project", {}).get("out_dir", "generated_files")
        return CompileArtifact(out_dir=out_dir, success=True)
