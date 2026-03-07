from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import time

import numpy as np

from fpgai.config.loader import FPGAIConfig
from fpgai.engine.result import CompileResult
from fpgai.engine.partition import single_device_plan
from fpgai.util.fs import ensure_clean_dir, write_text
from fpgai.util.binio import write_f32_bin


# -----------------------------------------------------------------------------
# Small config helper
# -----------------------------------------------------------------------------
def _cfg_get(raw: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = raw
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# -----------------------------------------------------------------------------
# Compiler
# -----------------------------------------------------------------------------
@dataclass
class Compiler:
    cfg: FPGAIConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Compiler":
        from fpgai.config.loader import load_config
        return cls(load_config(path))

    # --------------------------------------------
    # Main entrypoint
    # --------------------------------------------
    def compile(self) -> CompileResult:
        raw = self.cfg.raw
        t0 = time.time()

        out_dir = self._prepare_out_dir(raw)
        top_name = str(_cfg_get(raw, "pipeline.outputs.top_kernel_name", "deeplearn"))

        verbose = bool(_cfg_get(raw, "debug.verbose", False))
        emit_manifest = bool(_cfg_get(raw, "project.reproducibility.emit_manifest", True))

        enable_hls = bool(_cfg_get(raw, "backends.hls.enabled", True))
        enable_host = bool(_cfg_get(raw, "backends.host_cpp.enabled", True))

        # activation insertion (optional)
        act_kind, act_alpha, act_except_last = self._read_activation_insert_cfg(raw)

        # weights provisioning mode
        weights_mode = str(_cfg_get(raw, "data_movement.ps_pl.weights.mode", "embedded")).lower()

        # ------------------------------------------------------------------
        # 1) Import ONNX -> IR
        # ------------------------------------------------------------------
        g = self._import_and_prepare_graph(
            act_kind=act_kind,
            act_alpha=act_alpha,
            act_except_last=act_except_last,
        )

        # ------------------------------------------------------------------
        # 2) Emit IR artifacts + partition plan
        # ------------------------------------------------------------------
        self._emit_ir_artifacts(out_dir, g)

        # ------------------------------------------------------------------
        # 3) Emit dummy input.bin (for now)
        # ------------------------------------------------------------------
        self._emit_dummy_input(out_dir, g)

        # ------------------------------------------------------------------
        # 4) Emit backends
        # ------------------------------------------------------------------
        hls_dir: Optional[Path] = None
        if enable_hls:
            hls_dir = self._emit_hls(out_dir, g, top_name=top_name, weights_mode=weights_mode)

        host_dir: Optional[Path] = None
        if enable_host:
            host_dir = self._emit_hostcpp(out_dir, g, top_name=top_name)

        # ------------------------------------------------------------------
        # 5) Manifest
        # ------------------------------------------------------------------
        if emit_manifest:
            self._emit_manifest(
                out_dir=out_dir,
                top_name=top_name,
                weights_mode=weights_mode,
                graph=g,
                seconds=time.time() - t0,
            )

        if verbose:
            print("[FPGAI] out_dir:", out_dir)
            print("[FPGAI] top_name:", top_name)
            print("[FPGAI] weights_mode:", weights_mode)
            print("[FPGAI] ops:", [op.op_type for op in g.ops])

        return CompileResult(
            out_dir=out_dir,
            graph=g,
            hls_project_dir=hls_dir,
            host_project_dir=host_dir,
        )

    # --------------------------------------------
    # Stage helpers
    # --------------------------------------------
    def _prepare_out_dir(self, raw: Dict[str, Any]) -> Path:
        out_dir = Path(_cfg_get(raw, "project.out_dir", "build/fpgai")).resolve()
        clean = bool(_cfg_get(raw, "project.clean", True))
        ensure_clean_dir(out_dir, clean=clean)
        return out_dir

    def _read_activation_insert_cfg(self, raw: Dict[str, Any]) -> tuple[str, float, bool]:
        act_cfg = _cfg_get(raw, "operators.defaults.activation_insert", {}) or {}
        act_kind = str(act_cfg.get("kind", "none")).lower()
        act_alpha = float(act_cfg.get("alpha", 0.1))
        act_except_last = bool(act_cfg.get("except_last", True))
        return act_kind, act_alpha, act_except_last

    def _import_and_prepare_graph(self, *, act_kind: str, act_alpha: float, act_except_last: bool):
        from fpgai.frontend.onnx import import_onnx
        from fpgai.ir.passes import validate_allowlist, assign_stable_names, insert_activations

        g = import_onnx(self.cfg.model.path, canonicalize=True, infer_shapes=True)

        if act_kind != "none":
            g = insert_activations(g, kind=act_kind, alpha=act_alpha, except_last=act_except_last)

        g = assign_stable_names(g)

        validate_allowlist(g, self.cfg.operators.supported)
        return g

    def _emit_ir_artifacts(self, out_dir: Path, g) -> None:
        write_text(out_dir / "ir_summary.txt", g.summary())
        plan = single_device_plan(g, device_id="fpga0")
        write_text(out_dir / "partition_plan.json", json.dumps(plan.to_dict(), indent=2))

    def _emit_dummy_input(self, out_dir: Path, g) -> Path:
        # uses input last-dim as feature count
        x_name = g.inputs[0]
        x_spec = g.get_tensor(x_name)
        in_words = int(x_spec.shape[-1]) if (x_spec and x_spec.shape) else 1

        x = (np.arange(in_words, dtype=np.float32) + 1.0) * 0.1
        p = out_dir / "input.bin"
        write_f32_bin(p, x)
        return p

    def _emit_hls(self, out_dir: Path, g, *, top_name: str, weights_mode: str) -> Path:
        # --- CRITICAL FIX: Import from codegen, NOT the package root ---
        from fpgai.backends.hls.codegen import emit_hls_stub

        raw = self.cfg.raw
        part = str(_cfg_get(raw, "targets.platform.part", "xck26-sfvc784-2LV-c"))
        clk_mhz = float(_cfg_get(raw, "targets.platform.clocks.0.target_mhz", 200))

        proj = emit_hls_stub(
            graph=g,
            out_dir=out_dir,
            top_name=top_name,
            hls_options={
                "weights_mode": weights_mode,
                "part": part,
                "clk_mhz": int(clk_mhz),
                "proj_name": "fpgai_hls_proj",
            },
        )

        return proj.hls_dir

    def _emit_hostcpp(self, out_dir: Path, g, *, top_name: str) -> Path:
        from fpgai.backends.hostcpp.emit_host_model import emit_hostcpp_project

        return emit_hostcpp_project(g, out_dir, top_name=top_name)

    def _emit_manifest(self, *, out_dir: Path, top_name: str, weights_mode: str, graph, seconds: float) -> None:
        manifest = {
            "version": self.cfg.version,
            "model_path": self.cfg.model.path,
            "pipeline_mode": self.cfg.pipeline.mode,
            "top_kernel_name": top_name,
            "weights_mode": weights_mode,
            "out_dir": str(out_dir),
            "num_ops": len(graph.ops),
            "num_params": len(graph.params),
            "ops": [{"name": op.name, "type": op.op_type} for op in graph.ops],
            "seconds": round(float(seconds), 6),
        }
        write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2))