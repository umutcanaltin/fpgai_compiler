from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import time

import numpy as np

from fpgai.config.loader import FPGAIConfig
from fpgai.engine.analysis import analyze_graph
from fpgai.engine.communication import make_communication_plan
from fpgai.engine.memory import make_memory_plan
from fpgai.engine.planner import make_compile_plan
from fpgai.engine.result import CompileResult
from fpgai.engine.partition import single_device_plan
from fpgai.util.fs import ensure_clean_dir, write_text
from fpgai.util.binio import write_f32_bin


def _cfg_get(raw: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = raw
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@dataclass
class Compiler:
    cfg: FPGAIConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Compiler":
        from fpgai.config.loader import load_config
        return cls(load_config(path))

    def compile(self) -> CompileResult:
        raw = self.cfg.raw
        t0 = time.time()

        out_dir = self._prepare_out_dir(raw)
        top_name = str(_cfg_get(raw, "pipeline.outputs.top_kernel_name", "deeplearn"))

        verbose = bool(_cfg_get(raw, "debug.verbose", False))
        emit_manifest = bool(_cfg_get(raw, "project.reproducibility.emit_manifest", True))

        enable_hls = bool(_cfg_get(raw, "backends.hls.enabled", True))
        enable_host = bool(_cfg_get(raw, "backends.host_cpp.enabled", True))

        act_kind, act_alpha, act_except_last = self._read_activation_insert_cfg(raw)
        weights_mode = str(_cfg_get(raw, "data_movement.ps_pl.weights.mode", "embedded")).lower()

        # 1) Import ONNX -> IR
        g = self._import_and_prepare_graph(
            act_kind=act_kind,
            act_alpha=act_alpha,
            act_except_last=act_except_last,
        )

        # 2) Static analysis
        descriptors = analyze_graph(g)

        # 3) Heuristic planning
        compile_plan = make_compile_plan(self.cfg, descriptors)

        # 4) Memory planning
        memory_plan = make_memory_plan(g, descriptors, compile_plan)

        # 5) Communication planning
        communication_plan = make_communication_plan(self.cfg, memory_plan)

        # 6) Emit compiler-side artifacts
        self._emit_ir_artifacts(
            out_dir,
            g,
            descriptors,
            compile_plan,
            memory_plan,
            communication_plan,
        )

        # 7) Emit dummy input.bin
        self._emit_dummy_input(out_dir, g)

        # 8) Emit backends
        hls_dir: Optional[Path] = None
        if enable_hls:
            hls_dir = self._emit_hls(
                out_dir,
                g,
                top_name=top_name,
                weights_mode=weights_mode,
                compile_plan=compile_plan,
                memory_plan=memory_plan,
                communication_plan=communication_plan,
            )

        host_dir: Optional[Path] = None
        if enable_host:
            host_dir = self._emit_hostcpp(out_dir, g, top_name=top_name)

        # 9) Optionally run Vitis HLS
        hls_run = None
        if enable_hls and hls_dir is not None:
            hls_run = self._maybe_run_vitis_hls(hls_dir)

        # 10) Manifest
        if emit_manifest:
            self._emit_manifest(
                out_dir=out_dir,
                top_name=top_name,
                weights_mode=weights_mode,
                graph=g,
                descriptors=descriptors,
                compile_plan=compile_plan,
                memory_plan=memory_plan,
                communication_plan=communication_plan,
                hls_run=hls_run,
                seconds=time.time() - t0,
            )

        if verbose:
            print("[FPGAI] out_dir:", out_dir)
            print("[FPGAI] top_name:", top_name)
            print("[FPGAI] weights_mode:", weights_mode)
            print("[FPGAI] ops:", [op.op_type for op in g.ops])
            print("[FPGAI] descriptors:", len(descriptors))
            print("[FPGAI] planned layers:", len(compile_plan.layer_plans))
            print("[FPGAI] memory placements:", len(memory_plan.placements))
            print("[FPGAI] communication edges:", len(communication_plan.edges))
            if hls_run is not None:
                print("[FPGAI] vitis_hls ok:", hls_run.ok)
                print("[FPGAI] vitis_hls stdout:", hls_run.stdout_log)
                print("[FPGAI] vitis_hls stderr:", hls_run.stderr_log)

        return CompileResult(
            out_dir=out_dir,
            graph=g,
            hls_project_dir=hls_dir,
            host_project_dir=host_dir,
            hls_ran=(hls_run is not None),
            hls_ok=(hls_run.ok if hls_run is not None else None),
            hls_returncode=(hls_run.returncode if hls_run is not None else None),
            hls_stdout_log=(hls_run.stdout_log if hls_run is not None else None),
            hls_stderr_log=(hls_run.stderr_log if hls_run is not None else None),
            hls_csynth_report=(hls_run.csynth_report if hls_run is not None else None),
        )

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

    def _emit_ir_artifacts(
        self,
        out_dir: Path,
        g,
        descriptors,
        compile_plan,
        memory_plan,
        communication_plan,
    ) -> None:
        write_text(out_dir / "ir_summary.txt", g.summary())

        part_plan = single_device_plan(g, device_id="fpga0")
        write_text(out_dir / "partition_plan.json", json.dumps(part_plan.to_dict(), indent=2))

        ir_dir = out_dir / "ir"
        ir_dir.mkdir(parents=True, exist_ok=True)

        write_text(
            ir_dir / "descriptors.json",
            json.dumps([d.to_dict() for d in descriptors], indent=2),
        )
        write_text(
            ir_dir / "compile_plan.json",
            json.dumps(compile_plan.to_dict(), indent=2),
        )
        write_text(
            ir_dir / "memory_plan.json",
            json.dumps(memory_plan.to_dict(), indent=2),
        )
        write_text(
            ir_dir / "comm_plan.json",
            json.dumps(communication_plan.to_dict(), indent=2),
        )

    def _emit_dummy_input(self, out_dir: Path, g) -> Path:
        x_name = g.inputs[0]
        x_spec = g.get_tensor(x_name)

        if x_spec and x_spec.shape:
            in_words = 1
            for d in x_spec.shape:
                in_words *= int(d)
        else:
            in_words = 1

        x = (np.arange(in_words, dtype=np.float32) + 1.0) * 0.1
        p = out_dir / "input.bin"
        write_f32_bin(p, x)
        return p

    def _emit_hls(
        self,
        out_dir: Path,
        g,
        *,
        top_name: str,
        weights_mode: str,
        compile_plan=None,
        memory_plan=None,
        communication_plan=None,
    ) -> Path:
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
            compile_plan=compile_plan,
            memory_plan=memory_plan,
            communication_plan=communication_plan,
        )
        return proj.hls_dir

    def _maybe_run_vitis_hls(self, hls_dir: Path):
        raw = self.cfg.raw

        run_enabled = bool(_cfg_get(raw, "toolchain.vitis_hls.enabled", False))
        if not run_enabled:
            return None

        vitis_exe = str(
            _cfg_get(
                raw,
                "backends.hls.vitis.exe",
                _cfg_get(raw, "toolchain.vitis_hls.exe", "vitis_hls"),
            )
        )

        settings64 = _cfg_get(raw, "toolchain.vitis_hls.settings64", None)

        from fpgai.backends.hls.runner import run_vitis_hls

        return run_vitis_hls(
            hls_dir=hls_dir,
            vitis_hls_exe=vitis_exe,
            settings64=settings64,
        )

    def _emit_hostcpp(self, out_dir: Path, g, *, top_name: str) -> Path:
        from fpgai.backends.hostcpp.emit_host_model import emit_hostcpp_project
        return emit_hostcpp_project(g, out_dir, top_name=top_name)

    def _emit_manifest(
        self,
        *,
        out_dir: Path,
        top_name: str,
        weights_mode: str,
        graph,
        descriptors,
        compile_plan,
        memory_plan,
        communication_plan,
        hls_run,
        seconds: float,
    ) -> None:
        manifest = {
            "version": self.cfg.version,
            "model_path": self.cfg.model.path,
            "pipeline_mode": self.cfg.pipeline.mode,
            "top_kernel_name": top_name,
            "weights_mode": weights_mode,
            "out_dir": str(out_dir),
            "num_ops": len(graph.ops),
            "num_params": len(graph.params),
            "num_descriptors": len(descriptors),
            "num_layer_plans": len(compile_plan.layer_plans),
            "num_memory_placements": len(memory_plan.placements),
            "num_communication_edges": len(communication_plan.edges),
            "memory_totals": memory_plan.total_bytes_by_region,
            "ops": [{"name": op.name, "type": op.op_type} for op in graph.ops],
            "hls_ran": hls_run is not None,
            "hls_ok": (hls_run.ok if hls_run is not None else None),
            "hls_returncode": (hls_run.returncode if hls_run is not None else None),
            "hls_stdout_log": (hls_run.stdout_log if hls_run is not None else None),
            "hls_stderr_log": (hls_run.stderr_log if hls_run is not None else None),
            "hls_csynth_report": (hls_run.csynth_report if hls_run is not None else None),
            "seconds": round(float(seconds), 6),
        }
        write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2))