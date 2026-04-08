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
from fpgai.engine.layerwise_precision import resolve_layerwise_precision
from fpgai.engine.training import build_training_plan, emit_training_artifacts
from fpgai.analysis.quantization_report import run_quantization_report
from fpgai.analysis.precision_sweep import run_precision_sweep
from fpgai.analysis.design_space_report import run_design_space_report
from fpgai.analysis.hls_estimate_compare import run_estimate_vs_hls_compare
from fpgai.analysis.training_resource_estimate import run_training_resource_estimate
from fpgai.benchmark.training_reference import run_training_reference_step
from fpgai.benchmark.training_compare import compare_training_artifacts
from fpgai.util.fs import ensure_clean_dir, write_text
from fpgai.util.binio import write_f32_bin

from fpgai.backends.hls.emit.types_h import emit_types_h
from fpgai.backends.hls.emit.top_cpp import emit_top_cpp
from fpgai.backends.hls.emit.top_train_cpp import emit_top_train_cpp
from fpgai.backends.hls.emit.layers_dense import emit_dense_h, emit_dense_cpp
from fpgai.backends.hls.emit.layers_conv import emit_conv_h, emit_conv_cpp
from fpgai.backends.hls.emit.layers_pool import emit_pool_h, emit_pool_cpp
from fpgai.backends.hls.emit.layers_activations import emit_activations_h, emit_activations_cpp
from fpgai.backends.hls.emit.layers_batchnorm import emit_batchnorm_h, emit_batchnorm_cpp
from fpgai.backends.hls.emit.csim_tcl import emit_csim_tcl
from fpgai.backends.hls.emit.csim_train_tcl import emit_csim_train_tcl


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
        mode = str(self.cfg.pipeline.mode).lower()
        if mode == "inference":
            return self._compile_inference()
        if mode == "training_on_device":
            return self._compile_training()
        raise RuntimeError(f"Unsupported pipeline mode: {self.cfg.pipeline.mode}")

    def _compile_inference(self) -> CompileResult:
        raw = self.cfg.raw
        t0 = time.time()
        out_dir = self._prepare_out_dir(raw)
        top_name = str(_cfg_get(raw, "pipeline.outputs.top_kernel_name", "deeplearn"))
        verbose = bool(_cfg_get(raw, "debug.verbose", False))
        emit_manifest = bool(_cfg_get(raw, "project.reproducibility.emit_manifest", True))
        enable_hls = bool(_cfg_get(raw, "backends.hls.enabled", True))
        enable_host = bool(_cfg_get(raw, "backends.host_cpp.enabled", True))
        enable_quant_report = bool(_cfg_get(raw, "analysis.quantization_report.enabled", False))
        enable_precision_sweep = bool(_cfg_get(raw, "analysis.precision_sweep.enabled", False))
        enable_design_space = bool(_cfg_get(raw, "analysis.design_space.enabled", False))
        act_kind, act_alpha, act_except_last = self._read_activation_insert_cfg(raw)
        weights_mode = str(_cfg_get(raw, "data_movement.ps_pl.weights.mode", "embedded")).lower()

        g = self._import_and_prepare_graph(act_kind=act_kind, act_alpha=act_alpha, act_except_last=act_except_last)
        resolve_layerwise_precision(g, raw)

        descriptors = analyze_graph(g)
        compile_plan = make_compile_plan(self.cfg, descriptors)
        memory_plan = make_memory_plan(g, descriptors, compile_plan)
        communication_plan = make_communication_plan(self.cfg, memory_plan)

        self._emit_ir_artifacts(out_dir, g, descriptors, compile_plan, memory_plan, communication_plan)
        self._emit_dummy_input(out_dir, g)

        quant_result = run_quantization_report(model_path=self.cfg.model.path, raw_cfg=raw, out_dir=out_dir) if enable_quant_report else None
        sweep_result = run_precision_sweep(model_path=self.cfg.model.path, raw_cfg=raw, out_dir=out_dir) if enable_precision_sweep else None
        design_result = run_design_space_report(graph=g, model_path=self.cfg.model.path, raw_cfg=raw, out_dir=out_dir) if enable_design_space else None
        if design_result is not None and bool(_cfg_get(raw, "analysis.design_space.print_terminal_summary", True)):
            print("\n" + design_result.terminal_summary + "\n")

        hls_dir: Optional[Path] = self._emit_hls(out_dir, g, top_name=top_name, weights_mode=weights_mode, compile_plan=compile_plan, memory_plan=memory_plan, communication_plan=communication_plan) if enable_hls else None
        host_dir: Optional[Path] = self._emit_hostcpp(out_dir, g, top_name=top_name) if enable_host else None
        hls_run = self._maybe_run_vitis_hls(hls_dir) if enable_hls and hls_dir is not None else None

        estimate_vs_hls_result = None
        if design_result is not None:
            best = None
            try:
                ds_payload = json.loads(design_result.results_json.read_text(encoding="utf-8"))
                best = ds_payload.get("recommended_balanced") or ds_payload.get("recommended_smallest_valid") or ds_payload.get("recommended_best_accuracy")
            except Exception:
                best = None
            if best is not None:
                estimate_vs_hls_result = run_estimate_vs_hls_compare(
                    out_dir=out_dir,
                    design_space_summary=best,
                    csynth_report_path=(hls_run.csynth_report if hls_run is not None else None),
                    clock_mhz=float(_cfg_get(raw, "targets.platform.clocks.0.target_mhz", 200.0)),
                )
                print("\n" + estimate_vs_hls_result.terminal_summary + "\n")

        if emit_manifest:
            self._emit_manifest(
                out_dir=out_dir, top_name=top_name, weights_mode=weights_mode, graph=g, descriptors=descriptors,
                compile_plan=compile_plan, memory_plan=memory_plan, communication_plan=communication_plan,
                hls_run=hls_run, quant_result=quant_result, sweep_result=sweep_result, design_result=design_result,
                estimate_vs_hls_result=estimate_vs_hls_result, training_plan=None, training_reference_result=None,
                training_compare_result=None, training_estimate_result=None, seconds=time.time() - t0,
            )

        if verbose:
            if quant_result is not None:
                print("[FPGAI] quant_report:", quant_result.summary_txt)
            if sweep_result is not None:
                print("[FPGAI] precision_sweep:", sweep_result.summary_txt)
            if design_result is not None:
                print("[FPGAI] design_space:", design_result.summary_txt)
            if estimate_vs_hls_result is not None:
                print("[FPGAI] estimate_vs_hls:", estimate_vs_hls_result.summary_txt)

        return CompileResult(
            out_dir=out_dir, graph=g, hls_project_dir=hls_dir, host_project_dir=host_dir,
            hls_ran=(hls_run is not None), hls_ok=(hls_run.ok if hls_run is not None else None),
            hls_returncode=(hls_run.returncode if hls_run is not None else None),
            hls_stdout_log=(hls_run.stdout_log if hls_run is not None else None),
            hls_stderr_log=(hls_run.stderr_log if hls_run is not None else None),
            hls_csynth_report=(hls_run.csynth_report if hls_run is not None else None),
            quant_report_dir=(quant_result.out_dir if quant_result is not None else None),
            quant_metrics_json=(quant_result.metrics_json if quant_result is not None else None),
            quant_summary_txt=(quant_result.summary_txt if quant_result is not None else None),
            quant_layerwise_csv=(quant_result.layerwise_csv if quant_result is not None else None),
            precision_sweep_dir=(sweep_result.out_dir if sweep_result is not None else None),
            precision_sweep_results_json=(sweep_result.results_json if sweep_result is not None else None),
            precision_sweep_summary_txt=(sweep_result.summary_txt if sweep_result is not None else None),
            precision_sweep_results_csv=(sweep_result.results_csv if sweep_result is not None else None),
            design_space_dir=(design_result.out_dir if design_result is not None else None),
            design_space_results_json=(design_result.results_json if design_result is not None else None),
            design_space_summary_txt=(design_result.summary_txt if design_result is not None else None),
            design_space_results_csv=(design_result.results_csv if design_result is not None else None),
            design_space_terminal_summary=(design_result.terminal_summary if design_result is not None else None),
            training_plan_json=None, training_summary_txt=None,
        )

    def _compile_training(self) -> CompileResult:
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

        g = self._import_and_prepare_graph(act_kind=act_kind, act_alpha=act_alpha, act_except_last=act_except_last)
        resolve_layerwise_precision(g, raw)

        descriptors = analyze_graph(g)
        compile_plan = make_compile_plan(self.cfg, descriptors)
        memory_plan = make_memory_plan(g, descriptors, compile_plan)
        communication_plan = make_communication_plan(self.cfg, memory_plan)

        self._emit_ir_artifacts(out_dir, g, descriptors, compile_plan, memory_plan, communication_plan)

        training_plan = build_training_plan(g, raw, compile_plan=compile_plan, memory_plan=memory_plan, communication_plan=communication_plan)
        emit_training_artifacts(out_dir, training_plan)

        training_estimate_result = None
        if bool(training_plan.estimator.get("enabled", True)):
            training_estimate_result = run_training_resource_estimate(graph=g, training_plan=training_plan, out_dir=out_dir)
            print("\n" + training_estimate_result.summary_txt.read_text(encoding="utf-8") + "\n")

        input_path = self._emit_dummy_input(out_dir, g)
        target_path = self._emit_training_target(out_dir, g, raw)
        x_input = np.fromfile(input_path, dtype=np.float32)
        y_target = np.fromfile(target_path, dtype=np.float32)

        training_reference_result = run_training_reference_step(graph=g, raw_cfg=raw, out_dir=out_dir, x_input=x_input, target=y_target)

        hls_dir: Optional[Path] = self._emit_hls(out_dir, g, top_name=top_name, weights_mode=weights_mode, compile_plan=compile_plan, memory_plan=memory_plan, communication_plan=communication_plan) if enable_hls else None
        host_dir: Optional[Path] = self._emit_hostcpp(out_dir, g, top_name=top_name) if enable_host else None
        hls_run = self._maybe_run_vitis_hls(hls_dir) if enable_hls and hls_dir is not None else None

        training_compare_result = None
        if hls_run is not None and hls_dir is not None:
            hls_grads = self._find_file_recursive(hls_dir, "grads.bin")
            hls_w_before = self._find_file_recursive(hls_dir, "weights_before.bin")
            hls_w_after = self._find_file_recursive(hls_dir, "weights_after.bin")
            if hls_grads is not None and hls_w_before is not None and hls_w_after is not None:
                training_compare_result = compare_training_artifacts(
                    out_dir=out_dir,
                    ref_grads_bin=training_reference_result.grads_flat_path,
                    ref_weights_before_bin=training_reference_result.weights_before_flat_path,
                    ref_weights_after_bin=training_reference_result.weights_after_flat_path,
                    hls_grads_bin=hls_grads,
                    hls_weights_before_bin=hls_w_before,
                    hls_weights_after_bin=hls_w_after,
                )
                print("\n" + training_compare_result.summary_txt.read_text(encoding="utf-8") + "\n")

        if emit_manifest:
            self._emit_manifest(
                out_dir=out_dir, top_name=top_name, weights_mode=weights_mode, graph=g, descriptors=descriptors,
                compile_plan=compile_plan, memory_plan=memory_plan, communication_plan=communication_plan,
                hls_run=hls_run, quant_result=None, sweep_result=None, design_result=None,
                estimate_vs_hls_result=None, training_plan=training_plan,
                training_reference_result=training_reference_result, training_compare_result=training_compare_result,
                training_estimate_result=training_estimate_result, seconds=time.time() - t0,
            )

        if verbose:
            print("[FPGAI] training mode enabled")
            print(f"[FPGAI] training optimizer: {training_plan.optimizer_type}")
            print(f"[FPGAI] training loss: {training_plan.loss_type}")
            print(f"[FPGAI] training weights_mode: {training_plan.weights_mode}")
            print(f"[FPGAI] training parallel policy: {training_plan.planner_policy.get('parallel_policy')}")
            print(f"[FPGAI] training reference: {training_reference_result.summary_txt}")
            if training_estimate_result is not None:
                print(f"[FPGAI] training estimate: {training_estimate_result.summary_txt}")
            if training_compare_result is not None:
                print(f"[FPGAI] training compare: {training_compare_result.summary_txt}")

        return CompileResult(
            out_dir=out_dir, graph=g, hls_project_dir=hls_dir, host_project_dir=host_dir,
            hls_ran=(hls_run is not None), hls_ok=(hls_run.ok if hls_run is not None else None),
            hls_returncode=(hls_run.returncode if hls_run is not None else None),
            hls_stdout_log=(hls_run.stdout_log if hls_run is not None else None),
            hls_stderr_log=(hls_run.stderr_log if hls_run is not None else None),
            hls_csynth_report=(hls_run.csynth_report if hls_run is not None else None),
            quant_report_dir=None, quant_metrics_json=None, quant_summary_txt=None, quant_layerwise_csv=None,
            precision_sweep_dir=None, precision_sweep_results_json=None, precision_sweep_summary_txt=None, precision_sweep_results_csv=None,
            design_space_dir=None, design_space_results_json=None, design_space_summary_txt=None, design_space_results_csv=None,
            design_space_terminal_summary=None,
            training_plan_json=(out_dir / "training" / "training_plan.json"),
            training_summary_txt=(out_dir / "training" / "summary.txt"),
        )

    def _prepare_out_dir(self, raw: Dict[str, Any]) -> Path:
        out_dir = Path(_cfg_get(raw, "project.out_dir", "build/fpgai")).resolve()
        clean = bool(_cfg_get(raw, "project.clean", True))
        ensure_clean_dir(out_dir, clean=clean)
        return out_dir

    def _read_activation_insert_cfg(self, raw: Dict[str, Any]) -> tuple[str, float, bool]:
        act_cfg = _cfg_get(raw, "operators.defaults.activation_insert", {}) or {}
        return str(act_cfg.get("kind", "none")).lower(), float(act_cfg.get("alpha", 0.1)), bool(act_cfg.get("except_last", True))

    def _import_and_prepare_graph(self, *, act_kind: str, act_alpha: float, act_except_last: bool):
        from fpgai.frontend.onnx import import_onnx
        from fpgai.ir.passes import validate_allowlist, assign_stable_names, insert_activations
        g = import_onnx(self.cfg.model.path, canonicalize=True, infer_shapes=True)
        if act_kind != "none":
            g = insert_activations(g, kind=act_kind, alpha=act_alpha, except_last=act_except_last)
        g = assign_stable_names(g)
        validate_allowlist(g, self.cfg.operators.supported)
        return g

    def _emit_ir_artifacts(self, out_dir: Path, g, descriptors, compile_plan, memory_plan, communication_plan) -> None:
        write_text(out_dir / "ir_summary.txt", g.summary())
        part_plan = single_device_plan(g, device_id="fpga0")
        write_text(out_dir / "partition_plan.json", json.dumps(part_plan.to_dict(), indent=2))
        ir_dir = out_dir / "ir"
        ir_dir.mkdir(parents=True, exist_ok=True)
        write_text(ir_dir / "descriptors.json", json.dumps([d.to_dict() for d in descriptors], indent=2))
        write_text(ir_dir / "compile_plan.json", json.dumps(compile_plan.to_dict(), indent=2))
        write_text(ir_dir / "memory_plan.json", json.dumps(memory_plan.to_dict(), indent=2))
        write_text(ir_dir / "comm_plan.json", json.dumps(communication_plan.to_dict(), indent=2))
        prec_dump = []
        for idx, op in enumerate(g.ops):
            prec_dump.append({"index": idx, "name": op.name, "op_type": op.op_type, "precision": op.attrs.get("precision"), "precision_tag": op.attrs.get("precision_tag")})
        write_text(ir_dir / "layerwise_precision.json", json.dumps(prec_dump, indent=2))

    def _emit_dummy_input(self, out_dir: Path, g) -> Path:
        p = out_dir / "input.bin"
        if p.exists():
            return p
        x_name = g.inputs[0]
        x_spec = g.get_tensor(x_name)
        in_words = int(np.prod(tuple(int(d) for d in x_spec.shape))) if x_spec and x_spec.shape else 1
        x = (np.arange(in_words, dtype=np.float32) + 1.0) * 0.1
        write_f32_bin(p, x)
        return p

    def _emit_training_target(self, out_dir: Path, g, raw: Dict[str, Any]) -> Path:
        p = out_dir / "target.bin"
        if p.exists():
            return p
        y_name = g.outputs[0]
        y_spec = g.get_tensor(y_name)
        out_words = 1
        if y_spec is not None and getattr(y_spec, "shape", None):
            shape = tuple(int(x) for x in y_spec.shape)
            if len(shape) > 1 and shape[0] == 1:
                shape = shape[1:]
            out_words = int(np.prod(shape)) if shape else 1
        target = np.zeros((out_words,), dtype=np.float32)
        if out_words > 0:
            target[0] = 1.0
        write_f32_bin(p, target)
        return p

    def _find_file_recursive(self, root: Path, filename: str) -> Optional[Path]:
        for p in root.rglob(filename):
            return p
        return None

    def _emit_hls(self, out_dir: Path, g, *, top_name: str, weights_mode: str, compile_plan=None, memory_plan=None, communication_plan=None) -> Path:
        from fpgai.backends.hls.codegen import emit_hls_stub
        raw = self.cfg.raw
        part = str(_cfg_get(raw, "targets.platform.part", "xck26-sfvc784-2LV-c"))
        clk_mhz = float(_cfg_get(raw, "targets.platform.clocks.0.target_mhz", 200))
        intermediate_dump = bool(_cfg_get(raw, "benchmark.intermediate.enabled", False))
        pipeline_mode = str(self.cfg.pipeline.mode).lower()
        training_cfg = (_cfg_get(raw, "training", {}) or {})

        proj = emit_hls_stub(
            graph=g, out_dir=out_dir, top_name=top_name,
            hls_options={
                "weights_mode": weights_mode, "part": part, "clk_mhz": int(clk_mhz), "proj_name": "fpgai_hls_proj",
                "intermediate_dump": intermediate_dump, "pipeline_mode": pipeline_mode, "training_cfg": training_cfg, "raw_cfg": raw,
            },
            compile_plan=compile_plan, memory_plan=memory_plan, communication_plan=communication_plan,
        )
        hls_dir = proj.hls_dir
        inc_dir = hls_dir / "include"
        layers_inc_dir = inc_dir / "layers"
        src_dir = hls_dir / "src"
        layers_src_dir = src_dir / "layers"
        inc_dir.mkdir(parents=True, exist_ok=True)
        layers_inc_dir.mkdir(parents=True, exist_ok=True)
        src_dir.mkdir(parents=True, exist_ok=True)
        layers_src_dir.mkdir(parents=True, exist_ok=True)

        write_text(inc_dir / "fpgai_types.h", emit_types_h(g, top_name=top_name, raw_cfg=raw, compile_plan=compile_plan))
        write_text(layers_inc_dir / "dense.h", emit_dense_h())
        write_text(layers_src_dir / "dense.cpp", emit_dense_cpp())
        write_text(layers_inc_dir / "conv.h", emit_conv_h())
        write_text(layers_src_dir / "conv.cpp", emit_conv_cpp())
        write_text(layers_inc_dir / "pool.h", emit_pool_h())
        write_text(layers_src_dir / "pool.cpp", emit_pool_cpp())
        write_text(layers_inc_dir / "activations.h", emit_activations_h())
        write_text(layers_src_dir / "activations.cpp", emit_activations_cpp())
        write_text(layers_inc_dir / "batchnorm.h", emit_batchnorm_h())
        write_text(layers_src_dir / "batchnorm.cpp", emit_batchnorm_cpp())

        if pipeline_mode == "training_on_device":
            write_text(src_dir / f"{top_name}.cpp", emit_top_train_cpp(graph=g, top_name=top_name, weights_mode=weights_mode, training_cfg=training_cfg, compile_plan=compile_plan, memory_plan=memory_plan, communication_plan=communication_plan))
            input_bin = str((out_dir / "input.bin").resolve())
            target_bin = str((out_dir / "target.bin").resolve())
            write_text(hls_dir / "run_hls.tcl", emit_csim_train_tcl(top_name=top_name, part=part, input_bin_path=input_bin, target_bin_path=target_bin, weights_mode=weights_mode, intermediate_dump=intermediate_dump))
        else:
            write_text(src_dir / f"{top_name}.cpp", emit_top_cpp(g, top_name=top_name, weights_mode=weights_mode, compile_plan=compile_plan, memory_plan=memory_plan, communication_plan=communication_plan))
            input_bin = str((out_dir / "input.bin").resolve())
            write_text(hls_dir / "run_hls.tcl", emit_csim_tcl(top_name=top_name, part=part, clk_period_ns=(1000.0 / clk_mhz), input_bin_path=input_bin, weights_mode=weights_mode, intermediate_dump=intermediate_dump))
        return hls_dir

    def _maybe_run_vitis_hls(self, hls_dir: Path):
        raw = self.cfg.raw
        run_enabled = bool(_cfg_get(raw, "toolchain.vitis_hls.enabled", False))
        if not run_enabled:
            return None
        vitis_exe = str(_cfg_get(raw, "backends.hls.vitis.exe", _cfg_get(raw, "toolchain.vitis_hls.exe", "vitis_hls")))
        settings64 = _cfg_get(raw, "toolchain.vitis_hls.settings64", None)
        from fpgai.backends.hls.runner import run_vitis_hls
        return run_vitis_hls(hls_dir=hls_dir, vitis_hls_exe=vitis_exe, settings64=settings64)

    def _emit_hostcpp(self, out_dir: Path, g, *, top_name: str) -> Path:
        pipeline_mode = str(getattr(self.cfg.pipeline, "mode", "inference")).lower()
        if pipeline_mode == "training_on_device":
            from fpgai.backends.hostcpp.emit_host_train import emit_hostcpp_project_train
            return emit_hostcpp_project_train(g, out_dir, top_name=top_name, raw_cfg=self.cfg.raw)
        from fpgai.backends.hostcpp.emit_host_model import emit_hostcpp_project
        return emit_hostcpp_project(g, out_dir, top_name=top_name)

    def _emit_manifest(self, **kwargs) -> None:
        out_dir = kwargs["out_dir"]
        manifest = {
            "version": self.cfg.version,
            "model_path": self.cfg.model.path,
            "pipeline_mode": self.cfg.pipeline.mode,
            "top_kernel_name": kwargs["top_name"],
            "weights_mode": kwargs["weights_mode"],
            "out_dir": str(out_dir),
            "num_ops": len(kwargs["graph"].ops),
            "num_params": len(kwargs["graph"].params),
            "num_descriptors": len(kwargs["descriptors"]),
            "num_layer_plans": len(kwargs["compile_plan"].layer_plans),
            "num_memory_placements": len(kwargs["memory_plan"].placements),
            "num_communication_edges": len(kwargs["communication_plan"].edges),
            "memory_totals": kwargs["memory_plan"].total_bytes_by_region,
            "ops": [{"name": op.name, "type": op.op_type, "precision": op.attrs.get("precision"), "precision_tag": op.attrs.get("precision_tag")} for op in kwargs["graph"].ops],
            "training_plan": (None if kwargs["training_plan"] is None else kwargs["training_plan"].to_dict()),
            "training_reference": None if kwargs["training_reference_result"] is None else {
                "loss_before": kwargs["training_reference_result"].loss_before,
                "loss_after": kwargs["training_reference_result"].loss_after,
                "grads_ref_bin": str(kwargs["training_reference_result"].grads_flat_path),
                "weights_before_ref_bin": str(kwargs["training_reference_result"].weights_before_flat_path),
                "weights_after_ref_bin": str(kwargs["training_reference_result"].weights_after_flat_path),
                "summary_json": str(kwargs["training_reference_result"].summary_json),
                "summary_txt": str(kwargs["training_reference_result"].summary_txt),
            },
            "training_compare": None if kwargs["training_compare_result"] is None else {
                "out_dir": str(kwargs["training_compare_result"].out_dir),
                "results_json": str(kwargs["training_compare_result"].results_json),
                "summary_txt": str(kwargs["training_compare_result"].summary_txt),
                "grad_cosine": kwargs["training_compare_result"].grad_cosine,
                "weight_after_cosine": kwargs["training_compare_result"].weight_after_cosine,
                "weight_delta_cosine": kwargs["training_compare_result"].weight_delta_cosine,
                "grad_mae": kwargs["training_compare_result"].grad_mae,
                "grad_max_abs": kwargs["training_compare_result"].grad_max_abs,
                "weight_after_mae": kwargs["training_compare_result"].weight_after_mae,
                "weight_after_max_abs": kwargs["training_compare_result"].weight_after_max_abs,
            },
            "training_estimate": None if kwargs["training_estimate_result"] is None else {
                "out_dir": str(kwargs["training_estimate_result"].out_dir),
                "results_json": str(kwargs["training_estimate_result"].results_json),
                "summary_txt": str(kwargs["training_estimate_result"].summary_txt),
                "total_param_bytes": kwargs["training_estimate_result"].total_param_bytes,
                "total_activation_cache_bytes": kwargs["training_estimate_result"].total_activation_cache_bytes,
                "total_gradient_bytes": kwargs["training_estimate_result"].total_gradient_bytes,
                "total_optimizer_state_bytes": kwargs["training_estimate_result"].total_optimizer_state_bytes,
            },
            "quant_report": None if kwargs["quant_result"] is None else {
                "out_dir": str(kwargs["quant_result"].out_dir), "metrics_json": str(kwargs["quant_result"].metrics_json), "summary_txt": str(kwargs["quant_result"].summary_txt), "layerwise_csv": str(kwargs["quant_result"].layerwise_csv),
            },
            "precision_sweep": None if kwargs["sweep_result"] is None else {
                "out_dir": str(kwargs["sweep_result"].out_dir), "results_json": str(kwargs["sweep_result"].results_json), "summary_txt": str(kwargs["sweep_result"].summary_txt), "results_csv": str(kwargs["sweep_result"].results_csv),
            },
            "design_space": None if kwargs["design_result"] is None else {
                "out_dir": str(kwargs["design_result"].out_dir), "results_json": str(kwargs["design_result"].results_json), "summary_txt": str(kwargs["design_result"].summary_txt), "results_csv": str(kwargs["design_result"].results_csv),
            },
            "estimate_vs_hls": None if kwargs["estimate_vs_hls_result"] is None else {
                "out_dir": str(kwargs["estimate_vs_hls_result"].out_dir), "results_json": str(kwargs["estimate_vs_hls_result"].results_json), "summary_txt": str(kwargs["estimate_vs_hls_result"].summary_txt),
            },
            "hls_ran": kwargs["hls_run"] is not None,
            "hls_ok": (kwargs["hls_run"].ok if kwargs["hls_run"] is not None else None),
            "hls_returncode": (kwargs["hls_run"].returncode if kwargs["hls_run"] is not None else None),
            "hls_stdout_log": (str(kwargs["hls_run"].stdout_log) if kwargs["hls_run"] is not None and kwargs["hls_run"].stdout_log is not None else None),
            "hls_stderr_log": (str(kwargs["hls_run"].stderr_log) if kwargs["hls_run"] is not None and kwargs["hls_run"].stderr_log is not None else None),
            "hls_csynth_report": (str(kwargs["hls_run"].csynth_report) if kwargs["hls_run"] is not None and kwargs["hls_run"].csynth_report is not None else None),
            "seconds": round(float(kwargs["seconds"]), 6),
        }
        write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2))