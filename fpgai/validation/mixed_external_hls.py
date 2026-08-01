from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import shutil
import subprocess

import numpy as np

from fpgai.operators.external.external_api import ReferenceExecutionContext


@dataclass(frozen=True)
class MixedExternalValidationArtifacts:
    input_bin: Path
    expected_bin: Path
    output_bin: Path
    report_path: Path
    input_values: tuple[float, ...]
    expected_values: tuple[float, ...]


def _builtin_reference(op_type: str, value: np.ndarray, attrs: Mapping[str, Any]) -> np.ndarray:
    token = str(op_type).lower()
    if token in {"relu", "rectifiedlinear"}:
        return np.maximum(value, 0.0).astype(np.float32)
    if token in {"sigmoid", "logistic"}:
        return (1.0 / (1.0 + np.exp(-value))).astype(np.float32)
    if token in {"identity", "flatten", "reshape"}:
        return value.astype(np.float32, copy=True)
    if token == "leakyrelu":
        alpha = float(attrs.get("alpha", 0.01))
        return np.where(value >= 0.0, value, alpha * value).astype(np.float32)
    raise ValueError(f"No maintained mixed-HLS reference implementation for built-in operator {op_type!r}")


def execute_mixed_graph_reference(graph: Any, external_context: Any, input_values: np.ndarray) -> np.ndarray:
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise ValueError("Mixed external validation currently requires one graph input and one graph output")
    tensors: dict[str, np.ndarray] = {graph.inputs[0]: np.asarray(input_values, dtype=np.float32).reshape(-1)}
    for op in graph.ops:
        if len(op.inputs) != 1 or len(op.outputs) != 1:
            raise ValueError(f"Reference validation requires one input/output for node {op.name!r}")
        value = tensors[op.inputs[0]]
        provenance = op.attrs.get("_fpgai_external_operator")
        if isinstance(provenance, Mapping):
            operator_id = str(provenance.get("operator_id", ""))
            callback = external_context.reference_for(operator_id)
            if callback is None:
                raise ValueError(f"External operator {operator_id!r} has no numeric reference callback")
            clean_attrs = {k: v for k, v in op.attrs.items() if not str(k).startswith("_fpgai_")}
            result = callback(ReferenceExecutionContext(attributes=clean_attrs, inputs=(value,)))
            if len(result.outputs) != 1:
                raise ValueError(f"External reference for {operator_id!r} returned {len(result.outputs)} outputs")
            output = np.asarray(result.outputs[0], dtype=np.float32).reshape(-1)
        else:
            output = _builtin_reference(op.op_type, value, op.attrs)
        tensors[op.outputs[0]] = output
    return tensors[graph.outputs[0]].astype(np.float32, copy=False)


def _write_f32(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype=np.float32).tofile(path)


def prepare_mixed_external_validation(
    *, graph: Any, external_context: Any, out_dir: Path, config: Mapping[str, Any]
) -> MixedExternalValidationArtifacts:
    raw_values = config.get("input_values", (-2.0, -0.5, 0.5, 2.0))
    if not isinstance(raw_values, (list, tuple)) or not raw_values:
        raise ValueError("ecosystem.validation.input_values must be a non-empty list")
    input_values = np.asarray([float(v) for v in raw_values], dtype=np.float32)
    if not np.all(np.isfinite(input_values)):
        raise ValueError("ecosystem.validation.input_values must be finite")
    expected = execute_mixed_graph_reference(graph, external_context, input_values)
    input_bin = out_dir / "input.bin"
    expected_bin = out_dir / "reference_output.bin"
    output_bin = out_dir / "output.bin"
    report_path = out_dir / "reports" / "mixed_external_hls_validation.json"
    _write_f32(input_bin, input_values)
    _write_f32(expected_bin, expected)
    payload = {
        "schema": "fpgai.mixed-external-hls-validation/v1",
        "status": "prepared",
        "input_bin": str(input_bin),
        "reference_output_bin": str(expected_bin),
        "hls_output_bin": str(output_bin),
        "input_values": input_values.tolist(),
        "reference_values": expected.tolist(),
        "atol": float(config.get("atol", 1e-5)),
        "rtol": float(config.get("rtol", 1e-5)),
        "host_cpp": {"status": "generated_project_ready"},
        "vitis_csim": {"status": "not_run"},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return MixedExternalValidationArtifacts(
        input_bin, expected_bin, output_bin, report_path,
        tuple(float(v) for v in input_values), tuple(float(v) for v in expected),
    )


def finalize_mixed_external_validation(artifacts: MixedExternalValidationArtifacts, *, hls_run: Any | None, atol: float, rtol: float) -> dict[str, Any]:
    payload = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    payload["vitis_csim"] = {
        "status": "not_run" if hls_run is None else ("passed" if hls_run.csim_ok else "failed"),
        "ran": None if hls_run is None else hls_run.csim_ran,
        "ok": None if hls_run is None else hls_run.csim_ok,
        "stdout_log": None if hls_run is None else hls_run.stdout_log,
        "stderr_log": None if hls_run is None else hls_run.stderr_log,
    }
    if artifacts.output_bin.exists():
        actual = np.fromfile(artifacts.output_bin, dtype=np.float32)
        expected = np.asarray(artifacts.expected_values, dtype=np.float32)
        if actual.size != expected.size:
            comparison = {"status": "failed", "reason": "output_length_mismatch", "expected": int(expected.size), "actual": int(actual.size)}
        else:
            diff = np.abs(actual - expected)
            ok = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
            comparison = {
                "status": "passed" if ok else "failed",
                "outputs": int(actual.size),
                "max_abs": float(diff.max()) if diff.size else 0.0,
                "mean_abs": float(diff.mean()) if diff.size else 0.0,
                "actual_values": actual.tolist(),
            }
        payload["hls_vs_reference"] = comparison
        payload["status"] = comparison["status"]
    elif hls_run is not None:
        payload["status"] = "failed"
        payload["hls_vs_reference"] = {"status": "failed", "reason": "output_bin_missing"}
    artifacts.report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_portable_host_cpp_validation(
    *, graph: Any, composition_plan: Any, artifacts: MixedExternalValidationArtifacts, hls_dir: Path
) -> dict[str, Any]:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        return {"status": "skipped", "reason": "cxx_compiler_not_found"}
    work = artifacts.report_path.parent / "host_cpp"
    work.mkdir(parents=True, exist_ok=True)
    source = work / "mixed_graph_host.cpp"
    executable = work / "mixed_graph_host"
    host_output = work / "output.bin"
    input_values = list(artifacts.input_values)
    lines = [
        "#include <algorithm>", "#include <cmath>", "#include <cstdio>", "#include <fstream>", "#include <vector>", "",
    ]
    declared: set[str] = set()
    for binding in composition_plan.bindings:
        if binding.contract.top not in declared:
            integration = dict(binding.contract.metadata.get("integration", {})).get("hls", {})
            attrs = integration.get("attributes", []) if isinstance(integration, Mapping) else []
            attr_types = [str(item.get("cpp_type", "float")) for item in attrs if isinstance(item, Mapping)]
            suffix = "".join(f", {kind}" for kind in attr_types)
            lines.append(f"void {binding.contract.top}(const float*, float*, int{suffix});")
            declared.add(binding.contract.top)
    lines += ["", "int main() {", f"  std::vector<float> current = {{{', '.join(repr(float(v))+'f' for v in input_values)}}};"]
    for op in graph.ops:
        binding = composition_plan.binding_for_node(op.name)
        if binding is not None:
            integration = dict(binding.contract.metadata.get("integration", {})).get("hls", {})
            attr_specs = integration.get("attributes", []) if isinstance(integration, Mapping) else []
            args=[]
            for item in attr_specs:
                if not isinstance(item, Mapping):
                    continue
                name=str(item.get("name", "")); default=item.get("default", 0.0)
                args.append(repr(float(binding.attributes.get(name, default)))+"f")
            extra = "".join(", "+arg for arg in args)
            lines += [
                "  {", "    std::vector<float> next(current.size());",
                f"    {binding.contract.top}(current.data(), next.data(), (int)current.size(){extra});",
                "    current.swap(next);", "  }",
            ]
        elif str(op.op_type).lower() == "relu":
            lines.append("  for (float& value : current) value = std::max(0.0f, value);")
        elif str(op.op_type).lower() == "sigmoid":
            lines.append("  for (float& value : current) value = 1.0f / (1.0f + std::exp(-value));")
        elif str(op.op_type).lower() in {"identity", "flatten", "reshape"}:
            pass
        else:
            return {"status": "skipped", "reason": f"unsupported_host_operator:{op.op_type}"}
    lines += [
        f'  std::ofstream out("{host_output.as_posix()}", std::ios::binary);',
        "  out.write(reinterpret_cast<const char*>(current.data()), (std::streamsize)(current.size()*sizeof(float)));",
        "  return out ? 0 : 2;", "}",
    ]
    source.write_text("\n".join(lines)+"\n", encoding="utf-8")
    external_sources = sorted((hls_dir / "src" / "external").rglob("*.cpp"))
    include_dirs = sorted({path.parent for path in (hls_dir / "include" / "external").rglob("*.hpp")})
    command=[compiler,"-std=c++17","-O2",str(source),*(str(path) for path in external_sources)]
    for include_dir in include_dirs:
        command.extend(["-I",str(include_dir)])
    command.extend(["-o",str(executable)])
    try:
        subprocess.check_call(command, cwd=str(work))
        subprocess.check_call([str(executable)], cwd=str(work))
    except subprocess.CalledProcessError as exc:
        return {"status":"failed","returncode":int(exc.returncode),"source":str(source),"command":command}
    actual=np.fromfile(host_output,dtype=np.float32)
    expected=np.asarray(artifacts.expected_values,dtype=np.float32)
    diff=np.abs(actual-expected) if actual.size==expected.size else np.array([],dtype=np.float32)
    ok=actual.size==expected.size and bool(np.allclose(actual,expected,atol=1e-6,rtol=1e-6))
    return {
        "status":"passed" if ok else "failed", "source":str(source), "executable":str(executable),
        "output_bin":str(host_output), "outputs":int(actual.size),
        "max_abs":float(diff.max()) if diff.size else None, "mean_abs":float(diff.mean()) if diff.size else None,
    }
