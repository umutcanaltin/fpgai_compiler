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



def _shape_of(graph: Any, tensor_name: str) -> tuple[int, ...] | None:
    spec = getattr(graph, "tensors", {}).get(tensor_name) if hasattr(graph, "tensors") else None
    shape = getattr(spec, "shape", None)
    if not shape:
        return None
    try:
        dims = tuple(int(x) for x in shape)
    except (TypeError, ValueError):
        return None
    return dims if all(x > 0 for x in dims) else None


def _normalize_declared_tensor_shape(graph: Any, tensor_name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    declared = _shape_of(graph, tensor_name)
    if declared is None:
        return array
    expected_words = int(np.prod(declared))
    if array.size != expected_words:
        raise ValueError(
            f"Reference output for {tensor_name!r} has {array.size} values; "
            f"declared FPGAI tensor shape {declared} requires {expected_words}"
        )
    return array.reshape(declared)


def _conv2d_reference_nchw(
    value: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    attrs: Mapping[str, Any],
) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    if x.ndim != 4 or w.ndim != 4:
        raise ValueError("Maintained Conv reference currently requires NCHW input and OIHW weights")
    n, cin, ih, iw = x.shape
    cout, wcin, kh, kw = w.shape
    group = int(attrs.get("group", 1))
    if group != 1 or wcin != cin:
        raise ValueError("Maintained Conv reference currently supports group=1 with matching input channels")
    strides = tuple(int(v) for v in attrs.get("strides", (1, 1)))
    dilations = tuple(int(v) for v in attrs.get("dilations", (1, 1)))
    pads = tuple(int(v) for v in attrs.get("pads", (0, 0, 0, 0)))
    if len(strides) != 2 or len(dilations) != 2 or len(pads) != 4:
        raise ValueError("Maintained Conv reference requires 2D strides/dilations and four ONNX pads")
    sh, sw = strides; dh, dw = dilations
    pt, pl, pb, pr = pads
    eff_kh = dh * (kh - 1) + 1
    eff_kw = dw * (kw - 1) + 1
    oh = (ih + pt + pb - eff_kh) // sh + 1
    ow = (iw + pl + pr - eff_kw) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("Conv reference produced non-positive output dimensions")
    padded = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), mode="constant")
    out = np.zeros((n, cout, oh, ow), dtype=np.float32)
    for ni in range(n):
        for co in range(cout):
            for oy in range(oh):
                iy0 = oy * sh
                for ox in range(ow):
                    ix0 = ox * sw
                    acc = 0.0
                    for ci in range(cin):
                        for ky in range(kh):
                            iy = iy0 + ky * dh
                            for kx in range(kw):
                                ix = ix0 + kx * dw
                                acc += float(padded[ni, ci, iy, ix]) * float(w[co, ci, ky, kx])
                    if bias is not None:
                        acc += float(np.asarray(bias, dtype=np.float32).reshape(-1)[co])
                    out[ni, co, oy, ox] = acc
    return out

def execute_mixed_graph_trace(graph: Any, external_context: Any, input_values: np.ndarray) -> dict[str, np.ndarray]:
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise ValueError("Mixed external validation currently requires one graph input and one graph output")
    input_array = np.asarray(input_values, dtype=np.float32)
    input_shape = _shape_of(graph, graph.inputs[0])
    if input_shape is not None:
        expected_words = int(np.prod(input_shape))
        if input_array.size != expected_words:
            raise ValueError(
                f"Reference input for {graph.inputs[0]!r} has {input_array.size} values; expected {expected_words} from shape {input_shape}"
            )
        input_array = input_array.reshape(input_shape)
    tensors: dict[str, np.ndarray] = {graph.inputs[0]: input_array}
    constants_map = getattr(graph, "constants", {}) or {}
    constants = set(constants_map)
    for op in graph.ops:
        runtime_inputs = [name for name in op.inputs if name not in constants]
        provenance = op.attrs.get("_fpgai_external_operator")
        if isinstance(provenance, Mapping):
            if not runtime_inputs:
                raise ValueError(f"External reference node {op.name!r} requires at least one runtime input")
            missing = [name for name in runtime_inputs if name not in tensors]
            if missing:
                raise ValueError(f"External reference node {op.name!r} is missing runtime tensors {missing}")
            operator_id = str(provenance.get("operator_id", ""))
            callback = external_context.reference_for(operator_id)
            if callback is None:
                raise ValueError(f"External operator {operator_id!r} has no numeric reference callback")
            clean_attrs = {k: v for k, v in op.attrs.items() if not str(k).startswith("_fpgai_")}
            result = callback(ReferenceExecutionContext(
                attributes=clean_attrs,
                inputs=tuple(tensors[name] for name in runtime_inputs),
            ))
            if len(result.outputs) != len(op.outputs):
                raise ValueError(
                    f"External reference for {operator_id!r} returned {len(result.outputs)} outputs; "
                    f"node {op.name!r} declares {len(op.outputs)}"
                )
            for tensor_name, value in zip(op.outputs, result.outputs):
                tensors[tensor_name] = _normalize_declared_tensor_shape(graph, tensor_name, value)
            continue

        if len(op.outputs) != 1:
            raise ValueError(f"Built-in reference node {op.name!r} requires one output in the maintained profile")
        if str(op.op_type).lower() == "add":
            if len(runtime_inputs) != 2:
                raise ValueError(f"Add reference node {op.name!r} requires two runtime inputs")
            left = tensors[runtime_inputs[0]]
            right = tensors[runtime_inputs[1]]
            if left.shape != right.shape:
                raise ValueError(f"Add reference node {op.name!r} requires equal shapes")
            output = (left + right).astype(np.float32)
        elif str(op.op_type).lower() in {"conv", "conv2d"}:
            if len(runtime_inputs) != 1:
                raise ValueError(f"Conv reference node {op.name!r} requires one runtime activation input")
            constant_inputs = [name for name in op.inputs if name in constants]
            if not constant_inputs:
                raise ValueError(f"Conv reference node {op.name!r} requires constant weights")
            weight = constants_map[constant_inputs[0]]
            bias = constants_map[constant_inputs[1]] if len(constant_inputs) > 1 else None
            output = _conv2d_reference_nchw(tensors[runtime_inputs[0]], weight, bias, op.attrs)
        else:
            if len(runtime_inputs) != 1:
                raise ValueError(f"Reference validation requires one runtime input for node {op.name!r}")
            output = _builtin_reference(op.op_type, tensors[runtime_inputs[0]], op.attrs)
        tensors[op.outputs[0]] = _normalize_declared_tensor_shape(graph, op.outputs[0], output)
    return {name: np.asarray(value, dtype=np.float32) for name, value in tensors.items()}


def execute_mixed_graph_reference(graph: Any, external_context: Any, input_values: np.ndarray) -> np.ndarray:
    tensors = execute_mixed_graph_trace(graph, external_context, input_values)
    return tensors[graph.outputs[0]].astype(np.float32, copy=False).reshape(-1)

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

    import re

    def symbol(name: str) -> str:
        token = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
        return f"tensor_{token}"

    work = artifacts.report_path.parent / "host_cpp"
    work.mkdir(parents=True, exist_ok=True)
    source = work / "mixed_graph_host.cpp"
    executable = work / "mixed_graph_host"
    host_output = work / "output.bin"
    input_values = list(artifacts.input_values)
    constants = set(getattr(graph, "constants", {}) or {})

    lines = [
        "#include <algorithm>", "#include <cmath>", "#include <cstdio>",
        "#include <fstream>", "#include <vector>", "",
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

    input_var = symbol(graph.inputs[0])
    lines += [
        "",
        "int main() {",
        f"  std::vector<float> {input_var} = {{{', '.join(repr(float(v))+'f' for v in input_values)}}};",
    ]

    for op in graph.ops:
        runtime_inputs = [str(x) for x in op.inputs if str(x) not in constants]
        if len(op.outputs) != 1:
            return {"status": "skipped", "reason": f"unsupported_host_outputs:{op.name}"}
        out_var = symbol(op.outputs[0])
        binding = composition_plan.binding_for_node(op.name)
        if binding is not None:
            if len(runtime_inputs) != 1:
                return {"status": "skipped", "reason": f"unsupported_external_arity:{op.name}"}
            in_var = symbol(runtime_inputs[0])
            integration = dict(binding.contract.metadata.get("integration", {})).get("hls", {})
            attr_specs = integration.get("attributes", []) if isinstance(integration, Mapping) else []
            args = []
            for item in attr_specs:
                if not isinstance(item, Mapping):
                    continue
                name = str(item.get("name", ""))
                default = item.get("default", 0.0)
                args.append(repr(float(binding.attributes.get(name, default))) + "f")
            extra = "".join(", " + arg for arg in args)
            lines += [
                f"  std::vector<float> {out_var}({in_var}.size());",
                f"  {binding.contract.top}({in_var}.data(), {out_var}.data(), (int){in_var}.size(){extra});",
            ]
        elif str(op.op_type).lower() == "add":
            if len(runtime_inputs) != 2:
                return {"status": "skipped", "reason": f"unsupported_add_arity:{op.name}"}
            left = symbol(runtime_inputs[0])
            right = symbol(runtime_inputs[1])
            lines += [
                f"  if ({left}.size() != {right}.size()) return 3;",
                f"  std::vector<float> {out_var}({left}.size());",
                f"  for (size_t i = 0; i < {out_var}.size(); ++i) {out_var}[i] = {left}[i] + {right}[i];",
            ]
        elif str(op.op_type).lower() == "relu":
            in_var = symbol(runtime_inputs[0])
            lines += [
                f"  std::vector<float> {out_var} = {in_var};",
                f"  for (float& value : {out_var}) value = std::max(0.0f, value);",
            ]
        elif str(op.op_type).lower() == "sigmoid":
            in_var = symbol(runtime_inputs[0])
            lines += [
                f"  std::vector<float> {out_var} = {in_var};",
                f"  for (float& value : {out_var}) value = 1.0f / (1.0f + std::exp(-value));",
            ]
        elif str(op.op_type).lower() in {"identity", "flatten", "reshape"}:
            in_var = symbol(runtime_inputs[0])
            lines.append(f"  std::vector<float> {out_var} = {in_var};")
        else:
            return {"status": "skipped", "reason": f"unsupported_host_operator:{op.op_type}"}

    final_var = symbol(graph.outputs[0])
    lines += [
        f'  std::ofstream out("{host_output.as_posix()}", std::ios::binary);',
        f"  out.write(reinterpret_cast<const char*>({final_var}.data()), (std::streamsize)({final_var}.size()*sizeof(float)));",
        "  return out ? 0 : 2;", "}",
    ]
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    external_sources = sorted((hls_dir / "src" / "external").rglob("*.cpp"))
    include_dirs = sorted({path.parent for path in (hls_dir / "include" / "external").rglob("*.hpp")})
    command = [compiler, "-std=c++17", "-O2", str(source), *(str(path) for path in external_sources)]
    for include_dir in include_dirs:
        command.extend(["-I", str(include_dir)])
    command.extend(["-o", str(executable)])
    try:
        subprocess.check_call(command, cwd=str(work))
        subprocess.check_call([str(executable)], cwd=str(work))
    except subprocess.CalledProcessError as exc:
        return {"status": "failed", "returncode": int(exc.returncode), "source": str(source), "command": command}
    actual = np.fromfile(host_output, dtype=np.float32)
    expected = np.asarray(artifacts.expected_values, dtype=np.float32)
    diff = np.abs(actual - expected) if actual.size == expected.size else np.array([], dtype=np.float32)
    ok = actual.size == expected.size and bool(np.allclose(actual, expected, atol=1e-6, rtol=1e-6))
    return {
        "status": "passed" if ok else "failed", "source": str(source), "executable": str(executable),
        "output_bin": str(host_output), "outputs": int(actual.size),
        "max_abs": float(diff.max()) if diff.size else None, "mean_abs": float(diff.mean()) if diff.size else None,
    }

