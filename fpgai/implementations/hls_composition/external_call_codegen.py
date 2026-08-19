from __future__ import annotations
import math
from typing import Any, Mapping
from fpgai.implementations.hls_integration import parse_hls_abi, HLSFlatArrayABI, HLSTensorPortsABI

def _literal(value: Any, cpp_type: str) -> str:
    if cpp_type in {"int", "unsigned"}:
        return str(int(value)) + ("u" if cpp_type == "unsigned" else "")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("HLSCOMP010: non-finite external attribute")
    text = f"{number:.9g}"
    if "." not in text and "e" not in text.lower(): text += ".0"
    return text + ("f" if cpp_type == "float" else "")

def package_declarations(plan) -> list[str]:
    declarations=[]; seen=set()
    for binding in plan.bindings:
        c=binding.contract; key=(c.package_id,c.top)
        if key in seen: continue
        seen.add(key); abi=parse_hls_abi(c)
        if isinstance(abi,HLSFlatArrayABI):
            params=[f"const {abi.scalar_type}* input",f"{abi.scalar_type}* output","int count"]
        else:
            params=[
                *(f"const {abi.scalar_for(p)}* {p.name}" for p in abi.inputs),
                *(f"{abi.scalar_for(p)}* {p.name}" for p in abi.outputs),
            ]
            if abi.count_mode == "shared":
                params.append("int count")
            else:
                params.extend(f"int {p.name}_count" for p in (*abi.inputs, *abi.outputs))
        params.extend(f"{a.cpp_type} {a.name}" for a in abi.attributes)
        declarations.append(f"void {c.top}({', '.join(params)});")
    return declarations

def emit_external_call(binding, *, current_buffer: str, current_type: str, output_buffer: str, output_type: str) -> list[str]:
    abi=parse_hls_abi(binding.contract)
    if not isinstance(abi,HLSFlatArrayABI):
        raise ValueError("HLSCOMP014: emit_external_call is only for flat_array_v1")
    values=[_literal(binding.attributes.get(a.name,a.default),a.cpp_type) for a in abi.attributes]
    lines=[f"    // External implementation: {binding.contract.package_id}"]
    in_arg=current_buffer; out_arg=output_buffer
    if current_type!=abi.scalar_type:
        in_arg=f"{binding.wrapper_symbol}_input"; lines += [f"    {abi.scalar_type} {in_arg}[{binding.input_words}];",f"    for (int i = 0; i < {binding.input_words}; ++i) {{","#pragma HLS PIPELINE II=1",f"        {in_arg}[i] = ({abi.scalar_type}){current_buffer}[i];","    }"]
    if output_type!=abi.scalar_type:
        out_arg=f"{binding.wrapper_symbol}_output"; lines.append(f"    {abi.scalar_type} {out_arg}[{binding.output_words}];")
    lines.append(f"    {binding.contract.top}({', '.join([in_arg,out_arg,str(binding.output_words),*values])});")
    if output_type!=abi.scalar_type:
        lines += [f"    for (int i = 0; i < {binding.output_words}; ++i) {{","#pragma HLS PIPELINE II=1",f"        {output_buffer}[i] = ({output_type}){out_arg}[i];","    }"]
    return lines

def emit_external_tensor_ports_call(binding, *, input_buffers: Mapping[str,str], input_types: Mapping[str,str], output_buffers: Mapping[str,str], output_types: Mapping[str,str]) -> list[str]:
    abi=parse_hls_abi(binding.contract)
    if not isinstance(abi,HLSTensorPortsABI):
        raise ValueError("HLSCOMP015: tensor_ports_v1 ABI required")
    tensors_in=binding.input_tensors or (binding.input_tensor,)
    tensors_out=binding.output_tensors or (binding.output_tensor,)
    lines=[f"    // External tensor_ports_v1 implementation: {binding.contract.package_id}"]
    args=[]
    for port,tensor in zip(abi.inputs,tensors_in):
        words=int(binding.input_port_words.get(port.name, binding.port_words or binding.input_words))
        scalar=abi.scalar_for(port)
        src=input_buffers[tensor]; typ=input_types[tensor]
        if typ==scalar:
            args.append(src)
        else:
            tmp=f"{binding.wrapper_symbol}_{port.name}_in"
            lines += [f"    {scalar} {tmp}[{words}];",f"    for (int i = 0; i < {words}; ++i) {{","#pragma HLS PIPELINE II=1",f"        {tmp}[i] = ({scalar}){src}[i];","    }"]
            args.append(tmp)
    post=[]
    for port,tensor in zip(abi.outputs,tensors_out):
        words=int(binding.output_port_words.get(port.name, binding.port_words or binding.output_words))
        scalar=abi.scalar_for(port)
        dst=output_buffers[tensor]; typ=output_types[tensor]
        if typ==scalar:
            args.append(dst)
        else:
            tmp=f"{binding.wrapper_symbol}_{port.name}_out"
            lines.append(f"    {scalar} {tmp}[{words}];")
            args.append(tmp)
            post += [f"    for (int i = 0; i < {words}; ++i) {{","#pragma HLS PIPELINE II=1",f"        {dst}[i] = ({typ}){tmp}[i];","    }"]
    if abi.count_mode == "shared":
        shared=int(binding.port_words or binding.output_words)
        args.append(str(shared))
    else:
        args.extend(str(int(binding.input_port_words[p.name])) for p in abi.inputs)
        args.extend(str(int(binding.output_port_words[p.name])) for p in abi.outputs)
    values=[_literal(binding.attributes.get(a.name,a.default),a.cpp_type) for a in abi.attributes]
    lines.append(f"    {binding.contract.top}({', '.join([*args,*values])});")
    lines += post
    return lines


def _training_hls_config(binding) -> Mapping[str, Any]:
    integration = dict(getattr(binding.contract, "metadata", {}).get("integration", {}) or {})
    hls = integration.get("hls", {}) or {}
    if not isinstance(hls, Mapping):
        return {}
    training = hls.get("training", {}) or {}
    return training if isinstance(training, Mapping) else {}


def package_training_declarations(plan) -> list[str]:
    """Declare external training entrypoints for supported flat-array packages."""
    declarations: list[str] = []
    seen: set[tuple[str, str]] = set()
    for binding in plan.bindings:
        contract = binding.contract
        if not contract.training.backward_input:
            continue
        abi = parse_hls_abi(contract)
        if not isinstance(abi, HLSFlatArrayABI):
            raise ValueError(
                f"HLSCOMP020: external training currently supports flat_array_v1 only; node={binding.node_name!r}"
            )
        cfg = _training_hls_config(binding)
        top = str(cfg.get("backward_input_top", "")).strip()
        if not top:
            raise ValueError(
                f"HLSCOMP021: implementation {contract.package_id!r} declares training.backward_input=true "
                "but integration.hls.training.backward_input_top is missing"
            )
        key = (contract.package_id, top)
        if key in seen:
            continue
        seen.add(key)
        params = [f"const {abi.scalar_type}* grad_output", f"{abi.scalar_type}* grad_input", "int count"]
        params.extend(f"{a.cpp_type} {a.name}" for a in abi.attributes)
        declarations.append(f"void {top}({', '.join(params)});")
    return declarations


def emit_external_backward_input_call(
    binding,
    *,
    output_gradient: str,
    input_gradient: str,
    gradient_type: str = "grad_act_t",
) -> list[str]:
    """Emit the safe stateless external backward-input ABI."""
    contract = binding.contract
    if not contract.training.forward or not contract.training.backward_input:
        raise ValueError(
            f"HLSCOMP022: external node {binding.node_name!r} lacks training forward/backward-input capability"
        )
    if contract.training.parameter_gradients or contract.training.bias_gradients or contract.training.optimizer_update:
        raise ValueError(
            f"HLSCOMP023: trainable external parameter/update ABI is not implemented for node {binding.node_name!r}; "
            "use a stateless external implementation or a built-in trainable operator"
        )
    abi = parse_hls_abi(contract)
    if not isinstance(abi, HLSFlatArrayABI):
        raise ValueError("HLSCOMP020: external training currently supports flat_array_v1 only")
    cfg = _training_hls_config(binding)
    top = str(cfg.get("backward_input_top", "")).strip()
    if not top:
        raise ValueError(
            f"HLSCOMP021: integration.hls.training.backward_input_top is required for {contract.package_id!r}"
        )
    values = [_literal(binding.attributes.get(a.name, a.default), a.cpp_type) for a in abi.attributes]
    lines = [f"  // External training backward-input: {contract.package_id}"]
    grad_out_arg = output_gradient
    grad_in_arg = input_gradient
    if gradient_type != abi.scalar_type:
        grad_out_arg = f"{binding.wrapper_symbol}_grad_output"
        grad_in_arg = f"{binding.wrapper_symbol}_grad_input"
        lines += [
            f"  {abi.scalar_type} {grad_out_arg}[{binding.output_words}];",
            f"  {abi.scalar_type} {grad_in_arg}[{binding.input_words}];",
            f"  for (int i = 0; i < {binding.output_words}; ++i) {{",
            "#pragma HLS PIPELINE II=1",
            f"    {grad_out_arg}[i] = ({abi.scalar_type}){output_gradient}[i];",
            "  }",
        ]
    lines.append(f"  {top}({', '.join([grad_out_arg, grad_in_arg, str(binding.input_words), *values])});")
    if gradient_type != abi.scalar_type:
        lines += [
            f"  for (int i = 0; i < {binding.input_words}; ++i) {{",
            "#pragma HLS PIPELINE II=1",
            f"    {input_gradient}[i] += ({gradient_type}){grad_in_arg}[i];",
            "  }",
        ]
    return lines
