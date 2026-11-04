"""Compile actual generated C++ with host-only iteration-coordinate tracing.

Checks GCC-executed source against Phi; does not establish ap_fixed or Vitis
numerical, schedule or physical evidence. Trace is host-macro guarded.
"""
from __future__ import annotations
from hashlib import sha256
from math import prod
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from fpgai_research.architecture.resolver import coordinate
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.model_ir.contracts import LoweredModel
from fpgai_research.model_ir.semantics import verify_refinement


def trace_cpp_phi(model: LoweredModel, *, max_iterations: int = 20_000,
                  cxx: str = 'g++', emission=None) -> dict:
    if type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError('max_iterations must be a positive integer')
    verify_refinement(model)
    functional = model.architectural.functional
    if len(functional.inputs) != 1 or len(functional.outputs) != 1:
        raise ValueError('Host trace requires one input/output')
    relevant = {idx:point for idx,point in enumerate(model.architectural.points)
                if point.kind in ('MatMul','Gemm','Conv')}
    if not relevant:
        raise ValueError('At least one supported compute operator is required')
    logical_total = sum(prod(point.iteration_shape) for point in relevant.values())
    if logical_total > max_iterations:
        raise ValueError(f'Host trace iteration count {logical_total} exceeds bound {max_iterations}')
    resolved = shutil.which(cxx)
    if not resolved:
        raise FileNotFoundError(f'Host C++ compiler not installed: {cxx}')
    if emission is None:
        emission = emit_hls(model)
    else:
        from fpgai_research.model_ir.semantics import functional_fingerprint
        if emission.manifest.get('functional_sha256') != functional_fingerprint(functional):
            raise ValueError('Supplied HLS emission is not bound to Functional Model IR')
    input_size = prod(functional.tensor(functional.inputs[0]).shape)
    output_size = prod(functional.tensor(functional.outputs[0]).shape)
    runner = (f'\nint main() {{ float inputs[{input_size}] = {{}}; '
              f'float outputs[{output_size}] = {{}}; fpgai_model(inputs, outputs); return 0; }}\n')
    with TemporaryDirectory(prefix='fpgai_phi_trace_') as folder:
        root=Path(folder)
        cpp=root/'case.cpp'
        binary=root/'case'
        cpp.write_text(emission.cpp+runner)
        command=[resolved,'-std=c++17','-O0','-DFPGAI_HOST_FLOAT',
                 '-DFPGAI_TRACE_ITERATIONS','-Wno-unknown-pragmas',
                 str(cpp),'-o',str(binary)]
        compiled=subprocess.run(command,shell=False,capture_output=True,text=True,timeout=45)
        if compiled.returncode:
            raise RuntimeError('Generated C++ trace build failed: '+compiled.stderr[:1600])
        executed=subprocess.run([str(binary)],shell=False,capture_output=True,text=True,timeout=45)
        if executed.returncode:
            raise RuntimeError('Generated C++ trace execution failed: '+executed.stderr[:1600])
        lines=executed.stdout.splitlines()
    seen={number:set() for number in relevant}
    for line in lines:
        fields=line.split()
        if not fields or fields[0]!='FPGAI_ITER':
            raise AssertionError(f'Unexpected host trace line: {line[:120]}')
        try:
            values=tuple(int(value) for value in fields[1:])
        except ValueError as exc:
            raise AssertionError('Noninteger trace coordinate') from exc
        if not values or values[0] not in relevant:
            raise AssertionError('Unknown traced Model IR operator')
        number=values[0]
        point=relevant[number]
        rank=len(point.iteration_shape)
        if len(values)!=1+rank+4:
            raise AssertionError('Invalid emitted trace coordinate rank')
        index=values[1:1+rank]
        if index in seen[number]:
            raise AssertionError(f'Duplicate emitted C++ iteration: {point.operator}, {index}')
        try:
            expected=coordinate(point,index)
        except IndexError as exc:
            raise AssertionError('Generated C++ executed out-of-domain iteration') from exc
        replica,pe,lane,fold=values[1+rank:]
        fold_index=2 if point.kind in ('MatMul','Gemm') else 4
        if (replica,pe,lane,fold)!=(expected['replica'],expected['pe'],expected['lane'],expected['reuse_step'][fold_index]):
            raise AssertionError(f'Generated C++ disagrees with Phi: {point.operator}, {index}')
        seen[number].add(index)
    for number,point in relevant.items():
        if len(seen[number])!=prod(point.iteration_shape):
            raise AssertionError(f'C++ omitted iteration(s) in {point.operator}')
    return {'schema':'fpgai-research/actual-cpp-phi-trace/v1',
            'cpp_sha256':emission.manifest['cpp_sha256'],
            'operators':{point.operator:len(seen[number]) for number,point in relevant.items()},
            'logical_iterations':logical_total,
            'trace_sha256':sha256(executed.stdout.encode('utf-8')).hexdigest(),
            'source_phi_agreement':True,
            'evidence':'E1-executed-GCC-host-float-source-trace',
            'native_vitis_executed':False,
            'fixed_point_bit_exact':False,
            'limitations':'Executed GCC host-float C++ only; does not prove Vitis scheduling, HLS II or RTL mapping'}
