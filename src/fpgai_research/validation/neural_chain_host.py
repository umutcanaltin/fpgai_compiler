"""R2z actual GCC source intermediate trace vs independent NumPy CNN oracle.

Unlike a final-only classification comparison, checks every intermediate
Conv/Pool/Flatten/Dense/activation output element, including the final
model result. The original canonical emitter already provides opt-in
FPGAI_TRACE_TENSORS; no shadow source emitter or fake E3 result is used.

Host E1 float is NOT vendor ap_fixed bit-exact CSim or FPGA QoR evidence.
"""
from __future__ import annotations

from hashlib import sha256
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory
import re
import shutil
import subprocess

import numpy as np

from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.model_ir.contracts import FunctionalModel
from fpgai_research.model_ir.semantics import functional_fingerprint
from fpgai_research.validation.host import _float_literal
from fpgai_research.validation.reference import evaluate

SCHEMA = "fpgai-research/r2z-cnn-dense-actual-GCC-stage-E1/v1"
_TRACE = re.compile(
    r"^FPGAI_TENSOR ([0-9]+) ([0-9]+) (\S+)$"
)
_FINAL = re.compile(
    r"^FPGAI_NN_OUTPUT ([0-9]+) (\S+)$"
)


def verify_neural_chain_gcc(
    model: FunctionalModel, emission: Emission, runtime_input,
    *, rtol:float=0.0005, atol:float=0.00005,
    timeout_seconds:int=120,
) -> dict:
    if (shutil.which("g++") is None
            or type(timeout_seconds) is not int
            or not 1 <= timeout_seconds <= 600):
        raise RuntimeError("R2z host GCC and bounded compilation time required")
    if (not np.isfinite(rtol) or not np.isfinite(atol)
            or rtol <= 0 or atol <= 0
            or rtol > 0.01 or atol > 0.01):
        raise ValueError("R2z explicit bounded numerical tolerance required")
    model.validate()
    if (len(model.inputs)!=1 or len(model.outputs)!=1
            or sha256(emission.cpp.encode()).hexdigest()!=
                    emission.manifest.get("cpp_sha256")
            or emission.manifest.get("functional_sha256") !=
                    functional_fingerprint(model)):
        raise ValueError("R2z source identity, graph identity or boundaries invalid")
    x=np.asarray(runtime_input,dtype=np.float32)
    if x.shape != model.tensor(model.inputs[0]).shape:
        raise ValueError("R2z host input shape does not match Model IR")
    if not np.all(np.isfinite(x)):
        raise ValueError("R2z host runtime input must be finite")
    reference=evaluate(model,{model.inputs[0]:x},return_all=True)
    output_name=model.outputs[0]
    output_count=prod(model.tensor(output_name).shape)
    initializer=", ".join(_float_literal(v) for v in x.flat)
    runner=f"""
int main() {{
  float input[{x.size}] = {{{initializer}}};
  float output[{output_count}] = {{}};
  fpgai_model(input,output);
  for(int i=0; i<{output_count}; ++i)
    std::printf("FPGAI_NN_OUTPUT %d %.9g\\n", i, double(output[i]));
  return 0;
}}
"""
    with TemporaryDirectory() as temporary:
        cpp=Path(temporary)/"r2z_nn.cpp"
        exe=Path(temporary)/"r2z_nn"
        cpp.write_text(emission.cpp+"\n"+runner,encoding="utf-8")
        compiled=subprocess.run(
            ["g++","-std=c++17","-O0","-Wno-unknown-pragmas",
             "-DFPGAI_HOST_FLOAT","-DFPGAI_TRACE_TENSORS",
             str(cpp),"-o",str(exe)],
            capture_output=True,text=True,timeout=timeout_seconds,
        )
        if compiled.returncode!=0:
            raise RuntimeError("R2z generated GCC source compile failed: "+
                               compiled.stderr[-5000:])
        executed=subprocess.run(
            [str(exe)],capture_output=True,text=True,
            timeout=timeout_seconds,
        )
        if executed.returncode!=0:
            raise RuntimeError("R2z actual GCC source failed: "+
                               executed.stderr[-5000:])
        text=executed.stdout

    outputs={}
    final={}
    for line in text.splitlines():
        m=_TRACE.fullmatch(line)
        if m is not None:
            ordinal,index=int(m[1]),int(m[2])
            if not 0 <= ordinal < len(model.operators):
                raise ValueError("R2z actual C++ trace used unknown operator ordinal")
            name=model.operators[ordinal].outputs[0]
            if not 0 <= index < prod(model.tensor(name).shape):
                raise ValueError("R2z actual C++ trace used invalid element")
            key=(ordinal,index)
            if key in outputs:
                raise ValueError("R2z duplicate generated source stage element")
            outputs[key]=float(m[3])
            continue
        m=_FINAL.fullmatch(line)
        if m is not None:
            index=int(m[1])
            if not 0 <= index < output_count or index in final:
                raise ValueError("R2z duplicate or invalid final-output source witness")
            final[index]=float(m[2])
            continue
        raise ValueError("R2z unrecognized generated source trace line")
    required=sum(prod(model.tensor(op.outputs[0]).shape)
                 for op in model.operators)
    if len(outputs)!=required or len(final)!=output_count:
        raise ValueError("R2z incomplete generated Conv/Dense intermediate trace")
    stages=[]
    all_good=True
    for ordinal,op in enumerate(model.operators):
        name=op.outputs[0]
        count=prod(model.tensor(name).shape)
        observed=np.asarray(
            [outputs[(ordinal,i)] for i in range(count)],dtype=np.float64
        )
        expected=np.asarray(reference[name],dtype=np.float64).ravel()
        if not np.all(np.isfinite(observed)):
            raise ValueError("R2z nonfinite generated stage output")
        delta=np.abs(observed-expected)
        accepted=bool(np.allclose(
            observed,expected,atol=atol,rtol=rtol))
        all_good=all_good and accepted
        stages.append({
            "operator":op.name,"kind":op.kind,
            "shape":list(model.tensor(name).shape),
            "checked_elements":count,
            "max_absolute_error":float(np.max(delta)),
            "accepted":accepted,
        })
    actual_final=np.asarray([final[i] for i in range(output_count)],
                            dtype=np.float64)
    expected_final=np.asarray(reference[output_name],dtype=np.float64).ravel()
    if not np.all(np.isfinite(actual_final)):
        raise ValueError("R2z nonfinite final generated output")
    final_ok=bool(np.allclose(actual_final,expected_final,atol=atol,rtol=rtol))
    return {
        "schema":SCHEMA,
        "functional_sha256":functional_fingerprint(model),
        "source_cpp_sha256":emission.manifest["cpp_sha256"],
        "input_sha256":sha256(x.tobytes()).hexdigest(),
        "status":("R2Z_ALL_ACTUAL_GCC_NN_STAGE_VALUES_QUALIFIED_E1"
                  if all_good and final_ok else
                  "R2Z_ACTUAL_GENERATED_NN_SOURCE_NUMERIC_REJECTED_E1"),
        "actual_generated_GCC_executions":1,
        "all_stages_accepted":all_good,
        "final_output_accepted":final_ok,
        "stage_count":len(stages),
        "checked_stage_elements":required,
        "maximum_stage_absolute_error":max(
            r["max_absolute_error"] for r in stages),
        "final_max_absolute_error":float(
            np.max(np.abs(actual_final-expected_final))),
        "absolute_tolerance":atol,"relative_tolerance":rtol,
        "stages":stages,
        "new_original_Vitis_calls":0,
        "new_original_Vivado_calls":0,
        "physical_memory_measurements":None,
        "limits":[
            "Source GCC host-float, no fixed-point or HLS CSim proof",
            "Finite static network, fixed input samples, no full model accuracy claim",
            "Stages are fully materialized; no streaming or in-place scratch reuse",
            "No native resource, RAM-port, Fmax, power or device measurement",
        ],
    }
