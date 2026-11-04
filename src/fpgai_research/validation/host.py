"""Execute generated GCC host-float C++ against the NumPy reference.

This is *not* Vitis C simulation or fixed-point equivalence.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import subprocess
import numpy as np
from fpgai_research.model_ir.contracts import FunctionalModel
from fpgai_research.codegen.hls.emitter import Emission
from fpgai_research.validation.reference import evaluate

def _float_literal(v):
    x = f"{float(v):.9g}"
    return (x if "." in x or "e" in x else x + ".0") + "f"

def validate_host(model: FunctionalModel, emission: Emission, runtime_input,
                  *, timeout_seconds: int = 180, rtol: float = 1e-4, atol: float = 1e-5) -> dict:
    if shutil.which("g++") is None:
        raise RuntimeError("g++ is required for host-float verification")
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError("Host validation requires one input/output")
    x = np.asarray(runtime_input, dtype=np.float32)
    expected = evaluate(model, {model.inputs[0]: x})[model.outputs[0]].ravel().astype(np.float64)
    initializer = ",".join(_float_literal(v) for v in x.flat)
    runner = f'''
#include <cstdio>
int main() {{
  float input[{x.size}]={{{initializer}}};
  float output[{len(expected)}]={{}};
  fpgai_model(input,output);
  for(int i=0;i<{len(expected)};++i) std::printf("%.9g\\n",output[i]);
  return 0;
}}
'''
    with TemporaryDirectory() as directory:
        src,exe=Path(directory)/"model.cpp",Path(directory)/"run"
        src.write_text(emission.cpp+runner)
        build=subprocess.run(["g++","-std=c++17","-O0","-Wno-unknown-pragmas",
            "-DFPGAI_HOST_FLOAT",str(src),"-o",str(exe)],capture_output=True,text=True,
            timeout=timeout_seconds)
        if build.returncode:
            raise RuntimeError("Generated host C++ failed to compile:\n"+build.stderr[-6000:])
        run=subprocess.run([str(exe)],capture_output=True,text=True,timeout=timeout_seconds)
        if run.returncode:
            raise RuntimeError("Generated host executable failed:\n"+run.stderr[-6000:])
        observed=np.fromstring(run.stdout,sep=" ",dtype=np.float64)
    if observed.shape != expected.shape or not np.all(np.isfinite(observed)):
        raise ValueError("Nonfinite host values or wrong output cardinality")
    delta=observed-expected
    denominator=float(np.linalg.norm(expected)*np.linalg.norm(observed))
    accepted=bool(np.allclose(observed,expected,rtol=rtol,atol=atol))
    cosine=float(np.dot(expected,observed)/denominator) if denominator else (1.0 if accepted else 0.0)
    return {"schema":"fpgai-research/host-validation/v1",
            "mode":"gcc-host-float","compiler_command":"g++ -std=c++17 -O0 -DFPGAI_HOST_FLOAT",
            "input_sha256":sha256(x.tobytes()).hexdigest(),
            "cpp_sha256":emission.manifest["cpp_sha256"],
            "output_count":len(observed),"accepted":accepted,
            "cosine":cosine,"rmse":float(np.sqrt(np.mean(delta*delta))),
            "max_absolute_error":float(np.max(np.abs(delta))),
            "rtol":rtol,"atol":atol,"strongest_evidence":"E1-plus-host-float-check",
            "not_evidence_of":["fixed-point-equivalence","Vitis-csim",
                               "HLS-synthesis","place-and-route","board-runtime"]}
