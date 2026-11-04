"""Compile and run generated testbench using a FLOAT stub for ap_fixed headers.

This validates C++ syntax and experiment harness logic only; not vendor CSim
and emphatically not bit-accurate fixed-point simulation.
"""
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from fpgai_research import Tensor,Operator,FunctionalModel,ArchitectureRequest,resolve,lower
from fpgai_research.codegen.hls.emitter import emit_hls
from fpgai_research.toolchains.vitis_hls.package import make_testbench

@pytest.mark.skipif(shutil.which("g++") is None,reason="g++ not installed")
def test_multisample_bench_compiles_and_executes_in_float_stub_mode(tmp_path):
    model=FunctionalModel(
      (Tensor("x",(1,2)),Tensor("w",(2,1),data=(.25,.75)),Tensor("y",(1,1))),
      (Operator("mm","MatMul",("x","w"),("y",)),),("x",),("y",))
    cpp=emit_hls(lower(resolve(model,ArchitectureRequest())))
    bench=make_testbench(model,np.asarray([[[1.,2.]],[[3.,4.]]],dtype=np.float32),16,6)
    (tmp_path/"model.cpp").write_text(cpp.cpp)
    (tmp_path/"tb.cpp").write_text(bench)
    (tmp_path/"ap_fixed.h").write_text(
      "#pragma once\n#define AP_RND 0\n#define AP_SAT 0\n"
      "template<int W,int I,int R,int S> using ap_fixed=float;\n")
    (tmp_path/"hls_math.h").write_text(
      "#pragma once\n#include <cmath>\nnamespace hls {using std::exp; using std::sqrt;}\n")
    exe=tmp_path/"demo"
    build=subprocess.run(
      ["g++","-std=c++17","-I",str(tmp_path),str(tmp_path/"model.cpp"),
       str(tmp_path/"tb.cpp"),"-o",str(exe)],
      text=True,capture_output=True,timeout=90)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],text=True,capture_output=True,timeout=30)
    assert run.returncode==0,run.stderr
    assert "FPGAI_METRICS samples=2 outputs=1" in run.stdout
    assert "total_failures=0" in run.stdout
