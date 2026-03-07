import argparse
import torch
import torch.nn as nn
import torch.onnx
import os
import shutil
import subprocess
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import yaml
import time

# Add root to path
sys.path.append(os.getcwd())

# --- Test Registry ---
SUPPORTED_OPS = [
    "Dense", "Conv", "MaxPool", "Relu", "Softmax", "Flatten",
    # Uncomment these as you implement/verify them:
    # "AvgPool", "Sigmoid", "LeakyRelu" 
]

class OpModel(nn.Module):
    def __init__(self, op_type):
        super().__init__()
        self.op_type = op_type
        if op_type == "Dense":
            self.layer = nn.Linear(32, 16)
        elif op_type == "Conv":
            self.layer = nn.Conv2d(1, 4, 3, padding=1) # 16x16 -> 16x16
        elif op_type == "MaxPool":
            self.layer = nn.MaxPool2d(2, 2)
        elif op_type == "AvgPool":
            self.layer = nn.AvgPool2d(2, 2)
        elif op_type == "Relu":
            self.layer = nn.ReLU()
        elif op_type == "LeakyRelu":
            self.layer = nn.LeakyReLU(0.1)
        elif op_type == "Sigmoid":
            self.layer = nn.Sigmoid()
        elif op_type == "Softmax":
            self.layer = nn.Softmax(dim=1)
        elif op_type == "Flatten":
            self.layer = nn.Flatten()
        
    def forward(self, x):
        return self.layer(x)

def get_input_shape(op_type):
    if op_type in ["Conv", "MaxPool", "AvgPool"]: return (1, 1, 16, 16)
    if op_type == "Flatten": return (1, 4, 4, 4)
    # Dense/Relu/Softmax/etc usually take vectors in this simple test
    return (1, 32)

def run_cmd(cmd, cwd=None, verbose=False):
    vitis_settings = "/tools/Xilinx/Vitis_HLS/2023.2/settings64.sh"
    if ("vitis_hls" in cmd) and os.path.exists(vitis_settings):
        full_cmd = f"source {vitis_settings} && {cmd}"
        # Suppress output unless verbose
        kwargs = {} if verbose else {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL}
        subprocess.check_call(full_cmd, shell=True, cwd=cwd, executable="/bin/bash", **kwargs)
    else:
        kwargs = {} if verbose else {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL}
        subprocess.check_call(cmd, shell=True, cwd=cwd, **kwargs)

def run_single_test(op_name):
    print(f"Testing {op_name:<10} ... ", end="", flush=True)
    start_time = time.time()
    
    test_name = f"test_{op_name.lower()}"
    build_dir = Path("build") / test_name
    model_path = Path(f"tests/models/{op_name}.onnx")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Generate Model
        model = OpModel(op_name)
        model.eval()
        shape = get_input_shape(op_name)
        dummy_input = torch.randn(*shape)
        torch.onnx.export(model, dummy_input, str(model_path), 
                            input_names=['input'], output_names=['output'], 
                            opset_version=11) # <--- CRITICAL FIX

        # 2. Config
        config_str = f"""
project:
  name: {test_name}
  out_dir: build/{test_name}
model:
  format: onnx
  path: {model_path}
targets:
  platform:
    board: kv260
    part: xck26-sfvc784-2LV-c
    clock_mhz: 200
precision:
  activation: "ap_fixed<16,8>"
  weight: "ap_fixed<16,8>"
  bias: "ap_fixed<24,12>"
  accum: "ap_fixed<24,12>"
operators:
  supported: {SUPPORTED_OPS}
toolchain:
  vitis_hls: true
  settings64: "/tools/Xilinx/Vitis_HLS/2023.2/settings64.sh"
"""
        config_path = build_dir / "config.yml"
        build_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            f.write(config_str)

        # 3. Compile
        run_cmd(f"PYTHONPATH=. python main.py --config {config_path}")

        # 4. Reference
        sess = ort.InferenceSession(str(model_path))
        x_test = np.random.randn(*shape).astype(np.float32)
        x_test.tofile(build_dir / "input.bin")
        y_ref = sess.run(None, {'input': x_test})[0].flatten()

        # 5. HLS Sim
        hls_dir = build_dir / "hls"
        run_cmd("vitis_hls -f run_hls.tcl", cwd=hls_dir)

        # 6. Compare
        hls_out_bin = hls_dir / "fpgai_hls_proj" / "sol1" / "csim" / "build" / "output.bin"
        if not hls_out_bin.exists(): hls_out_bin = hls_dir / "output.bin"
        
        if not hls_out_bin.exists():
            print("FAIL ❌ (No Output)")
            return False, 0.0

        y_hls = np.fromfile(hls_out_bin, dtype=np.float32)
        n = min(len(y_ref), len(y_hls))
        diff = np.abs(y_ref[:n] - y_hls[:n])
        max_err = diff.max()
        
        duration = time.time() - start_time
        
        if max_err < 0.05:
            print(f"PASS ✅ (Err: {max_err:.6f}, {duration:.1f}s)")
            return True, max_err
        else:
            print(f"FAIL ❌ (Err: {max_err:.6f})")
            return False, max_err

    except Exception as e:
        print(f"CRASH 💥 ({str(e)})")
        return False, 999.9

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default="all", help="Op name or 'all'")
    args = parser.parse_args()

    ops_to_run = SUPPORTED_OPS if args.op == "all" else [args.op]
    
    print("\n" + "="*60)
    print(f" FPGAI REGRESSION SUITE - {len(ops_to_run)} Tests")
    print("="*60)

    results = []
    for op in ops_to_run:
        passed, err = run_single_test(op)
        results.append((op, passed, err))

    print("\n" + "="*60)
    print(" SUMMARY REPORT")
    print("="*60)
    print(f"{'Layer':<15} | {'Status':<10} | {'Max Error':<15}")
    print("-" * 60)
    
    success_count = 0
    for op, passed, err in results:
        status = "PASS ✅" if passed else "FAIL ❌"
        if passed: success_count += 1
        print(f"{op:<15} | {status:<10} | {err:.6f}")
    
    print("-" * 60)
    print(f"Total Passed: {success_count} / {len(results)}")
    print("="*60 + "\n")
    
    if success_count < len(results):
        sys.exit(1)

if __name__ == "__main__":
    main()