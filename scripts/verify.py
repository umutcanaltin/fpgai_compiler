import argparse
import subprocess
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import yaml
import xml.etree.ElementTree as ET
import os

def load_bin(path):
    return np.fromfile(path, dtype=np.float32)

def run_cmd(cmd, cwd=None, settings_path=None):
    """
    Runs a shell command. 
    If a Vitis settings path is provided and valid, prepends 'source ...'
    """
    # Only source if we are running Xilinx tools and have a valid settings file
    if ("vitis_hls" in cmd or "vivado" in cmd) and settings_path and os.path.exists(settings_path):
        print(f"[INFO] Auto-sourcing: {settings_path}")
        # Use bash explicitly because 'source' is a bash built-in
        full_cmd = f"source {settings_path} && {cmd}"
        print(f"[CMD] {full_cmd}")
        subprocess.check_call(full_cmd, shell=True, cwd=cwd, executable="/bin/bash")
    else:
        print(f"[CMD] {cmd}")
        subprocess.check_call(cmd, shell=True, cwd=cwd)

def parse_hls_report(report_path):
    if not report_path.exists():
        print(f"[WARN] Report not found: {report_path}")
        return

    tree = ET.parse(report_path)
    root = tree.getroot()

    perf = root.find("PerformanceEstimates/SummaryOfOverallLatency/Average-case")
    latency = perf.find("cycles").text if perf is not None else "N/A"
    
    area = root.find("AreaEstimates/Resources")
    dsp = area.find("DSP").text if area is not None else "0"
    lut = area.find("LUT").text if area is not None else "0"
    ff = area.find("FF").text if area is not None else "0"
    bram = area.find("BRAM_18K").text if area is not None else "0"

    print("\n" + "="*40)
    print("  HLS SYNTHESIS REPORT")
    print("="*40)
    print(f"  Latency (Cycles) : {latency}")
    print(f"  DSP Usage        : {dsp}")
    print(f"  LUT Usage        : {lut}")
    print(f"  FF  Usage        : {ff}")
    print(f"  BRAM Usage       : {bram}")
    print("="*40 + "\n")

def ensure_input_bin(sess, input_bin_path):
    inp_obj = sess.get_inputs()[0]
    expected_shape = inp_obj.shape
    
    total_elements = 1
    concrete_shape = []
    for d in expected_shape:
        if isinstance(d, int) and d > 0:
            total_elements *= d
            concrete_shape.append(d)
        else:
            concrete_shape.append(1)
            
    if input_bin_path.exists():
        existing_data = np.fromfile(input_bin_path, dtype=np.float32)
        if existing_data.size == total_elements:
            return existing_data
        print(f"[INFO] Regenerating input.bin (Size mismatch)")
    else:
        print(f"[INFO] Generating new input.bin ({concrete_shape})")

    new_data = np.random.randn(total_elements).astype(np.float32) * 0.5
    new_data.tofile(input_bin_path)
    return new_data

def run_onnx_robust(sess, x_flat):
    inp_obj = sess.get_inputs()[0]
    inp_name = inp_obj.name
    expected_shape = inp_obj.shape
    
    concrete_shape = []
    for d in expected_shape:
        val = d if (isinstance(d, int) and d > 0) else 1
        concrete_shape.append(val)
    
    candidates = []
    try: candidates.append(x_flat.reshape(concrete_shape))
    except: pass
    candidates.append(x_flat)
    try: candidates.append(x_flat.reshape(1, -1))
    except: pass

    for x_candidate in candidates:
        try:
            return sess.run(None, {inp_name: x_candidate})[0].flatten()
        except Exception:
            continue
    raise RuntimeError(f"Shape mismatch for {inp_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="fpgai.yml")
    args = parser.parse_args()

    # Load Config FIRST
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --- UPDATED: Handle Nested Config Structure ---
    tc = cfg.get("toolchain", {})
    vitis_settings = None
    
    # 1. Check nested: toolchain -> vitis_hls -> settings64
    if "vitis_hls" in tc and isinstance(tc["vitis_hls"], dict):
         vitis_settings = tc["vitis_hls"].get("settings64")
    
    # 2. Fallback: Flat structure
    if not vitis_settings:
         vitis_settings = tc.get("settings64") or tc.get("settings_path")

    # 1. Compile
    print("\n[1/4] Compiling Project...")
    run_cmd(f"PYTHONPATH=. python main.py --config {args.config}")
    
    out_dir = Path(cfg["project"]["out_dir"]).resolve()
    model_path = cfg["model"]["path"]
    input_bin = out_dir / "input.bin"

    # 2. Reference (Python)
    print("\n[2/4] Running ONNX Reference...")
    sess = ort.InferenceSession(model_path)
    x_flat = ensure_input_bin(sess, input_bin)
    y_onnx = run_onnx_robust(sess, x_flat)
    print(f"   ONNX Output: {y_onnx[:4]} ...")

    # 3. HLS Sim & Synth
    print("\n[3/4] Running HLS Simulation & Synthesis...")
    hls_dir = out_dir / "hls"
    
    try:
        # Pass the extracted settings path
        run_cmd("vitis_hls -f run_hls.tcl", cwd=hls_dir, settings_path=vitis_settings)
    except subprocess.CalledProcessError:
        print("\n[Error] Vitis HLS failed.")
        if not vitis_settings:
            print(f"[Tip] Could not find 'settings64' in fpgai.yml. Checked: toolchain.vitis_hls.settings64")
        return

    # Report & Compare
    report_path = hls_dir / "fpgai_hls_proj" / "sol1" / "syn" / "report" / "deeplearn_csynth.xml"
    parse_hls_report(report_path)

    hls_out_bin = hls_dir / "fpgai_hls_proj" / "sol1" / "csim" / "build" / "output.bin"
    if not hls_out_bin.exists():
        hls_out_bin = hls_dir / "output.bin"

    if hls_out_bin.exists():
        y_hls = load_bin(hls_out_bin)
        print(f"   HLS Output:  {y_hls[:4]} ...")
        
        print("\n[4/4] Final Comparison Results")
        print("-" * 40)
        print(f"{'Metric':<20} | {'HLS vs ONNX':<15}")
        print("-" * 40)
        
        n = min(len(y_onnx), len(y_hls))
        diff_hls  = np.abs(y_hls[:n]  - y_onnx[:n])
        
        print(f"{'Max Error':<20} | {diff_hls.max():<15.6f}")
        print(f"{'Mean Error':<20} | {diff_hls.mean():<15.6f}")
        print("-" * 40)
        
        if diff_hls.max() < 0.1: 
            print("\nSUCCESS: HLS matches Reference!")
        else:
            print("\nWARNING: Large error detected.")
    else:
        print("Warning: HLS output.bin not found.")

if __name__ == "__main__":
    main()