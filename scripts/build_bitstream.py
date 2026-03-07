import argparse
import subprocess
from pathlib import Path
import sys
import yaml
from fpgai.backends.vivado.tcl_generator import emit_vivado_tcl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="fpgai.yml")
    args = parser.parse_args()

    # Load Config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    out_dir = Path(cfg["project"]["out_dir"]).resolve()
    project_name = "fpgai_vivado_proj"
    part = cfg["targets"]["platform"]["part"]
    board_part = "xilinx.com:kv260_som:part0:1.4" 

    hls_impl_dir = out_dir / "hls" / "fpgai_hls_proj" / "sol1" / "impl"
    ip_repo = hls_impl_dir / "ip" 

    if not ip_repo.exists():
        print(f"Error: IP repository not found at {ip_repo}")
        return

    # Generate TCL
    tcl_content = emit_vivado_tcl(project_name, str(out_dir), part, board_part, str(ip_repo))
    tcl_path = out_dir / "build_bitstream.tcl"
    tcl_path.write_text(tcl_content)

    print("\n[FPGAI] Starting Vivado Bitstream Generation...")
    print(f"   Mode: Real-time Output")
    
    # Run Vivado and stream output to console
    cmd = f"vivado -mode batch -source {tcl_path.name}"
    
    process = subprocess.Popen(
        cmd, 
        cwd=out_dir, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True
    )

    # Read output line by line and print it
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip())

    if process.returncode == 0:
        print("\nSUCCESS: Bitstream generated.")
        print(f"Location: {out_dir}/{project_name}.runs/impl_1/system_wrapper.bit")
    else:
        print("\n[ERROR] Vivado failed.")

if __name__ == "__main__":
    main()