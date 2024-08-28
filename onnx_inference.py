import onnxruntime as ort
import numpy as np

# Load the ONNX model
onnx_model_path = "mlp.onnx"  # Replace with your ONNX model path
ort_session = ort.InferenceSession(onnx_model_path)

# Print input details to check the expected shape
for input_meta in ort_session.get_inputs():
    print(f"Input name: {input_meta.name}")
    print(f"Input shape: {input_meta.shape}")
    print(f"Input type: {input_meta.type}")

# Prepare input data
input_shape = ort_session.get_inputs()[0].shape

# Example input shape of (1, 8)
input_shape = (1, 8)

# Generate your data as before
increment = 0.1
total_elements = np.prod(input_shape)
input_data = np.arange(0.1, total_elements * increment + 0.1, increment, dtype=np.float32)

# Reshape to match the expected 4D input shape (assuming batch_size=1, channels=1, height=8, width=1)
input_data = input_data.reshape(1, 1, 1,8)

#input_data = np.ones(input_shape, dtype=np.float32) / 2  # Fill input with 0.5
print(f"Input data: {input_data}")

# Get the names of all output layers
output_names = [output.name for output in ort_session.get_outputs()]
print("Output layer names:", output_names)

# Run inference
outputs = ort_session.run(output_names, {ort_session.get_inputs()[0].name: input_data})

# Print outputs of all layers
for name, output in zip(output_names, outputs):
    print(f"Layer '{name}' output: {output}")


import sys
sys.path.append('home/umutcanaltin//Desktop/github_projects/fpgai_compiler/') 
from tcl_helpers import vitis_project_compiler
tcl_loc = vitis_project_compiler.create_vitis_tcl_file_for_simulation(tcl_filename = "generated_tcl_vitis.tcl",project_name="generated_project",project_dir="/home/umutcanaltin/Desktop/generated_dir",main_files=["deeplearn.h","main.cpp"],test_file="testbench.cpp",file_location ="/home/umutcanaltin/Desktop/tcl_scripts")
vitis_project_compiler.run_tcl_with_sourced_env(tcl_script_path=tcl_loc,vitis_path="/tools/Xilinx/Vitis_HLS/2023.2/")
from test_folder.create_testbench import create_inference_test_script
create_inference_test_script(input_data[0])
