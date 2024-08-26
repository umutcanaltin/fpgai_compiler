# Load the ONNX model
import onnxruntime as ort
import numpy as np
onnx_model_path = "mlp.onnx"  # Replace with your ONNX model path
ort_session = ort.InferenceSession(onnx_model_path)

# Print input details to check the expected shape
for input_meta in ort_session.get_inputs():
    print(f"Input name: {input_meta.name}")
    print(f"Input shape: {input_meta.shape}")
    print(f"Input type: {input_meta.type}")

# Assuming the model expects a shape [97, 3, 224, 224]
input_shape = ort_session.get_inputs()[0].shape
input_data = np.ones(input_shape, dtype=np.float32)  # Fill input with ones # Generate random input with correct shape
print(input_data)

# Get input and output names
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Run inference
outputs = ort_session.run([output_name], {input_name: input_data})

# Print output results
print(f"Output: {outputs}")