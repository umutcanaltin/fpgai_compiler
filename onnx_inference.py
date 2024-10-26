import onnx
import onnxruntime as ort
import numpy as np
import onnx.numpy_helper as numpy_helper

def onnx_inference_pytorch(onnx_model,input_data):
    # Load the ONNX model
    onnx_model_path = onnx_model  # Replace with your ONNX model path
    ort_session = ort.InferenceSession(onnx_model_path)
    # Get the names of all output layers
    output_names = [output.name for output in ort_session.get_outputs()]


    # Run inference
    outputs = ort_session.run(output_names, {ort_session.get_inputs()[0].name: input_data})

    # Print outputs of all layers
    for name, output in zip(output_names, outputs):
        print(f"Layer '{name}' output: {output}")
    return outputs


# Function to extract weights from an ONNX model
def extract_weights_from_onnx(onnx_model):
    model = onnx.load(onnx_model)
    weights = {}
    for initializer in model.graph.initializer:
        weights[initializer.name] = numpy_helper.to_array(initializer)
    return weights

# Function to update weights in an ONNX model
def update_weights_in_onnx(onnx_model, new_weights, save_path):
    model = onnx.load(onnx_model)
    for initializer in model.graph.initializer:
        if initializer.name in new_weights:
            updated_tensor = numpy_helper.from_array(new_weights[initializer.name], initializer.name)
            initializer.CopyFrom(updated_tensor)

    # Save the updated model
    onnx.save(model, save_path)
    print(f"Model saved at {save_path}")

# Function for a simple training step
def onnx_train_pytorch(onnx_model, input_data, target_output, learning_rate=0.01):
    # Load the ONNX model weights
    weights = extract_weights_from_onnx(onnx_model)

    # Forward pass (inference)
    predicted_output = onnx_inference_pytorch(onnx_model, input_data)
    print(predicted_output)
    predicted_output = np.array(predicted_output).flatten()  # Assuming flattened outputs

    # Loss calculation (mean squared error)
    loss = np.mean((predicted_output - target_output) ** 2)
    print(f"Loss: {loss}")

    # Backpropagation (manual gradient calculation)
    grad_output = 2 * (predicted_output - target_output)  # Gradient of MSE with respect to output

    # Updating weights manually (you'll need to update weights based on the gradient)
    updated_weights = {}
    for layer_name, layer_weights in weights.items():
        # Simple gradient descent weight update (assuming fully connected layer weights)
        grad_w = np.outer(input_data, grad_output)  # Gradient with respect to the input
        updated_weights[layer_name] = layer_weights - learning_rate * grad_w

    # Save updated weights back to ONNX
    print(updated_weights)

# Example usage
#input_data = np.array([0.5, 0.2, -0.3], dtype=np.float32)  # Example input
#target_output = np.array([1.0], dtype=np.float32)  # Example target

#onnx_model_path = "mlp.onnx"  # Path to your ONNX model
#onnx_train_pytorch(onnx_model_path, input_data, target_output, learning_rate=0.01)