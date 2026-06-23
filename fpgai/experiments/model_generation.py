import torch
import torch.nn as nn
import torch.onnx
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# --- Model 1: Simple MLP (What you have been using) ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(8, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)
        # Note: We usually don't include Softmax in the ONNX export for training
        # but for FPGA inference we might want it explicit.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

def export_mlp():
    model = SimpleMLP()
    model.eval()
    # Dummy input: Batch size 1, 8 input features
    dummy_input = torch.randn(1, 8)
    
    torch.onnx.export(model, dummy_input, "models/mlp.onnx",
                      input_names=['input'], output_names=['output'],
                      opset_version=13)
    print("[OK] Generated models/mlp.onnx")

# --- Model 2: Small CNN (Conv + Pool + Flat + Dense) ---
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        # Input: 1 channel (grayscale), 28x28 image
        # Conv: 1 -> 4 channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0) 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # After Conv (26x26) -> Pool (13x13) -> 4 channels
        # Flatten size = 4 * 13 * 13 = 676
        self.fc = nn.Linear(4 * 13 * 13, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.softmax(x)

def export_cnn():
    model = TinyCNN()
    model.eval()
    # Dummy input: Batch 1, 1 Channel, 28x28 (MNIST size)
    dummy_input = torch.randn(1, 1, 28, 28)
    
    torch.onnx.export(model, dummy_input, "models/cnn_mnist.onnx",
                      input_names=['input'], output_names=['output'],
                      opset_version=13)
    print("[OK] Generated models/cnn_mnist.onnx")

if __name__ == "__main__":
    export_mlp()
    export_cnn()