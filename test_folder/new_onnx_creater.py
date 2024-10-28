import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime

class MyModel1(nn.Module):
    def __init__(self):
        super(MyModel1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2))
        self.fc1 = nn.Linear(16, 12)
        self.fc2 = nn.Linear(12, 5)
        self.fc3 = nn.Linear(5, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create and initialize model
torch_model = MyModel1()
torch_input = torch.randn(1, 1, 5, 5)

# Export model using torch.onnx.export
onnx_path = "image_classifier_1_exported.onnx"
torch.onnx.export(torch_model, 
                  torch_input, 
                  onnx_path, 
                  export_params=True, 
                  opset_version=11,  # You may need to adjust this version
                  do_constant_folding=True,
                  input_names=['input'], 
                  output_names=['output'])

# Load and check the ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# Print model graph to inspect structure and weights
print(onnx.helper.printable_graph(onnx_model.graph))

# Verify with ONNX Runtime
ort_session = onnxruntime.InferenceSession(onnx_path)
ort_inputs = {ort_session.get_inputs()[0].name: torch_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)
print("ONNX Model Output:", ort_outs)






class MyModel2(nn.Module):
    def __init__(self):
        super(MyModel2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2))


    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)

        return x

# Create and initialize model
torch_model = MyModel2()
torch_input = torch.randn(1, 1, 5, 5)

# Export model using torch.onnx.export
onnx_path = "image_classifier_2_exported.onnx"
torch.onnx.export(torch_model, 
                  torch_input, 
                  onnx_path, 
                  export_params=True, 
                  opset_version=11,  # You may need to adjust this version
                  do_constant_folding=True,
                  input_names=['input'], 
                  output_names=['output'])

# Load and check the ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# Print model graph to inspect structure and weights
print(onnx.helper.printable_graph(onnx_model.graph))

# Verify with ONNX Runtime
ort_session = onnxruntime.InferenceSession(onnx_path)
ort_inputs = {ort_session.get_inputs()[0].name: torch_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)
print("ONNX Model Output:", ort_outs)



class MyModel3(nn.Module):
    def __init__(self):
        super(MyModel3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2))


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)

        return x

# Create and initialize model
torch_model = MyModel3()
torch_input = torch.randn(1, 1, 5, 5)

# Export model using torch.onnx.export
onnx_path = "image_classifier_2_exported.onnx"
torch.onnx.export(torch_model, 
                  torch_input, 
                  onnx_path, 
                  export_params=True, 
                  opset_version=11,  # You may need to adjust this version
                  do_constant_folding=True,
                  input_names=['input'], 
                  output_names=['output'])

# Load and check the ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

# Print model graph to inspect structure and weights
print(onnx.helper.printable_graph(onnx_model.graph))

# Verify with ONNX Runtime
ort_session = onnxruntime.InferenceSession(onnx_path)
ort_inputs = {ort_session.get_inputs()[0].name: torch_input.numpy()}
ort_outs = ort_session.run(None, ort_inputs)
print("ONNX Model Output:", ort_outs)