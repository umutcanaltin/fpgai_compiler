import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3)
        self.fc1 = nn.Linear(30*30*2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
torch_model = MyModel()
torch_input = torch.randn(1, 1, 32, 32)
onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
onnx_program.save("my_image_classifier.onnx")


import torch
import torch.nn as nn

class MLPModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.fc0 = nn.Linear(8, 8, bias=True)
      self.fc1 = nn.Linear(8, 4, bias=True)
      self.fc2 = nn.Linear(4, 2, bias=True)
      self.fc3 = nn.Linear(2, 2, bias=True)

  def forward(self, tensor_x: torch.Tensor):
      tensor_x = self.fc0(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      tensor_x = self.fc1(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      tensor_x = self.fc2(tensor_x)
      tensor_x = torch.sigmoid(tensor_x)
      output = self.fc3(tensor_x)
      return output

model = MLPModel()
tensor_x = torch.rand((97, 8), dtype=torch.float32)
onnx_program = torch.onnx.dynamo_export(model, tensor_x)
onnx_program.save("mlp.onnx")
