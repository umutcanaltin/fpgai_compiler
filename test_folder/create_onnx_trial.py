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

import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(8, 8, bias=True)
        self.fc1 = nn.Linear(8, 4, bias=True)

    def forward(self, tensor_x: torch.Tensor):
        tensor_x = self.fc0(tensor_x)
        output = self.fc1(tensor_x)
        return output

torch_model = MLPModel()
torch_input = torch.randn(1, 1, 1, 8)
onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
onnx_program.save("mlp.onnx")





class MyModel1(nn.Module):
    def __init__(self):
        super(MyModel1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 1 ,out_channels=1,kernel_size=(2, 2))
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
torch_model = MyModel1()
torch_input = torch.randn(1, 1, 5, 5)
onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
onnx_program.save("image_classifier_1.onnx")
