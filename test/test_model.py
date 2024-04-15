import torch
import torch.nn as nn
import torch.nn.functional as F

class Test_Model(nn.Module):
    def __init__(self):
        super(Test_Model, self).__init__()
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



class Test_Onnx_Model():
    def __init__(self, onnx_file_name = "default.onnx"):
        self.onnx_file_name = onnx_file_name

