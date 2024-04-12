---
title: 'fpgai, Engine for Neural Network Training and Inference'
tags:
  - acceleration
  - fpga
  - deep learning
  - engine
authors:
 - name: Umut Can Altin
   orcid: 0000-0001-6841-1058
   affiliation: "1"
 - name: Nasir Ahmad
   orcid: 0000-0002-0457-0153
   affiliation: "2"
 - name: Marcel van Gerven
   affiliation: "3"
affiliations:
 - name: Donders Institute for Brain, Cognition and Behaviour
   index: 1

date: 22 July 2024
bibliography: paper.bib
---
# Summary
Deep learning (DL) has revolutionized diverse fields, but the computational demands of DL models present challenges for real-time inference and training. Field-programmable gate arrays (FPGAs) offer a solution due to their flexibility, yet designing for FPGAs is complex. We introduce a novel software framework converting ONNX models to FPGA implementations, optimized for on-chip inference and training. Leveraging ONNX's interoperability, our framework seamlessly integrates DL models with FPGA hardware, supporting various architectures.This advancement represents a comprehensive approach, addressing both inference and training, enabling DL practitioners to leverage FPGA hardware effectively.
![fpgai overview](hls.jpg)

# Statement of Need

Designing an FPGA with VHDL or HLS presents significant challenges due to the complexity of both the languages and the hardware. Requires a deep understanding of digital logic and FPGA architecture, making it challenging to translate high-level design concepts into efficient architecture. Optimizing performance and minimizing power consumption involves intricate trade-offs between design complexity and resource utilization. 

Debugging FPGA designs also can be challenging, as traditional software debugging techniques may not apply directly to hardware designs. Additionally, FPGA development requires a steep learning curve, demanding time and effort to acquire proficiency in designing FPGA architecture. Overall, designing hardware on FPGA requires expertise, patience, and a systematic approach to overcome these challenges.

In response to the widespread demand within academic and industrial spheres, there is an evident necessity for a simplified approach to deploying DL on FPGAs. We addresses this requirement by developing a engine capable of efficiently converting deep learning models from ONNX format to hardware design for FPGA execution. Leveraging the user-friendly ONNX format, renowned for its adaptability across diverse DL models.

# Supported Features
Our engine provides support for essential DL operations such as feedforward and convolution. Feedforward operations enable the flow of data through neural network layers from input to output, facilitating tasks like classification and regression. Convolution operations, on the other hand, are fundamental for tasks involving spatial relationships, such as image processing and feature extraction. With our support for these operations, users can efficiently implement a wide range of deep learning models and applications, empowering them to address complex tasks effectively.

Our engine supports DMA (Direct Memory Access) usage for efficient data (image stream) transfer and BRAM (Block RAM) usage for storing weights and parameters of the model. DMA usage enables data transfers between different memory locations which is vital for accelerating data-intensive tasks. Meanwhile, BRAM usage ensures efficient utilization of on-chip memory resources, reducing access latency and improving overall performance. By leveraging DMA and BRAM usage, our engine optimizes resource utilization, maximizing hardware efficiency and facilitating faster and more efficient execution of deep learning tasks on FPGA platforms.

# Example Usage with Pytorch 
```python
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
```

```console
main.py --onnx-file deneme.onnx
```

# Test Setup and Results
The testing feature in the fpgai engine allows users to configure and customize their testing models according to their specific requirements. User can use our examples and define test model in test/test_model.py file with any framework. Engine will use this model to compare network outputs with compiled HLS implementation.


# Future work

PySM 3 opens the way to implement a new category of models at much higher resolution. However, instead of just upgrading the current models to smaller scales, we want to also update them with the latest knowledge of Galactic emission and gather feedback from each of the numerous CMB experiments. For this reason we are collaborating with the Panexperiment Galactic Science group to lead the development of the new class of models to be included in PySM 3.

# How to cite

If you are using fpgai for your work, please cite this paper for the software itself.

# Acknowledgments

* This work was supported in part by DBI2 grant.


# References
