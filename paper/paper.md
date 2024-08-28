---
title: 'FPGAI Engine for Neural Network Training and Inference'
tags:
  - acceleration
  - fpga
  - deep learning
  - system on chip
  - compilers
authors:
 - name: Umut Can Altin
   orcid: 0000-0001-6841-1058
   affiliation: "1"
 - name: Marcel van Gerven
   orcid: 0000-0002-2206-9098
   affiliation: "1"
affiliations:
 - name: Donders Institute for Brain, Cognition and Behaviour
   index: 1

date: 28 August 2024
bibliography: paper.bib
---
# Summary
Deep learning (DL) has revolutionized diverse fields, but the computational demands of DL models present challenges for real-time inference and training. Field-programmable gate arrays (FPGAs) offer a solution due to their flexibility, yet designing for FPGAs is complex. We introduce our FPGA for AI (FPGAI) engine as a novel software framework for converting AI models to FPGA implementations that are optimized for on-chip inference and training (see Figure 1). Leveraging ONNX's interoperability, our framework integrates DL models with FPGA hardware, supporting various architectures. This advancement represents a comprehensive approach, addressing both inference and training, enabling DL practitioners to leverage FPGA hardware effectively.

![fpgai overview](hls.jpg)
Figure 1: FPGAI Engine.

# Statement of need
Designing an FPGA with VHDL or HLS presents significant challenges due to the complexity of both the languages and the hardware. It requires a deep understanding of digital logic and FPGA architectures, making it challenging to translate high-level AI models and algorithms into efficient architectures. Optimizing performance and minimizing power consumption involves intricate trade-offs between design complexity and resource utilization. 

Debugging FPGA designs can also be challenging, as traditional software debugging techniques may not apply directly to hardware designs. Additionally, FPGA development requires a steep learning curve, demanding time and effort to acquire proficiency in designing FPGA architecture. Overall, designing hardware on FPGA requires expertise, patience, and a systematic approach to overcome these challenges.

In the realm of FPGA-based deep learning engines, most tools[source:1] [source:2] rely on LLVM IR representation for optimization, limiting hardware-specific optimizations. However, our tool takes a different route, focusing on a software-driven approach to hardware design. This unique perspective makes it easier to implement future optimizations tailored to deep learning architectures. Additionally, our engine stands out by supporting on-chip training for deep learning models, expanding its capabilities beyond conventional inference tasks. This approach not only promises improved performance but also opens doors for advancing FPGA-based deep learning systems.

In response to the growing demand within academic and industrial circles, there is a clear need for a simplified method for deploying DL models on FPGAs. This need is increasingly evident when surveying[source:3] the existing literature and observing the challenges faced by researchers and practitioners alike in effectively utilizing FPGA-accelerated DL solutions.

# Supported features
Our FPGAI engine provides support for essential DL operations such as feedforward processing and convolution. Feedforward operations enable the flow of data through neural network layers from input to output, facilitating tasks like classification and regression. Convolution operations, on the other hand, are fundamental for tasks involving spatial relationships, such as image processing and feature extraction. With our support for these operations, users can efficiently implement a wide range of deep learning models and applications, empowering them to address complex tasks effectively.

Our engine supports DMA (Direct Memory Access) usage for efficient data (image stream) transfer and BRAM (Block RAM) usage for storing weights and parameters of the model. DMA usage enables data transfer  between different memory locations, which is vital for accelerating data-intensive tasks. Meanwhile, BRAM usage ensures efficient utilization of on-chip memory resources, reducing access latency and improving overall performance. By leveraging DMA and BRAM usage, our engine optimizes resource utilization, maximizing hardware efficiency and facilitating faster and more efficient execution of deep learning tasks on FPGA platforms.

## Deployment of ONNX files via the console
The following example shows how to convert a Pytorch model to ONNX. The ONNX file is subsequently converted for deployment on FPGA.

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
python3 main.py --onnx-file-name my_image_classifier.onnx --precision float  --dma-usage True
```

## Deployment of ONNX files in Python
The following example shows how ONNX models can be deployed on FPGA in Python using the FPGAI engine.

```python
from fpgai_engine import fpgai_engine

# Read ONNX file
fpgai_engine_object = fpgai_engine(onnx_file_name="my_onnx_file.onnx", precision = "float", vitis_hls_location="/tools/Xilinx/Vitis_HLS/2023.2",

# Create project
hls_project_name="trial_project", hls_solution_name="solution1", memory_option_weights="BRAM", use_DMA=True,user_DDR=True)

# Generate HLS files for model inference and training
fpgai_engine_object.generate_hls_file(mode="inference", file_location="/home/desktop/my_hls_project")
fpgai_engine_object.generate_hls_file(mode="training", file_location="/home/desktop/my_hls_project")

# Compile HLS files
results = fpgai_engine_object.compile_hls_file(file_location="/home/desktop/my_hardware_project")

# Create CPP files to BIT file
fpgai_engine_object.compile_hardware_files(file_location="/home/desktop/my_hardware_project")
```

## Deployment of models via Python
The following example shows how DL models can be deployed on FPGA in Python using the FPGAI engine.

```python
from architectures.convolution_layer import ConvolutionLayer
from architectures.dense_layer import DenseLayer
from implementations import dense_layer_imp, conv_layer_imp
from utils.random_generator import weight_generator
from utils.model_to import model_to_hls, model_to_cpp

class MyModel():
    def __init__(self):
      self.layers = []

      # Layers must be in order to compile code!
      # Input and output shape information for the layer must be with weights argument!

      # Convolution layer 1
      self.layers.append(ConvolutionLayer(weights=weight_generator(layer_type="convolution", input_shape=10, shape=(5,10,10), precision="float")))

      # Dense layer 1 with ReLU activation
      self.layers.append(DenseLayer(activation_function="relu", precision="float", name_of_layer='my_first_dense_layer', weights=weight_generator(layer_type="dense", precision="float", input_shape=100, output_shape=10)))

      # Dense layer 3 with linear activation
      dense_layer_3 = DenseLayer(activation_function="linear", precision="float")
      random_generated_weights = np.random(100,10)
      dense_layer_3.inject_weights(weights=random_generated_weights)
      self.layers.append(dense_layer_3)

    def get_hls_codes(self):
      return model_to_hls(self)
    
    def get_cpp_codes(self):
      return model_to_cpp(self)

# Create CPP files to BIT file
model = MyModel()
model.compile(file_location="/home/desktop/my_hardware_project")
```

Important notes for using the Python library:
- One has to create self.layers inside the model object and the list of the layers should be in order. The library will compile these layers in order.
- The default activation function for a layer is a linear function. One should declare the defined activation function with an argument when calling the layer class.
- The user can export cpp or HLS files from the model. HLS files include pragmas differently from cpp files.

## How to change or add functions
The following example shows how functions can be changed or added.

Linear Activation function without pointer declaration:
location: /activation/activation_functions.py
```python
if(self.activation_function == "linear"):
  activation_string = self.precision + " activate_"+self.name_of_layer+ "( "+self.precision +" x) { return x; } \n"+self.precision + " dactivate_"+self.name_of_layer+"( "+self.precision +" x) { return 1; }"
```
Add new activation function named Leaky Relu:
```python
if(self.activation_function == "LRelu"):
  activation_string = self.precision + " activate_"+self.name_of_layer+ "( "+self.precision +" x)
  {
    if(0.01*x < x){
      return x;}
    else{
      return 0.01*x;
    }
  }
  \n"
```

Users have to declare the derivative of the activation function for backpropagation!

### UA: EXAMPLE

## FPGA deployment
The following example shows how to use the BIT file inside PYNQ for inference.

```python
import numpy as np
from pynq import Xlnk
from pynq import Overlay

# Inference with integer 20 input and 5 output without any mode (weight injection)
overlay = Overlay('deeplearn.bit')
dma = overlay.axi_dma
xlnk = Xlnk()
input_buffer = xlnk.cma_array(shape=(20,), dtype=np.uint32)
output_buffer = xlnk.cma_array(shape=(5,), dtype=np.uint32) 

```

The following more advanced example shows how to use the BIT file inside PYNQ for training in an object-oriented manner.
Users can directly use this code structure inside PYNQ. Check mode descriptions(import/export/etc.) before usage!
```python

class DeepLearnModeDriver(DefaultIP):
    def __init__(self, description):
        super().__init__(description=description)

    bindto = ['Xilinx:hls:deeplearn:1.0']
    @property
    def mode(self):
        return self.read(0x10)

    @mode.setter
    def mode(self, value):
        self.write(0x10, value)

from pynq import Xlnk
import numpy as np

xlnk = Xlnk()
in_buffer = xlnk.cma_array(shape=(5,), dtype=np.uint32)
out_buffer = xlnk.cma_array(shape=(5,), dtype=np.uint32)

for i in range(5):
    in_buffer[i] = i

deeplearn.mode = 1
dma.sendchannel.transfer(in_buffer) #in_buffer can also be weights for injection to BRAMs depending on the mode selection
dma.recvchannel.transfer(out_buffer) #out_buffer can be trained weights from BRAMs to your local computer depending on mode selection
dma.sendchannel.wait()
dma.recvchannel.wait()

out_buffer     
```

# Results
![Results Table](https://github.com/umutcanaltin/fpgai_compiler/blob/main/paper/results.png?raw=true)
Table 1: Performance and resource usage for running backpropagation on FPGA via the FPGAI engine.

Table 1 provides validation results for our FPGAI engine. Results show performance and resource usage when running feedforward and convolutional neural networks on a ZYNQ FPGA board (UA: ADD PL DETAILS). The feedforward neural network consists of 3x3x3 neurons. The convolutional neural network consists of 28x28->2 and 19x19x2->10 convolutional layers.

## Report generation
The output report generated by the HLS compiler will be parsed and presented to the user, providing visibility into the hardware components utilized by the engine. This feature enables users to understand the resource consumption of the engine, empowering them to make informed decisions regarding hardware allocation and optimization strategies.

UA:EXAMPLE REPORT

# Future work
Our current engine supports basic architectures like feedforward and convolutional layers. Moving forward, we plan to include more complex models such as recurrent neural networks (RNNs). Additionally, our design is flexible enough to easily integrate new hardware optimization techniques. This adaptability ensures our engine remains at the forefront of FPGA-accelerated deep learning advancements in academia and industry. The authors welcome contributions and collaborations to accelerate the impact and use of our FPGAI engine.

# How to cite
If you are using FPGAI for your work, please cite this paper.

# Acknowledgments
This work is supported by the project Dutch Brain Interface Initiative (DBI2) with project number 024.005.022 of the research programme Gravitation which is (partly) financed by the Dutch Research Council (NWO).

# References
