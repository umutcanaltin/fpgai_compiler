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
 - name: Marcel van Gerven
   orcid: 0000-0002-2206-9098
   affiliation: "1"
affiliations:
 - name: Donders Institute for Brain, Cognition and Behaviour
   index: 1

date: 22 July 2024
bibliography: paper.bib
---
# Summary
Deep learning (DL) has revolutionized diverse fields, but the computational demands of DL models present challenges for real-time inference and training. Field-programmable gate arrays (FPGAs) offer a solution due to their flexibility, yet designing for FPGAs is complex. We introduce a novel software framework converting ONNX models to FPGA implementations, optimized for on-chip inference and training. Leveraging ONNX's interoperability, our framework seamlessly integrates DL models with FPGA hardware, supporting various architectures. This advancement represents a comprehensive approach, addressing both inference and training, enabling DL practitioners to leverage FPGA hardware effectively.
![fpgai overview](hls.jpg)

# Statement of Need

Designing an FPGA with VHDL or HLS presents significant challenges due to the complexity of both the languages and the hardware. It requires a deep understanding of digital logic and FPGA architectures, making it challenging to translate high-level design concepts into efficient architecture. Optimizing performance and minimizing power consumption involves intricate trade-offs between design complexity and resource utilization. 

Debugging FPGA designs also can be challenging, as traditional software debugging techniques may not apply directly to hardware designs. Additionally, FPGA development requires a steep learning curve, demanding time and effort to acquire proficiency in designing FPGA architecture. Overall, designing hardware on FPGA requires expertise, patience, and a systematic approach to overcome these challenges.

In the realm of FPGA-based deep learning engines, most tools[source:1] [source:2] rely on LLVM IR representation for optimization, limiting hardware-specific optimizations. However, our tool takes a different route, focusing on a software-driven approach to hardware design. This unique perspective makes it easier to implement future optimizations tailored to deep learning architectures. Additionally, our engine stands out by supporting on-chip training for deep learning models, expanding its capabilities beyond conventional inference tasks. This approach not only promises improved performance but also opens doors for advancing FPGA-based deep learning systems.

In response to the growing demand within academic and industrial circles, there is a clear need for a simplified method for deploying DL models on FPGAs. This need is increasingly evident when surveying[source:3] the existing literature and observing the challenges faced by researchers and practitioners alike in effectively utilizing FPGA-accelerated DL solutions.

# Supported Features
Our engine provides support for essential DL operations such as feedforward processing and convolution. Feedforward operations enable the flow of data through neural network layers from input to output, facilitating tasks like classification and regression. Convolution operations, on the other hand, are fundamental for tasks involving spatial relationships, such as image processing and feature extraction. With our support for these operations, users can efficiently implement a wide range of deep learning models and applications, empowering them to address complex tasks effectively.

Our engine supports DMA (Direct Memory Access) usage for efficient data (image stream) transfer and BRAM (Block RAM) usage for storing weights and parameters of the model. DMA usage enables data transfer  between different memory locations, which is vital for accelerating data-intensive tasks. Meanwhile, BRAM usage ensures efficient utilization of on-chip memory resources, reducing access latency and improving overall performance. By leveraging DMA and BRAM usage, our engine optimizes resource utilization, maximizing hardware efficiency and facilitating faster and more efficient execution of deep learning tasks on FPGA platforms.

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
python3 main.py --onnx-file-name my_image_classifier.onnx --precision float  --dma-usage True
```

# Test Setup and Results
The testing feature in the fpgai engine allows users to configure and customize their testing models according to their specific requirements. User can use our examples and define test model in test/test_model.py file with any framework. The engine will use this model to compare network outputs with compiled HLS implementation.

The output report generated by the HLS compiler will be parsed and presented to the user, providing visibility into the hardware components utilized by the engine. This feature enables users to understand the resource consumption of the engine, empowering them to make informed decisions regarding hardware allocation and optimization strategies.

# Modular Usage and Optimization Configuration


## Usage with ONNX file

```python
from fpgai_engine import fpgai_engine

    fpgai_engine_object = fpgai_engine(onnx_file_name="my_onnx_file.onnx", precision = "float",vitis_hls_location="/tools/Xilinx/Vitis_HLS/2023.2", hls_project_name= "trial_project", hls_solution_name= "solution1",memory_option_weights="BRAM" , use_DMA=True,user_DDR=True)
    fpgai_engine_object.generate_hls_file(mode="inference",file_location= "/home/desktop/my_hls_project")
    fpgai_engine_object.generate_hls_file(mode="training",file_location= "/home/desktop/my_hls_project")

    results = fpgai_engine_object.compile_hls_file(file_location= "/home/desktop/my_hardware_project")
    print(results)
    fpgai_engine_object.compile_hardware_files(file_location= "/home/desktop/my_hardware_project")

```


## Usage without ONNX file as python library

Important notes for using python library:
- Have to create self.layers inside the model object and list of the layers should be in order. Library will compile this layers in order.

- Default activation function for a layer is linear function. You should decleare the defined activation function with an argumant when you call the layer class.

- User can export cpp or HLS files from model. HLS files includes pragmas differently from cpp files.

```python
from architectures.convolution_layer import ConvolutionLayer
from architectures.dense_layer import DenseLayer
from implementations import dense_layer_imp, conv_layer_imp
from utils.random_generator import weight_generator
from utils.model_to import model_to_hls, model_to_cpp

class My_Model():
    def __init__(self):
      self.layers = []
      #Layers must be in order to compile codes!
      #Input and output shape information for the layer must be with weights argument!
      self.layers.append(ConvolutionLayer(weights= weight_generator(layer_type = "convolution", input_shape=, shape=(5,10,10),precision = "float")))
      # Convolution Layer 1
      self.layers.append(DenseLayer(activation_function = "relu", precision = "float",name_of_layer = 'my_first_dense_layer', weights= weight_generator(layer_type = "dense",precision = "float", input_shape= 100, output_shape = 10)))
      # Dense Layer 1 with relu activation

      dense_layer_3 = DenseLayer(activation_function = "linear",precision = "float")
      random_generatad_weights = np.random(100,10)
      dense_layer_3.inject_weights(weights = random_generatad_weights)
      self.layers.append(dense_layer_3)
      # Dense Layer 3 with linear activation


    def get_hls_codes(self):
      return model_to_hls(self)
    
    def get_cpp_codes(self):
      return model_to_cpp(self)

```

## How to Change  or Add Functions (Activation example)

Linear Activation function without pointer decleration:
location: /activation/activation_functions.py
```python
        if(self.activation_function == "linear"):
            activation_string = self.precision + " activate_"+self.name_of_layer+ "( "+self.precision +" x) { return x; } \n"+self.precision + " dactivate_"+self.name_of_layer+"( "+self.precision +" x) { return 1; }"
```
Add new activation funtion named Leaky Relu:

```python
        if(self.activation_function == "linear"):
            activation_string = self.precision + " activate_"+self.name_of_layer+ "( "+self.precision +" x) { return x; } \n"+self.precision + " dactivate_"+self.name_of_layer+"( "+self.precision +" x) { return 1; }"
```

## How to Use After Compile the Model
```python
import numpy as np
from pynq import Xlnk
from pynq import Overlay


#inference with integer 20 input and 5 output without any mode (weight injection)
overlay = Overlay('deeplearn.bit')
dma = overlay.axi_dma
xlnk = Xlnk()
input_buffer = xlnk.cma_array(shape=(20,), dtype=np.uint32)
output_buffer = xlnk.cma_array(shape=(5,), dtype=np.uint32) 

```
## Python Driver for PYNQ model with modes(inject weughts or get trained weights)
You can directly use this code structure inside PYNQ!
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
dma.sendchannel.transfer(in_buffer) #in_buffer can also be weighs for injection to BRAMs depends on the mode selection
dma.recvchannel.transfer(out_buffer)
dma.sendchannel.wait()
dma.recvchannel.wait()

out_buffer     
```
# Future work

Our current engine supports basic architectures like feedforward and convolutional layers. Moving forward, we plan to include more complex models such as recurrent neural networks (RNNs). Additionally, our design is flexible enough to easily integrate new hardware optimization techniques. This adaptability ensures our engine remains at the forefront of FPGA-accelerated deep learning advancements in academia and industry.

# How to cite

If you are using fpgai for your work, please cite this paper.

# Acknowledgments

* This work is supported by the project Dutch Brain Interface Initiative (DBI2) with project number 024.005.022 of the research programme Gravitation which is (partly) financed by the Dutch Research Council (NWO).

# References
