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

date: 30 October 2024
bibliography: paper.bib
---
# Summary
Deep learning (DL) has revolutionized diverse fields, but the computational demands of DL models present challenges for real-time inference and training. Field Programmable Gate Arrays (FPGAs) offer a solution due to their flexibility, yet designing for FPGAs is complex. We introduce our FPGA for AI (FPGAI) engine as a novel software framework for converting AI models to FPGA implementations that are optimized for on-chip inference and training (see Figure 1). Leveraging ONNX's interoperability, our framework integrates DL models with FPGA hardware, supporting various architectures. This advancement represents a comprehensive approach, addressing both inference and training, enabling DL practitioners to leverage FPGA hardware effectively.
![Visualisation of the engine structure](assets/hls.pdf)

# Statement of need
Designing an FPGA with VHDL or HLS presents significant challenges due to the complexity of both the languages and the hardware. It requires a deep understanding of digital logic and FPGA architectures, making it challenging to translate high-level AI models and algorithms into efficient architectures. Optimizing performance and minimizing power consumption involves intricate trade-offs between design complexity and resource utilization. 

Debugging FPGA designs can also be challenging, as traditional software debugging techniques may not apply directly to hardware designs. Additionally, FPGA development requires a steep learning curve, demanding time and effort to acquire proficiency in designing FPGA architecture. Overall, designing hardware on FPGA requires expertise, patience, and a systematic approach to overcome these challenges.

In the realm of FPGA-based deep learning engines, most tools[@Noronha_et_al_2018] [@Umuroglu_et_al_2017] rely on LLVM IR representation for optimization, limiting hardware-specific optimizations. However, our tool takes a different route, focusing on a software-driven approach to hardware design. This unique perspective makes it easier to implement future optimizations tailored to deep learning architectures. Additionally, our engine stands out by supporting on-chip training for deep learning models, expanding its capabilities beyond conventional inference tasks. This approach not only promises improved performance but also opens doors for advancing FPGA-based deep learning systems.

In response to the growing demand within academic and industrial circles, there is a clear need for a simplified method for deploying DL models on FPGAs. This need is increasingly evident when surveying[@Rahimifar_et_al_2022] the existing literature and observing the challenges faced by researchers and practitioners alike in effectively utilizing FPGA-accelerated DL solutions.

# Supported Features
Our FPGAI engine provides support for essential DL operations such as feedforward processing and convolution. Feedforward operations enable the flow of data through neural network layers from input to output, facilitating tasks like classification and regression. Convolution operations, on the other hand, are fundamental for tasks involving spatial relationships, such as image processing and feature extraction. With our support for these operations, users can efficiently implement a wide range of deep learning models and applications, empowering them to address complex tasks effectively.

Our engine supports DMA (Direct Memory Access) usage for efficient data (image stream) transfer and BRAM (Block RAM) usage for storing weights and parameters of the model. DMA usage enables data transfer  between different memory locations, which is vital for accelerating data-intensive tasks. Meanwhile, BRAM usage ensures efficient utilization of on-chip memory resources, reducing access latency and improving overall performance. By leveraging DMA and BRAM usage, our engine optimizes resource utilization, maximizing hardware efficiency and facilitating faster and more efficient execution of deep learning tasks on FPGA platforms.


## Supported Activation Functions

Our engine currently supports two primary activation functions essential for deep learning inference and training: **Linear** and **ReLU (Rectified Linear Unit)**.

1. **Linear Activation**: A straightforward function where the output is a scaled version of the input, useful for maintaining linearity in layers or during output generation.

2. **ReLU (Rectified Linear Unit)**: A widely used non-linear activation function defined as $f(x) = \max(0, x)$, effectively introducing non-linearity to the network and helping mitigate vanishing gradients during training.

These activation functions are optimized for hardware efficiency, ensuring low latency and resource utilization on FPGA platforms. Further support for additional functions is planned in future iterations of our engine.

## Supported Loss Functions

Our FPGA-based engine currently supports the **Mean Squared Error (MSE) Loss** function, a core metric for regression tasks that calculates the average squared differences between predicted and actual values. This function provides a smooth gradient, ideal for backpropagation and optimized for FPGA implementation, ensuring minimal latency and efficient resource usage.

The MSE Loss function is defined by:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

where $y_i$ represents the true values, and $\hat{y}_i$ represents the predicted values.


## Testing Feature for FPGA Inference and Testbench

The testing feature of our engine integrates PyTorch for inference alongside an HLS testbench, allowing simultaneous validation of outputs.

This setup enables users to execute inference on a neural network model within PyTorch while concurrently running an HLS-based testbench that mirrors the same computations. This dual approach facilitates real-time comparison of outputs from both environments, ensuring that the HLS implementation aligns with the PyTorch results. Users can easily identify discrepancies, optimize performance, and validate functionality across both platforms efficiently.


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
        x = F.relureturn 1(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

torch_model = MyModel()
torch_input = torch.randn(1, 1, 32, 32)
onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
onnx_program.save("my_image_classifier.onnx")
```

```console
python3 main.py --onnx-file-name my_image_classifier.onnx 
```

Before using the main.py file, the user must declare the input and output data types in the main file. This step is crucial for ensuring that the data passed to the model is correctly formatted, allowing for accurate inference and processing within the engine. By specifying these types, users can avoid runtime errors and enhance the compatibility of their data with the FPGA implementation.




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
The engine systematically generates five distinct files within the generate_files directory:

1. deeplearn.h: This header file encompasses the functions and parameter definitions essential for the engine's operation.
2. main.cpp: This serves as the primary C++ file designed for integration with Vitis HLS.
3. testbench.cpp: This file facilitates the testing features associated with Vitis HLS, enabling comprehensive output validation.
4. tcl_for_vitis.tcl: A TCL script designed to execute Vitis HLS without the graphical user interface, streamlining the compilation process.
5. tcl_for_vivado.tcl: This script automates the execution of Vivado, establishing necessary hardware configurations, such as Direct Memory Access (DMA) connections, and generating requisite hardware configurations.


After using the FPGAI compiler, you can directly run the generated files with the following script. It sources the Vitis HLS environment and executes the generated TCL script, automating the compilation:

```sh
#!/bin/bash

# Source the Vitis environment setup script
source /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh

# Run the Vitis HLS command with the TCL script
vitis_hls -f /home/umutcanaltin/Desktop/github_projects/fpgai_compiler/generated_tcl_vitis.tcl
```

To use the provided shell script with your FPGAI compiler's generated files, remember to adjust the paths to suit your specific system configuration. Update the <version> in settings64.sh and set the path to your generated_tcl_vitis.tcl file as necessary. This ensures the script sources the Vitis environment and runs without errors on your setup.

To switch to Vivado, use the same script, updating the source path to the Vivado environment setup (e.g., settings64.sh for Vivado) and replacing the TCL file path with generated_tcl_vivado.tcl for the Vivado-specific commands.

# Results
![Visualisation of performance and resource usage](assets/results.pdf)
Table 1: Performance and resource usage for running backpropagation on FPGA via the FPGAI engine.

Table 1 provides validation results for our FPGAI engine. Results show performance and resource usage when running feedforward and convolutional neural networks on a ZYNQ FPGA board. The feedforward neural network consists of 3x3x3 neurons. The convolutional neural network consists of 28x28->2 and 19x19x2->10 convolutional layers.

## Report generation
The output report generated by the HLS compiler will be parsed and presented to the user, providing visibility into the hardware components utilized by the engine. This feature enables users to understand the resource consumption of the engine, empowering them to make informed decisions regarding hardware allocation and optimization strategies. User can find this feature in connections tab.


# Future work
Our current engine supports basic architectures like feedforward and convolutional layers. Moving forward, we plan to include more complex models such as recurrent neural networks (RNNs). Additionally, our design is flexible enough to easily integrate new hardware optimization techniques. This adaptability ensures our engine remains at the forefront of FPGA-accelerated deep learning advancements in academia and industry. The authors welcome contributions and collaborations to accelerate the impact and use of our FPGAI engine.

# How to cite
If you are using FPGAI for your work, please cite this paper.

# Acknowledgments
This work is supported by the project Dutch Brain Interface Initiative (DBI2) with project number 024.005.022 of the research programme Gravitation which is (partly) financed by the Dutch Research Council (NWO).

# References
