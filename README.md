FPGAI Engine for Neural Network Training and Inference
Overview
FPGAI Engine is an innovative framework designed to enable seamless deployment, training, and inference of neural networks on FPGA hardware. By converting models from high-level AI frameworks to optimized FPGA implementations, the engine empowers deep learning researchers to leverage FPGAs for high-performance, resource-efficient computing.

Key Features
Support for ONNX Models: Integrates with ONNX for streamlined AI model deployment on FPGAs.
Efficient Data Transfer: Utilizes DMA for input/output data handling and BRAM for model weights, optimizing on-chip resources.
Testing Support: Includes a PyTorch-based inference setup and an HLS testbench for concurrent output validation.
Supported Functions: Implements linear and ReLU activation functions, with MSE loss for regression tasks.
Getting Started
Clone the repository and navigate to the project directory.
Run the fpgai_compiler to generate necessary files for Vitis HLS.
Execute the following script to compile in Vitis HLS, adjusting the file paths to your environment:


```sh
#!/bin/bash
source /path/to/Xilinx/Vitis_HLS/2023.2/settings64.sh
vitis_hls -f /path/to/generated_tcl_vitis.tcl

```
Documentation
Full documentation and examples are available in the paper directory. This includes instructions for advanced deployment with PYNQ, detailed usage of the engine’s features, and integration tips.

Contributing
Contributions are welcome. Please see CONTRIBUTING.md for guidelines and open an issue for any suggestions or feedback.

