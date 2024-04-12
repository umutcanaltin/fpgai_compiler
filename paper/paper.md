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
 - name: Nasir
   orcid: 0000-0002-0457-0153
   affiliation: "2"
 - name: Marcel
   affiliation: "3"
affiliations:
 - name: Donders Institute
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

# Optimization

# Performance

As an example of the performance improvements achieved with fpgai, we run the following configuration:

* A network model consist of 2 hidden layers with relu activation functions inside. (Inference and Training Results)
* Anetwork model consist of 5 hidden layers with relu and linear activation functions (Inference and Training Results)
* A network model consist of 2 convolutional layer and 3 hidden layers with relu and linear activation functions (Inference and Training Results)


The following tables shows the hardware components usage and timig results from simulation.

| Output $N_{side}$ | PySM 3        | PySM 2        |
|-------------------|---------------|---------------|
| 512               | 1m 0.7 GB     | 1m40s 1.45 GB |
| 1024              | 3m30s 2.3 GB  | 7m20s 5.5 GB  |
| 2048              | 16m10s 8.5 GB | Out of memory |

The models at $N_{side}=512$ have been tested to be equal given a relative tolerance of `1e-5`.

At the moment it is not very useful to run at resolutions higher than $N_{side}=512$ because there is no actual template signal at smaller scales. However, this demonstrates the performance improvements that will make working with higher resolution templates possible.

# Future work

PySM 3 opens the way to implement a new category of models at much higher resolution. However, instead of just upgrading the current models to smaller scales, we want to also update them with the latest knowledge of Galactic emission and gather feedback from each of the numerous CMB experiments. For this reason we are collaborating with the Panexperiment Galactic Science group to lead the development of the new class of models to be included in PySM 3.

# How to cite

If you are using PySM 3 for your work, please cite this paper for the software itself; for the actual emission modeling please also cite the original PySM 2 paper [@pysm17]. There will be a future paper on the generation of new PySM 3 astrophysical models.

# Acknowledgments

* This work was supported in part by NASA grant `80NSSC18K1487`.
* The software was tested, in part, on facilities run by the Scientific Computing Core of the Flatiron Institute.
* This research used resources of the National Energy Research Scientific Computing Center (NERSC), a U.S. Department of Energy Office of Science User Facility located at Lawrence Berkeley National Laboratory, operated under Contract No. `DE-AC02-05CH11231`.

# References
