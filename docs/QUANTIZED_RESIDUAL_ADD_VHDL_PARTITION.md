# Quantized residual Add VHDL partition

FPGAI can partition the maintained PTQ residual CNN so the convolution body remains in Vitis HLS while the residual Add and terminal ReLU execute in VHDL.

The physical DAG explicitly duplicates the packed input stream. One copy enters the HLS Conv/ReLU/Conv body and the other is retained as the residual skip branch. The VHDL Add consumes the HLS main branch and the skip branch with grouped ready/valid semantics.

The Add VHDL is generated from the calibrated tensor contracts and the same quantized Add lowering metadata used by the integer HLS reference: independent input zero-points, multiplier/shift requantization, output zero-point, rounding, and signed int8 saturation. The terminal VHDL ReLU then applies the final activation.

Use `configs/examples/quantized_residual_cnn_ptq_mixed_add.yml`. Reports distinguish the HLS-body reference, residual-sum reference, and full-model reference.
