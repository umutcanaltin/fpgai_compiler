#pragma once

void scale_bias_hls(
    const float* input,
    float* output,
    int count,
    float scale,
    float bias
);

void scale_bias_backward_input_hls(
    const float* grad_output,
    float* grad_input,
    int count,
    float scale,
    float bias
);
