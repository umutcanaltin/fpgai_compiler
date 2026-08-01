#pragma once

void scale_bias_hls(
    const float* input,
    float* output,
    int count,
    float scale,
    float bias
);
