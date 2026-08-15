#pragma once

void split_scale_tensor_ports_hls(
    const float* input,
    float* identity,
    float* scaled,
    int count,
    float scale
);
